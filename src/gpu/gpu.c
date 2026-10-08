/* GPU section driver: sizes, calibration, timing policy and reporting. The
 * backends (gpu_metal.m, gpu_vulkan.c) build resources and time command
 * buffers; see gpu_backend.h. */
#include "gpu/gpu.h"
#include "gpu/gpu_backend.h"
#include "core/report.h"
#include "core/timer.h"

#include <stdio.h>
#include <string.h>

#define TARGET_DISPATCH_NS 20e6    /* calibrated work per dispatch */
#define TARGET_CB_NS       100e6   /* dispatches per command buffer -> ~100 ms */
#define CAL_MIN_NS         2e6     /* shortest run trusted for calibration */
#define MAX_DISPATCH       64
#define TRIALS             3       /* timed command buffers; the fastest is reported */

static const u64 lat_sizes[] = {
    16ULL << 10, 256ULL << 10, 4ULL << 20, 16ULL << 20, 64ULL << 20, 256ULL << 20,
};

typedef struct {
    sb_gpu_dev  *dev;
    sb_gpu_caps  caps;
    bool         warm;    /* section-level GPU warmup done */
} ctx;

static f64 clampf(f64 v, f64 lo, f64 hi) {
    if (v < lo) { return lo; }
    if (v > hi) { return hi; }
    return v;
}

/* Runs until SB_WARMUP_NS of host time has passed, so the GPU clock has ramped. */
static sb_status_e warmup(ctx *c, u32 n, u32 nd) {
    u64 t0 = sb_timer_now_ns();
    while (sb_timer_now_ns() - t0 < SB_WARMUP_NS) {
        f64 ns = 0;
        sb_status_e s = sb_gpu_run(c->dev, n, nd, &ns);
        if (s != SB_OK) { return s; }
    }
    c->warm = true;
    return SB_OK;
}

/* Best-of-TRIALS ns per dispatch for a fixed `n`; `t1` is one dispatch's time. */
static sb_status_e trials(ctx *c, u32 n, f64 t1, f64 *out_ns) {
    u32 nd = (u32)clampf(TARGET_CB_NS / (t1 > 1.0 ? t1 : 1.0), 1, MAX_DISPATCH);
    if (!c->warm) {
        sb_status_e s = warmup(c, n, nd);
        if (s != SB_OK) { return s; }
    }
    f64 best = 0;
    for (u32 i = 0; i < TRIALS; i++) {
        f64 ns = 0;
        sb_status_e s = sb_gpu_run(c->dev, n, nd, &ns);
        if (s != SB_OK) { return s; }
        if (i == 0 || ns < best) { best = ns; }
    }
    if (best <= 0) { return SB_ERR_RANGE; }
    *out_ns = best / nd;
    return SB_OK;
}

/* Scalable kernels: grow `n` until a dispatch is long enough to time, scale it
 * to TARGET_DISPATCH_NS, then time. Returns the chosen n and ns per dispatch. */
static sb_status_e measure_scaled(ctx *c, u32 n_start, u32 n_max, u32 *out_n, f64 *out_ns) {
    u32 n  = n_start;
    f64 t  = 0;
    for (;;) {
        sb_status_e s = sb_gpu_run(c->dev, n, 1, &t);
        if (s != SB_OK) { return s; }
        if (t >= CAL_MIN_NS || n >= n_max) { break; }
        f64 grow = t > 0 ? CAL_MIN_NS * 1.5 / t : 16.0;
        n = (u32)clampf((f64)n * clampf(grow, 2.0, 1024.0), 1, n_max);
    }
    if (!c->warm) {    /* calibrating on a cold GPU picks too little work */
        sb_status_e s = warmup(c, n, 1);
        if (s == SB_OK) { s = sb_gpu_run(c->dev, n, 1, &t); }
        if (s != SB_OK) { return s; }
    }
    /* Scale to the target; repeat once if the scaled run misses it by > 2x. */
    for (u32 pass = 0; pass < 2; pass++) {
        n = (u32)clampf((f64)n * TARGET_DISPATCH_NS / (t > 1.0 ? t : 1.0), 1, n_max);
        sb_status_e s = sb_gpu_run(c->dev, n, 1, &t);
        if (s != SB_OK) { return s; }
        if (t > TARGET_DISPATCH_NS / 2 && t < TARGET_DISPATCH_NS * 2) { break; }
    }
    *out_n = n;
    return trials(c, n, t, out_ns);
}

/* Fixed-work kernels (one dispatch reads a whole buffer/texture). */
static sb_status_e measure_fixed(ctx *c, u32 n, f64 *out_ns) {
    f64 t = 0;
    for (u32 i = 0; i < 2; i++) {    /* first run faults pages in / fills caches */
        sb_status_e s = sb_gpu_run(c->dev, n, 1, &t);
        if (s != SB_OK) { return s; }
    }
    return trials(c, n, t, out_ns);
}

static void report_status(const char *test, sb_status_e s) {
    if (s == SB_ERR_UNSUPPORTED) { sb_report_skip(test, "not supported by this device"); }
    else                         { sb_report_error(test, s); }
}

/* --- Compute --- */

static void run_compute(ctx *c, sb_gpu_kernel_e k, const char *test) {
    if (k == SB_GPU_K_FP16 && !c->caps.fp16) {
        sb_report_skip(test, "no half-precision arithmetic");
        return;
    }
    sb_gpu_test t = {
        .kernel  = k,
        .threads = c->caps.is_cpu ? (1U << 14) : (1U << 22),
    };
    sb_status_e s = sb_gpu_prepare(c->dev, &t);
    u32 n  = 0;
    f64 ns = 0;
    if (s == SB_OK) { s = measure_scaled(c, 1, 1U << 20, &n, &ns); }
    sb_gpu_release(c->dev);
    if (s != SB_OK) {
        report_status(test, s);
        return;
    }
    f64 per_iter = k == SB_GPU_K_INT32 ? SB_GPU_INT_OPS_ITER : SB_GPU_FMA_OPS_ITER;
    f64 ops      = (f64)t.threads * (f64)n * per_iter;
    sb_report_value(test, ops / ns, k == SB_GPU_K_INT32 ? "GOPS" : "GFLOPS", SB_KIND_PEAK);
}

/* --- Memory bandwidth --- */

static u64 bw_bytes(const ctx *c) {
    u64 want = c->caps.is_cpu ? (64ULL << 20) : (1ULL << 30);
    while (want > c->caps.max_buffer && want > (1ULL << 20)) { want >>= 1; }
    return want;
}

static void run_bandwidth(ctx *c, sb_gpu_kernel_e k, const char *label) {
    u64  bytes = bw_bytes(c);
    char sz[32], test[64];
    sb_fmt_size(bytes, sz, sizeof(sz));
    snprintf(test, sizeof(test), "%s, %s", label, sz);
    if (k == SB_GPU_K_BW_HOST && c->caps.unified) {
        sb_report_skip(test, "unified memory: same DRAM as device-local");
        return;
    }
    sb_gpu_test t = {
        .kernel  = k,
        .threads = c->caps.is_cpu ? (1U << 14) : (1U << 20),
        .bytes   = bytes,
    };
    sb_status_e s  = sb_gpu_prepare(c->dev, &t);
    f64         ns = 0;
    if (s == SB_OK) { s = measure_fixed(c, (u32)(bytes / 16), &ns); }
    sb_gpu_release(c->dev);
    if (s != SB_OK) {
        report_status(test, s);
        return;
    }
    sb_report_value(test, (f64)bytes / ns, "GB/s", SB_KIND_PEAK);
}

/* --- Latency: single random cycle over nodes SB_GPU_LAT_STRIDE bytes apart --- */

static u64 splitmix64(u64 *state) {
    u64 z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Fills words[0..bytes/4) with a Sattolo cycle: word i*stride holds the index
 * of the next node's word. Starting at word 0 visits every node once. */
static sb_status_e build_chain(u64 bytes, u32 **out) {
    u64 nodes  = bytes / SB_GPU_LAT_STRIDE;
    u32 stride = SB_GPU_LAT_STRIDE / 4;
    if (nodes < 2 || bytes / 4 > UINT32_MAX) { return SB_ERR_INVALID; }
    u32 *words = calloc(bytes / 4, sizeof(u32));
    u32 *perm  = SB_MALLOC(nodes * sizeof(u32));
    if (words == NULL || perm == NULL) {
        SB_FREE(words);
        SB_FREE(perm);
        return SB_ERR_NOMEM;
    }
    for (u64 i = 0; i < nodes; i++) { perm[i] = (u32)i; }
    u64 seed = 0x5EED5EEDULL ^ bytes;
    for (u64 i = nodes - 1; i > 0; i--) {
        u64 j   = splitmix64(&seed) % i;    /* j < i: Sattolo, one cycle */
        u32 tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }
    for (u64 i = 0; i < nodes; i++) { words[i * stride] = perm[i] * stride; }
    SB_FREE(perm);
    *out = words;
    return SB_OK;
}

static void run_latency(ctx *c, u64 bytes) {
    char sz[32], test[64];
    sb_fmt_size(bytes, sz, sizeof(sz));
    snprintf(test, sizeof(test), "Chase %s", sz);
    if (bytes > c->caps.max_buffer) {
        sb_report_skip(test, "larger than the device's buffer limit");
        return;
    }
    u32 *chain = NULL;
    sb_status_e s = build_chain(bytes, &chain);
    if (s != SB_OK) {
        sb_report_error(test, s);
        return;
    }
    sb_gpu_test t = {
        .kernel  = SB_GPU_K_LATENCY,
        .threads = 1,
        .bytes   = bytes,
        .chain   = chain,
    };
    s = sb_gpu_prepare(c->dev, &t);
    SB_FREE(chain);
    u32 n  = 0;
    f64 ns = 0;
    if (s == SB_OK) { s = measure_scaled(c, 1024, 1U << 30, &n, &ns); }
    sb_gpu_release(c->dev);
    if (s != SB_OK) {
        report_status(test, s);
        return;
    }
    sb_report_value(test, ns / (f64)n, "ns", SB_KIND_MEASURED);
}

/* --- Threadgroup memory --- */

static void run_tgmem(ctx *c) {
    const char *test = "Read, float4 conflict-free";
    sb_gpu_test t = {
        .kernel  = SB_GPU_K_TGMEM,
        .threads = c->caps.is_cpu ? (1U << 12) : (1U << 20),
    };
    sb_status_e s = sb_gpu_prepare(c->dev, &t);
    u32 n  = 0;
    f64 ns = 0;
    if (s == SB_OK) { s = measure_scaled(c, 1, 1U << 20, &n, &ns); }
    sb_gpu_release(c->dev);
    if (s != SB_OK) {
        report_status(test, s);
        return;
    }
    f64 bytes = (f64)t.threads * (f64)n * SB_GPU_TG_BYTES_ITER;
    sb_report_value(test, bytes / ns, "GB/s", SB_KIND_PEAK);
}

/* --- Texture --- */

static void run_tex_cached(ctx *c) {
    char test[64];
    u64  bytes = (u64)SB_GPU_TEX_CACHED * SB_GPU_TEX_CACHED * 4;
    char sz[32];
    snprintf(test, sizeof(test), "Cache-resident, %s", sb_fmt_size(bytes, sz, sizeof(sz)));
    sb_gpu_test t = {
        .kernel  = SB_GPU_K_TEX_CACHED,
        .threads = c->caps.is_cpu ? (1U << 12) : (1U << 20),
        .tex_dim = SB_GPU_TEX_CACHED,
    };
    sb_status_e s = sb_gpu_prepare(c->dev, &t);
    u32 n  = 0;
    f64 ns = 0;
    if (s == SB_OK) { s = measure_scaled(c, 1, 1U << 20, &n, &ns); }
    sb_gpu_release(c->dev);
    if (s != SB_OK) {
        report_status(test, s);
        return;
    }
    f64 samples = (f64)t.threads * (f64)n * 2.0;    /* two samples per loop iteration */
    sb_report_value(test, samples / ns, "Gtexel/s", SB_KIND_PEAK);
}

static void run_tex_stream(ctx *c) {
    u32 dim = c->caps.is_cpu ? 1024 : 8192;
    while (dim > c->caps.max_tex_dim && dim > 256) { dim >>= 1; }
    u64  bytes = (u64)dim * dim * 4;
    char sz[32], test[64];
    snprintf(test, sizeof(test), "Streaming, %s", sb_fmt_size(bytes, sz, sizeof(sz)));
    sb_gpu_test t = {
        .kernel  = SB_GPU_K_TEX_STREAM,
        .threads = dim * (dim / SB_GPU_TEX_ROWS),
        .tex_dim = dim,
    };
    sb_status_e s  = sb_gpu_prepare(c->dev, &t);
    f64         ns = 0;
    if (s == SB_OK) { s = measure_fixed(c, SB_GPU_TEX_ROWS / 2, &ns); }
    sb_gpu_release(c->dev);
    if (s != SB_OK) {
        report_status(test, s);
        return;
    }
    sb_report_value(test, (f64)bytes / ns, "GB/s", SB_KIND_MEASURED);
}

static sb_status_e gpu_run(void) {
    ctx c = {0};
    sb_status_e s = sb_gpu_open(&c.dev, &c.caps);
    if (s != SB_OK) {
        sb_report_skip("GPU", s == SB_ERR_NOTFOUND || s == SB_ERR_UNSUPPORTED ? "no usable GPU device"
                                                                                : sb_status_str(s));
        return SB_OK;
    }
    sb_report_info("Device: %s (%s)%s", c.caps.name, c.caps.api, c.caps.is_cpu ? " [software: work reduced]" : "");
    for (u32 i = 0; i < SB_ARRAY_LEN(c.caps.notes); i++) {
        if (c.caps.notes[i][0] != '\0') { sb_report_info("%s", c.caps.notes[i]); }
    }
    sb_report_info("Timing: %s; fastest of %d command buffers of ~100 ms after calibration",
                   c.caps.gpu_timer ? "GPU timestamps" : "host wall time (no GPU timestamps)", TRIALS);

    sb_report_group("Compute (32 independent chains per thread)");
    run_compute(&c, SB_GPU_K_FP32, "FP32 FMA (float2)");
    run_compute(&c, SB_GPU_K_FP16, "FP16 FMA (half2)");
    run_compute(&c, SB_GPU_K_INT32, "INT32 multiply-add");

    sb_report_group("Memory read bandwidth (coalesced float4)");
    run_bandwidth(&c, SB_GPU_K_BW_DEVICE, "Device-local");
    run_bandwidth(&c, SB_GPU_K_BW_HOST, "Host-visible");

    sb_report_group("Memory latency (1 thread, 128 B stride, random cycle)");
    for (u32 i = 0; i < SB_ARRAY_LEN(lat_sizes); i++) {
        if (c.caps.is_cpu && lat_sizes[i] > (16ULL << 20)) { break; }
        run_latency(&c, lat_sizes[i]);
    }

    sb_report_group("Threadgroup / shared memory");
    run_tgmem(&c);

    sb_report_group("Texture sampling (RGBA8, bilinear)");
    run_tex_cached(&c);
    run_tex_stream(&c);

    sb_gpu_close(c.dev);
    return SB_OK;
}

const sb_section sb_section_gpu = {
    .name       = "gpu",
    .title      = "GPU",
    .help       =
        "  Metal on macOS, Vulkan 1.1 on Linux; the same kernels and rows on both.\n"
        "  Timed with GPU timestamps over ~100 ms command buffers (fastest of 3).\n"
        "  Compute: FMA on float2/half2 with 32 independent chains per thread, and\n"
        "  INT32 multiply-add with runtime operands.\n"
        "  Bandwidth: coalesced grid-stride float4 reads of a 1 GiB buffer;\n"
        "  host-visible system memory is measured only where it differs from\n"
        "  device-local memory (discrete GPUs).\n"
        "  Latency: one thread chasing a random cycle of 128 B-spaced nodes,\n"
        "  16 KiB to 256 MiB (includes TLB misses at large sizes).\n"
        "  Threadgroup memory: conflict-free, SIMD-aligned float4 reads (best case;\n"
        "  misaligned or strided patterns are several times slower).\n"
        "  Texture: bilinear RGBA8 at 1 texel per thread from a 16 KiB texture\n"
        "  (filter rate), and a 256 MiB texture read once (texture path from DRAM).\n"
        "  Vulkan prefers a discrete GPU; SB_GPU_DEVICE=<index|name> overrides.\n",
    .run        = gpu_run,
    .repeatable = true,
};
