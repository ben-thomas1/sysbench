/* Bandwidth: single-thread read sweep, single-thread stores, and DRAM read and
 * copy bandwidth with several threads (time-based, sb_par_run). */
#include "mem/mem_internal.h"
#include "core/platform.h"
#include "core/report.h"
#include "core/thread.h"

#include <stdio.h>
#include <string.h>

#define READ_MIN_BYTES  (4ULL << 10)
#define PAR_BLOCK       (4ULL << 20)      /* bytes per sb_par_run work call */
#define PAR_WINDOW_NS   500'000'000ULL

static const u64 store_sizes[] = { 32ULL << 10, 256ULL << 10, 4ULL << 20, 64ULL << 20, 512ULL << 20 };

/* --- Single thread --- */

typedef struct {
    void  *buf;
    size_t bytes;
} bw_ctx;

static void read_rep(void *ctx, u64 reps) {
    bw_ctx *c = ctx;
    mem_kern_read(c->buf, c->bytes, reps);
}

static void store128_rep(void *ctx, u64 reps) {
    bw_ctx *c = ctx;
    mem_kern_store128(c->buf, c->bytes, reps);
    mem_kern_drain();
}

static void store256_rep(void *ctx, u64 reps) {
    bw_ctx *c = ctx;
    mem_kern_store256(c->buf, c->bytes, reps);
    mem_kern_drain();
}

static void store_nt_rep(void *ctx, u64 reps) {
    bw_ctx *c = ctx;
    mem_kern_store_nt(c->buf, c->bytes, reps);
    mem_kern_drain();
}

/* GB/s (10^9 B/s) of one pass of `bytes` repeated by fn. */
static sb_status_e measure_gbps(mem_rep_fn fn, void *buf, size_t bytes, f64 *out) {
    bw_ctx      c  = { .buf = buf, .bytes = bytes };
    f64         ns = 0;
    sb_status_e s  = mem_time_best(fn, &c, &ns);
    if (s != SB_OK) { return s; }
    *out = (f64)bytes / ns;
    return SB_OK;
}

static void read_sweep(const mem_arena *a) {
    char title[96], name[64], sz[24];
    snprintf(title, sizeof(title), "Read bandwidth: 1 thread, %s loads, 8 chains", mem_kern_isa);
    sb_report_group(title);
    for (u64 bytes = READ_MIN_BYTES; bytes <= a->bytes; bytes *= 4) {
        snprintf(name, sizeof(name), "Read %s", sb_fmt_size(bytes, sz, sizeof(sz)));
        f64 v = 0;
        sb_status_e s = measure_gbps(read_rep, a->base, (size_t)bytes, &v);
        if (s == SB_OK) { sb_report_value(name, v, "GB/s", SB_KIND_PEAK); }
        else            { sb_report_error(name, s); }
    }
}

static void store_sweep(const mem_arena *a) {
    char name[64], sz[24];
    sb_report_group("Store bandwidth: 1 thread, best of 128 and 256 B/iter loops");
    sb_report_info("No barrier between passes; one dsb sy / mfence before the timer stops.");
    f64 big128 = 0, big256 = 0;
    u64 big    = 0;
    for (size_t i = 0; i < SB_ARRAY_LEN(store_sizes); i++) {
        u64 bytes = store_sizes[i];
        if (bytes > a->bytes) { break; }
        snprintf(name, sizeof(name), "Store %s", sb_fmt_size(bytes, sz, sizeof(sz)));
        f64 v128 = 0, v256 = 0;
        sb_status_e s = measure_gbps(store128_rep, a->base, (size_t)bytes, &v128);
        if (s == SB_OK) { s = measure_gbps(store256_rep, a->base, (size_t)bytes, &v256); }
        if (s != SB_OK) {
            sb_report_error(name, s);
            continue;
        }
        sb_report_value(name, v128 > v256 ? v128 : v256, "GB/s", SB_KIND_PEAK);
        big128 = v128;
        big256 = v256;
        big    = bytes;
    }
    if (big == 0) { return; }

    sb_fmt_size(big, sz, sizeof(sz));
    snprintf(name, sizeof(name), "Store %s, 128 B/iter loop", sz);
    sb_report_value(name, big128, "GB/s", SB_KIND_MEASURED);
    snprintf(name, sizeof(name), "Store %s, 256 B/iter loop", sz);
    sb_report_value(name, big256, "GB/s", SB_KIND_MEASURED);
#if defined(__aarch64__)
    sb_report_info("Apple cores skip reading a line before overwriting it only for some loop");
    sb_report_info("shapes (here 128 B/iter); reading each line first halves DRAM store speed.");
#else
    sb_report_info("Cached x86 stores read each line before writing it (RFO); non-temporal don't.");
#endif

    snprintf(name, sizeof(name), "Store %s, non-temporal", sz);
    f64 v = 0;
    sb_status_e s = measure_gbps(store_nt_rep, a->base, (size_t)big, &v);
    if (s == SB_OK) { sb_report_value(name, v, "GB/s", SB_KIND_MEASURED); }
    else            { sb_report_error(name, s); }
#if defined(__aarch64__)
    sb_report_info("Non-temporal = stnp, which is only a hint on AArch64.");
#else
    sb_report_info("Non-temporal = movntdq-family streaming stores, drained by mfence.");
#endif
}

/* --- Multithreaded DRAM (whole arena, one slice per thread) --- */

typedef struct {
    u8    *base;
    size_t slice;                 /* bytes per thread, multiple of PAR_BLOCK */
    size_t half;                  /* copy: destination offset */
    size_t off[SB_MAX_CPUS];      /* per-thread position inside its slice */
} par_ctx;

static u64 par_read_work(void *ctx, u32 tid, u64 chunk) {
    par_ctx *c  = ctx;
    size_t   o  = c->off[tid];
    mem_kern_read(c->base + (size_t)tid * c->slice + o, (size_t)chunk, 1);
    o += (size_t)chunk;
    c->off[tid] = o >= c->slice ? 0 : o;
    return chunk;
}

/* Counts read + write bytes: 2 x the bytes copied. */
static u64 par_copy_work(void *ctx, u32 tid, u64 chunk) {
    par_ctx *c   = ctx;
    size_t   o   = c->off[tid];
    u8      *src = c->base + (size_t)tid * c->slice + o;
    memcpy(src + c->half, src, (size_t)chunk);
    __asm__ volatile("" ::: "memory");
    o += (size_t)chunk;
    c->off[tid] = o >= c->slice ? 0 : o;
    return 2 * chunk;
}

static void par_measure(const mem_arena *a, const char *name, u32 nthreads, sb_core_e place, bool copy) {
    par_ctx *c = SB_MALLOC(sizeof(*c));
    if (c == NULL) {
        sb_report_error(name, SB_ERR_NOMEM);
        return;
    }
    memset(c, 0, sizeof(*c));
    size_t span = copy ? a->bytes / 2 : a->bytes;
    c->base  = a->base;
    c->half  = a->bytes / 2;
    c->slice = (span / nthreads) / PAR_BLOCK * PAR_BLOCK;
    if (c->slice == 0) {
        SB_FREE(c);
        sb_report_skip(name, "buffer too small for thread count");
        return;
    }
    sb_par_cfg cfg = {
        .nthreads  = nthreads,
        .place     = place,
        .window_ns = PAR_WINDOW_NS,
        .chunk     = PAR_BLOCK,
        .fn        = copy ? par_copy_work : par_read_work,
        .ctx       = c,
    };
    sb_par_result r;
    sb_status_e   s = sb_par_run(&cfg, &r);
    SB_FREE(c);
    if (s != SB_OK) {
        sb_report_error(name, s);
        return;
    }
    sb_report_value(name, r.units_per_sec / 1e9, "GB/s", copy ? SB_KIND_MEASURED : SB_KIND_PEAK);
}

static void par_sweep(const mem_arena *a) {
    const sb_platform *p = sb_platform_get();
    char title[96], name[64], sz[24];
    snprintf(title, sizeof(title), "DRAM bandwidth: %s buffer, one slice per thread, time-based",
             sb_fmt_size(a->bytes, sz, sizeof(sz)));
    sb_report_group(title);

    u32 counts[4];
    u32 ncounts = 0;
    u32 cand[4] = { 2, 4, p->nperf, p->ncpu };
    for (u32 i = 0; i < 4; i++) {
        u32 n = cand[i];
        if (n < 2 || n > p->ncpu) { continue; }
        bool dup = false;
        for (u32 k = 0; k < ncounts; k++) { dup = dup || counts[k] == n; }
        if (!dup) { counts[ncounts++] = n; }
    }

    for (u32 i = 0; i < ncounts; i++) {
        u32 n = counts[i];
        bool all_p = p->nperf > 0 && n == p->nperf;
        snprintf(name, sizeof(name), "Read DRAM, %u threads%s", n,
                 n == p->ncpu ? " (all)" : all_p ? " (P-cores)" : "");
        par_measure(a, name, n, all_p ? SB_CORE_PERF : SB_CORE_ANY, false);
    }
    if (p->neff > 0) {
        snprintf(name, sizeof(name), "Read DRAM, %u E-core threads", p->neff);
        par_measure(a, name, p->neff, SB_CORE_EFF, false);
    }

    sb_report_info("memcpy rows count bytes read + bytes written (2 x bytes copied).");
    par_measure(a, "Copy DRAM (memcpy), 1 thread", 1, SB_CORE_ANY, true);
    if (p->ncpu > 1) {
        snprintf(name, sizeof(name), "Copy DRAM (memcpy), %u threads", p->ncpu);
        par_measure(a, name, p->ncpu, SB_CORE_ANY, true);
    }
}

sb_status_e mem_run_bandwidth(const mem_arena *a) {
    read_sweep(a);
    store_sweep(a);
    par_sweep(a);
    return SB_OK;
}
