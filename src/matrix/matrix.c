#include "matrix/matrix.h"
#include "core/platform.h"
#include "core/report.h"
#include "core/stats.h"
#include "core/timer.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifdef HAS_BLAS

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <sys/sysctl.h>
#else
#include <cblas.h>
#endif

#define WINDOW_NS    750'000'000ULL    /* timed window per size, after warmup */
#define BATCH_NS     5'000'000ULL      /* target length of one timed batch */
#define MAX_BATCHES  512
#define MIN_BATCHES  3
#define MAX_CALL_NS  2'000'000'000ULL  /* skip a size whose single call would exceed this */
#define FULL_CHECK_N 256               /* check every element up to this size */
#define CHECK_SAMPLES 256              /* sampled elements above it */

typedef enum { PREC_F32, PREC_F64 } prec_e;

typedef struct {
    prec_e prec;
    u32    n;
    void  *a;
    void  *b;
    void  *c;
} gemm_job;

typedef struct {
    f64 gflops;   /* median over batches */
    f64 call_ns;  /* median time of one call */
} gemm_result;

/* The timed operation: C = A * B, row-major, no transposes. */
static void gemm_call(const gemm_job *j) {
    int n = (int)j->n;
    if (j->prec == PREC_F32) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n,
                    1.0f, j->a, n, j->b, n, 0.0f, j->c, n);
    } else {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n,
                    1.0, j->a, n, j->b, n, 0.0, j->c, n);
    }
}

/* xorshift64*: deterministic inputs in [-1, 1), no libc rand() state. */
static f64 rnd_unit(u64 *s) {
    *s ^= *s >> 12;
    *s ^= *s << 25;
    *s ^= *s >> 27;
    u64 r = *s * 0x2545F4914F6CDD1DULL;
    return (f64)(r >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

static void fill(prec_e p, void *buf, size_t count, u64 seed) {
    u64 s = seed;
    if (p == PREC_F32) {
        f32 *x = buf;
        for (size_t i = 0; i < count; i++) { x[i] = (f32)rnd_unit(&s); }
    } else {
        f64 *x = buf;
        for (size_t i = 0; i < count; i++) { x[i] = rnd_unit(&s); }
    }
}

static f64 load(prec_e p, const void *buf, size_t idx) {
    return p == PREC_F32 ? (f64)((const f32 *)buf)[idx] : ((const f64 *)buf)[idx];
}

/* NaN/Inf test on the bit pattern: -ffast-math lets the compiler assume
 * isfinite() is always true. */
static bool finite_bits(prec_e p, const void *buf, size_t idx) {
    if (p == PREC_F32) {
        u32 u;
        memcpy(&u, (const f32 *)buf + idx, sizeof(u));
        return (u & 0x7f80'0000u) != 0x7f80'0000u;
    }
    u64 u;
    memcpy(&u, (const f64 *)buf + idx, sizeof(u));
    return (u & 0x7ff0'0000'0000'0000ULL) != 0x7ff0'0000'0000'0000ULL;
}

/* Compare C[i][j] with an FP64 dot product. Error bound per element:
 * (n + 2) * eps * sum_k |a_ik * b_kj|, which covers any summation order
 * (the classic gamma_n bound is n * eps / 2). Returns the worst
 * error / bound ratio in *ratio (-1 for NaN/Inf); passes when it is <= 1. */
static bool check_elem(const gemm_job *j, u32 i, u32 col, f64 eps, f64 *ratio) {
    size_t n   = j->n;
    f64    ref = 0.0;
    f64    mag = 0.0;
    for (size_t k = 0; k < n; k++) {
        f64 t = load(j->prec, j->a, i * n + k) * load(j->prec, j->b, k * n + col);
        ref += t;
        mag += fabs(t);
    }
    size_t idx = i * n + col;
    if (!finite_bits(j->prec, j->c, idx)) {
        *ratio = -1.0;  /* marks NaN/Inf */
        return false;
    }
    f64 err   = fabs(load(j->prec, j->c, idx) - ref);
    f64 bound = (f64)(n + 2) * eps * mag + DBL_MIN;
    f64 r     = err / bound;
    if (r > *ratio) { *ratio = r; }
    return r <= 1.0;
}

static bool verify(const gemm_job *j, f64 *out_ratio) {
    f64  eps   = j->prec == PREC_F32 ? (f64)FLT_EPSILON : DBL_EPSILON;
    f64  ratio = 0.0;
    bool ok    = true;
    u32  n     = j->n;
    if (n <= FULL_CHECK_N) {
        for (u32 i = 0; i < n && ok; i++) {
            for (u32 c = 0; c < n && ok; c++) { ok = check_elem(j, i, c, eps, &ratio); }
        }
    } else {
        u64 s = 0x9E37'79B9'7F4A'7C15ULL ^ n;
        ok = check_elem(j, 0, 0, eps, &ratio) && check_elem(j, n - 1, n - 1, eps, &ratio);
        for (u32 k = 0; k < CHECK_SAMPLES && ok; k++) {
            u32 i = (u32)((rnd_unit(&s) + 1.0) * 0.5 * n) % n;
            u32 c = (u32)((rnd_unit(&s) + 1.0) * 0.5 * n) % n;
            ok = check_elem(j, i, c, eps, &ratio);
        }
    }
    *out_ratio = ratio;
    return ok;
}

/* Warm up for SB_WARMUP_NS (at least one call), then time batches of calls
 * for WINDOW_NS (at least MIN_BATCHES) and take the median batch rate. A
 * batch is sized to ~BATCH_NS so small sizes are not timer-bound. */
static sb_status_e measure(const gemm_job *j, gemm_result *out) {
    u64 best = UINT64_MAX;
    u64 w0   = sb_timer_now_ns();
    u64 now  = w0;
    do {
        u64 t0 = sb_timer_now_ns();
        gemm_call(j);
        now = sb_timer_now_ns();
        if (now - t0 < best) { best = now - t0; }
    } while (now - w0 < SB_WARMUP_NS);
    if (best == 0) { best = 1; }

    u64 batch = BATCH_NS / best;
    if (batch < 1) { batch = 1; }

    f64 rates[MAX_BATCHES];
    f64 calls[MAX_BATCHES];
    u32 nb  = 0;
    u64 m0  = sb_timer_now_ns();
    do {
        u64 t0 = sb_timer_now_ns();
        for (u64 k = 0; k < batch; k++) { gemm_call(j); }
        u64 t1 = sb_timer_now_ns();
        if (t1 <= t0) { return SB_ERR_RANGE; }
        f64 dt    = (f64)(t1 - t0);
        calls[nb] = dt / (f64)batch;
        rates[nb] = 2.0 * (f64)j->n * (f64)j->n * (f64)j->n * (f64)batch / dt;  /* flop/ns = GFLOPS */
        nb++;
        now = t1;
    } while (nb < MAX_BATCHES && (nb < MIN_BATCHES || now - m0 < WINDOW_NS));

    sb_stats sr;
    sb_stats sc;
    sb_status_e s = sb_stats_compute(rates, nb, &sr);
    if (s != SB_OK) { return s; }
    s = sb_stats_compute(calls, nb, &sc);
    if (s != SB_OK) { return s; }
    out->gflops  = sr.median;
    out->call_ns = sc.median;
    return SB_OK;
}

/* --- Library threading ---------------------------------------------------- */

#ifdef __APPLE__

static bool sysctl_flag(const char *name) {
    i32    v   = 0;
    size_t len = sizeof(v);
    return sysctlbyname(name, &v, &len, NULL, 0) == 0 && v != 0;
}

static void report_library(void) {
    sb_report_info("Library: Accelerate; thread count is library-controlled (no API to read it)");
    if (sysctl_flag("hw.optional.arm.FEAT_SME")) {
        sb_report_info("CPU has SME%s: Accelerate runs GEMM on the SME matrix unit(s), so results can",
                       sysctl_flag("hw.optional.arm.FEAT_SME2") ? "/SME2" : "");
        sb_report_info("exceed the all-core NEON FMA peak (cpu section)");
    }
}

static sb_status_e set_single_thread(bool single) {
    if (__builtin_available(macOS 15.0, *)) {
        int r = BLASSetThreading(single ? BLAS_THREADING_SINGLE_THREADED : BLAS_THREADING_MULTI_THREADED);
        return r == 0 ? SB_OK : SB_ERR_UNSUPPORTED;
    }
    return SB_ERR_UNSUPPORTED;
}

#else

static int default_threads;

static void report_library(void) {
    const sb_platform *p   = sb_platform_get();
    int                par = openblas_get_parallel();
    default_threads        = openblas_get_num_threads();
    sb_report_info("Library: OpenBLAS (%s)", openblas_get_config());
    sb_report_info("Threads: %d (%s build), %u CPUs available; OPENBLAS_NUM_THREADS overrides",
                   default_threads, par == 0 ? "sequential" : par == 1 ? "pthreads" : "OpenMP", p->ncpu);
    if (p->ncpu > 0 && default_threads > (int)p->ncpu) {
        sb_report_info("WARNING: more BLAS threads than CPUs; small sizes will suffer from oversubscription");
    }
}

static sb_status_e set_single_thread(bool single) {
    openblas_set_num_threads(single ? 1 : default_threads);
    return SB_OK;
}

#endif

/* --- Groups --------------------------------------------------------------- */

typedef struct {
    const char *prefix;    /* "SGEMM" / "DGEMM" */
    const char *suffix;    /* appended to the test name, "" or " 1T" */
    prec_e      prec;
    const u32  *sizes;
    u32         nsizes;
} gemm_group;

static void run_group(const gemm_group *g, void *a, void *b, void *c) {
    f64 prev_ns = 0.0;
    u32 prev_n  = 0;
    for (u32 i = 0; i < g->nsizes; i++) {
        u32  n = g->sizes[i];
        char test[64];
        snprintf(test, sizeof(test), "%s %ux%u%s", g->prefix, n, n, g->suffix);

        if (prev_n != 0) {
            f64 r   = (f64)n / (f64)prev_n;
            f64 est = prev_ns * r * r * r;
            if (est > (f64)MAX_CALL_NS) {
                char why[64];
                snprintf(why, sizeof(why), "~%.1f s per call; over the time budget", est / 1e9);
                sb_report_skip(test, why);
                continue;
            }
        }

        gemm_job    j = { .prec = g->prec, .n = n, .a = a, .b = b, .c = c };
        gemm_result r;
        sb_status_e s = measure(&j, &r);
        if (s != SB_OK) {
            sb_report_error(test, s);
            continue;
        }
        prev_ns = r.call_ns;
        prev_n  = n;

        f64 ratio = 0.0;
        if (!verify(&j, &ratio)) {
            char why[64];
            if (ratio < 0.0) { snprintf(why, sizeof(why), "WRONG RESULT (NaN/Inf element)"); }
            else             { snprintf(why, sizeof(why), "WRONG RESULT (error %.3g x bound)", ratio); }
            sb_report_skip(test, why);
            continue;
        }
        sb_report_value(test, r.gflops, "GFLOPS", SB_KIND_MEASURED);
    }
}

static const u32 sgemm_sizes[]  = { 64, 128, 256, 512, 1024, 2048, 4096 };
static const u32 single_sizes[] = { 128, 1024 };
static const u32 dgemm_sizes[]  = { 256, 1024, 2048 };

static sb_status_e matrix_run(void) {
    report_library();
    sb_report_info("C = A*B, square row-major; GFLOPS = 2n^3 / time, median of ~5 ms batches");
    sb_report_info("over 0.75 s after a 0.25 s warmup. Each result is checked against an FP64");
    sb_report_info("reference outside the timed region (all elements for n <= %u, else %u samples)",
                   FULL_CHECK_N, CHECK_SAMPLES);

    u32    nmax  = sgemm_sizes[SB_ARRAY_LEN(sgemm_sizes) - 1];
    size_t count = (size_t)nmax * nmax;
    size_t bytes = count * sizeof(f32);
    u32    dmax  = dgemm_sizes[SB_ARRAY_LEN(dgemm_sizes) - 1];
    if ((size_t)dmax * dmax * sizeof(f64) > bytes) { bytes = (size_t)dmax * dmax * sizeof(f64); }

    void *a = SB_MALLOC(bytes);
    void *b = SB_MALLOC(bytes);
    void *c = SB_MALLOC(bytes);
    if (a == NULL || b == NULL || c == NULL) {
        SB_FREE(a);
        SB_FREE(b);
        SB_FREE(c);
        sb_report_error("SGEMM", SB_ERR_NOMEM);
        return SB_ERR_NOMEM;
    }

    fill(PREC_F32, a, count, 1);
    fill(PREC_F32, b, count, 2);
    memset(c, 0, bytes);

    sb_report_group("SGEMM FP32, library threading");
    gemm_group sg = { "SGEMM", "", PREC_F32, sgemm_sizes, SB_ARRAY_LEN(sgemm_sizes) };
    run_group(&sg, a, b, c);

    sb_report_group("SGEMM FP32, single thread");
    if (set_single_thread(true) == SB_OK) {
        gemm_group st = { "SGEMM", " 1T", PREC_F32, single_sizes, SB_ARRAY_LEN(single_sizes) };
        run_group(&st, a, b, c);
        (void)set_single_thread(false);
    } else {
        sb_report_skip("SGEMM 1T", "library cannot be limited to one thread");
    }

    size_t dcount = (size_t)dmax * dmax;
    fill(PREC_F64, a, dcount, 3);
    fill(PREC_F64, b, dcount, 4);

    sb_report_group("DGEMM FP64, library threading");
    gemm_group dg = { "DGEMM", "", PREC_F64, dgemm_sizes, SB_ARRAY_LEN(dgemm_sizes) };
    run_group(&dg, a, b, c);

    SB_FREE(a);
    SB_FREE(b);
    SB_FREE(c);
    return SB_OK;
}

#else

static sb_status_e matrix_run(void) {
    sb_report_skip("SGEMM", "no BLAS library (Linux: install OpenBLAS and rebuild)");
    return SB_OK;
}

#endif

const sb_section sb_section_matrix = {
    .name       = "matrix",
    .title      = "Matrix Multiply",
    .help       = "  SGEMM (FP32) from 64x64 to 4096x4096 with the library's default threading,\n"
                  "  SGEMM limited to one thread at 128 and 1024, and DGEMM (FP64) at 256-2048.\n"
                  "  Accelerate on macOS (SME on M4), OpenBLAS on Linux. GFLOPS are measured\n"
                  "  library throughput (2n^3 / time), not hardware peak; every result is\n"
                  "  checked against an FP64 reference outside the timed region.\n",
    .run        = matrix_run,
    .repeatable = true,
};
