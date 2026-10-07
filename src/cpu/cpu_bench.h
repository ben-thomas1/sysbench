#pragma once

#include <stdint.h>

#define BENCH_NCHAINS 10

typedef double (*bench_fn)(uint64_t iters);

struct cpu_bench {
    const char *name;
    const char *unit;     /* "GFLOPS" or "GINTOPS" */
    bench_fn scalar;
    bench_fn simd;        /* NULL if unavailable */
    unsigned simd_lanes;
};

extern const struct cpu_bench *cpu_benches;
extern int cpu_bench_count;
extern const char *cpu_simd_name;

/* Platform asm constraint macros for register barriers */
#if defined(__aarch64__)
#define FP_REG(x) "+w"(x)
#elif defined(__x86_64__)
#define FP_REG(x) "+x"(x)
#else
#define FP_REG(x) "+r"(x)
#endif
#define GP_REG(x) "+r"(x)

/*
 * Scalar benchmark generators.
 * Each generates a static function: double fname(uint64_t iters)
 * returning throughput in GOPS.
 */

#define DEFINE_FP_SCALAR(fname, type, k1v, k2v)                              \
static double fname(uint64_t iters) {                                        \
    type a0=1.0, a1=1.01, a2=1.02, a3=1.03, a4=1.04;                        \
    type a5=1.05, a6=1.06, a7=1.07, a8=1.08, a9=1.09;                       \
    const type k1 = k1v, k2 = k2v;                                          \
    uint64_t t0 = timer_ns();                                                \
    for (uint64_t i = 0; i < iters; i++) {                                   \
        a0=a0*k1+k2; a1=a1*k1+k2; a2=a2*k1+k2; a3=a3*k1+k2; a4=a4*k1+k2;  \
        a5=a5*k1+k2; a6=a6*k1+k2; a7=a7*k1+k2; a8=a8*k1+k2; a9=a9*k1+k2;  \
        __asm__ volatile("" : FP_REG(a0),FP_REG(a1),FP_REG(a2),FP_REG(a3),FP_REG(a4), \
                              FP_REG(a5),FP_REG(a6),FP_REG(a7),FP_REG(a8),FP_REG(a9));\
    }                                                                        \
    uint64_t t1 = timer_ns();                                                \
    return (double)iters * BENCH_NCHAINS * 2.0 / ((double)(t1-t0) / 1e9) / 1e9; \
}

#define DEFINE_INT_SCALAR(fname, type, k1v, k2v)                             \
static double fname(uint64_t iters) {                                        \
    type a0=1, a1=3, a2=5, a3=7, a4=9;                                      \
    type a5=11, a6=13, a7=15, a8=17, a9=19;                                  \
    const uint64_t k1 = k1v, k2 = k2v;                                      \
    uint64_t t0 = timer_ns();                                                \
    for (uint64_t i = 0; i < iters; i++) {                                   \
        a0=a0*k1+k2; a1=a1*k1+k2; a2=a2*k1+k2; a3=a3*k1+k2; a4=a4*k1+k2;  \
        a5=a5*k1+k2; a6=a6*k1+k2; a7=a7*k1+k2; a8=a8*k1+k2; a9=a9*k1+k2;  \
        __asm__ volatile("" : GP_REG(a0),GP_REG(a1),GP_REG(a2),GP_REG(a3),GP_REG(a4), \
                              GP_REG(a5),GP_REG(a6),GP_REG(a7),GP_REG(a8),GP_REG(a9));\
    }                                                                        \
    uint64_t t1 = timer_ns();                                                \
    return (double)iters * BENCH_NCHAINS * 2.0 / ((double)(t1-t0) / 1e9) / 1e9; \
}
