#if defined(__AVX2__) && !defined(__AVX512F__)

#include "timer.h"
#include "cpu_bench.h"

#include <stdint.h>
#include <immintrin.h>

/* --- Scalar benchmarks --- */

DEFINE_FP_SCALAR(fp64_scalar, double, 0.9999999, 0.0000001)
DEFINE_FP_SCALAR(fp32_scalar, float,  0.9999999f, 0.0000001f)

DEFINE_INT_SCALAR(int64_scalar, uint64_t, 0x2545F4914F6CDD1DULL, 0x9E3779B97F4A7C15ULL)
DEFINE_INT_SCALAR(int32_scalar, uint32_t, 0x4F6CDD1Du, 0x7F4A7C15u)
DEFINE_INT_SCALAR(int16_scalar, uint16_t, 0xDD1Du, 0x7C15u)
DEFINE_INT_SCALAR(int8_scalar,  uint8_t,  0x1Du, 0x15u)

/* --- AVX2 SIMD generators --- */

#define DEFINE_FP_SIMD_AVX2(fname, vtype, set, fma, lanes, k1v, k2v)         \
static double fname(uint64_t iters) {                                         \
    vtype a0=set(1.0),  a1=set(1.01), a2=set(1.02), a3=set(1.03), a4=set(1.04); \
    vtype a5=set(1.05), a6=set(1.06), a7=set(1.07), a8=set(1.08), a9=set(1.09); \
    vtype vk1 = set(k1v), vk2 = set(k2v);                                    \
    uint64_t t0 = timer_ns();                                                 \
    for (uint64_t i = 0; i < iters; i++) {                                    \
        a0=fma(a0,vk1,vk2); a1=fma(a1,vk1,vk2); a2=fma(a2,vk1,vk2);         \
        a3=fma(a3,vk1,vk2); a4=fma(a4,vk1,vk2); a5=fma(a5,vk1,vk2);         \
        a6=fma(a6,vk1,vk2); a7=fma(a7,vk1,vk2); a8=fma(a8,vk1,vk2);         \
        a9=fma(a9,vk1,vk2);                                                  \
        __asm__ volatile("" : "+x"(a0),"+x"(a1),"+x"(a2),"+x"(a3),"+x"(a4), \
                              "+x"(a5),"+x"(a6),"+x"(a7),"+x"(a8),"+x"(a9));\
    }                                                                         \
    uint64_t t1 = timer_ns();                                                 \
    return (double)iters * BENCH_NCHAINS * (lanes) * 2.0                      \
           / ((double)(t1-t0) / 1e9) / 1e9;                                  \
}

#define DEFINE_INT_SIMD_AVX2(fname, vtype, set, mul, add, lanes, k1v, k2v)   \
static double fname(uint64_t iters) {                                         \
    vtype a0=set(1),  a1=set(3),  a2=set(5),  a3=set(7),  a4=set(9);         \
    vtype a5=set(11), a6=set(13), a7=set(15), a8=set(17), a9=set(19);        \
    vtype vk1 = set(k1v), vk2 = set(k2v);                                    \
    uint64_t t0 = timer_ns();                                                 \
    for (uint64_t i = 0; i < iters; i++) {                                    \
        a0=add(mul(a0,vk1),vk2); a1=add(mul(a1,vk1),vk2);                    \
        a2=add(mul(a2,vk1),vk2); a3=add(mul(a3,vk1),vk2);                    \
        a4=add(mul(a4,vk1),vk2); a5=add(mul(a5,vk1),vk2);                    \
        a6=add(mul(a6,vk1),vk2); a7=add(mul(a7,vk1),vk2);                    \
        a8=add(mul(a8,vk1),vk2); a9=add(mul(a9,vk1),vk2);                    \
        __asm__ volatile("" : "+x"(a0),"+x"(a1),"+x"(a2),"+x"(a3),"+x"(a4), \
                              "+x"(a5),"+x"(a6),"+x"(a7),"+x"(a8),"+x"(a9));\
    }                                                                         \
    uint64_t t1 = timer_ns();                                                 \
    return (double)iters * BENCH_NCHAINS * (lanes) * 2.0                      \
           / ((double)(t1-t0) / 1e9) / 1e9;                                  \
}

/* --- AVX2 benchmarks --- */

#ifdef __FMA__
DEFINE_FP_SIMD_AVX2(fp64_simd, __m256d, _mm256_set1_pd,
                    _mm256_fmadd_pd, 4, 0.9999999, 0.0000001)
DEFINE_FP_SIMD_AVX2(fp32_simd, __m256,  _mm256_set1_ps,
                    _mm256_fmadd_ps, 8, 0.9999999f, 0.0000001f)
#endif
DEFINE_INT_SIMD_AVX2(int32_simd, __m256i, _mm256_set1_epi32,
                     _mm256_mullo_epi32, _mm256_add_epi32, 8, 0x4F6CDD1Du, 0x7F4A7C15u)
DEFINE_INT_SIMD_AVX2(int16_simd, __m256i, _mm256_set1_epi16,
                     _mm256_mullo_epi16, _mm256_add_epi16, 16, 0xDD1Du, 0x7C15u)

/* --- Dispatch table --- */

const struct cpu_bench cpu_benches_avx2[] = {
#ifdef __FMA__
    {"FP64",  "GFLOPS",  fp64_scalar, fp64_simd,  4},
    {"FP32",  "GFLOPS",  fp32_scalar, fp32_simd,  8},
#else
    {"FP64",  "GFLOPS",  fp64_scalar, NULL,       0},
    {"FP32",  "GFLOPS",  fp32_scalar, NULL,       0},
#endif
    {"INT64", "GINTOPS", int64_scalar, NULL,       0},
    {"INT32", "GINTOPS", int32_scalar, int32_simd, 8},
    {"INT16", "GINTOPS", int16_scalar, int16_simd,16},
    {"INT8",  "GINTOPS", int8_scalar,  NULL,       0},
};

const int cpu_bench_count_avx2 = sizeof(cpu_benches_avx2) / sizeof(cpu_benches_avx2[0]);

#endif
