#if defined(__AVX512F__)

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

/* --- AVX-512 SIMD generators ---
 *   Use "+v" constraint for ZMM registers (clang/gcc).
 */

#define DEFINE_FP_SIMD_512(fname, vtype, set, fma, lanes, k1v, k2v)          \
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
        __asm__ volatile("" : "+v"(a0),"+v"(a1),"+v"(a2),"+v"(a3),"+v"(a4),  \
                              "+v"(a5),"+v"(a6),"+v"(a7),"+v"(a8),"+v"(a9)); \
    }                                                                         \
    uint64_t t1 = timer_ns();                                                 \
    return (double)iters * BENCH_NCHAINS * (lanes) * 2.0                      \
           / ((double)(t1-t0) / 1e9) / 1e9;                                  \
}

#define DEFINE_INT_SIMD_512(fname, vtype, set, mul, add, lanes, k1v, k2v)    \
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
        __asm__ volatile("" : "+v"(a0),"+v"(a1),"+v"(a2),"+v"(a3),"+v"(a4),  \
                              "+v"(a5),"+v"(a6),"+v"(a7),"+v"(a8),"+v"(a9)); \
    }                                                                         \
    uint64_t t1 = timer_ns();                                                 \
    return (double)iters * BENCH_NCHAINS * (lanes) * 2.0                      \
           / ((double)(t1-t0) / 1e9) / 1e9;                                  \
}

/* --- AVX-512 benchmarks --- */

DEFINE_FP_SIMD_512(fp64_simd, __m512d, _mm512_set1_pd,
                   _mm512_fmadd_pd, 8, 0.9999999, 0.0000001)
DEFINE_FP_SIMD_512(fp32_simd, __m512,  _mm512_set1_ps,
                   _mm512_fmadd_ps, 16, 0.9999999f, 0.0000001f)
DEFINE_INT_SIMD_512(int32_simd, __m512i, _mm512_set1_epi32,
                    _mm512_mullo_epi32, _mm512_add_epi32, 16, 0x4F6CDD1Du, 0x7F4A7C15u)
#ifdef __AVX512BW__
DEFINE_INT_SIMD_512(int16_simd, __m512i, _mm512_set1_epi16,
                    _mm512_mullo_epi16, _mm512_add_epi16, 32, 0xDD1Du, 0x7C15u)
#endif

/* --- Dispatch table --- */

const struct cpu_bench cpu_benches_avx512[] = {
    {"FP64",  "GFLOPS",  fp64_scalar, fp64_simd,  8},
    {"FP32",  "GFLOPS",  fp32_scalar, fp32_simd, 16},
    {"INT64", "GINTOPS", int64_scalar, NULL,       0},
    {"INT32", "GINTOPS", int32_scalar, int32_simd,16},
#ifdef __AVX512BW__
    {"INT16", "GINTOPS", int16_scalar, int16_simd,32},
#else
    {"INT16", "GINTOPS", int16_scalar, NULL,       0},
#endif
    {"INT8",  "GINTOPS", int8_scalar,  NULL,       0},
};

const int cpu_bench_count_avx512 = sizeof(cpu_benches_avx512) / sizeof(cpu_benches_avx512[0]);

#endif
