#include "timer.h"
#include "cpu_bench.h"

#include <stdint.h>
#include <arm_neon.h>

/* --- Scalar benchmarks --- */

DEFINE_FP_SCALAR(fp64_scalar, double, 0.9999999, 0.0000001)
DEFINE_FP_SCALAR(fp32_scalar, float,  0.9999999f, 0.0000001f)

DEFINE_INT_SCALAR(int64_scalar, uint64_t, 0x2545F4914F6CDD1DULL, 0x9E3779B97F4A7C15ULL)
DEFINE_INT_SCALAR(int32_scalar, uint32_t, 0x4F6CDD1Du, 0x7F4A7C15u)
DEFINE_INT_SCALAR(int16_scalar, uint16_t, 0xDD1Du, 0x7C15u)
DEFINE_INT_SCALAR(int8_scalar,  uint8_t,  0x1Du, 0x15u)

/* --- NEON benchmark generator ---
 *   op(addend, val, multiplier): vfmaq/vmlaq intrinsics all use this order.
 *   k1v = multiplier, k2v = addend (matches scalar convention).
 */

#define DEFINE_FP_NEON(fname, vtype, dup, op, lanes, k1v, k2v)              \
static double fname(uint64_t iters) {                                        \
    vtype a0=dup(1.0),  a1=dup(1.01), a2=dup(1.02), a3=dup(1.03), a4=dup(1.04); \
    vtype a5=dup(1.05), a6=dup(1.06), a7=dup(1.07), a8=dup(1.08), a9=dup(1.09); \
    vtype vk1 = dup(k1v), vk2 = dup(k2v);                                   \
    uint64_t t0 = timer_ns();                                                \
    for (uint64_t i = 0; i < iters; i++) {                                   \
        a0=op(vk2,a0,vk1); a1=op(vk2,a1,vk1); a2=op(vk2,a2,vk1);           \
        a3=op(vk2,a3,vk1); a4=op(vk2,a4,vk1); a5=op(vk2,a5,vk1);           \
        a6=op(vk2,a6,vk1); a7=op(vk2,a7,vk1); a8=op(vk2,a8,vk1);           \
        a9=op(vk2,a9,vk1);                                                  \
        __asm__ volatile("" : "+w"(a0),"+w"(a1),"+w"(a2),"+w"(a3),"+w"(a4), \
                              "+w"(a5),"+w"(a6),"+w"(a7),"+w"(a8),"+w"(a9));\
    }                                                                        \
    uint64_t t1 = timer_ns();                                                \
    return (double)iters * BENCH_NCHAINS * (lanes) * 2.0                     \
           / ((double)(t1-t0) / 1e9) / 1e9;                                 \
}

#define DEFINE_INT_NEON(fname, vtype, dup, op, lanes, k1v, k2v)              \
static double fname(uint64_t iters) {                                        \
    vtype a0=dup(1),  a1=dup(3),  a2=dup(5),  a3=dup(7),  a4=dup(9);        \
    vtype a5=dup(11), a6=dup(13), a7=dup(15), a8=dup(17), a9=dup(19);       \
    vtype vk1 = dup(k1v), vk2 = dup(k2v);                                   \
    uint64_t t0 = timer_ns();                                                \
    for (uint64_t i = 0; i < iters; i++) {                                   \
        a0=op(vk2,a0,vk1); a1=op(vk2,a1,vk1); a2=op(vk2,a2,vk1);           \
        a3=op(vk2,a3,vk1); a4=op(vk2,a4,vk1); a5=op(vk2,a5,vk1);           \
        a6=op(vk2,a6,vk1); a7=op(vk2,a7,vk1); a8=op(vk2,a8,vk1);           \
        a9=op(vk2,a9,vk1);                                                  \
        __asm__ volatile("" : "+w"(a0),"+w"(a1),"+w"(a2),"+w"(a3),"+w"(a4), \
                              "+w"(a5),"+w"(a6),"+w"(a7),"+w"(a8),"+w"(a9));\
    }                                                                        \
    uint64_t t1 = timer_ns();                                                \
    return (double)iters * BENCH_NCHAINS * (lanes) * 2.0                     \
           / ((double)(t1-t0) / 1e9) / 1e9;                                 \
}

/* --- NEON benchmarks --- */

/*                  name        type           dup              op          lanes  k1(mult)     k2(add)     */
DEFINE_FP_NEON( fp64_neon,  float64x2_t, vdupq_n_f64, vfmaq_f64,  2,  0.9999999,   0.0000001)
DEFINE_FP_NEON( fp32_neon,  float32x4_t, vdupq_n_f32, vfmaq_f32,  4,  0.9999999f,  0.0000001f)
DEFINE_INT_NEON(int32_neon, uint32x4_t,  vdupq_n_u32, vmlaq_u32,  4,  0x4F6CDD1Du, 0x7F4A7C15u)
DEFINE_INT_NEON(int16_neon, uint16x8_t,  vdupq_n_u16, vmlaq_u16,  8,  0xDD1Du,     0x7C15u)
DEFINE_INT_NEON(int8_neon,  uint8x16_t,  vdupq_n_u8,  vmlaq_u8,  16,  0x1Du,       0x15u)

/* --- Dispatch table --- */

const char *cpu_simd_name = "NEON";

static const struct cpu_bench cpu_benches_neon[] = {
    {"FP64",  "GFLOPS",  fp64_scalar, fp64_neon,  2},
    {"FP32",  "GFLOPS",  fp32_scalar, fp32_neon,  4},
    {"INT64", "GINTOPS", int64_scalar, NULL,       0},
    {"INT32", "GINTOPS", int32_scalar, int32_neon, 4},
    {"INT16", "GINTOPS", int16_scalar, int16_neon, 8},
    {"INT8",  "GINTOPS", int8_scalar,  int8_neon, 16},
};

const struct cpu_bench *cpu_benches = cpu_benches_neon;
int cpu_bench_count = sizeof(cpu_benches_neon) / sizeof(cpu_benches_neon[0]);
