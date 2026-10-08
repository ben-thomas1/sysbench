#include "cpu/kern.h"

#include <stdio.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#endif

/* AArch64 kernels. Chain counts come from a sweep on the M4 Pro
 * (sysbench-validation/results/agent-cpu/chains.csv): FP/SIMD multiply-add
 * has 3-4 cycles of latency and 4 pipes, so 16 chains reach ~92% and 20 reach
 * ~97-100%; 24 or more chains drop to ~80%. Scalar `madd` saturates at 8-12
 * accumulator-form chains. Accumulators: v0-v19 (SIMD/FP), x0-x11 (integer);
 * operands: v30/v31, x16/x17. */

#define REP12(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8) M(9) M(10) M(11)
#define REP16(M) REP12(M) M(12) M(13) M(14) M(15)
#define REP20(M) REP16(M) M(16) M(17) M(18) M(19)
#define REP40(M) REP20(M) REP20(M)

#define VCLOB "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", \
              "v14", "v15", "v16", "v17", "v18", "v19", "v30", "v31"
#define XCLOB12 "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x16", "x17"
#define XCLOB16 XCLOB12, "x12", "x13", "x14", "x15"

/* Assembler extensions for FP16 and dot-product instructions: Apple clang's
 * -march=native does not enable them, so availability is checked at runtime. */
#define EXT ".arch_extension fp16\n\t.arch_extension dotprod\n\t"

#define SIMD_CHAINS 20
#define SIMD_REPS   2
#define SIMD_INSNS  (SIMD_CHAINS * SIMD_REPS)
#define INT_CHAINS  12
#define INT_REPS    4
#define INT_INSNS   (INT_CHAINS * INT_REPS)

/* One kernel: `init` loads operands and accumulators, then `chunk`
 * iterations of `body`. */
#define KERNEL(fn, init, body, ...)                                                 \
    static u64 fn(void *ctx, u32 tid, u64 chunk) {                                  \
        (void)ctx;                                                                  \
        (void)tid;                                                                  \
        u64 n = chunk;                                                              \
        __asm__ volatile(EXT init ".p2align 6\n1:\n\t" body "subs %x[n], %x[n], #1\n\tb.ne 1b\n\t" \
                         : [n] "+r"(n) : : "cc", __VA_ARGS__);                          \
        return chunk;                                                               \
    }

/* --- Accumulator initialisers --- */
#define I_F32(d) "fmov v" #d ".4s, #1.0\n\t"
#define I_F64(d) "fmov v" #d ".2d, #1.0\n\t"
#define I_F16(d) "fmov v" #d ".8h, #1.0\n\t"
#define I_SD(d)  "fmov d" #d ", #1.0\n\t"
#define I_SS(d)  "fmov s" #d ", #1.0\n\t"
#define I_INT(d) "movi v" #d ".16b, #1\n\t"
#define I_X(d)   "mov x" #d ", #1\n\t"

/* Operands: products are 1/64 so FP accumulators grow slowly and stay finite. */
#define K_F32 "fmov v30.4s, #0.125\n\tfmov v31.4s, #0.125\n\t"
#define K_F64 "fmov v30.2d, #0.125\n\tfmov v31.2d, #0.125\n\t"
#define K_F16 "fmov v30.8h, #0.125\n\tfmov v31.8h, #0.125\n\t"
#define K_INT "movi v30.16b, #3\n\tmovi v31.16b, #5\n\t"
#define K_X   "mov x16, #3\n\tmov x17, #5\n\t"

/* --- Accumulator-form operations: acc += a * b --- */
#define O_FMADD_D(d) "fmadd d" #d ", d30, d31, d" #d "\n\t"
#define O_FMADD_S(d) "fmadd s" #d ", s30, s31, s" #d "\n\t"
#define O_MADD_X(d)  "madd x" #d ", x16, x17, x" #d "\n\t"
#define O_FMLA_2D(d) "fmla v" #d ".2d, v30.2d, v31.2d\n\t"
#define O_FMLA_4S(d) "fmla v" #d ".4s, v30.4s, v31.4s\n\t"
#define O_FMLA_8H(d) "fmla v" #d ".8h, v30.8h, v31.8h\n\t"
#define O_MLA_4S(d)  "mla v" #d ".4s, v30.4s, v31.4s\n\t"
#define O_MLA_8H(d)  "mla v" #d ".8h, v30.8h, v31.8h\n\t"
#define O_MLA_16B(d) "mla v" #d ".16b, v30.16b, v31.16b\n\t"
#define O_UDOT(d)    "udot v" #d ".4s, v30.16b, v31.16b\n\t"

#define SIMD_BODY(op) REP20(op) REP20(op)
#define INT_BODY(op)  REP12(op) REP12(op) REP12(op) REP12(op)

KERNEL(k_fp64_scalar, K_F64 REP20(I_SD),  SIMD_BODY(O_FMADD_D), VCLOB)
KERNEL(k_fp32_scalar, K_F32 REP20(I_SS),  SIMD_BODY(O_FMADD_S), VCLOB)
KERNEL(k_int_scalar,  K_X REP12(I_X),     INT_BODY(O_MADD_X),   XCLOB12)
KERNEL(k_fp64_neon,   K_F64 REP20(I_F64), SIMD_BODY(O_FMLA_2D), VCLOB)
KERNEL(k_fp32_neon,   K_F32 REP20(I_F32), SIMD_BODY(O_FMLA_4S), VCLOB)
KERNEL(k_fp16_neon,   K_F16 REP20(I_F16), SIMD_BODY(O_FMLA_8H), VCLOB)
KERNEL(k_int32_neon,  K_INT REP20(I_INT), SIMD_BODY(O_MLA_4S),  VCLOB)
KERNEL(k_int16_neon,  K_INT REP20(I_INT), SIMD_BODY(O_MLA_8H),  VCLOB)
KERNEL(k_int8_neon,   K_INT REP20(I_INT), SIMD_BODY(O_MLA_16B), VCLOB)
KERNEL(k_int8_udot,   K_INT REP20(I_INT), SIMD_BODY(O_UDOT),    VCLOB)

/* --- Ops-per-cycle kernels --- */

/* Clock reference: one register-register add chain, 64 per iteration
 * (register operand, so no immediate-add shortcuts apply). */
#define O_DEP_ADD(d) "add x0, x0, x17\n\t" "add x0, x0, x17\n\t" "add x0, x0, x17\n\t" "add x0, x0, x17\n\t"
#define DEP_ADD_N 64
KERNEL(k_dep_add, K_X I_X(0), REP16(O_DEP_ADD), "x0", "x16", "x17")

#define O_ADD_X(d) "add x" #d ", x" #d ", x17\n\t"
#define IND_ADD_CHAINS 16
#define IND_ADD_N      (IND_ADD_CHAINS * 3)
KERNEL(k_ind_add, K_X REP16(I_X), REP16(O_ADD_X) REP16(O_ADD_X) REP16(O_ADD_X), XCLOB16)

#define O_FMLA_4S_0(d) "fmla v0.4s, v30.4s, v31.4s\n\t"
#define FMA_LAT_N 40
KERNEL(k_fma_lat, K_F32 I_F32(0), REP40(O_FMLA_4S_0), "v0", "v30", "v31")

/* --- Runtime features --- */

static bool has_feature(const char *mac_key, unsigned long linux_bit) {
#if defined(__APPLE__)
    (void)linux_bit;
    int    v  = 0;
    size_t sz = sizeof(v);
    if (sysctlbyname(mac_key, &v, &sz, NULL, 0) != 0) { return false; }
    return v != 0;
#elif defined(__linux__)
    (void)mac_key;
    return (getauxval(AT_HWCAP) & linux_bit) != 0;
#else
    (void)mac_key;
    (void)linux_bit;
    return false;
#endif
}

/* Linux AT_HWCAP bits (asm/hwcap.h): ASIMDHP = 1 << 10, ASIMDDP = 1 << 20. */
static bool has_fp16(void) { return has_feature("hw.optional.arm.FEAT_FP16", 1UL << 10); }
static bool has_dotprod(void) { return has_feature("hw.optional.arm.FEAT_DotProd", 1UL << 20); }

u32 sb_cpu_kernels_get(sb_cpu_kernel *out) {
    const f64 s = SIMD_INSNS;
    bool fp16 = has_fp16();
    bool dot  = has_dotprod();
    const sb_cpu_kernel k[] = {
        { "FP64 scalar fmadd",  "GFLOPS", k_fp64_scalar, SIMD_INSNS, s * 2,      NULL },
        { "FP32 scalar fmadd",  "GFLOPS", k_fp32_scalar, SIMD_INSNS, s * 2,      NULL },
        { "INT64 scalar madd",  "GOPS",   k_int_scalar,  INT_INSNS,  INT_INSNS * 2.0, NULL },
        { "FP64 NEON fmla.2d",  "GFLOPS", k_fp64_neon,   SIMD_INSNS, s * 2 * 2,  NULL },
        { "FP32 NEON fmla.4s",  "GFLOPS", k_fp32_neon,   SIMD_INSNS, s * 4 * 2,  NULL },
        { "FP16 NEON fmla.8h",  "GFLOPS", fp16 ? k_fp16_neon : NULL, SIMD_INSNS, s * 8 * 2, "CPU lacks FEAT_FP16" },
        { "INT32 NEON mla.4s",  "GOPS",   k_int32_neon,  SIMD_INSNS, s * 4 * 2,  NULL },
        { "INT16 NEON mla.8h",  "GOPS",   k_int16_neon,  SIMD_INSNS, s * 8 * 2,  NULL },
        { "INT8 NEON mla.16b",  "GOPS",   k_int8_neon,   SIMD_INSNS, s * 16 * 2, NULL },
        { "INT8 dot udot.4s",   "GOPS",   dot ? k_int8_udot : NULL, SIMD_INSNS, s * 16 * 2, "CPU lacks FEAT_DotProd" },
    };
    static_assert(SB_ARRAY_LEN(k) <= SB_CPU_MAX_KERNELS);
    for (u32 i = 0; i < SB_ARRAY_LEN(k); i++) { out[i] = k[i]; }
    return (u32)SB_ARRAY_LEN(k);
}

void sb_cpu_ipc_get(sb_cpu_ipc *out) {
    *out = (sb_cpu_ipc){
        .dep_add        = k_dep_add,
        .dep_add_n      = DEP_ADD_N,
        .ind_add        = k_ind_add,
        .ind_add_n      = IND_ADD_N,
        .ind_add_chains = IND_ADD_CHAINS,
        .fma_lat        = k_fma_lat,
        .fma_lat_n      = FMA_LAT_N,
        .fma_tput       = k_fp32_neon,
        .fma_tput_n     = SIMD_INSNS,
        .fma_chains     = SIMD_CHAINS,
        .fma_name       = "fmla.4s",
    };
}

const char *sb_cpu_simd_name(void) {
    static char buf[64];
    snprintf(buf, sizeof(buf), "NEON%s%s", has_fp16() ? " +FP16" : "", has_dotprod() ? " +DotProd" : "");
    return buf;
}

const char *sb_cpu_kernel_note(void) {
    return "inline-asm loops, acc += a*b; 20 FP/SIMD chains, 12 scalar-int chains";
}
