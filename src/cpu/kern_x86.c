#include "cpu/kern.h"

/* x86-64 kernels. The SIMD tier is chosen at build time from -march:
 *   AVX-512 (__AVX512F__): zmm, 16 chains (32 registers), operands zmm30/31
 *   AVX2    (__AVX2__):    ymm, 12 chains (16 registers), operands ymm14/15
 *   SSE2    (baseline):    xmm, 12 chains, operands xmm14/15
 * FMA latency is 4 cycles on current Intel and AMD cores with two FMA pipes,
 * so 8 chains saturate; 12 and 16 leave margin. With FMA the kernels use the
 * accumulator form (vfmadd231: acc += a * b). Without a fused instruction
 * (SSE2, integer lanes) each step is `acc = acc * a; acc = acc + b`, which
 * keeps both operations on the chain so neither can be hoisted, and the
 * integer multiply latency (up to 10 cycles) is covered by the chain count.
 * Not validated on hardware yet; see FINDINGS §14. */

#define REP8(M)  M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)
#define REP12(M) REP8(M) M(8) M(9) M(10) M(11)
#define REP16(M) REP12(M) M(12) M(13) M(14) M(15)
#define REP40(M) REP16(M) REP16(M) REP8(M)

#define XCLOB12 "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", \
                "xmm10", "xmm11", "xmm14", "xmm15"

#if defined(__AVX512F__)
#define TIER   "AVX-512"
#define VR     "zmm"
#define KA     "30"
#define KB     "31"
#define VMOV   "vmovups"
#define VCOPY  "vmovaps"
#define REPV   REP16
#define VCH    16
#define VBYTES 64
#define VEND   "vzeroupper\n\t"
#define VCLOB  "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", \
               "xmm11", "xmm12", "xmm13", "xmm14", "xmm15", "xmm30", "xmm31"
#elif defined(__AVX2__)
#define TIER   "AVX2"
#define VR     "ymm"
#define KA     "14"
#define KB     "15"
#define VMOV   "vmovups"
#define VCOPY  "vmovaps"
#define REPV   REP12
#define VCH    12
#define VBYTES 32
#define VEND   "vzeroupper\n\t"
#define VCLOB  XCLOB12
#else
#define TIER   "SSE2"
#define VR     "xmm"
#define KA     "14"
#define KB     "15"
#define VMOV   "movups"
#define VCOPY  "movaps"
#define REPV   REP12
#define VCH    12
#define VBYTES 16
#define VEND   ""
#define VCLOB  XCLOB12
#endif

#if defined(__AVX__)
#define SEND "vzeroupper\n\t"
#define SMOV "vmovups"
#else
#define SEND ""
#define SMOV "movups"
#endif

#define VREPS 2
#define VSTEPS (VCH * VREPS)   /* multiply-add steps per iteration */
#define SCH    12
#define SSTEPS (SCH * VREPS)

/* Operand tables: row 0 = a, row 1 = b. FP: a = 0.5, b = 0.25 (bit patterns,
 * so the FP16 table needs no _Float16). Integer values are arbitrary odd. */
alignas(64) static const u32 kf32[2][16] = {
    { [0 ... 15] = 0x3F000000u }, { [0 ... 15] = 0x3E800000u } };
alignas(64) static const u64 kf64[2][8] = {
    { [0 ... 7] = 0x3FE0000000000000ull }, { [0 ... 7] = 0x3FD0000000000000ull } };
#if defined(__AVX512FP16__)
alignas(64) static const u16 kf16[2][32] = { { [0 ... 31] = 0x3800u }, { [0 ... 31] = 0x3400u } };
#endif
alignas(64) static const u32 kint[2][16] = {
    { [0 ... 15] = 0x03050307u }, { [0 ... 15] = 0x01010101u } };

/* One kernel: load operands a/b, copy a into every accumulator, then `chunk`
 * iterations of `body`. */
#define KERNEL(fn, tab, load, init, body, end, ...)                                 \
    static u64 fn(void *ctx, u32 tid, u64 chunk) {                                  \
        (void)ctx;                                                                  \
        (void)tid;                                                                  \
        u64 n = chunk;                                                              \
        __asm__ volatile(load init ".p2align 6\n1:\n\t" body "subq $1, %[n]\n\tjnz 1b\n\t" end    \
                         : [n] "+r"(n)                                              \
                         : [a] "m"(tab[0]), [b] "m"(tab[1])                         \
                         : "cc", __VA_ARGS__);                                           \
        return chunk;                                                               \
    }

/* --- Vector (tier width) --- */
#define V_LOAD   VMOV " %[a], %%" VR KA "\n\t" VMOV " %[b], %%" VR KB "\n\t"
#define V_INIT(d) VCOPY " %%" VR KA ", %%" VR #d "\n\t"
#define V_BODY(op) REPV(op) REPV(op)

#define V_FMA_PS(d) "vfmadd231ps %%" VR KA ", %%" VR KB ", %%" VR #d "\n\t"
#define V_FMA_PD(d) "vfmadd231pd %%" VR KA ", %%" VR KB ", %%" VR #d "\n\t"
#define V_FMA_PH(d) "vfmadd231ph %%" VR KA ", %%" VR KB ", %%" VR #d "\n\t"
#if defined(__AVX__)
#define V_MULADD(mul, add, d) "v" mul " %%" VR KA ", %%" VR #d ", %%" VR #d "\n\t" \
                              "v" add " %%" VR KB ", %%" VR #d ", %%" VR #d "\n\t"
#else
#define V_MULADD(mul, add, d) mul " %%" VR KA ", %%" VR #d "\n\t" add " %%" VR KB ", %%" VR #d "\n\t"
#endif
#define V_MA_PS(d)  V_MULADD("mulps", "addps", d)
#define V_MA_PD(d)  V_MULADD("mulpd", "addpd", d)
#define V_MA_D(d)   V_MULADD("pmulld", "paddd", d)
#define V_MA_W(d)   V_MULADD("pmullw", "paddw", d)
#define V_MA_Q(d)   V_MULADD("pmullq", "paddq", d)
#if defined(__AVX512VNNI__)
#define V_DPBUSD(d) "vpdpbusd %%" VR KA ", %%" VR KB ", %%" VR #d "\n\t"
#elif defined(__AVXVNNI__)
#define V_DPBUSD(d) "%{vex%} vpdpbusd %%" VR KA ", %%" VR KB ", %%" VR #d "\n\t"
#endif

/* --- Scalar (xmm low lane / general-purpose registers) --- */
#define S_LOAD   SMOV " %[a], %%xmm14\n\t" SMOV " %[b], %%xmm15\n\t"
#if defined(__AVX__)
#define S_INIT(d) "vmovaps %%xmm14, %%xmm" #d "\n\t"
#define S_MULADD(mul, add, d) "v" mul " %%xmm14, %%xmm" #d ", %%xmm" #d "\n\t" \
                              "v" add " %%xmm15, %%xmm" #d ", %%xmm" #d "\n\t"
#else
#define S_INIT(d) "movaps %%xmm14, %%xmm" #d "\n\t"
#define S_MULADD(mul, add, d) mul " %%xmm14, %%xmm" #d "\n\t" add " %%xmm15, %%xmm" #d "\n\t"
#endif
#define S_BODY(op) REP12(op) REP12(op)
#define S_FMA_SD(d) "vfmadd231sd %%xmm14, %%xmm15, %%xmm" #d "\n\t"
#define S_FMA_SS(d) "vfmadd231ss %%xmm14, %%xmm15, %%xmm" #d "\n\t"
#define S_MA_SD(d)  S_MULADD("mulsd", "addsd", d)
#define S_MA_SS(d)  S_MULADD("mulss", "addss", d)

#if defined(__FMA__)
#define HAS_FMA 1
KERNEL(k_fp64_scalar, kf64, S_LOAD, REP12(S_INIT), S_BODY(S_FMA_SD), SEND, XCLOB12)
KERNEL(k_fp32_scalar, kf32, S_LOAD, REP12(S_INIT), S_BODY(S_FMA_SS), SEND, XCLOB12)
KERNEL(k_fp64_simd,   kf64, V_LOAD, REPV(V_INIT),  V_BODY(V_FMA_PD), VEND, VCLOB)
KERNEL(k_fp32_simd,   kf32, V_LOAD, REPV(V_INIT),  V_BODY(V_FMA_PS), VEND, VCLOB)
#define FP_STEP_INSNS 1
#define FMA_NAME "vfmadd231ps"
#define FP_TAG   "FMA"
#define O_LAT(d) "vfmadd231ps %%xmm14, %%xmm15, %%xmm0\n\t"
#else
#define HAS_FMA 0
KERNEL(k_fp64_scalar, kf64, S_LOAD, REP12(S_INIT), S_BODY(S_MA_SD), SEND, XCLOB12)
KERNEL(k_fp32_scalar, kf32, S_LOAD, REP12(S_INIT), S_BODY(S_MA_SS), SEND, XCLOB12)
KERNEL(k_fp64_simd,   kf64, V_LOAD, REPV(V_INIT),  V_BODY(V_MA_PD), VEND, VCLOB)
KERNEL(k_fp32_simd,   kf32, V_LOAD, REPV(V_INIT),  V_BODY(V_MA_PS), VEND, VCLOB)
#define FP_STEP_INSNS 2
#define FMA_NAME "mulps+addps"
#define FP_TAG   "mul+add"
#define O_LAT(d) S_MULADD("mulps", "addps", 0)
#endif

#if defined(__SSE4_1__)
#define HAS_INT32 1
KERNEL(k_int32_simd, kint, V_LOAD, REPV(V_INIT), V_BODY(V_MA_D), VEND, VCLOB)
#endif
#if !defined(__AVX512F__) || defined(__AVX512BW__)
#define HAS_INT16 1
KERNEL(k_int16_simd, kint, V_LOAD, REPV(V_INIT), V_BODY(V_MA_W), VEND, VCLOB)
#endif
#if defined(__AVX512DQ__)
#define HAS_INT64 1
KERNEL(k_int64_simd, kint, V_LOAD, REPV(V_INIT), V_BODY(V_MA_Q), VEND, VCLOB)
#endif
#if defined(V_DPBUSD)
#define HAS_VNNI 1
KERNEL(k_int8_dot, kint, V_LOAD, REPV(V_INIT), V_BODY(V_DPBUSD), VEND, VCLOB)
#endif
#if defined(__AVX512FP16__)
#define HAS_FP16 1
KERNEL(k_fp16_simd, kf16, V_LOAD, REPV(V_INIT), V_BODY(V_FMA_PH), VEND, VCLOB)
#endif

/* Scalar integer: 8 chains of `acc = acc * a + 7` in general-purpose registers
 * (imul latency 3, add 1: 8 chains cover the single multiply port twice). */
#define GP8(M) M(rax) M(rcx) M(rdx) M(rsi) M(rdi) M(r8) M(r9) M(r10)
#define G_INIT(r) "movq %%r11, %%" #r "\n\t"
#define G_MA(r)   "imulq %%r11, %%" #r "\n\taddq $7, %%" #r "\n\t"
#define GCLOB8 "rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11"
#define GSTEPS (8 * 6)
KERNEL(k_int_scalar, kint, "movq %[a], %%r11\n\t", GP8(G_INIT), GP8(G_MA) GP8(G_MA) GP8(G_MA)
       GP8(G_MA) GP8(G_MA) GP8(G_MA), "", GCLOB8)

/* --- Ops-per-cycle kernels --- */

/* Clock reference: one register-register add chain, 64 per iteration
 * (register operand: some cores shortcut dependent adds of small immediates). */
#define O_DEP_ADD(d) "addq %%r11, %%rax\n\taddq %%r11, %%rax\n\taddq %%r11, %%rax\n\taddq %%r11, %%rax\n\t"
#define DEP_ADD_N 64
KERNEL(k_dep_add, kint, "movq %[b], %%r11\n\txorl %%eax, %%eax\n\t", "", REP16(O_DEP_ADD), "", "rax", "r11")

#define GP12(M) M(rax) M(rcx) M(rdx) M(rsi) M(rdi) M(r8) M(r9) M(r10) M(rbx) M(r12) M(r13) M(r14)
#define G_ADD(r) "addq %%r11, %%" #r "\n\t"
#define IND_ADD_CHAINS 12
#define IND_ADD_N      (IND_ADD_CHAINS * 4)
KERNEL(k_ind_add, kint, "movq %[b], %%r11\n\t", GP12(G_INIT), GP12(G_ADD) GP12(G_ADD) GP12(G_ADD) GP12(G_ADD),
       "", "rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "rbx", "r12", "r13", "r14", "r11")

#define FMA_LAT_N 40
KERNEL(k_fma_lat, kf32, S_LOAD, S_INIT(0), REP40(O_LAT), SEND, "xmm0", "xmm14", "xmm15")

#define LANES32 (VBYTES / 4)

u32 sb_cpu_kernels_get(sb_cpu_kernel *out) {
    const f64 v  = VSTEPS;
    const f64 s  = SSTEPS;
    const u32 vi = VSTEPS * FP_STEP_INSNS;
    const u32 si = SSTEPS * FP_STEP_INSNS;
    const sb_cpu_kernel k[] = {
        { "FP64 scalar " FP_TAG, "GFLOPS", k_fp64_scalar, si, s * 2, NULL },
        { "FP32 scalar " FP_TAG, "GFLOPS", k_fp32_scalar, si, s * 2, NULL },
        { "INT64 scalar imul+add", "GOPS",   k_int_scalar,  GSTEPS * 2, GSTEPS * 2.0, NULL },
        { "FP64 " TIER " " FP_TAG, "GFLOPS", k_fp64_simd, vi, v * (LANES32 / 2) * 2, NULL },
        { "FP32 " TIER " " FP_TAG, "GFLOPS", k_fp32_simd, vi, v * LANES32 * 2, NULL },
#if defined(HAS_FP16)
        { "FP16 " TIER " vfmadd231ph", "GFLOPS", k_fp16_simd, VSTEPS, v * LANES32 * 2 * 2, NULL },
#else
        { "FP16 " TIER " vfmadd231ph", "GFLOPS", NULL, 0, 0, "needs AVX512-FP16 build" },
#endif
#if defined(HAS_INT64)
        { "INT64 " TIER " pmullq+add", "GOPS", k_int64_simd, VSTEPS * 2, v * (LANES32 / 2) * 2, NULL },
#else
        { "INT64 " TIER " pmullq+add", "GOPS", NULL, 0, 0, "needs AVX512DQ build" },
#endif
#if defined(HAS_INT32)
        { "INT32 " TIER " pmulld+add", "GOPS", k_int32_simd, VSTEPS * 2, v * LANES32 * 2, NULL },
#else
        { "INT32 " TIER " pmulld+add", "GOPS", NULL, 0, 0, "needs SSE4.1 build" },
#endif
#if defined(HAS_INT16)
        { "INT16 " TIER " pmullw+add", "GOPS", k_int16_simd, VSTEPS * 2, v * LANES32 * 2 * 2, NULL },
#else
        { "INT16 " TIER " pmullw+add", "GOPS", NULL, 0, 0, "needs AVX512BW build" },
#endif
#if defined(HAS_VNNI)
        /* vpdpbusd: 4 u8 x s8 products summed into each 32-bit lane. */
        { "INT8 dot " TIER " vpdpbusd", "GOPS", k_int8_dot, VSTEPS, v * LANES32 * 4 * 2, NULL },
#else
        { "INT8 dot " TIER " vpdpbusd", "GOPS", NULL, 0, 0, "needs AVX-VNNI/AVX512-VNNI build" },
#endif
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
        .fma_tput       = k_fp32_simd,
        .fma_tput_n     = VSTEPS * FP_STEP_INSNS,
        .fma_chains     = VCH,
        .fma_name       = FMA_NAME,
    };
}

const char *sb_cpu_simd_name(void) {
    return TIER
#if HAS_FMA
        " +FMA"
#endif
#if defined(__AVX512DQ__)
        " +DQ"
#endif
#if defined(__AVX512BW__)
        " +BW"
#endif
#if defined(__AVX512VNNI__) || defined(__AVXVNNI__)
        " +VNNI"
#endif
#if defined(__AVX512FP16__)
        " +FP16"
#endif
        " (build-time tier)";
}

const char *sb_cpu_kernel_note(void) {
#if defined(__AVX512F__)
    return "inline-asm loops; 16 SIMD chains, 12 scalar-FP chains, 8 scalar-int chains";
#else
    return "inline-asm loops; 12 SIMD and scalar-FP chains, 8 scalar-int chains";
#endif
}
