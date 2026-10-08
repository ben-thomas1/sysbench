/* AArch64 bandwidth kernels in inline asm, so the hot loops are exactly what
 * is written here regardless of compiler flags (-ffast-math, LTO, unrolling).
 * Loops are 64-byte aligned so code placement can't shift the results. */
#include "mem/mem_internal.h"

const char *const mem_kern_isa = "NEON";

/* 8 x ldp q (256 B) per iteration into 8 independent 128-bit add chains. */
void mem_kern_read(const void *buf, size_t bytes, u64 passes) {
    for (u64 p = 0; p < passes; p++) {
        const u8 *q = buf;
        const u8 *e = q + bytes;
        __asm__ volatile(
            "movi v16.16b, #0\n\t" "movi v17.16b, #0\n\t" "movi v18.16b, #0\n\t" "movi v19.16b, #0\n\t"
            "movi v20.16b, #0\n\t" "movi v21.16b, #0\n\t" "movi v22.16b, #0\n\t" "movi v23.16b, #0\n\t"
            ".p2align 6\n"
            "1:\n\t"
            "ldp q0, q1, [%[q]]\n\t"
            "ldp q2, q3, [%[q], #32]\n\t"
            "ldp q4, q5, [%[q], #64]\n\t"
            "ldp q6, q7, [%[q], #96]\n\t"
            "add v16.2d, v16.2d, v0.2d\n\t" "add v17.2d, v17.2d, v1.2d\n\t"
            "add v18.2d, v18.2d, v2.2d\n\t" "add v19.2d, v19.2d, v3.2d\n\t"
            "add v20.2d, v20.2d, v4.2d\n\t" "add v21.2d, v21.2d, v5.2d\n\t"
            "add v22.2d, v22.2d, v6.2d\n\t" "add v23.2d, v23.2d, v7.2d\n\t"
            "ldp q0, q1, [%[q], #128]\n\t"
            "ldp q2, q3, [%[q], #160]\n\t"
            "ldp q4, q5, [%[q], #192]\n\t"
            "ldp q6, q7, [%[q], #224]\n\t"
            "add v16.2d, v16.2d, v0.2d\n\t" "add v17.2d, v17.2d, v1.2d\n\t"
            "add v18.2d, v18.2d, v2.2d\n\t" "add v19.2d, v19.2d, v3.2d\n\t"
            "add v20.2d, v20.2d, v4.2d\n\t" "add v21.2d, v21.2d, v5.2d\n\t"
            "add v22.2d, v22.2d, v6.2d\n\t" "add v23.2d, v23.2d, v7.2d\n\t"
            "add %[q], %[q], #256\n\t"
            "cmp %[q], %[e]\n\t"
            "b.lo 1b\n\t"
            : [q] "+r"(q)
            : [e] "r"(e)
            : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
    }
}

/* 4 x stp q (128 B) per iteration. On Apple cores this shape streams past L1
 * without reading lines first (~140 GB/s to DRAM), but in L1 its rate varies
 * with timing (85-140 GB/s). */
void mem_kern_store128(void *buf, size_t bytes, u64 passes) {
    for (u64 p = 0; p < passes; p++) {
        u8       *q = buf;
        const u8 *e = q + bytes;
        __asm__ volatile(
            "movi v0.16b, #0x5a\n\t"
            "movi v1.16b, #0xa5\n\t"
            ".p2align 6\n"
            "1:\n\t"
            "stp q0, q1, [%[q]]\n\t"
            "stp q0, q1, [%[q], #32]\n\t"
            "stp q0, q1, [%[q], #64]\n\t"
            "stp q0, q1, [%[q], #96]\n\t"
            "add %[q], %[q], #128\n\t"
            "cmp %[q], %[e]\n\t"
            "b.lo 1b\n\t"
            : [q] "+r"(q)
            : [e] "r"(e)
            : "memory", "cc", "v0", "v1");
    }
}

/* 8 x stp q (256 B) per iteration. Stable full rate in L1 (32 B/cycle), but on
 * Apple cores each line is read before it is written once it misses L1. */
void mem_kern_store256(void *buf, size_t bytes, u64 passes) {
    for (u64 p = 0; p < passes; p++) {
        u8       *q = buf;
        const u8 *e = q + bytes;
        __asm__ volatile(
            "movi v0.16b, #0x5a\n\t"
            "movi v1.16b, #0xa5\n\t"
            ".p2align 6\n"
            "1:\n\t"
            "stp q0, q1, [%[q]]\n\t"
            "stp q0, q1, [%[q], #32]\n\t"
            "stp q0, q1, [%[q], #64]\n\t"
            "stp q0, q1, [%[q], #96]\n\t"
            "stp q0, q1, [%[q], #128]\n\t"
            "stp q0, q1, [%[q], #160]\n\t"
            "stp q0, q1, [%[q], #192]\n\t"
            "stp q0, q1, [%[q], #224]\n\t"
            "add %[q], %[q], #256\n\t"
            "cmp %[q], %[e]\n\t"
            "b.lo 1b\n\t"
            : [q] "+r"(q)
            : [e] "r"(e)
            : "memory", "cc", "v0", "v1");
    }
}

/* stnp is only a hint; Apple cores treat it like a normal store at large sizes. */
void mem_kern_store_nt(void *buf, size_t bytes, u64 passes) {
    for (u64 p = 0; p < passes; p++) {
        u8       *q = buf;
        const u8 *e = q + bytes;
        __asm__ volatile(
            "movi v0.16b, #0x5a\n\t"
            "movi v1.16b, #0xa5\n\t"
            ".p2align 6\n"
            "1:\n\t"
            "stnp q0, q1, [%[q]]\n\t"
            "stnp q0, q1, [%[q], #32]\n\t"
            "stnp q0, q1, [%[q], #64]\n\t"
            "stnp q0, q1, [%[q], #96]\n\t"
            "add %[q], %[q], #128\n\t"
            "cmp %[q], %[e]\n\t"
            "b.lo 1b\n\t"
            : [q] "+r"(q)
            : [e] "r"(e)
            : "memory", "cc", "v0", "v1");
    }
}

void mem_kern_drain(void) { __asm__ volatile("dsb sy" ::: "memory"); }
