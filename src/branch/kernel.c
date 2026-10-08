#include "branch/kernel.h"

/* Layout of the conditional part (arm64; x86-64 is the same with jnz / jmp /
 * jz and flag-preserving lea):
 *
 *        tbnz  t, #0, 2f     value 1: taken          value 0: not taken
 *        add   acc, #7       value 0
 *        b     3f            value 0: taken
 *     2: add   acc, #3       value 1
 *        tbz   t, #0, 3f     value 1: never taken (bit 0 is set here)
 *     3: subs  n; b.ne 1b    loop back-edge, taken
 *
 * So pattern value 1 means "the measured branch is taken". The old kernel
 * ran an extra taken branch on one outcome, so its "extra time" mixed path
 * cost with mispredict cost.
 *
 * noinline: LTO otherwise clones the loop once per pattern, and every
 * pattern should run the same code at the same address. */

#if defined(__aarch64__)

[[gnu::noinline]] u64 sb_branch_kernel(const u8 *pat, u64 mask, u64 n, u64 acc) {
    u64 idx = 0;
    u32 t;
    __asm__ volatile(
        ".p2align 6\n"
        "1:\n\t"
        "ldrb   %w[t], [%[pat], %[idx]]\n\t"
        "add    %[idx], %[idx], #1\n\t"
        "and    %[idx], %[idx], %[mask]\n\t"
        "tbnz   %w[t], #0, 2f\n\t"
        "add    %[acc], %[acc], #7\n\t"
        "b      3f\n"
        "2:\n\t"
        "add    %[acc], %[acc], #3\n\t"
        "tbz    %w[t], #0, 3f\n"
        "3:\n\t"
        "subs   %[n], %[n], #1\n\t"
        "b.ne   1b\n\t"
        : [acc] "+r"(acc), [idx] "+r"(idx), [n] "+r"(n), [t] "=&r"(t)
        : [pat] "r"(pat), [mask] "r"(mask)
        : "cc", "memory");
    return acc;
}

[[gnu::noinline]] u64 sb_branch_add_chain(u64 n, u64 acc) {
    __asm__ volatile(
        ".p2align 6\n"
        "1:\n\t"
        ".rept 100\n\t"
        "add    %[a], %[a], #1\n\t"
        ".endr\n\t"
        "subs   %[n], %[n], #1\n\t"
        "b.ne   1b\n\t"
        : [a] "+r"(acc), [n] "+r"(n)
        :
        : "cc");
    return acc;
}

#elif defined(__x86_64__)

[[gnu::noinline]] u64 sb_branch_kernel(const u8 *pat, u64 mask, u64 n, u64 acc) {
    u64 idx = 0;
    u32 t;
    __asm__ volatile(
        ".p2align 6\n"
        "1:\n\t"
        "movzbl (%[pat],%[idx]), %k[t]\n\t"
        "addq   $1, %[idx]\n\t"
        "andq   %[mask], %[idx]\n\t"
        "testl  $1, %k[t]\n\t"
        "jnz    2f\n\t"
        "leaq   7(%[acc]), %[acc]\n\t"
        "jmp    3f\n"
        "2:\n\t"
        "leaq   3(%[acc]), %[acc]\n\t"
        "jz     3f\n"                   /* ZF is still 0 here: never taken */
        "3:\n\t"
        "subq   $1, %[n]\n\t"
        "jnz    1b\n\t"
        : [acc] "+r"(acc), [idx] "+r"(idx), [n] "+r"(n), [t] "=&r"(t)
        : [pat] "r"(pat), [mask] "r"(mask)
        : "cc", "memory");
    return acc;
}

[[gnu::noinline]] u64 sb_branch_add_chain(u64 n, u64 acc) {
    __asm__ volatile(
        ".p2align 6\n"
        "1:\n\t"
        ".rept 100\n\t"
        "addq   $1, %[a]\n\t"
        ".endr\n\t"
        "subq   $1, %[n]\n\t"
        "jnz    1b\n\t"
        : [a] "+r"(acc), [n] "+r"(n)
        :
        : "cc");
    return acc;
}

#else
#error "branch: unsupported architecture (arm64 and x86-64 only)"
#endif

static_assert(SB_BRANCH_CHAIN_LEN == 100, ".rept count in the asm must match SB_BRANCH_CHAIN_LEN");
