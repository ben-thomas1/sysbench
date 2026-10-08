#pragma once

#include "core/types.h"

/* Branch loop: for i in [0, n): v = pat[i & mask]; one conditional branch,
 * taken when bit 0 of v is set; acc += v ? 3 : 7. Written entirely in inline asm so release
 * flags cannot reshape it. Both outcomes execute the same number of
 * instructions (3 after the test), the same number of branches (2) and the
 * same number of taken branches (1). With the loop back-edge every iteration
 * retires 3 branches, 2 of them taken, whatever the outcome.
 * Preconditions: n >= 1, mask + 1 is a power of two, pat holds mask + 1 bytes. */
u64 sb_branch_kernel(const u8 *pat, u64 mask, u64 n, u64 acc);

/* Clock reference: n iterations of SB_BRANCH_CHAIN_LEN dependent integer adds
 * (1 cycle each on every supported core). Precondition: n >= 1. */
#define SB_BRANCH_CHAIN_LEN 100
u64 sb_branch_add_chain(u64 n, u64 acc);
