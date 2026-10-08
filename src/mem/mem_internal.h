#pragma once

#include "core/status.h"
#include "core/types.h"

/* Shared state and helpers for the mem section (mem.c) and its parts:
 * mem_lat.c (latency, TLB), mem_bw.c (read/store/copy bandwidth),
 * mem_atomic.c (atomics), mem_alloc.c (malloc/free). */

/* Padding between independently written words. At least one cache line on
 * every supported CPU (Apple 128 B, x86 64 B); not used as a hardware fact. */
#define MEM_PAD 128u

/* Every buffer length passed to a bandwidth kernel is a multiple of this. */
#define MEM_KERN_GRAIN 512u

/* One prefaulted, page-aligned buffer shared by all single-thread and DRAM tests.
 * Size is a power of two (1 GiB, or less on small-memory machines). */
typedef struct {
    u8    *base;
    size_t bytes;
} mem_arena;

/* Timing: runs fn(ctx, reps) with reps doubled until one call takes >= 2 ms
 * (this also warms caches and TLBs), scales reps to ~25 ms, then keeps the
 * best (shortest) of 3 timed calls. Single thread, the caller's thread. */
typedef void (*mem_rep_fn)(void *ctx, u64 reps);
sb_status_e mem_time_best(mem_rep_fn fn, void *ctx, f64 *out_ns_per_rep);

/* Deterministic xorshift64* stream; seed must be non-zero. */
u64 mem_rng_next(u64 *state);
/* Uniform in [0, bound), bound >= 1. */
u32 mem_rng_below(u64 *state, u32 bound);

/* Section parts; each reports its own rows and returns only fatal errors. */
sb_status_e mem_run_latency(const mem_arena *a);
sb_status_e mem_run_bandwidth(const mem_arena *a);
sb_status_e mem_run_atomics(void);
sb_status_e mem_run_alloc(void);

/* --- Kernels (mem_kern_arm.c / mem_kern_x86.c) ---
 * buf is page-aligned, bytes is a non-zero multiple of MEM_KERN_GRAIN, and each
 * call makes `passes` sequential passes over [buf, buf + bytes). No hardware
 * barrier is issued between passes; stores are only ordered by a compiler
 * barrier. Call mem_kern_drain() once before stopping a store timer. */
extern const char *const mem_kern_isa;   /* "NEON", "AVX-512", "AVX2", "SSE2" */
void mem_kern_read(const void *buf, size_t bytes, u64 passes);
void mem_kern_store128(void *buf, size_t bytes, u64 passes);   /* 128 B per loop iteration */
void mem_kern_store256(void *buf, size_t bytes, u64 passes);   /* 256 B per loop iteration */
void mem_kern_store_nt(void *buf, size_t bytes, u64 passes);   /* non-temporal hint stores */
void mem_kern_drain(void);                                     /* wait for buffered stores */
