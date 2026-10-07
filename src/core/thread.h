#pragma once

#include "core/platform.h"
#include "core/status.h"
#include "core/types.h"

/* Work callback: perform `chunk` units of work on behalf of thread `tid` and
 * return the number of units actually done. Called repeatedly until the
 * measurement window closes, so a chunk should take ~0.1-10 ms. */
typedef u64 (*sb_work_fn)(void *ctx, u32 tid, u64 chunk);

typedef struct {
    u32        nthreads;
    sb_core_e  place;        /* SB_CORE_ANY, or restrict to one core type */
    u64        warmup_ns;    /* per-thread untimed spin+work before the window; 0 = SB_WARMUP_NS */
    u64        window_ns;    /* timed window; 0 = 1 s */
    u64        chunk;        /* units per callback */
    sb_work_fn fn;
    void      *ctx;
} sb_par_cfg;

typedef struct {
    u32 nthreads;
    f64 units;            /* total units completed in the window, all threads */
    f64 units_per_sec;    /* sum of per-thread rates */
    f64 min_rate;         /* slowest thread, units/s */
    f64 max_rate;         /* fastest thread, units/s */
} sb_par_result;

/* Time-based measurement: all threads start together, run `fn` for the same
 * wall-clock window, and each thread's rate is measured over its own time.
 * Thread startup and teardown are outside the window. */
sb_status_e sb_par_run(const sb_par_cfg *cfg, sb_par_result *out);

/* Bias/pin the calling thread. `idx` selects among CPUs of that type on Linux
 * (round-robin); macOS uses QoS classes (PERF -> USER_INTERACTIVE,
 * EFF -> BACKGROUND, ANY -> USER_INTERACTIVE) since pinning is unavailable. */
sb_status_e sb_thread_place_self(sb_core_e place, u32 idx);

/* Number of CPUs usable for `place` (falls back to all CPUs if not hybrid). */
u32 sb_thread_count_for(sb_core_e place);
