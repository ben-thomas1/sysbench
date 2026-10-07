#pragma once

#include "core/types.h"

#include <time.h>

/* Monotonic nanoseconds. macOS: CLOCK_UPTIME_RAW (mach_absolute_time in ns,
 * 24 MHz / ~42 ns resolution on Apple Silicon). Linux: CLOCK_MONOTONIC_RAW. */
static inline u64 sb_timer_now_ns(void) {
#if defined(__APPLE__)
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (u64)ts.tv_sec * 1'000'000'000ULL + (u64)ts.tv_nsec;
#endif
}

/* Busy-spin the calling thread for `ns` so the core reaches its sustained
 * clock before a timed region (DVFS ramp takes ~100-200 ms). */
void sb_timer_spin(u64 ns);

#define SB_WARMUP_NS 250'000'000ULL
