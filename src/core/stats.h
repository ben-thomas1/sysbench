#pragma once

#include "core/status.h"
#include "core/types.h"

typedef struct {
    u32 n;
    f64 median;
    f64 min;
    f64 max;
    f64 mean;
} sb_stats;

/* Summary statistics of v[0..n). */
sb_status_e sb_stats_compute(const f64 *v, u32 n, sb_stats *out);
