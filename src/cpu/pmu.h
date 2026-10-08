#pragma once

#include "core/status.h"
#include "core/types.h"

/* User-mode cycle and instruction counters for the calling thread.
 * Linux: perf_event_open (needs perf_event_paranoid <= 2 and no seccomp
 * block). Other platforms: SB_ERR_UNSUPPORTED (macOS counters need root). */
typedef struct {
    int fd_cycles;
    int fd_insns;
} sb_cpu_pmu;

sb_status_e sb_cpu_pmu_init(sb_cpu_pmu *p);
void        sb_cpu_pmu_start(sb_cpu_pmu *p);
sb_status_e sb_cpu_pmu_stop(sb_cpu_pmu *p, u64 *out_cycles, u64 *out_insns);
void        sb_cpu_pmu_free(sb_cpu_pmu *p);
