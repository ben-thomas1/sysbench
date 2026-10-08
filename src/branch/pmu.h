#pragma once

#include "core/status.h"
#include "core/types.h"

/* Per-thread branch counters for the calling thread: Linux perf_event
 * (user space only) or macOS kperf (root only, private framework loaded with
 * dlopen). Everything here degrades to SB_ERR_UNSUPPORTED / SB_ERR_RANGE so
 * the caller can fall back to estimates. */

typedef struct {
    u64  cycles;
    u64  branches;
    u64  misses;
    bool has_cycles;
} sb_branch_pmu_counts;

typedef struct {
    const char *name;           /* "perf_event" / "kperf" once initialised */
    char        reason[96];     /* why init failed */
    bool        has_cycles;
#if defined(__APPLE__)
    u32         classes;
    size_t      map[3];         /* counter index of cycles, branches, misses */
    u64         start[3];
    int         force_prev;
    void       *db;
    void       *cfg;
#else
    int         fd_branches;    /* group leader */
    int         fd_misses;
    int         fd_cycles;      /* -1 when the cycles event is unavailable */
#endif
} sb_branch_pmu;

/* Must be called on the thread that will be measured. */
sb_status_e sb_branch_pmu_init(sb_branch_pmu *p);
sb_status_e sb_branch_pmu_start(sb_branch_pmu *p);
/* SB_ERR_RANGE when the counts are partial (multiplexed, or the thread ran on
 * a PMU that does not carry the events). */
sb_status_e sb_branch_pmu_stop(sb_branch_pmu *p, sb_branch_pmu_counts *out);
void        sb_branch_pmu_free(sb_branch_pmu *p);
