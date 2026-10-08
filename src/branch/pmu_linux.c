#include "branch/pmu.h"

#include <errno.h>
#include <linux/perf_event.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

/* Group read layout for PERF_FORMAT_GROUP | TOTAL_TIME_ENABLED | TOTAL_TIME_RUNNING. */
typedef struct {
    u64 nr;
    u64 time_enabled;
    u64 time_running;
    u64 values[3];
} group_read;

static int open_hw(u64 config, int group_fd) {
    struct perf_event_attr a;
    memset(&a, 0, sizeof(a));
    a.type           = PERF_TYPE_HARDWARE;
    a.size           = sizeof(a);
    a.config         = config;
    a.disabled       = group_fd < 0 ? 1 : 0;
    a.exclude_kernel = 1;
    a.exclude_hv     = 1;
    a.read_format    = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED |
                       PERF_FORMAT_TOTAL_TIME_RUNNING;
    return (int)syscall(SYS_perf_event_open, &a, 0, -1, group_fd, 0UL);
}

static void close_fd(int *fd) {
    if (*fd >= 0) { close(*fd); }
    *fd = -1;
}

sb_status_e sb_branch_pmu_init(sb_branch_pmu *p) {
    memset(p, 0, sizeof(*p));
    p->fd_branches = -1;
    p->fd_misses   = -1;
    p->fd_cycles   = -1;

    p->fd_branches = open_hw(PERF_COUNT_HW_BRANCH_INSTRUCTIONS, -1);
    if (p->fd_branches < 0) {
        snprintf(p->reason, sizeof(p->reason), "perf_event_open: %s", strerror(errno));
        return SB_ERR_UNSUPPORTED;
    }
    p->fd_misses = open_hw(PERF_COUNT_HW_BRANCH_MISSES, p->fd_branches);
    if (p->fd_misses < 0) {
        snprintf(p->reason, sizeof(p->reason), "perf branch-misses: %s", strerror(errno));
        close_fd(&p->fd_branches);
        return SB_ERR_UNSUPPORTED;
    }
    /* Cycles are optional (some VMs expose branch events only). */
    p->fd_cycles  = open_hw(PERF_COUNT_HW_CPU_CYCLES, p->fd_branches);
    p->has_cycles = p->fd_cycles >= 0;
    p->name       = "perf_event";
    return SB_OK;
}

sb_status_e sb_branch_pmu_start(sb_branch_pmu *p) {
    if (ioctl(p->fd_branches, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) < 0) { return SB_ERR_SYS; }
    if (ioctl(p->fd_branches, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) < 0) { return SB_ERR_SYS; }
    return SB_OK;
}

sb_status_e sb_branch_pmu_stop(sb_branch_pmu *p, sb_branch_pmu_counts *out) {
    if (ioctl(p->fd_branches, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) < 0) { return SB_ERR_SYS; }

    u64 want = p->has_cycles ? 3 : 2;
    group_read g;
    memset(&g, 0, sizeof(g));
    ssize_t size = (ssize_t)((3 + want) * sizeof(u64));
    if (read(p->fd_branches, &g, sizeof(g)) != size) { return SB_ERR_IO; }
    if (g.nr != want) { return SB_ERR_RANGE; }
    /* Partial counts: the group was multiplexed with other events, or the
     * thread ran on a core whose PMU does not carry these events (hybrid). */
    if (g.time_running == 0 || g.time_running != g.time_enabled) { return SB_ERR_RANGE; }

    /* Values come in group order: branches, misses, [cycles]. */
    out->branches   = g.values[0];
    out->misses     = g.values[1];
    out->cycles     = p->has_cycles ? g.values[2] : 0;
    out->has_cycles = p->has_cycles;
    return SB_OK;
}

void sb_branch_pmu_free(sb_branch_pmu *p) {
    close_fd(&p->fd_cycles);
    close_fd(&p->fd_misses);
    close_fd(&p->fd_branches);
    p->name = NULL;
}
