#include "cpu/pmu.h"

#if defined(__linux__)

#include <linux/perf_event.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

static int open_counter(u64 config, int group) {
    struct perf_event_attr a;
    memset(&a, 0, sizeof(a));
    a.type           = PERF_TYPE_HARDWARE;
    a.size           = sizeof(a);
    a.config         = config;
    a.exclude_kernel = 1;
    a.exclude_hv     = 1;
    if (group < 0) { a.disabled = 1; } /* members follow the leader */
    return (int)syscall(SYS_perf_event_open, &a, 0, -1, group, 0);
}

sb_status_e sb_cpu_pmu_init(sb_cpu_pmu *p) {
    p->fd_cycles = open_counter(PERF_COUNT_HW_CPU_CYCLES, -1);
    if (p->fd_cycles < 0) {
        p->fd_insns = -1;
        return SB_ERR_UNSUPPORTED;
    }
    p->fd_insns = open_counter(PERF_COUNT_HW_INSTRUCTIONS, p->fd_cycles);
    if (p->fd_insns < 0) {
        sb_cpu_pmu_free(p);
        return SB_ERR_UNSUPPORTED;
    }
    return SB_OK;
}

void sb_cpu_pmu_start(sb_cpu_pmu *p) {
    ioctl(p->fd_cycles, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
    ioctl(p->fd_cycles, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
}

sb_status_e sb_cpu_pmu_stop(sb_cpu_pmu *p, u64 *out_cycles, u64 *out_insns) {
    ioctl(p->fd_cycles, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    u64 c = 0;
    u64 i = 0;
    if (read(p->fd_cycles, &c, sizeof(c)) != (ssize_t)sizeof(c)) { return SB_ERR_IO; }
    if (read(p->fd_insns, &i, sizeof(i)) != (ssize_t)sizeof(i)) { return SB_ERR_IO; }
    if (c == 0) { return SB_ERR_RANGE; }
    *out_cycles = c;
    *out_insns  = i;
    return SB_OK;
}

void sb_cpu_pmu_free(sb_cpu_pmu *p) {
    if (p->fd_insns >= 0) { close(p->fd_insns); }
    if (p->fd_cycles >= 0) { close(p->fd_cycles); }
    p->fd_cycles = -1;
    p->fd_insns  = -1;
}

#else

sb_status_e sb_cpu_pmu_init(sb_cpu_pmu *p) {
    p->fd_cycles = -1;
    p->fd_insns  = -1;
    return SB_ERR_UNSUPPORTED;
}

void sb_cpu_pmu_start(sb_cpu_pmu *p) { (void)p; }

sb_status_e sb_cpu_pmu_stop(sb_cpu_pmu *p, u64 *out_cycles, u64 *out_insns) {
    (void)p;
    (void)out_cycles;
    (void)out_insns;
    return SB_ERR_UNSUPPORTED;
}

void sb_cpu_pmu_free(sb_cpu_pmu *p) {
    p->fd_cycles = -1;
    p->fd_insns  = -1;
}

#endif
