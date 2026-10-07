#ifdef __APPLE__

#include "sysinfo_platform.h"
#include "timer.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/sysctl.h>
#include <sys/mount.h>

static uint64_t sysctl_u64(const char *name) {
    uint64_t val = 0;
    size_t sz = sizeof(val);
    sysctlbyname(name, &val, &sz, NULL, 0);
    return val;
}

static int sysctl_int(const char *name) {
    int val = 0;
    size_t sz = sizeof(val);
    sysctlbyname(name, &val, &sz, NULL, 0);
    return val;
}

static void sysctl_str(const char *name, char *buf, size_t bufsz) {
    size_t sz = bufsz;
    if (sysctlbyname(name, buf, &sz, NULL, 0) != 0)
        buf[0] = '\0';
    else if (bufsz > 0)
        buf[bufsz - 1] = '\0';
}

void query_cpu_info_platform(struct cpu_info *info) {
    sysctl_str("machdep.cpu.brand_string", info->name, sizeof(info->name));
    info->freq_hz    = sysctl_u64("hw.cpufrequency");
    info->l1d_bytes  = sysctl_u64("hw.l1dcachesize");
    info->l2_bytes   = sysctl_u64("hw.l2cachesize");
    info->l3_bytes   = sysctl_u64("hw.l3cachesize");
    /* Generic cache keys describe the smaller cluster on Apple Silicon. */
    uint64_t perf_l1 = sysctl_u64("hw.perflevel0.l1dcachesize");
    uint64_t perf_l2 = sysctl_u64("hw.perflevel0.l2cachesize");
    if (perf_l1) info->l1d_bytes = perf_l1;
    if (perf_l2) info->l2_bytes = perf_l2;
    info->cores_perf = sysctl_int("hw.perflevel0.physicalcpu");
    info->cores_eff  = sysctl_int("hw.perflevel1.physicalcpu");
}

void query_mem_info_platform(struct mem_info *info) {
    info->total_bytes = sysctl_u64("hw.memsize");
}

void query_disk_info_platform(struct disk_info *info, const char *path) {
    struct statfs sfs;
    if (statfs(path, &sfs) == 0)
        snprintf(info->fs_type, sizeof(info->fs_type), "%s", sfs.f_fstypename);
}

double get_cpu_freq_ghz(void) {
    uint64_t freq = 0;
    size_t sz = sizeof(freq);
    if (sysctlbyname("hw.cpufrequency", &freq, &sz, NULL, 0) == 0 && freq > 0)
        return (double)freq / 1e9;
    if (sysctlbyname("hw.cpufrequency_max", &freq, &sz, NULL, 0) == 0 && freq > 0)
        return (double)freq / 1e9;

    /* Fallback: calibrate with a known loop */
    uint64_t a = 1;
    uint64_t t0 = timer_ns();
    for (uint64_t i = 0; i < 100000000ULL; i++) {
        a += 1;
        __asm__ volatile("" : "+r"(a));
    }
    uint64_t t1 = timer_ns();
    double elapsed_s = (double)(t1 - t0) / 1e9;
    return 100000000.0 / elapsed_s / 1e9;
}

#endif
