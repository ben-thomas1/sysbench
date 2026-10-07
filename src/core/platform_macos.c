#include "core/platform.h"

#include <stdio.h>
#include <string.h>
#include <sys/sysctl.h>

void sb_platform_query_os(sb_platform *out);

static u64 sysctl_u64(const char *name) {
    u64    v  = 0;
    size_t sz = sizeof(v);
    if (sysctlbyname(name, &v, &sz, NULL, 0) != 0) { return 0; }
    /* some keys are 32-bit */
    if (sz == sizeof(u32)) { return (u64)(u32)v; }
    return v;
}

void sb_platform_query_os(sb_platform *out) {
    size_t sz = sizeof(out->cpu);
    if (sysctlbyname("machdep.cpu.brand_string", out->cpu, &sz, NULL, 0) != 0) { out->cpu[0] = '\0'; }
    out->cpu[sizeof(out->cpu) - 1] = '\0';

    char prod[32] = "";
    size_t psz = sizeof(prod);
    if (sysctlbyname("kern.osproductversion", prod, &psz, NULL, 0) == 0) {
        snprintf(out->os, sizeof(out->os), "macOS %s", prod);
    }

    u64 n = sysctl_u64("hw.logicalcpu");
    out->ncpu = n > SB_MAX_CPUS ? SB_MAX_CPUS : (u32)n;
    for (u32 i = 0; i < out->ncpu; i++) { out->cpu_ids[i] = (u16)i; }

    /* Apple Silicon: perflevel0 = performance, perflevel1 = efficiency. */
    if (sysctl_u64("hw.nperflevels") >= 2) {
        out->nperf = (u32)sysctl_u64("hw.perflevel0.logicalcpu");
        out->neff  = (u32)sysctl_u64("hw.perflevel1.logicalcpu");
    }

    out->cache_line = (u32)sysctl_u64("hw.cachelinesize");
    out->l1d_bytes  = sysctl_u64("hw.perflevel0.l1dcachesize");
    out->l2_bytes   = sysctl_u64("hw.perflevel0.l2cachesize");
    if (out->l1d_bytes == 0) { out->l1d_bytes = sysctl_u64("hw.l1dcachesize"); }
    if (out->l2_bytes == 0) { out->l2_bytes = sysctl_u64("hw.l2cachesize"); }
    out->l3_bytes    = sysctl_u64("hw.l3cachesize");
    out->mem_bytes   = sysctl_u64("hw.memsize");
    /* Not exposed on Apple Silicon; present on Intel Macs. */
    out->freq_max_hz = sysctl_u64("hw.cpufrequency_max");
}
