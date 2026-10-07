#include "core/platform.h"

#include <stdio.h>
#include <string.h>
#include <sys/utsname.h>
#include <unistd.h>

/* Implemented in platform_macos.c / platform_linux.c. */
void sb_platform_query_os(sb_platform *out);

static sb_platform g_platform;
static bool        g_platform_ready;

sb_status_e sb_platform_query(sb_platform *out) {
    memset(out, 0, sizeof(*out));

    struct utsname un;
    if (uname(&un) == 0) {
        snprintf(out->os, sizeof(out->os), "%s", un.sysname);
        snprintf(out->os_release, sizeof(out->os_release), "%s", un.release);
    }
#if defined(__aarch64__)
    snprintf(out->arch, sizeof(out->arch), "arm64");
#elif defined(__x86_64__)
    snprintf(out->arch, sizeof(out->arch), "x86_64");
#else
    snprintf(out->arch, sizeof(out->arch), "unknown");
#endif

    long page = sysconf(_SC_PAGESIZE);
    out->page_size = page > 0 ? (u32)page : 4096;

    sb_platform_query_os(out);

    if (out->ncpu == 0) {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        out->ncpu = n > 0 ? (u32)n : 1;
        if (out->ncpu > SB_MAX_CPUS) { out->ncpu = SB_MAX_CPUS; }
        for (u32 i = 0; i < out->ncpu; i++) { out->cpu_ids[i] = (u16)i; }
    }
    if (out->cache_line == 0) { out->cache_line = 64; }
    return SB_OK;
}

const sb_platform *sb_platform_get(void) {
    if (!g_platform_ready) {
        (void)sb_platform_query(&g_platform);
        g_platform_ready = true;
    }
    return &g_platform;
}
