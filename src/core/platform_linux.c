#include "core/platform.h"

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void sb_platform_query_os(sb_platform *out);

/* Read the first line of a sysfs/procfs file into buf; returns false if missing. */
static bool read_line(const char *path, char *buf, size_t cap) {
    FILE *f = fopen(path, "r");
    if (f == NULL) { return false; }
    bool ok = fgets(buf, (int)cap, f) != NULL;
    fclose(f);
    if (ok) { buf[strcspn(buf, "\n")] = '\0'; }
    return ok;
}

static u64 read_u64(const char *path) {
    char buf[64];
    if (!read_line(path, buf, sizeof(buf))) { return 0; }
    return strtoull(buf, NULL, 10);
}

/* "512K" / "32M" style sysfs cache sizes. */
static u64 read_size(const char *path) {
    char buf[64];
    if (!read_line(path, buf, sizeof(buf))) { return 0; }
    char *end = NULL;
    u64 v = strtoull(buf, &end, 10);
    if (end != NULL && (*end == 'K' || *end == 'k')) { v *= 1024; }
    if (end != NULL && (*end == 'M' || *end == 'm')) { v *= 1024 * 1024; }
    return v;
}

/* Parse a CPU list like "0-7,16-23" into ids; returns count. */
static u32 parse_cpu_list(const char *s, u16 *ids, u32 cap) {
    u32 n = 0;
    while (*s != '\0' && n < cap) {
        char *end = NULL;
        unsigned long lo = strtoul(s, &end, 10);
        if (end == s) { break; }
        unsigned long hi = lo;
        s = end;
        if (*s == '-') {
            hi = strtoul(s + 1, &end, 10);
            s  = end;
        }
        for (unsigned long c = lo; c <= hi && n < cap; c++) { ids[n++] = (u16)c; }
        if (*s == ',') { s++; }
    }
    return n;
}

static bool cpu_allowed(const sb_platform *p, u16 id) {
    for (u32 i = 0; i < p->ncpu; i++) {
        if (p->cpu_ids[i] == id) { return true; }
    }
    return false;
}

/* Keep only ids that are in the process affinity mask. */
static u32 filter_allowed(const sb_platform *p, u16 *ids, u32 n) {
    u32 k = 0;
    for (u32 i = 0; i < n; i++) {
        if (cpu_allowed(p, ids[i])) { ids[k++] = ids[i]; }
    }
    return k;
}

void sb_platform_query_os(sb_platform *out) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f != NULL) {
        char line[512];
        while (fgets(line, sizeof(line), f) != NULL) {
            if (strncmp(line, "model name", 10) == 0) {
                char *p = strchr(line, ':');
                if (p != NULL) {
                    p++;
                    while (*p == ' ') { p++; }
                    p[strcspn(p, "\n")] = '\0';
                    snprintf(out->cpu, sizeof(out->cpu), "%s", p);
                }
                break;
            }
        }
        fclose(f);
    }

    char pretty[128];
    f = fopen("/etc/os-release", "r");
    if (f != NULL) {
        char line[256];
        while (fgets(line, sizeof(line), f) != NULL) {
            if (strncmp(line, "PRETTY_NAME=", 12) == 0) {
                char *v = line + 12;
                v[strcspn(v, "\n")] = '\0';
                if (*v == '"') { v++; v[strcspn(v, "\"")] = '\0'; }
                snprintf(pretty, sizeof(pretty), "%s", v);
                snprintf(out->os, sizeof(out->os), "%s", pretty);
                break;
            }
        }
        fclose(f);
    }

    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (u32 c = 0; c < CPU_SETSIZE && out->ncpu < SB_MAX_CPUS; c++) {
            if (CPU_ISSET(c, &set)) { out->cpu_ids[out->ncpu++] = (u16)c; }
        }
    }

    /* Intel hybrid exposes separate PMUs listing each core type. */
    char list[1024];
    if (read_line("/sys/devices/cpu_core/cpus", list, sizeof(list))) {
        out->nperf = filter_allowed(out, out->perf_ids, parse_cpu_list(list, out->perf_ids, SB_MAX_CPUS));
    }
    if (read_line("/sys/devices/cpu_atom/cpus", list, sizeof(list))) {
        out->neff = filter_allowed(out, out->eff_ids, parse_cpu_list(list, out->eff_ids, SB_MAX_CPUS));
    }
    if (out->nperf == 0 || out->neff == 0) { out->nperf = 0; out->neff = 0; }

    out->cache_line = (u32)read_u64("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size");

    /* Describe the first allowed (performance, if hybrid) CPU. */
    u16 ref = out->nperf > 0 ? out->perf_ids[0] : (out->ncpu > 0 ? out->cpu_ids[0] : 0);
    for (u32 idx = 0; idx < 8; idx++) {
        char path[160], type[32];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u/cache/index%u/type", ref, idx);
        if (!read_line(path, type, sizeof(type))) { continue; }
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u/cache/index%u/level", ref, idx);
        u64 level = read_u64(path);
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u/cache/index%u/size", ref, idx);
        u64 size = read_size(path);
        if (level == 1 && strcmp(type, "Data") == 0) { out->l1d_bytes = size; }
        if (level == 2) { out->l2_bytes = size; }
        if (level == 3) { out->l3_bytes = size; }
    }

    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u/cpufreq/cpuinfo_max_freq", ref);
    out->freq_max_hz = read_u64(path) * 1000;

    long pages = sysconf(_SC_PHYS_PAGES);
    if (pages > 0) { out->mem_bytes = (u64)pages * out->page_size; }
}
