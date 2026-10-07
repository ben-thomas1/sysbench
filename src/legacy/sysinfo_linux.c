#ifdef __linux__

#include "sysinfo_platform.h"
#include "timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <sys/statvfs.h>

void query_cpu_info_platform(struct cpu_info *info) {
    /* Parse /proc/cpuinfo for model name */
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "model name", 10) == 0) {
                char *p = strchr(line, ':');
                if (p) {
                    p++;
                    while (*p == ' ') p++;
                    char *nl = strchr(p, '\n');
                    if (nl) *nl = '\0';
                    strncpy(info->name, p, sizeof(info->name) - 1);
                }
                break;
            }
        }
        fclose(f);
    }

    /* CPU frequency */
    f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "r");
    if (f) {
        unsigned long khz = 0;
        if (fscanf(f, "%lu", &khz) == 1)
            info->freq_hz = khz * 1000ULL;
        fclose(f);
    }

    /* Cache sizes */
    char path[256];
    for (int idx = 0; idx < 4; idx++) {
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu0/cache/index%d/type", idx);
        f = fopen(path, "r");
        if (!f) continue;
        char type[32] = {0};
        if (fgets(type, sizeof(type), f)) {
            char *nl = strchr(type, '\n');
            if (nl) *nl = '\0';
        }
        fclose(f);

        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu0/cache/index%d/size", idx);
        f = fopen(path, "r");
        if (!f) continue;
        unsigned long sz = 0;
        char unit = 'K';
        if (fscanf(f, "%lu%c", &sz, &unit) >= 1) {
            uint64_t bytes = sz * 1024;
            if (unit == 'M') bytes = sz * 1024 * 1024;

            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu0/cache/index%d/level", idx);
            FILE *lf = fopen(path, "r");
            int level = 0;
            if (lf) { fscanf(lf, "%d", &level); fclose(lf); }

            if (level == 1 && strcmp(type, "Data") == 0)
                info->l1d_bytes = bytes;
            else if (level == 2)
                info->l2_bytes = bytes;
            else if (level == 3)
                info->l3_bytes = bytes;
        }
        fclose(f);
    }

    /* Frequency bins are not a reliable indication of physical core type. */
}

void query_mem_info_platform(struct mem_info *info) {
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0)
        info->total_bytes = (uint64_t)pages * (uint64_t)page_size;
}

void query_disk_info_platform(struct disk_info *info, const char *path) {
    char resolved[PATH_MAX];
    const char *match_path = path;
    if (realpath(path, resolved))
        match_path = resolved;

    /* Read /proc/mounts to find the filesystem type */
    FILE *f = fopen("/proc/mounts", "r");
    if (f) {
        char line[512];
        char best_mount[256] = "/";
        while (fgets(line, sizeof(line), f)) {
            char dev[256], mount[256], fstype[64];
            if (sscanf(line, "%255s %255s %63s", dev, mount, fstype) == 3) {
                size_t mlen = strlen(mount);
                if (strncmp(match_path, mount, mlen) == 0 &&
                    (match_path[mlen] == '\0' || match_path[mlen] == '/' ||
                     strcmp(mount, "/") == 0) &&
                    mlen >= strlen(best_mount)) {
                    snprintf(best_mount, sizeof(best_mount), "%s", mount);
                    snprintf(info->fs_type, sizeof(info->fs_type), "%s", fstype);
                }
            }
        }
        fclose(f);
    }
}

double get_cpu_freq_ghz(void) {
    /* Try sysfs first */
    FILE *f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "r");
    if (f) {
        unsigned long khz = 0;
        if (fscanf(f, "%lu", &khz) == 1 && khz > 0) {
            fclose(f);
            return (double)(khz * 1000ULL) / 1e9;
        }
        fclose(f);
    }

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
