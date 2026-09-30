#include "sysinfo.h"
#include "sysinfo_platform.h"

#include <string.h>
#include <unistd.h>

void query_cpu_info(struct cpu_info *info) {
    memset(info, 0, sizeof(*info));

#ifdef __aarch64__
    strncpy(info->arch, "arm64", sizeof(info->arch));
#elif defined(__x86_64__)
    strncpy(info->arch, "x86_64", sizeof(info->arch));
#else
    strncpy(info->arch, "unknown", sizeof(info->arch));
#endif

    info->cores_total = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (info->cores_total < 1) info->cores_total = 1;

    query_cpu_info_platform(info);
}

void query_mem_info(struct mem_info *info) {
    memset(info, 0, sizeof(*info));
    info->page_size = (size_t)sysconf(_SC_PAGESIZE);
    query_mem_info_platform(info);
}

void query_disk_info(struct disk_info *info, const char *path) {
    memset(info, 0, sizeof(*info));
    query_disk_info_platform(info, path);
}
