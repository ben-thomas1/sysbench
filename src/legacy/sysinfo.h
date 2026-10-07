#pragma once

#include <stdint.h>
#include <stddef.h>

struct cpu_info {
    char name[256];
    char arch[16];
    int cores_total;
    int cores_perf;
    int cores_eff;
    uint64_t freq_hz;
    uint64_t l1d_bytes;
    uint64_t l2_bytes;
    uint64_t l3_bytes;
};

struct mem_info {
    uint64_t total_bytes;
    size_t page_size;
};

struct disk_info {
    char fs_type[32];
    char device_model[128];
};

void query_cpu_info(struct cpu_info *info);
void query_mem_info(struct mem_info *info);
void query_disk_info(struct disk_info *info, const char *path);
