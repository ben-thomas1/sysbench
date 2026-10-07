#pragma once

#include "sysinfo.h"

/* Platform-specific implementations — one of sysinfo_macos.c or sysinfo_linux.c
   is compiled in via the Makefile. */

void query_cpu_info_platform(struct cpu_info *info);
void query_mem_info_platform(struct mem_info *info);
void query_disk_info_platform(struct disk_info *info, const char *path);

/* CPU frequency in GHz — used by IPC estimation in cpu.c */
double get_cpu_freq_ghz(void);
