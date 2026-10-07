#pragma once

#include "core/status.h"
#include "core/types.h"

#define SB_MAX_CPUS 256

typedef enum {
    SB_CORE_ANY = 0,
    SB_CORE_PERF,
    SB_CORE_EFF,
} sb_core_e;

/* Hardware/OS facts read from the platform; fields are 0 / "" when unknown.
 * Cache sizes describe the performance cores. */
typedef struct {
    char os[64];
    char os_release[64];
    char cpu[128];
    char arch[16];

    u32  ncpu;                      /* logical CPUs available to this process */
    u32  nperf;                     /* hybrid: performance cores (0 if not hybrid / unknown) */
    u32  neff;                      /* hybrid: efficiency cores */
    u16  cpu_ids[SB_MAX_CPUS];      /* Linux: allowed CPU ids; macOS: 0..ncpu-1 */
    u16  perf_ids[SB_MAX_CPUS];     /* Linux hybrid: CPU ids of each type */
    u16  eff_ids[SB_MAX_CPUS];

    u32  cache_line;
    u32  page_size;
    u64  l1d_bytes;
    u64  l2_bytes;
    u64  l3_bytes;
    u64  mem_bytes;
    u64  freq_max_hz;               /* reported maximum, 0 if the OS does not expose it */
} sb_platform;

/* Fills `out`; never fails hard — unknown facts stay zero. */
sb_status_e sb_platform_query(sb_platform *out);

/* Process-wide platform facts, queried once by main before any section runs. */
const sb_platform *sb_platform_get(void);
