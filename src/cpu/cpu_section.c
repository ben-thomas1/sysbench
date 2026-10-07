/* Legacy section adapter: the old module prints its own output.
 * Replace this file when the module is rewritten on the core API. */
#include "cpu/cpu.h"
#include "bench.h"

static sb_status_e run(void) {
    bench_cpu();
    return SB_OK;
}

const sb_section sb_section_cpu = {
    .name       = "cpu",
    .title      = "CPU Compute Throughput",
    .help       =
                  "  FP64/FP32 and INT64/INT32/INT16/INT8 multiply-add throughput.\n"
                  "  Single-thread scalar/SIMD and separate multicore runs. Multiply + add\n"
                  "  counts as two operations. Unsupported SIMD types print n/a.\n"
                  "  Backends: NEON, SSE2, AVX2, AVX-512, selected at build time.\n"
                  "  Multicore: equal work per worker, total work / wall time, up to 64 workers.\n"
                  "  Linux respects allowed CPU IDs; macOS placement is scheduler-controlled.\n"
                  "  IPC estimates use a reference clock, not hardware cycle counters.\n",
    .run        = run,
    .repeatable = false,
};
