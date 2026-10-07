/* Legacy section adapter: the old module prints its own output.
 * Replace this file when the module is rewritten on the core API. */
#include "mem/mem.h"
#include "bench.h"

static sb_status_e run(void) {
    bench_memory();
    return SB_OK;
}

const sb_section sb_section_mem = {
    .name       = "mem",
    .title      = "Memory Latency & Bandwidth",
    .help       =
                  "  Shuffled pointer chase with 64-byte stride, 4 KiB to 1 GiB.\n"
                  "  Single-thread read and cached/streaming store bandwidth.\n"
                  "  Atomics with 1, 2, 4, and 8 threads; malloc/free at 64, 256, 4096 bytes.\n"
                  "  Streaming stores use architecture-specific hints and completion fences.\n",
    .run        = run,
    .repeatable = false,
};
