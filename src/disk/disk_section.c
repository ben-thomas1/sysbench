/* Legacy section adapter: the old module prints its own output.
 * Replace this file when the module is rewritten on the core API. */
#include "disk/disk.h"
#include "bench.h"

static sb_status_e run(void) {
    bench_disk();
    return SB_OK;
}

const sb_section sb_section_disk = {
    .name       = "disk",
    .title      = "SSD / Storage",
    .help       =
                  "  Unique 2 GiB temporary file under ./build/ on the current filesystem.\n"
                  "  F_NOCACHE (macOS) or O_DIRECT (Linux) bypasses OS page caching.\n"
                  "  Sequential 1 MiB I/O; random 4 KiB reads and writes; 1/2/4/8/16 readers.\n"
                  "  Burst writes sync once at the end; the fsync row syncs every write.\n"
                  "  Drive caches may participate. macOS fsync is not F_FULLFSYNC.\n"
                  "  Several GiB are written. The file is removed on normal completion.\n",
    .run        = run,
    .repeatable = false,
};
