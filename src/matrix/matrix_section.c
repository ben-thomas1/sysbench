/* Legacy section adapter: the old module prints its own output.
 * Replace this file when the module is rewritten on the core API. */
#include "matrix/matrix.h"
#include "bench.h"

static sb_status_e run(void) {
    bench_matrix();
    return SB_OK;
}

const sb_section sb_section_matrix = {
    .name       = "matrix",
    .title      = "Matrix Multiply",
    .help       =
                  "  SGEMM: square FP32 matrices from 64x64 through 4096x4096.\n"
                  "  Accelerate on macOS, optional OpenBLAS on Linux.\n",
    .run        = run,
    .repeatable = false,
};
