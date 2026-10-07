/* Legacy section adapter: the old module prints its own output.
 * Replace this file when the module is rewritten on the core API. */
#include "branch/branch.h"
#include "bench.h"

static sb_status_e run(void) {
    bench_branch();
    return SB_OK;
}

const sb_section sb_section_branch = {
    .name       = "branch",
    .title      = "Branch Prediction",
    .help       =
                  "  Forced conditional branches over a 1 MiB outcome stream.\n"
                  "  Predictable, periodic, and random-biased patterns; best of five trials.\n"
                  "  Penalty estimates divide extra time by measured or assumed miss rates.\n"
                  "  Linux perf counters may be unavailable or multiplexed.\n",
    .run        = run,
    .repeatable = false,
};
