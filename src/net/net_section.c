/* Legacy section adapter: the old module prints its own output.
 * Replace this file when the module is rewritten on the core API. */
#include "net/net.h"
#include "bench.h"

static sb_status_e run(void) {
    bench_net();
    return SB_OK;
}

const sb_section sb_section_net = {
    .name       = "net",
    .title      = "Network (TCP loopback)",
    .help       =
                  "  TCP loopback: 2 GiB bulk transfer and one-byte TCP_NODELAY ping-pong.\n"
                  "  Measures the local OS stack, not NIC throughput.\n",
    .run        = run,
    .repeatable = false,
};
