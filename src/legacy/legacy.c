/* Shared helpers for modules not yet rewritten on the core API.
 * Delete this directory once every module has been rewritten. */
#include "bench.h"

#include <stdio.h>

const char *fmt_size(size_t bytes) {
    static char buf[32];
    if (bytes >= 1024UL * 1024 * 1024)
        snprintf(buf, sizeof(buf), "%zu GiB", bytes / (1024UL * 1024 * 1024));
    else if (bytes >= 1024 * 1024)
        snprintf(buf, sizeof(buf), "%zu MiB", bytes / (1024 * 1024));
    else
        snprintf(buf, sizeof(buf), "%zu KiB", bytes / 1024);
    return buf;
}
