#include "core/stats.h"

#include <string.h>

static int cmp_f64(const void *a, const void *b) {
    f64 x = *(const f64 *)a, y = *(const f64 *)b;
    return (x > y) - (x < y);
}

sb_status_e sb_stats_compute(const f64 *v, u32 n, sb_stats *out) {
    memset(out, 0, sizeof(*out));
    if (n == 0) { return SB_ERR_INVALID; }
    f64 *s = SB_MALLOC((size_t)n * sizeof(f64)); /* u32 count: cannot overflow size_t */
    if (s == NULL) { return SB_ERR_NOMEM; }
    memcpy(s, v, n * sizeof(f64));
    qsort(s, n, sizeof(f64), cmp_f64);

    f64 sum = 0;
    for (u32 i = 0; i < n; i++) { sum += s[i]; }
    out->n      = n;
    out->min    = s[0];
    out->max    = s[n - 1];
    out->mean   = sum / (f64)n;
    out->median = (n % 2) ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
    SB_FREE(s);
    return SB_OK;
}
