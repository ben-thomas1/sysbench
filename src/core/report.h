#pragma once

#include "core/status.h"
#include "core/types.h"

/* What a number means; printed next to every value. */
typedef enum {
    SB_KIND_MEASURED = 0,  /* directly measured behaviour of this workload */
    SB_KIND_PEAK,          /* hardware ceiling: kernel built to saturate one unit */
    SB_KIND_EFFECTIVE,     /* nominal work / time (e.g. model ops, not executed ops) */
    SB_KIND_ESTIMATE,      /* derived from assumptions (assumed clock, assumed miss rate, ...) */
} sb_kind_e;

/* Called by main around each section. With passes > 1 every value is
 * collected and printed as median/min/max at section end; with passes == 1
 * rows print immediately. */
void sb_report_section_begin(const char *title, u32 passes);
void sb_report_pass_begin(u32 pass);
void sb_report_section_end(void);

/* Called by benchmarks. Info lines and group headers print on the first pass only. */
[[gnu::format(printf, 1, 2)]]
void sb_report_info(const char *fmt, ...);
void sb_report_group(const char *title);
void sb_report_value(const char *test, f64 value, const char *unit, sb_kind_e kind);
void sb_report_error(const char *test, sb_status_e s);
void sb_report_skip(const char *test, const char *reason);

/* "4 KiB", "16 MiB", "1 GiB" into buf. */
const char *sb_fmt_size(u64 bytes, char *buf, size_t cap);
