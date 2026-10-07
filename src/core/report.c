#include "core/report.h"
#include "core/stats.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define ROW_FMT  "  %-34s %12.2f %-10s %s\n"
#define ROW_SFMT "  %-34s %12s %s\n"

typedef enum { ENTRY_VALUE, ENTRY_GROUP, ENTRY_ERROR, ENTRY_SKIP } entry_e;

typedef struct {
    entry_e     type;
    char        test[64];
    char        unit[16];
    char        text[64];   /* group title, error text, or skip reason */
    sb_kind_e   kind;
    f64        *vals;
    u32         nvals;
} entry;

static struct {
    u32    passes;
    u32    pass;
    entry *e;
    u32    len;
    u32    cap;
    u32    cursor;          /* next entry index to match on passes > 0 */
} g;

static const char *kind_tag(sb_kind_e k);

/* Unit column is padded only when a kind tag follows, so rows carry no trailing spaces. */
static void print_row(const char *test, f64 value, const char *unit, const char *tag) {
    if (tag[0] != '\0') { printf(ROW_FMT, test, value, unit, tag); }
    else                 { printf("  %-34s %12.2f %s\n", test, value, unit); }
}

static const char *kind_tag(sb_kind_e k) {
    switch (k) {
    case SB_KIND_MEASURED:  return "";
    case SB_KIND_PEAK:      return "[peak]";
    case SB_KIND_EFFECTIVE: return "[effective]";
    case SB_KIND_ESTIMATE:  return "[estimate]";
    }
    return "";
}

const char *sb_fmt_size(u64 bytes, char *buf, size_t cap) {
    if (bytes >= (1ULL << 30) && bytes % (1ULL << 30) == 0) {
        snprintf(buf, cap, "%llu GiB", (unsigned long long)(bytes >> 30));
    } else if (bytes >= (1ULL << 20) && bytes % (1ULL << 20) == 0) {
        snprintf(buf, cap, "%llu MiB", (unsigned long long)(bytes >> 20));
    } else if (bytes >= 1024 && bytes % 1024 == 0) {
        snprintf(buf, cap, "%llu KiB", (unsigned long long)(bytes >> 10));
    } else {
        snprintf(buf, cap, "%llu B", (unsigned long long)bytes);
    }
    return buf;
}

static void free_entries(void) {
    for (u32 i = 0; i < g.len; i++) { SB_FREE(g.e[i].vals); }
    SB_FREE(g.e);
    g.e   = NULL;
    g.len = 0;
    g.cap = 0;
}

/* Find the entry for this (type, test) on later passes; append on pass 0. */
static entry *slot(entry_e type, const char *test) {
    if (g.pass > 0) {
        for (u32 k = 0; k < g.len; k++) {
            u32 i = (g.cursor + k) % g.len;
            if (g.e[i].type == type && strcmp(g.e[i].test, test) == 0) {
                g.cursor = i + 1;
                return &g.e[i];
            }
        }
        return NULL;  /* test appeared only on a later pass: ignore */
    }
    if (g.len == g.cap) {
        u32 ncap = g.cap ? g.cap * 2 : 32;
        entry *ne = SB_REALLOC(g.e, (size_t)ncap * sizeof(entry));
        if (ne == NULL) { return NULL; }
        g.e   = ne;
        g.cap = ncap;
    }
    entry *e = &g.e[g.len++];
    memset(e, 0, sizeof(*e));
    e->type = type;
    snprintf(e->test, sizeof(e->test), "%s", test);
    return e;
}

void sb_report_section_begin(const char *title, u32 passes) {
    free_entries();
    g.passes = passes ? passes : 1;
    g.pass   = 0;
    g.cursor = 0;
    printf("=== %s ===\n", title);
    fflush(stdout);
}

void sb_report_pass_begin(u32 pass) {
    g.pass   = pass;
    g.cursor = 0;
    if (g.passes > 1) {
        fprintf(stderr, "  [pass %u/%u]\n", pass + 1, g.passes);
    }
}

void sb_report_info(const char *fmt, ...) {
    if (g.pass > 0) { return; }
    va_list ap;
    va_start(ap, fmt);
    fputs("  ", stdout);
    vprintf(fmt, ap);
    fputc('\n', stdout);
    va_end(ap);
    fflush(stdout);
}

void sb_report_group(const char *title) {
    if (g.passes > 1) {
        entry *e = slot(ENTRY_GROUP, title);
        if (e != NULL && g.pass == 0) { snprintf(e->text, sizeof(e->text), "%s", title); }
        return;
    }
    printf("\n  --- %s ---\n", title);
    fflush(stdout);
}

void sb_report_value(const char *test, f64 value, const char *unit, sb_kind_e kind) {
    if (g.passes == 1) {
        print_row(test, value, unit, kind_tag(kind));
        fflush(stdout);
        return;
    }
    entry *e = slot(ENTRY_VALUE, test);
    if (e == NULL) { return; }
    if (e->vals == NULL) {
        e->vals = SB_MALLOC((size_t)g.passes * sizeof(f64));
        if (e->vals == NULL) { return; }
        snprintf(e->unit, sizeof(e->unit), "%s", unit);
        e->kind = kind;
    }
    if (e->nvals < g.passes) { e->vals[e->nvals++] = value; }
}

void sb_report_error(const char *test, sb_status_e s) {
    if (g.passes == 1) {
        printf(ROW_SFMT, test, "error", sb_status_str(s));
        fflush(stdout);
        return;
    }
    entry *e = slot(ENTRY_ERROR, test);
    if (e != NULL) { snprintf(e->text, sizeof(e->text), "%s", sb_status_str(s)); }
}

void sb_report_skip(const char *test, const char *reason) {
    if (g.passes == 1) {
        printf(ROW_SFMT, test, "n/a", reason);
        fflush(stdout);
        return;
    }
    entry *e = slot(ENTRY_SKIP, test);
    if (e != NULL) { snprintf(e->text, sizeof(e->text), "%s", reason); }
}

void sb_report_section_end(void) {
    if (g.passes > 1) {
        printf("  %-34s %12s %12s %12s %s\n", "Test", "Median", "Min", "Max", "Unit");
        for (u32 i = 0; i < g.len; i++) {
            entry *e = &g.e[i];
            switch (e->type) {
            case ENTRY_GROUP:
                printf("\n  --- %s ---\n", e->text);
                break;
            case ENTRY_VALUE: {
                sb_stats s;
                if (sb_stats_compute(e->vals, e->nvals, &s) != SB_OK) { break; }
                const char *tag = kind_tag(e->kind);
                printf("  %-34s %12.2f %12.2f %12.2f %-10s %s%s(n=%u)\n",
                       e->test, s.median, s.min, s.max, e->unit, tag, tag[0] ? " " : "", s.n);
                break;
            }
            case ENTRY_ERROR:
                printf(ROW_SFMT, e->test, "error", e->text);
                break;
            case ENTRY_SKIP:
                printf(ROW_SFMT, e->test, "n/a", e->text);
                break;
            }
        }
    }
    free_entries();
    fflush(stdout);
}
