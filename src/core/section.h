#pragma once

#include "core/status.h"

/* One selectable benchmark section. Each module defines its descriptor,
 * e.g. `const sb_section sb_section_sys` in src/sys/sys.c; main lists them. */
typedef struct {
    const char *name;     /* CLI name: --only/--skip */
    const char *title;    /* section header */
    const char *help;     /* --help paragraph (lines start with two spaces) */
    sb_status_e (*run)(void);
    bool        repeatable; /* false: legacy code that prints directly; runs once */
} sb_section;
