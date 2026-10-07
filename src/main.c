#include "core/platform.h"
#include "core/report.h"
#include "core/section.h"
#include "core/types.h"

#include "branch/branch.h"
#include "cpu/cpu.h"
#include "disk/disk.h"
#include "gpu/gpu.h"
#include "matrix/matrix.h"
#include "mem/mem.h"
#include "net/net.h"
#include "npu/npu.h"
#include "sys/sys.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const sb_section *const sections[] = {
    &sb_section_cpu,
    &sb_section_branch,
    &sb_section_gpu,
    &sb_section_mem,
    &sb_section_disk,
    &sb_section_sys,
    &sb_section_net,
    &sb_section_matrix,
    &sb_section_npu,
};

#define NSECTIONS SB_ARRAY_LEN(sections)
#define MAX_REPEAT 1000

static int find_section(const char *name) {
    for (u32 i = 0; i < NSECTIONS; i++) {
        if (strcmp(name, sections[i]->name) == 0) { return (int)i; }
    }
    return -1;
}

/* Parse "a,b,c" and set mark[i] for each named section. */
static sb_status_e parse_list(const char *list, bool mark[NSECTIONS]) {
    char   buf[256];
    size_t len = strlen(list);
    if (len == 0 || len >= sizeof(buf) || list[0] == ',' || list[len - 1] == ',' || strstr(list, ",,") != NULL) {
        fprintf(stderr, "Invalid section list: expected comma-separated names\n");
        return SB_ERR_INVALID;
    }
    memcpy(buf, list, len + 1);
    for (char *tok = strtok(buf, ","); tok != NULL; tok = strtok(NULL, ",")) {
        int idx = find_section(tok);
        if (idx < 0) {
            fprintf(stderr, "Unknown section: %s\n", tok);
            return SB_ERR_INVALID;
        }
        mark[idx] = true;
    }
    return SB_OK;
}

static void print_names(FILE *f) {
    for (u32 i = 0; i < NSECTIONS; i++) { fprintf(f, "%s%s", i ? ", " : "", sections[i]->name); }
    fputc('\n', f);
}

static void usage(void) {
    fprintf(stderr,
        "Usage: bench [--only <list> | --skip <list>] [--repeat N] [--list] [--help]\n"
        "\n"
        "  --only <list>  Run only the listed sections (comma-separated)\n"
        "  --skip <list>  Run all sections except the listed ones\n"
        "  --repeat N     Run each section N times; report median/min/max\n"
        "  --list         List section names\n"
        "  --help         Describe every section\n"
        "\nSections: ");
    print_names(stderr);
}

static void detailed_help(void) {
    printf("bench - system benchmark suite\n\n"
           "Usage: bench [--only <list> | --skip <list>] [--repeat N] [--list] [--help]\n"
           "Default is a single quick pass of every section. --repeat N runs each\n"
           "section N times and reports median, min and max per result.\n"
           "Values tagged [peak] are hardware ceilings, [effective] are nominal work\n"
           "divided by time, [estimate] depend on stated assumptions.\n\n");
    for (u32 i = 0; i < NSECTIONS; i++) {
        printf("%s (%s)\n%s\n", sections[i]->title, sections[i]->name, sections[i]->help);
    }
    printf("See README.md for dependencies and measurement limitations.\n");
}

int main(int argc, char **argv) {
    bool selected[NSECTIONS];
    bool have_only = false, have_skip = false;
    u32  repeat    = 1;
    for (u32 i = 0; i < NSECTIONS; i++) { selected[i] = true; }

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "--only") == 0 || strcmp(a, "--skip") == 0) {
            bool only = a[2] == 'o';
            if (++i >= argc) {
                fprintf(stderr, "%s requires a comma-separated list\n", a);
                usage();
                return 1;
            }
            if (have_only || have_skip) {
                fprintf(stderr, "Use one --only or --skip\n");
                return 1;
            }
            bool mark[NSECTIONS] = {};
            if (parse_list(argv[i], mark) != SB_OK) {
                usage();
                return 1;
            }
            for (u32 j = 0; j < NSECTIONS; j++) { selected[j] = only ? mark[j] : !mark[j]; }
            have_only = only;
            have_skip = !only;
        } else if (strcmp(a, "--repeat") == 0) {
            char *end = NULL;
            if (++i >= argc) {
                fprintf(stderr, "--repeat requires a count\n");
                return 1;
            }
            unsigned long n = strtoul(argv[i], &end, 10);
            if (end == argv[i] || *end != '\0' || n < 1 || n > MAX_REPEAT) {
                fprintf(stderr, "--repeat expects 1..%d\n", MAX_REPEAT);
                return 1;
            }
            repeat = (u32)n;
        } else if (strcmp(a, "--list") == 0) {
            print_names(stdout);
            return 0;
        } else if (strcmp(a, "--help") == 0) {
            detailed_help();
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", a);
            usage();
            return 1;
        }
    }

    setvbuf(stdout, NULL, _IOLBF, 0);
    (void)sb_platform_get();
    /* Legacy modules use rand() for shuffles and offsets. */
    srand(0x5eed);

    bool first = true;
    for (u32 i = 0; i < NSECTIONS; i++) {
        const sb_section *s = sections[i];
        if (!selected[i]) { continue; }
        if (!first) { printf("\n"); }
        first = false;

        if (!s->repeatable) {
            /* Legacy module: prints its own header and table; runs once. */
            if (repeat > 1) { fprintf(stderr, "  [%s: --repeat not supported yet, running once]\n", s->name); }
            sb_status_e st = s->run();
            if (st != SB_OK) { printf("  %s failed: %s\n", s->name, sb_status_str(st)); }
            continue;
        }
        sb_report_section_begin(s->title, repeat);
        for (u32 p = 0; p < repeat; p++) {
            sb_report_pass_begin(p);
            sb_status_e st = s->run();
            if (st != SB_OK) {
                printf("  %s failed: %s\n", s->name, sb_status_str(st));
                break;
            }
        }
        sb_report_section_end();
    }
    return 0;
}
