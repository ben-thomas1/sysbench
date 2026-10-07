#include "bench.h"
#include "timer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#define BRANCH_PATTERN_LEN (1U << 20)
#define BRANCH_TARGET_NS  200000000ULL
#define BRANCH_TRIALS     5

static_assert((BRANCH_PATTERN_LEN & (BRANCH_PATTERN_LEN - 1)) == 0,
               "BRANCH_PATTERN_LEN must be a power of two");

enum branch_pattern {
    PATTERN_ALWAYS_TAKEN,
    PATTERN_ALTERNATING,
    PATTERN_PERIODIC_8,
    PATTERN_PERIODIC_64,
    PATTERN_RANDOM_50,
    PATTERN_RANDOM_90,
    PATTERN_RANDOM_99,
};

struct branch_case {
    const char *name;
    enum branch_pattern pattern;
    double expected_miss_rate;
};

struct branch_result {
    double branch_ns;
    double extra_ns;
    double penalty_ns;
    uint64_t hw_branches;
    uint64_t hw_misses;
    size_t logical_branches;
    int have_counters;
};

#ifdef __linux__
struct branch_perf {
    int branches_fd;
    int misses_fd;
    int ok;
};

struct branch_perf_read {
    uint64_t nr;
    uint64_t time_enabled;
    uint64_t time_running;
    uint64_t values[2];
};

static int perf_event_open_hw(uint64_t config, int group_fd, int disabled) {
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_HARDWARE;
    attr.size = sizeof(attr);
    attr.config = config;
    attr.disabled = disabled;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED |
                       PERF_FORMAT_TOTAL_TIME_RUNNING;

    return (int)syscall(__NR_perf_event_open, &attr, 0, -1, group_fd, 0);
}

static void branch_perf_close(struct branch_perf *p) {
    if (p->branches_fd >= 0) close(p->branches_fd);
    if (p->misses_fd >= 0) close(p->misses_fd);
    p->branches_fd = -1;
    p->misses_fd = -1;
    p->ok = 0;
}

static struct branch_perf branch_perf_open(void) {
    struct branch_perf p = {-1, -1, 0};

    p.branches_fd = perf_event_open_hw(PERF_COUNT_HW_BRANCH_INSTRUCTIONS, -1, 1);
    if (p.branches_fd < 0)
        return p;

    p.misses_fd = perf_event_open_hw(PERF_COUNT_HW_BRANCH_MISSES, p.branches_fd, 0);
    if (p.misses_fd < 0) {
        branch_perf_close(&p);
        return p;
    }

    p.ok = 1;
    return p;
}

static int branch_perf_start(struct branch_perf *p) {
    if (!p || !p->ok)
        return 0;
    if (ioctl(p->branches_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) < 0)
        return 0;
    if (ioctl(p->branches_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) < 0)
        return 0;
    return 1;
}

static int branch_perf_stop(struct branch_perf *p,
                            uint64_t *branches, uint64_t *misses) {
    struct branch_perf_read data;

    if (!p || !p->ok)
        return 0;

    if (ioctl(p->branches_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) < 0)
        return 0;

    memset(&data, 0, sizeof(data));
    if (read(p->branches_fd, &data, sizeof(data)) != (ssize_t)sizeof(data))
        return 0;
    /* Reject partial counts after multiplexing or migration between PMUs. */
    if (data.nr != 2 || data.time_running == 0 || data.time_running != data.time_enabled)
        return 0;

    *branches = data.values[0];
    *misses = data.values[1];
    return 1;
}
#else
struct branch_perf {
    int ok;
};

static void branch_perf_close(struct branch_perf *p) {
    (void)p;
}

static struct branch_perf branch_perf_open(void) {
    struct branch_perf p = {0};
    return p;
}

static int branch_perf_start(struct branch_perf *p) {
    (void)p;
    return 0;
}

static int branch_perf_stop(struct branch_perf *p,
                            uint64_t *branches, uint64_t *misses) {
    (void)p;
    (void)branches;
    (void)misses;
    return 0;
}
#endif

#if defined(__aarch64__)
static inline void forced_branch(uint8_t value, uint64_t *acc) {
    uint64_t a = *acc;
    uint32_t v = value;
    __asm__ volatile(
        "tbz %w[v], #0, 1f\n\t"
        "add %[a], %[a], #3\n\t"
        "b 2f\n\t"
        "1: add %[a], %[a], #7\n\t"
        "b 2f\n\t"
        "2:"
        : [a] "+r"(a)
        : [v] "r"(v)
        : "cc");
    *acc = a;
}
#elif defined(__x86_64__)
static inline void forced_branch(uint8_t value, uint64_t *acc) {
    uint64_t a = *acc;
    __asm__ volatile(
        "testb $1, %[v]\n\t"
        "jz 1f\n\t"
        "addq $3, %[a]\n\t"
        "jmp 2f\n\t"
        "1: addq $7, %[a]\n\t"
        "jmp 2f\n\t"
        "2:"
        : [a] "+r"(a)
        : [v] "q"(value)
        : "cc");
    *acc = a;
}
#else
static inline void forced_branch(uint8_t value, uint64_t *acc) {
    if (value)
        *acc += 3;
    else
        *acc += 7;
}
#endif

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static void fill_pattern(uint8_t *out, size_t n, enum branch_pattern pattern) {
    uint32_t rng = 0x12345678u;

    for (size_t i = 0; i < n; i++) {
        switch (pattern) {
        case PATTERN_ALWAYS_TAKEN:
            out[i] = 1;
            break;
        case PATTERN_ALTERNATING:
            out[i] = (uint8_t)(i & 1);
            break;
        case PATTERN_PERIODIC_8:
            out[i] = (uint8_t)((i & 7) == 0);
            break;
        case PATTERN_PERIODIC_64:
            out[i] = (uint8_t)((i & 63) == 0);
            break;
        case PATTERN_RANDOM_50:
            out[i] = (uint8_t)(xorshift32(&rng) >> 31);
            break;
        case PATTERN_RANDOM_90:
            out[i] = (uint8_t)((xorshift32(&rng) % 10) != 0);
            break;
        case PATTERN_RANDOM_99:
            out[i] = (uint8_t)((xorshift32(&rng) % 100) != 0);
            break;
        }
    }
}

static uint64_t run_branchy(const uint8_t *pattern, size_t len,
                            size_t iters, uint64_t seed) {
    uint64_t acc = seed;
    size_t idx = 0;

    for (size_t i = 0; i < iters; i++) {
        uint8_t v = pattern[idx];
        idx = (idx + 1) & (len - 1);

        forced_branch(v, &acc);
        __asm__ volatile("" : "+r"(acc));
    }

    return acc;
}

static double time_branchy_ns(const uint8_t *pattern, size_t len, size_t iters) {
    uint64_t t0 = timer_ns();
    uint64_t acc = run_branchy(pattern, len, iters, 1);
    uint64_t t1 = timer_ns();
    __asm__ volatile("" : "+r"(acc));
    if (t1 <= t0) return -1;
    return (double)(t1 - t0) / (double)iters;
}

static double time_branchy_with_perf_ns(const uint8_t *pattern, size_t len,
                                        size_t iters, struct branch_perf *perf,
                                        uint64_t *branches, uint64_t *misses,
                                        int *have_counters) {
    uint64_t acc;
    uint64_t t0;
    uint64_t t1;

    *branches = 0;
    *misses = 0;
    *have_counters = 0;

    t0 = timer_ns();
    int counters_started = branch_perf_start(perf);
    acc = run_branchy(pattern, len, iters, 1);
    if (counters_started && branch_perf_stop(perf, branches, misses))
        *have_counters = 1;
    t1 = timer_ns();

    __asm__ volatile("" : "+r"(acc));
    if (t1 <= t0) return -1;
    return (double)(t1 - t0) / (double)iters;
}

static size_t calibrate_branch_iters(const uint8_t *pattern, size_t len) {
    size_t iters = 1000000;
    double ns = time_branchy_ns(pattern, len, iters);
    if (ns <= 0) return iters;

    size_t target = (size_t)((double)BRANCH_TARGET_NS / ns);
    if (target < iters) target = iters;
    return target;
}

static struct branch_result measure_case(const struct branch_case *c,
                                         uint8_t *pattern, size_t len,
                                         double baseline_ns,
                                         struct branch_perf *perf) {
    struct branch_result r = {-1, -1, -1, 0, 0, 0, 0};
    fill_pattern(pattern, len, c->pattern);

    size_t iters = calibrate_branch_iters(pattern, len);
    run_branchy(pattern, len, len * 2, 1);

    double best_branch = 1e30;
    uint64_t best_branches = 0;
    uint64_t best_misses = 0;
    int best_have_counters = 0;

    for (int trial = 0; trial < BRANCH_TRIALS; trial++) {
        uint64_t branches = 0;
        uint64_t misses = 0;
        int have_counters = 0;
        double ns = time_branchy_with_perf_ns(pattern, len, iters, perf,
                                             &branches, &misses, &have_counters);
        if (ns > 0 && ns < best_branch) {
            best_branch = ns;
            best_branches = branches;
            best_misses = misses;
            best_have_counters = have_counters;
        }
    }

    if (best_branch >= 1e29)
        return r;

    r.branch_ns = best_branch;
    r.extra_ns = best_branch - baseline_ns;
    if (r.extra_ns < 0) r.extra_ns = 0;
    r.hw_branches = best_branches;
    r.hw_misses = best_misses;
    r.logical_branches = iters;
    r.have_counters = best_have_counters;
    if (r.have_counters && r.hw_misses > 0) {
        double misses_per_logical_branch =
            (double)r.hw_misses / (double)r.logical_branches;
        r.penalty_ns = r.extra_ns / misses_per_logical_branch;
    } else {
        r.penalty_ns = c->expected_miss_rate > 0 ? r.extra_ns / c->expected_miss_rate : 0;
    }
    return r;
}

void bench_branch(void) {
    static const struct branch_case cases[] = {
        {"Always true",  PATTERN_ALWAYS_TAKEN, 0.0},
        {"Alternating",  PATTERN_ALTERNATING,  0.0},
        {"Periodic 1/8", PATTERN_PERIODIC_8,   0.0},
        {"Periodic 1/64", PATTERN_PERIODIC_64, 0.0},
        {"Random 50/50", PATTERN_RANDOM_50,    0.5},
        {"Random 90/10", PATTERN_RANDOM_90,    0.1},
        {"Random 99/1",  PATTERN_RANDOM_99,    0.01},
    };
    struct branch_perf perf = branch_perf_open();

    uint8_t *pattern = malloc(BRANCH_PATTERN_LEN);
    printf("=== Branch Prediction ===\n");
    printf("  Pattern length: %s outcomes\n", fmt_size(BRANCH_PATTERN_LEN));
#ifdef __linux__
    if (!perf.ok)
        printf("  Hardware counters unavailable; using assumed miss rates.\n");
#else
    printf("  Hardware counters unavailable on this OS; using assumed miss rates.\n");
#endif
    printf("%-16s %12s %12s %10s %12s %12s %12s %12s\n",
           "Pattern", "Branch", "Extra", "Assumed", "HW branches",
           "HW misses", "Miss/logical", "Penalty");
    printf("%-16s %12s %12s %10s %12s %12s %12s %12s\n",
           "-------", "------", "-----", "-------", "-----------",
           "---------", "---------", "-------");

    if (!pattern) {
        printf("  Failed to allocate pattern buffer\n");
        branch_perf_close(&perf);
        return;
    }

    struct branch_result baseline =
        measure_case(&cases[0], pattern, BRANCH_PATTERN_LEN, 0, &perf);
    double baseline_ns = baseline.branch_ns > 0 ? baseline.branch_ns : 0;

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        struct branch_result r = measure_case(&cases[i], pattern, BRANCH_PATTERN_LEN,
                                              baseline_ns, &perf);
        if (r.branch_ns < 0) {
            printf("%-16s %12s %12s %10s %12s %12s %12s %12s\n",
                   cases[i].name, "error", "error", "error", "error",
                   "error", "error", "error");
        } else {
            char assumed[32];
            char hw_branches[32];
            char hw_misses[32];
            char logical_miss_rate[32];
            char penalty[32];

            if (cases[i].expected_miss_rate > 0)
                snprintf(assumed, sizeof(assumed), "%.1f%%",
                         cases[i].expected_miss_rate * 100.0);
            else
                snprintf(assumed, sizeof(assumed), "%s", "learned");

            if (r.have_counters && r.hw_branches > 0) {
                snprintf(hw_branches, sizeof(hw_branches), "%llu",
                         (unsigned long long)r.hw_branches);
                snprintf(hw_misses, sizeof(hw_misses), "%llu",
                         (unsigned long long)r.hw_misses);
                snprintf(logical_miss_rate, sizeof(logical_miss_rate), "%.2f%%",
                         100.0 * (double)r.hw_misses /
                         (double)r.logical_branches);
            } else {
                snprintf(hw_branches, sizeof(hw_branches), "%s", "n/a");
                snprintf(hw_misses, sizeof(hw_misses), "%s", "n/a");
                snprintf(logical_miss_rate, sizeof(logical_miss_rate), "%s", "n/a");
            }

            if ((r.have_counters && r.hw_misses > 0) ||
                (!r.have_counters && cases[i].expected_miss_rate > 0)) {
                snprintf(penalty, sizeof(penalty), "%.2f ns", r.penalty_ns);
            } else {
                snprintf(penalty, sizeof(penalty), "%s", "n/a");
            }

            printf("%-16s %8.2f ns %8.2f ns %10s %12s %12s %12s %12s\n",
                   cases[i].name, r.branch_ns, r.extra_ns, assumed,
                   hw_branches, hw_misses, logical_miss_rate, penalty);
        }
        fflush(stdout);
    }

    free(pattern);
    branch_perf_close(&perf);
}
