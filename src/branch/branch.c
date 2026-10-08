#include "branch/branch.h"
#include "branch/kernel.h"
#include "branch/pmu.h"
#include "core/report.h"
#include "core/thread.h"
#include "core/timer.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>

#define PAT_LEN      (1u << 20)        /* outcomes per stream: far beyond predictor history */
#define ROUNDS       21                /* interleaved trials per pattern; best one counts */
#define TRIAL_NS     15'000'000ULL
#define CHAIN_NS     10'000'000ULL
#define CALIB_ITERS  1'000'000ULL

static_assert((PAT_LEN & (PAT_LEN - 1)) == 0, "PAT_LEN must be a power of two");

/* Outcome streams; value 1 = the measured branch is taken. */
typedef enum {
    PAT_TAKEN,
    PAT_NOT_TAKEN,
    PAT_ALT,
    PAT_PERIOD8,
    PAT_PERIOD64,
    PAT_RAND1,
    PAT_RAND10,
    PAT_RAND90,
    PAT_RAND50,
    PAT_COUNT,
} pat_e;

typedef struct {
    const char *name;
    bool        random;    /* minority outcomes are unpredictable */
} pat_info;

static const pat_info pats[PAT_COUNT] = {
    [PAT_TAKEN]     = { "Always taken",              false },
    [PAT_NOT_TAKEN] = { "Never taken",               false },
    [PAT_ALT]       = { "Alternating",               false },
    [PAT_PERIOD8]   = { "Period 8 (1 taken in 8)",   false },
    [PAT_PERIOD64]  = { "Period 64 (1 taken in 64)", false },
    [PAT_RAND1]     = { "Random, 1% taken",          true  },
    [PAT_RAND10]    = { "Random, 10% taken",         true  },
    [PAT_RAND90]    = { "Random, 90% taken",         true  },
    [PAT_RAND50]    = { "Random, 50% taken",         true  },
};

/* Penalty = extra time / extra misses of one random pattern. The direction
 * matters on Apple M4: missing a branch that is usually taken costs much
 * more than missing one that is usually not taken. 10% rather than 1%
 * minority outcomes: 1% leaves ~0.04 ns of signal per iteration and varied
 * by +-20% between runs; 10% still has misses far enough apart that each
 * one pays the full refill. */
typedef struct {
    const char *name;
    pat_e       pat;
} penalty_info;

static const penalty_info penalties[] = {
    { "Taken, predicted not-taken", PAT_RAND10 },
    { "Not-taken, predicted taken", PAT_RAND90 },
    { "Mixed (50% taken)",          PAT_RAND50 },
};

typedef struct {
    u8                  *pat[PAT_COUNT];
    f64                  assumed[PAT_COUNT];   /* random: assumed misses/iteration */
    u64                  iters[PAT_COUNT];
    f64                  ns[PAT_COUNT];        /* best ns/iteration */
    bool                 have_counts[PAT_COUNT];
    sb_branch_pmu_counts counts[PAT_COUNT];    /* from the best trial with valid counts */
    f64                  count_ns[PAT_COUNT];
    f64                  add_ns;               /* best ns per dependent add */
    bool                 pmu_ok;
    const char          *pmu_name;
    char                 pmu_reason[96];
    sb_status_e          status;
} bench_ctx;

static u64 splitmix64(u64 *s) {
    u64 z = (*s += 0x9E37'79B9'7F4A'7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58'476D'1CE4'E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D0'49BB'1331'11EBULL;
    return z ^ (z >> 31);
}

/* splitmix64 rather than xorshift: the predictor partly learns xorshift's
 * linear bit structure (FINDINGS: 47.7% instead of 50% misses). */
static f64 fill(u8 *pat, pat_e p) {
    u64 rng  = 0x5EED'0000'0000'0000ULL + (u64)p;
    u64 ones = 0;
    for (u32 i = 0; i < PAT_LEN; i++) {
        bool v = false;
        switch (p) {
        case PAT_TAKEN:     v = true;                                      break;
        case PAT_NOT_TAKEN: v = false;                                     break;
        case PAT_ALT:       v = (i & 1) != 0;                              break;
        case PAT_PERIOD8:   v = i % 8 == 0;                                break;
        case PAT_PERIOD64:  v = i % 64 == 0;                               break;
        case PAT_RAND1:     v = splitmix64(&rng) < UINT64_MAX / 100;       break;
        case PAT_RAND10:    v = splitmix64(&rng) < UINT64_MAX / 10;        break;
        case PAT_RAND90:    v = splitmix64(&rng) >= UINT64_MAX / 10;       break;
        case PAT_RAND50:    v = (splitmix64(&rng) >> 63) != 0;             break;
        case PAT_COUNT:                                                    break;
        }
        pat[i] = v ? 1 : 0;
        ones  += v ? 1 : 0;
    }
    if (p == PAT_RAND50) { return 0.5; }  /* no predictor beats a coin flip */
    u64 minority = ones < PAT_LEN - ones ? ones : PAT_LEN - ones;
    return (f64)minority / (f64)PAT_LEN;
}

static f64 time_pattern(const bench_ctx *c, pat_e p, u64 iters, sb_branch_pmu *pmu,
                        sb_branch_pmu_counts *out, bool *out_ok) {
    *out_ok = false;
    bool counting = pmu != NULL && sb_branch_pmu_start(pmu) == SB_OK;
    u64 t0  = sb_timer_now_ns();
    u64 acc = sb_branch_kernel(c->pat[p], PAT_LEN - 1, iters, 1);
    u64 t1  = sb_timer_now_ns();
    if (counting && sb_branch_pmu_stop(pmu, out) == SB_OK) { *out_ok = true; }
    __asm__ volatile("" : : "r"(acc));
    return (f64)(t1 - t0) / (f64)iters;
}

static f64 time_chain(u64 n) {
    u64 t0  = sb_timer_now_ns();
    u64 acc = sb_branch_add_chain(n, 0);
    u64 t1  = sb_timer_now_ns();
    __asm__ volatile("" : : "r"(acc));
    return (f64)(t1 - t0) / ((f64)n * SB_BRANCH_CHAIN_LEN);
}

static u64 iters_for(f64 ns_per_iter, u64 target_ns) {
    if (ns_per_iter <= 0) { return CALIB_ITERS; }
    f64 n = (f64)target_ns / ns_per_iter;
    return n < (f64)CALIB_ITERS ? CALIB_ITERS : (u64)n;
}

/* Runs on its own thread so placement (QoS / affinity) and the per-thread
 * counters do not leak into the main thread. */
static void *measure(void *arg) {
    bench_ctx *c = arg;
    sb_thread_place_self(SB_CORE_PERF, 0);

    sb_branch_pmu pmu;
    c->pmu_ok = sb_branch_pmu_init(&pmu) == SB_OK;
    if (c->pmu_ok) { c->pmu_name = pmu.name; }
    else           { snprintf(c->pmu_reason, sizeof(c->pmu_reason), "%s", pmu.reason); }
    sb_branch_pmu *pp = c->pmu_ok ? &pmu : NULL;

    sb_timer_spin(SB_WARMUP_NS);

    /* Calibration doubles as warmup of each stream (L2-resident, 1 MiB each). */
    sb_branch_pmu_counts cnt;
    bool ok;
    for (u32 p = 0; p < PAT_COUNT; p++) {
        f64 ns = time_pattern(c, (pat_e)p, CALIB_ITERS, NULL, &cnt, &ok);
        c->iters[p]    = iters_for(ns, TRIAL_NS);
        c->ns[p]       = 1e30;
        c->count_ns[p] = 1e30;
    }
    u64 chain_n = iters_for(time_chain(100'000) * SB_BRANCH_CHAIN_LEN, CHAIN_NS);
    c->add_ns = 1e30;

    /* Interleave patterns so slow clock drift hits all of them alike. */
    for (u32 r = 0; r < ROUNDS; r++) {
        f64 a = time_chain(chain_n);
        if (a > 0 && a < c->add_ns) { c->add_ns = a; }
        for (u32 p = 0; p < PAT_COUNT; p++) {
            f64 ns = time_pattern(c, (pat_e)p, c->iters[p], pp, &cnt, &ok);
            if (ns > 0 && ns < c->ns[p]) { c->ns[p] = ns; }
            if (ok && ns > 0 && ns < c->count_ns[p]) {
                c->count_ns[p]    = ns;
                c->counts[p]      = cnt;
                c->have_counts[p] = true;
            }
        }
    }

    if (c->pmu_ok) { sb_branch_pmu_free(&pmu); }
    c->status = SB_OK;
    for (u32 p = 0; p < PAT_COUNT; p++) {
        if (c->ns[p] >= 1e29) { c->status = SB_ERR_RANGE; }
    }
    if (c->add_ns >= 1e29) { c->status = SB_ERR_RANGE; }
    return NULL;
}

/* Per-iteration rates derived from either counters or timing. */
typedef struct {
    bool counted;          /* misses (and maybe cycles) come from the PMU */
    bool counted_cycles;
    f64  ghz;              /* add-chain clock estimate */
    f64  extra_ns[PAT_COUNT];
    f64  miss[PAT_COUNT];  /* misses/iteration above the baseline (counted mode) */
    f64  extra_cyc[PAT_COUNT];
} derived;

static void derive(const bench_ctx *c, derived *d) {
    memset(d, 0, sizeof(*d));
    d->ghz = 1.0 / c->add_ns;
    f64 base_ns = (c->ns[PAT_TAKEN] + c->ns[PAT_NOT_TAKEN]) / 2.0;
    for (u32 p = 0; p < PAT_COUNT; p++) { d->extra_ns[p] = c->ns[p] - base_ns; }

    d->counted = c->pmu_ok;
    for (u32 p = 0; p < PAT_COUNT; p++) { d->counted = d->counted && c->have_counts[p]; }
    if (!d->counted) { return; }

    f64 miss[PAT_COUNT];
    f64 cyc[PAT_COUNT];
    d->counted_cycles = true;
    for (u32 p = 0; p < PAT_COUNT; p++) {
        miss[p] = (f64)c->counts[p].misses / (f64)c->iters[p];
        cyc[p]  = (f64)c->counts[p].cycles / (f64)c->iters[p];
        d->counted_cycles = d->counted_cycles && c->counts[p].has_cycles;
    }
    f64 base_miss = (miss[PAT_TAKEN] + miss[PAT_NOT_TAKEN]) / 2.0;
    f64 base_cyc  = (cyc[PAT_TAKEN] + cyc[PAT_NOT_TAKEN]) / 2.0;
    for (u32 p = 0; p < PAT_COUNT; p++) {
        d->miss[p]      = miss[p] - base_miss;
        d->extra_cyc[p] = cyc[p] - base_cyc;
    }
}

static void report_counts(const bench_ctx *c, const derived *d, f64 taken_pen_ns) {
    sb_report_group("Mispredicts per 1000 iterations");
    for (u32 p = 0; p < PAT_COUNT; p++) {
        if (d->counted) {
            f64 total = (f64)c->counts[p].misses / (f64)c->iters[p];
            sb_report_value(pats[p].name, 1000.0 * total, "/1k iter", SB_KIND_MEASURED);
        } else if (pats[p].random) {
            sb_report_value(pats[p].name, 1000.0 * c->assumed[p], "/1k iter", SB_KIND_ESTIMATE);
        } else if (taken_pen_ns > 0) {
            /* Baseline patterns sit at the mean, so noise can make extra time
             * slightly negative; a miss count cannot be. */
            f64 implied = d->extra_ns[p] > 0 ? d->extra_ns[p] / taken_pen_ns : 0;
            sb_report_value(pats[p].name, 1000.0 * implied, "/1k iter", SB_KIND_ESTIMATE);
        } else {
            sb_report_error(pats[p].name, SB_ERR_RANGE);
        }
    }
}

static void report_penalties(const bench_ctx *c, const derived *d) {
    sb_report_group("Mispredict penalty");
    for (u32 i = 0; i < SB_ARRAY_LEN(penalties); i++) {
        const penalty_info *pi = &penalties[i];
        f64 misses = d->counted ? d->miss[pi->pat] : c->assumed[pi->pat];
        if (misses <= 1e-6 || d->extra_ns[pi->pat] <= 0) {
            sb_report_error(pi->name, SB_ERR_RANGE);
            sb_report_error(pi->name, SB_ERR_RANGE);
            continue;
        }
        f64 pen_ns = d->extra_ns[pi->pat] / misses;
        sb_report_value(pi->name, pen_ns, "ns", d->counted ? SB_KIND_MEASURED : SB_KIND_ESTIMATE);
        if (d->counted_cycles) {
            sb_report_value(pi->name, d->extra_cyc[pi->pat] / misses, "cycles", SB_KIND_MEASURED);
        } else {
            sb_report_value(pi->name, pen_ns * d->ghz, "cycles", SB_KIND_ESTIMATE);
        }
    }
    sb_report_value("Clock (dependent add chain)", d->ghz, "GHz", SB_KIND_ESTIMATE);
}

static void report_info(const bench_ctx *c, const derived *d) {
    char size[32];
    sb_report_info("Kernel: asm loop with one measured branch per iteration; both outcomes run");
    sb_report_info("  3 instructions with 1 taken branch (3 branches/iter incl. loop back-edge)");
    sb_report_info("Streams: %s outcomes each (splitmix64), best of %u interleaved trials",
                   sb_fmt_size(PAT_LEN, size, sizeof(size)), ROUNDS);
    sb_report_info("Extra time is measured against the always/never-taken mean");
    if (d->counted) {
        f64 br = (f64)c->counts[PAT_TAKEN].branches / (f64)c->iters[PAT_TAKEN];
        sb_report_info("Counters: %s, %.2f branches/iter retired (expect 3)", c->pmu_name, br);
        sb_report_info("Penalty = extra time (or cycles) / misses above the always/never-taken mean");
        if (!d->counted_cycles) { sb_report_info("Cycles = ns x clock estimate (no cycle counter)"); }
    } else {
        if (c->pmu_ok) { sb_report_info("Counters: %s, but reads were partial (multiplexed?)", c->pmu_name); }
        else           { sb_report_info("Counters: unavailable (%s)", c->pmu_reason); }
        sb_report_info("Assumed: random streams miss once per minority outcome (50%%: every");
        sb_report_info("  other); other miss counts = extra time / taken-branch penalty");
        sb_report_info("Cycles = ns x clock estimate (dependent adds at 1/cycle)");
    }
}

static sb_status_e branch_run(void) {
    bench_ctx *c = SB_MALLOC(sizeof(*c));
    if (c == NULL) { return SB_ERR_NOMEM; }
    memset(c, 0, sizeof(*c));

    sb_status_e s = SB_OK;
    for (u32 p = 0; p < PAT_COUNT; p++) {
        c->pat[p] = SB_MALLOC(PAT_LEN);
        if (c->pat[p] == NULL) {
            s = SB_ERR_NOMEM;
            goto cleanup;
        }
        c->assumed[p] = fill(c->pat[p], (pat_e)p);
    }

    pthread_t t;
    if (pthread_create(&t, NULL, measure, c) != 0) {
        s = SB_ERR_SYS;
        goto cleanup;
    }
    pthread_join(t, NULL);
    if (c->status != SB_OK) {
        s = c->status;
        goto cleanup;
    }

    derived d;
    derive(c, &d);
    report_info(c, &d);

    sb_report_group("Time per iteration");
    for (u32 p = 0; p < PAT_COUNT; p++) {
        sb_report_value(pats[p].name, 1000.0 * c->ns[p], "ps", SB_KIND_MEASURED);
    }

    f64 taken_pen_ns = c->assumed[PAT_RAND10] > 0 ? d.extra_ns[PAT_RAND10] / c->assumed[PAT_RAND10] : 0;
    report_counts(c, &d, taken_pen_ns);
    report_penalties(c, &d);

cleanup:
    if (s != SB_OK) { sb_report_error("Branch prediction", s); }
    for (u32 p = 0; p < PAT_COUNT; p++) { SB_FREE(c->pat[p]); }
    SB_FREE(c);
    return s;
}

const sb_section sb_section_branch = {
    .name       = "branch",
    .title      = "Branch Prediction",
    .help       = "  One data-dependent branch per iteration of an inline-asm loop, fed\n"
                  "  from 1 MiB outcome streams. Both outcomes run the same instructions\n"
                  "  and taken branches, so extra time over the always/never-taken mean\n"
                  "  is mispredict cost only. Miss counts come from Linux perf_event or\n"
                  "  macOS kperf (root only); otherwise they are estimates. Penalties are\n"
                  "  split by direction (a usually-taken branch costs more to miss on\n"
                  "  Apple M4) and given in ns and cycles (clock from a dependent add chain).\n",
    .run        = branch_run,
    .repeatable = true,
};
