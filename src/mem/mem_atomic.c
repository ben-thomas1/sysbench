/* Atomics on 64-bit counters.
 *   Uncontended, one thread:
 *     fetch_add latency   seq_cst fetch_add on one line; each op adds the value the
 *                         previous one returned (LDADDAL / lock xadd), ns per op.
 *     add throughput      posted adds, result unused, round-robin over 8 lines so
 *                         they don't serialise on one address (STADD / lock add),
 *                         ns per op.
 *     CAS latency         seq_cst compare-exchange on one line; each new value is
 *                         the value the previous CAS read + 1 (CASAL / lock
 *                         cmpxchg), so the chain also holds one add (1 cycle).
 *   Contended, N threads on one line (sb_par_run, startup outside the window):
 *     the same dependent fetch_add, reported as aggregate Mops/s over all
 *     threads and as per-thread ns/op = N / aggregate rate (how long each thread
 *     waits per increment). */
#include "mem/mem_internal.h"
#include "core/platform.h"
#include "core/report.h"
#include "core/thread.h"

#include <stdatomic.h>
#include <stdio.h>

#define OPS_PER_REP     64u
#define THRU_LINES      8u
#define CONTEND_CHUNK   1000u
#define CONTEND_WINDOW  400'000'000ULL

typedef struct {
    alignas(MEM_PAD) _Atomic u64 v;
} padded_ctr;

static padded_ctr ctrs[THRU_LINES];

static void fetch_add_lat_rep(void *ctx, u64 reps) {
    _Atomic u64 *c = ctx;
    u64          r = 1;
    for (u64 i = 0; i < reps * OPS_PER_REP; i++) {
        /* Adding the previous result: a data dependency with no extra ALU op
         * in the chain. The counter value is meaningless (it wraps). */
        r = atomic_fetch_add_explicit(c, r, memory_order_seq_cst);
    }
    __asm__ volatile("" : : "r"(r));
}

/* An add whose result is not needed. Compilers emit LDADD into a scratch
 * register for this, so AArch64 with LSE spells out the posted STADD. */
static inline void posted_add(_Atomic u64 *p) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_ATOMICS)
    __asm__ volatile("stadd %x[v], [%[p]]" : : [v] "r"((u64)1), [p] "r"(p) : "memory");
#else
    atomic_fetch_add_explicit(p, 1, memory_order_relaxed);
#endif
}

static void add_thru_rep(void *ctx, u64 reps) {
    padded_ctr *c = ctx;
    for (u64 i = 0; i < reps * (OPS_PER_REP / THRU_LINES); i++) {
        for (u32 k = 0; k < THRU_LINES; k++) { posted_add(&c[k].v); }
    }
}

static void cas_lat_rep(void *ctx, u64 reps) {
    _Atomic u64 *c = ctx;
    u64          v = atomic_load_explicit(c, memory_order_relaxed);
    for (u64 i = 0; i < reps * OPS_PER_REP; i++) {
        u64 seen = v;
        (void)atomic_compare_exchange_strong_explicit(c, &seen, seen + 1, memory_order_seq_cst,
                                                      memory_order_seq_cst);
        /* Always succeeds (one thread); the opaque copy keeps the next CAS
         * dependent on the value this one returned. */
        __asm__ volatile("" : "+r"(seen));
        v = seen + 1;
    }
}

static void uncontended(void) {
    struct {
        const char *name;
        mem_rep_fn  fn;
        void       *ctx;
    } t[] = {
        { "fetch_add latency, 1 line",   fetch_add_lat_rep, &ctrs[0].v },
        { "add throughput, 8 lines",     add_thru_rep,      ctrs },
        { "CAS latency, 1 line",         cas_lat_rep,       &ctrs[0].v },
    };
    for (size_t i = 0; i < SB_ARRAY_LEN(t); i++) {
        f64 ns = 0;
        sb_status_e s = mem_time_best(t[i].fn, t[i].ctx, &ns);
        if (s == SB_OK) { sb_report_value(t[i].name, ns / OPS_PER_REP, "ns/op", SB_KIND_MEASURED); }
        else            { sb_report_error(t[i].name, s); }
    }
}

static u64 contend_work(void *ctx, u32 tid, u64 chunk) {
    _Atomic u64 *c = ctx;
    u64          r = tid + 1;
    for (u64 i = 0; i < chunk; i++) { r = atomic_fetch_add_explicit(c, r, memory_order_seq_cst); }
    __asm__ volatile("" : : "r"(r));
    return chunk;
}

static void contended(u32 n) {
    char agg[64], per[64];
    snprintf(agg, sizeof(agg), "Contended fetch_add, %u threads", n);
    snprintf(per, sizeof(per), "  per thread, %u threads", n);
    sb_par_cfg cfg = {
        .nthreads  = n,
        .place     = SB_CORE_ANY,
        .window_ns = CONTEND_WINDOW,
        .chunk     = CONTEND_CHUNK,
        .fn        = contend_work,
        .ctx       = &ctrs[0].v,
    };
    sb_par_result r;
    sb_status_e   s = sb_par_run(&cfg, &r);
    if (s == SB_OK && r.units_per_sec <= 0) { s = SB_ERR_RANGE; }
    if (s != SB_OK) {
        sb_report_error(agg, s);
        return;
    }
    sb_report_value(agg, r.units_per_sec / 1e6, "Mops/s", SB_KIND_MEASURED);
    sb_report_value(per, (f64)n * 1e9 / r.units_per_sec, "ns/op", SB_KIND_MEASURED);
}

sb_status_e mem_run_atomics(void) {
    sb_report_group("Atomics: 64-bit counters, uncontended (1 thread)");
    uncontended();

    const sb_platform *p = sb_platform_get();
    if (p->ncpu < 2) { return SB_OK; }
    sb_report_group("Atomics: contended, all threads on one line");
    sb_report_info("Aggregate = all threads' increments per second; per thread = N / aggregate.");
    u32 cand[3] = { 2, 4, p->ncpu };
    u32 last    = 0;
    for (u32 i = 0; i < 3; i++) {
        if (cand[i] <= last || cand[i] > p->ncpu) { continue; }
        contended(cand[i]);
        last = cand[i];
    }
    return SB_OK;
}
