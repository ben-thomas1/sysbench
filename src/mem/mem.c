#include "mem/mem.h"
#include "mem/mem_internal.h"
#include "core/platform.h"
#include "core/report.h"
#include "core/timer.h"

#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

#define ARENA_MAX     (1ULL << 30)
#define CALIB_NS      2'000'000ULL
#define TARGET_NS     25'000'000ULL
#define TIMED_REPS    3

/* --- Shared helpers --- */

u64 mem_rng_next(u64 *state) {
    u64 x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

u32 mem_rng_below(u64 *state, u32 bound) {
    u64 r = mem_rng_next(state) >> 32;
    return (u32)((r * bound) >> 32);
}

sb_status_e mem_time_best(mem_rep_fn fn, void *ctx, f64 *out_ns_per_rep) {
    u64 reps = 1;
    u64 dt   = 0;
    for (;;) {
        u64 t0 = sb_timer_now_ns();
        fn(ctx, reps);
        dt = sb_timer_now_ns() - t0;
        if (dt >= CALIB_NS) { break; }
        if (reps > (UINT64_MAX >> 2)) { return SB_ERR_RANGE; }
        reps *= 2;
    }
    f64 scaled = (f64)reps * (f64)TARGET_NS / (f64)dt;
    reps = scaled < 1.0 ? 1 : (u64)scaled;

    u64 best = UINT64_MAX;
    for (u32 r = 0; r < TIMED_REPS; r++) {
        u64 t0 = sb_timer_now_ns();
        fn(ctx, reps);
        dt = sb_timer_now_ns() - t0;
        if (dt < best) { best = dt; }
    }
    if (best == 0) { return SB_ERR_RANGE; }
    *out_ns_per_rep = (f64)best / (f64)reps;
    return SB_OK;
}

/* --- Arena: one prefaulted mapping reused by every test --- */

static sb_status_e arena_init(mem_arena *a) {
    const sb_platform *p = sb_platform_get();
    u64 want = ARENA_MAX;
    /* Stay under a quarter of RAM on small machines. */
    while (p->mem_bytes > 0 && want > p->mem_bytes / 4 && want > (64ULL << 20)) { want >>= 1; }

    void *m = mmap(NULL, (size_t)want, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (m == MAP_FAILED) { return SB_ERR_NOMEM; }
#if defined(__linux__) && defined(MADV_NOHUGEPAGE)
    /* Base pages on every system, whatever the THP setting, so the TLB rows mean the same thing. */
    (void)madvise(m, (size_t)want, MADV_NOHUGEPAGE);
#endif
    memset(m, 0x11, (size_t)want);  /* fault every page in before any timing */
    a->base  = m;
    a->bytes = (size_t)want;
    return SB_OK;
}

static void arena_free(mem_arena *a) {
    if (a->base != NULL) { munmap(a->base, a->bytes); }
    a->base  = NULL;
    a->bytes = 0;
}

static void report_platform(void) {
    const sb_platform *p = sb_platform_get();
    char l1[24], l2[24], l3[24], ram[24], pg[24];
    snprintf(ram, sizeof(ram), "%.1f GiB", (f64)p->mem_bytes / (f64)(1ULL << 30));
    sb_fmt_size(p->l1d_bytes, l1, sizeof(l1));
    sb_fmt_size(p->l2_bytes, l2, sizeof(l2));
    sb_fmt_size(p->l3_bytes, l3, sizeof(l3));
    sb_fmt_size(p->page_size, pg, sizeof(pg));
    sb_report_info("Cache line %u B, page %s, L1d %s, L2 %s%s%s, RAM %s; kernels: %s",
                   p->cache_line, pg, p->l1d_bytes ? l1 : "?", p->l2_bytes ? l2 : "?",
                   p->l3_bytes ? ", L3 " : "", p->l3_bytes ? l3 : "", p->mem_bytes ? ram : "?",
                   mem_kern_isa);
}

static sb_status_e mem_run(void) {
    report_platform();

    mem_arena   a = {0};
    sb_status_e s = arena_init(&a);
    if (s != SB_OK) {
        sb_report_error("Memory arena", s);
        return s;
    }
    char sz[24];
    sb_report_info("Single-thread tests: best of 3 x ~25 ms after warm-up, in a %s prefaulted buffer.",
                   sb_fmt_size(a.bytes, sz, sizeof(sz)));
    sb_timer_spin(SB_WARMUP_NS);

    s = mem_run_latency(&a);
    if (s == SB_OK) { s = mem_run_bandwidth(&a); }
    arena_free(&a);
    if (s == SB_OK) { s = mem_run_atomics(); }
    if (s == SB_OK) { s = mem_run_alloc(); }
    return s;
}

const sb_section sb_section_mem = {
    .name       = "mem",
    .title      = "Memory Latency & Bandwidth",
    .help       =
        "  Latency: pointer chase through one random cycle, stride = cache line,\n"
        "  4 KiB to 1 GiB; TLB rows touch one line per page to show page-walk cost.\n"
        "  Read bandwidth: one thread over a size sweep, then DRAM with 2, 4, all\n"
        "  P-cores, all cores and E-cores (time-based); memcpy counts read + write bytes.\n"
        "  Store bandwidth: best of two loop shapes per size, plus the slower shape\n"
        "  and non-temporal stores at DRAM size. Atomics: uncontended fetch_add\n"
        "  latency, independent-add throughput and CAS latency; contended fetch_add\n"
        "  on one line as aggregate Mops/s and per-thread ns/op. malloc+free pairs at\n"
        "  64 B, 4 KiB and 1 MiB, and a held batch of mixed sizes freed in random order.\n",
    .run        = mem_run,
    .repeatable = true,
};
