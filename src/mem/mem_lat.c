/* Load-to-use latency: a pointer chase through one random cycle over the
 * nodes, so every load depends on the previous one and the order defeats the
 * prefetchers. Stride = the platform's cache line, so no two nodes share a line.
 *
 * The chase loop is unrolled by an odd count (61). Node counts are powers of
 * two, so each load instruction walks the whole cycle instead of revisiting one
 * address: with 64 loads per iteration and 32 or 64 nodes, each load PC always
 * saw the same address and Apple's load-address predictor broke the dependency
 * chain (0.15 ns at 4 KiB, below one cycle). */
#include "mem/mem_internal.h"
#include "core/platform.h"
#include "core/report.h"

#include <stdio.h>

#define LAT_MIN_BYTES   (4ULL << 10)
#define CHASE_PER_REP   61u
#define TLB_MAX_PAGES   32768u
#define RNG_SEED        0x9E3779B97F4A7C15ULL

static const u32 tlb_pages[] = { 16, 64, 256, 1024, 2048, 4096, 8192, TLB_MAX_PAGES };

typedef struct {
    void **p;
} chase_ctx;

#define CHASE4 p = (void **)*p; p = (void **)*p; p = (void **)*p; p = (void **)*p;

static void chase_rep(void *ctx, u64 reps) {
    chase_ctx *c = ctx;
    void     **p = c->p;
    for (u64 i = 0; i < reps; i++) {
        static_assert(CHASE_PER_REP == 61, "unrolled 15 x 4 + 1");
        CHASE4 CHASE4 CHASE4 CHASE4 CHASE4 CHASE4 CHASE4 CHASE4
        CHASE4 CHASE4 CHASE4 CHASE4 CHASE4 CHASE4 CHASE4
        p = (void **)*p;
        __asm__ volatile("" : "+r"(p));
    }
    c->p = p;
}

#undef CHASE4

/* Link n nodes at base + i * stride into one cycle in Sattolo order (a uniformly
 * random cyclic permutation, so the chase visits every node). perm has n slots. */
static void **build_cycle(u8 *base, u32 n, size_t stride, u32 *perm, u64 *rng) {
    for (u32 i = 0; i < n; i++) { perm[i] = i; }
    for (u32 i = n - 1; i > 0; i--) {
        u32 j   = mem_rng_below(rng, i);
        u32 t   = perm[i];
        perm[i] = perm[j];
        perm[j] = t;
    }
    for (u32 i = 0; i < n; i++) {
        void **node = (void **)(void *)(base + (size_t)i * stride);
        *node       = base + (size_t)perm[i] * stride;
    }
    return (void **)(void *)base;
}

static sb_status_e measure_chain(void **start, f64 *out_ns) {
    chase_ctx   c = { .p = start };
    f64         ns_rep = 0;
    sb_status_e s = mem_time_best(chase_rep, &c, &ns_rep);
    if (s != SB_OK) { return s; }
    *out_ns = ns_rep / CHASE_PER_REP;
    return SB_OK;
}

sb_status_e mem_run_latency(const mem_arena *a) {
    const sb_platform *p    = sb_platform_get();
    size_t             line = p->cache_line;
    size_t             page = p->page_size;
    u64                rng  = RNG_SEED;

    size_t max_nodes = a->bytes / line;
    if (max_nodes > UINT32_MAX) { return SB_ERR_RANGE; }
    u32 *perm = SB_MALLOC(max_nodes * sizeof(u32));
    if (perm == NULL) { return SB_ERR_NOMEM; }

    char title[96], name[64], sz[24];
    snprintf(title, sizeof(title), "Latency: random pointer chase, %zu B stride", line);
    sb_report_group(title);

    /* Latency at the size whose node count matches the largest TLB row; the
     * same lines packed contiguously, which is the TLB rows' no-walk baseline. */
    f64 packed_ns = 0;
    u32 tlb_max   = TLB_MAX_PAGES;
    while ((size_t)tlb_max * (page + line) > a->bytes) { tlb_max /= 2; }

    for (u64 bytes = LAT_MIN_BYTES; bytes <= a->bytes; bytes *= 2) {
        snprintf(name, sizeof(name), "Latency %s", sb_fmt_size(bytes, sz, sizeof(sz)));
        u32  n     = (u32)(bytes / line);
        void **st  = build_cycle(a->base, n, line, perm, &rng);
        f64  ns    = 0;
        sb_status_e s = measure_chain(st, &ns);
        if (s != SB_OK) {
            sb_report_error(name, s);
            continue;
        }
        sb_report_value(name, ns, "ns", SB_KIND_MEASURED);
        if (n == tlb_max) { packed_ns = ns; }
    }

    /* TLB: one line per page, stepping page + line so lines also spread over
     * cache sets. The data is at most tlb_max lines (4 MiB at 128 B), so it
     * stays in L1/L2 while the page count grows past the TLB reach. */
    sb_fmt_size(page, sz, sizeof(sz));
    snprintf(title, sizeof(title), "TLB: one line per %s page, random order", sz);
    sb_report_group(title);
    f64 plateau_ns = 0;
    for (size_t i = 0; i < SB_ARRAY_LEN(tlb_pages) && tlb_pages[i] <= tlb_max; i++) {
        u32 n = tlb_pages[i];
        snprintf(name, sizeof(name), "TLB %u pages (%s span)", n,
                 sb_fmt_size((u64)n * page, sz, sizeof(sz)));
        void **st = build_cycle(a->base, n, page + line, perm, &rng);
        f64    ns = 0;
        sb_status_e s = measure_chain(st, &ns);
        if (s != SB_OK) {
            sb_report_error(name, s);
            continue;
        }
        sb_report_value(name, ns, "ns", SB_KIND_MEASURED);
        if (n == tlb_max) { plateau_ns = ns; }
    }
    if (plateau_ns > 0 && packed_ns > 0) {
        sb_report_info("Page walk = TLB %u pages minus Latency %s (same %u lines, packed).",
                       tlb_max, sb_fmt_size((u64)tlb_max * line, sz, sizeof(sz)), tlb_max);
        sb_report_info("Large-size latency rows include this walk on every load.");
        sb_report_value("Page walk (TLB miss), estimate", plateau_ns - packed_ns, "ns", SB_KIND_ESTIMATE);
    }

    SB_FREE(perm);
    return SB_OK;
}
