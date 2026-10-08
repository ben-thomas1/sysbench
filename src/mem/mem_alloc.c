/* System allocator cost, in ns per malloc+free pair. Calls libc malloc/free
 * directly (not SB_MALLOC), since the C library allocator is what is measured.
 *   pair <size>   malloc then free of one block, repeated: the same-size fast
 *                 path (thread cache / free-list hit). 1 MiB shows the
 *                 large-block path (mmap or a large free list, by platform).
 *   held batch    allocate BATCH blocks of random sizes 16 B-4 KiB (writing one
 *                 byte to each), then free them in a random order: a
 *                 fragmenting pattern that misses the fast path more often. */
#include "mem/mem_internal.h"
#include "core/report.h"

#include <stdio.h>
#include <stdlib.h>

#define PAIRS_PER_REP 64u
#define BATCH         16384u
#define BATCH_MIN     16u
#define BATCH_MAX     4096u
#define RNG_SEED      0xD1B54A32D192ED03ULL

static const size_t pair_sizes[] = { 64, 4096, 1u << 20 };

typedef struct {
    size_t size;
    bool   failed;
} pair_ctx;

static void pair_rep(void *ctx, u64 reps) {
    pair_ctx *c = ctx;
    for (u64 i = 0; i < reps * PAIRS_PER_REP; i++) {
        void *p = malloc(c->size);
        if (p == NULL) {
            c->failed = true;
            return;
        }
        __asm__ volatile("" : "+r"(p) : : "memory");
        free(p);
    }
}

typedef struct {
    void **ptr;
    u32   *size;
    u32   *order;
    bool   failed;
} batch_ctx;

static void batch_rep(void *ctx, u64 reps) {
    batch_ctx *c = ctx;
    for (u64 r = 0; r < reps; r++) {
        for (u32 i = 0; i < BATCH; i++) {
            u8 *p = malloc(c->size[i]);
            if (p == NULL) {
                c->failed = true;
                for (u32 k = 0; k < i; k++) { free(c->ptr[k]); }
                return;
            }
            p[0]      = (u8)i;
            c->ptr[i] = p;
        }
        for (u32 i = 0; i < BATCH; i++) { free(c->ptr[c->order[i]]); }
    }
}

static void measure_batch(void) {
    const char *name = "Held batch, 16 B-4 KiB, random free";
    batch_ctx   c    = {
        .ptr   = SB_MALLOC(BATCH * sizeof(void *)),
        .size  = SB_MALLOC(BATCH * sizeof(u32)),
        .order = SB_MALLOC(BATCH * sizeof(u32)),
    };
    sb_status_e s = SB_OK;
    if (c.ptr == NULL || c.size == NULL || c.order == NULL) {
        s = SB_ERR_NOMEM;
        goto out;
    }
    u64 rng = RNG_SEED;
    for (u32 i = 0; i < BATCH; i++) {
        c.size[i]  = BATCH_MIN + mem_rng_below(&rng, BATCH_MAX - BATCH_MIN + 1);
        c.order[i] = i;
    }
    for (u32 i = BATCH - 1; i > 0; i--) {
        u32 j      = mem_rng_below(&rng, i + 1);
        u32 t      = c.order[i];
        c.order[i] = c.order[j];
        c.order[j] = t;
    }
    f64 ns = 0;
    s = mem_time_best(batch_rep, &c, &ns);
    if (s == SB_OK && c.failed) { s = SB_ERR_NOMEM; }
    if (s == SB_OK) { sb_report_value(name, ns / BATCH, "ns/pair", SB_KIND_MEASURED); }
out:
    if (s != SB_OK) { sb_report_error(name, s); }
    SB_FREE(c.ptr);
    SB_FREE(c.size);
    SB_FREE(c.order);
}

sb_status_e mem_run_alloc(void) {
    sb_report_group("Allocation: libc malloc+free, 1 thread");
    char name[64], sz[24];
    for (size_t i = 0; i < SB_ARRAY_LEN(pair_sizes); i++) {
        snprintf(name, sizeof(name), "malloc+free pair, %s", sb_fmt_size(pair_sizes[i], sz, sizeof(sz)));
        pair_ctx    c  = { .size = pair_sizes[i] };
        f64         ns = 0;
        sb_status_e s  = mem_time_best(pair_rep, &c, &ns);
        if (s == SB_OK && c.failed) { s = SB_ERR_NOMEM; }
        if (s == SB_OK) { sb_report_value(name, ns / PAIRS_PER_REP, "ns/pair", SB_KIND_MEASURED); }
        else            { sb_report_error(name, s); }
    }
    measure_batch();
    return SB_OK;
}
