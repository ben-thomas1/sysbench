#include "core/thread.h"
#include "core/timer.h"

#include <pthread.h>
#include <stdatomic.h>
#include <string.h>
#include <time.h>

#if defined(__APPLE__)
#include <pthread/qos.h>
#else
#include <sched.h>
#endif

u32 sb_thread_count_for(sb_core_e place) {
    const sb_platform *p = sb_platform_get();
    if (place == SB_CORE_PERF && p->nperf > 0) { return p->nperf; }
    if (place == SB_CORE_EFF && p->neff > 0) { return p->neff; }
    return p->ncpu;
}

sb_status_e sb_thread_place_self(sb_core_e place, u32 idx) {
#if defined(__APPLE__)
    (void)idx;
    qos_class_t q = place == SB_CORE_EFF ? QOS_CLASS_BACKGROUND : QOS_CLASS_USER_INTERACTIVE;
    return pthread_set_qos_class_self_np(q, 0) == 0 ? SB_OK : SB_ERR_SYS;
#else
    const sb_platform *p = sb_platform_get();
    const u16 *ids = p->cpu_ids;
    u32 n = p->ncpu;
    if (place == SB_CORE_PERF && p->nperf > 0) { ids = p->perf_ids; n = p->nperf; }
    if (place == SB_CORE_EFF && p->neff > 0) { ids = p->eff_ids; n = p->neff; }
    if (n == 0) { return SB_ERR_UNSUPPORTED; }
    if (place == SB_CORE_ANY) { return SB_OK; } /* let the scheduler spread threads */
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(ids[idx % n], &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0 ? SB_OK : SB_ERR_SYS;
#endif
}

typedef struct {
    const sb_par_cfg *cfg;
    u32               tid;
    atomic_uint      *ready;
    atomic_bool      *go;
    atomic_bool      *stop;
    u64               units;
    u64               ns;
    sb_status_e       status;
} worker_arg;

static void *worker(void *p) {
    worker_arg       *a   = p;
    const sb_par_cfg *cfg = a->cfg;

    a->status = sb_thread_place_self(cfg->place, a->tid);
    if (a->status == SB_ERR_SYS && cfg->place == SB_CORE_ANY) { a->status = SB_OK; }

    /* Warm up with the real work so caches/TLBs and clocks are in steady state. */
    u64 warm = cfg->warmup_ns ? cfg->warmup_ns : SB_WARMUP_NS;
    u64 t0   = sb_timer_now_ns();
    while (sb_timer_now_ns() - t0 < warm) { (void)cfg->fn(cfg->ctx, a->tid, cfg->chunk); }

    atomic_fetch_add(a->ready, 1);
    while (!atomic_load_explicit(a->go, memory_order_acquire)) {
        (void)cfg->fn(cfg->ctx, a->tid, cfg->chunk); /* keep the core busy while others warm up */
    }

    u64 units = 0;
    t0 = sb_timer_now_ns();
    while (!atomic_load_explicit(a->stop, memory_order_relaxed)) {
        units += cfg->fn(cfg->ctx, a->tid, cfg->chunk);
    }
    a->ns    = sb_timer_now_ns() - t0;
    a->units = units;
    return NULL;
}

sb_status_e sb_par_run(const sb_par_cfg *cfg, sb_par_result *out) {
    memset(out, 0, sizeof(*out));
    if (cfg->fn == NULL || cfg->nthreads == 0 || cfg->chunk == 0) { return SB_ERR_INVALID; }
    if (cfg->nthreads > SB_MAX_CPUS) { return SB_ERR_RANGE; }

    u32          n    = cfg->nthreads;
    worker_arg  *args = SB_MALLOC(n * sizeof(*args));
    pthread_t   *th   = SB_MALLOC(n * sizeof(*th));
    if (args == NULL || th == NULL) {
        SB_FREE(args);
        SB_FREE(th);
        return SB_ERR_NOMEM;
    }

    atomic_uint ready = 0;
    atomic_bool go    = false;
    atomic_bool stop  = false;
    u32 created = 0;
    sb_status_e st = SB_OK;
    for (u32 i = 0; i < n; i++) {
        args[i] = (worker_arg){ .cfg = cfg, .tid = i, .ready = &ready, .go = &go, .stop = &stop };
        if (pthread_create(&th[i], NULL, worker, &args[i]) != 0) {
            st = SB_ERR_SYS;
            break;
        }
        created++;
    }

    if (st == SB_OK) {
        while (atomic_load(&ready) < n) {
            struct timespec ts = { .tv_sec = 0, .tv_nsec = 1'000'000 };
            nanosleep(&ts, NULL);
        }
        atomic_store_explicit(&go, true, memory_order_release);
        u64 window = cfg->window_ns ? cfg->window_ns : 1'000'000'000ULL;
        struct timespec ts = { .tv_sec = (time_t)(window / 1'000'000'000ULL),
                               .tv_nsec = (long)(window % 1'000'000'000ULL) };
        nanosleep(&ts, NULL);
    }
    /* On creation failure, release whatever started so it can exit. */
    atomic_store_explicit(&go, true, memory_order_release);
    atomic_store_explicit(&stop, true, memory_order_relaxed);
    for (u32 i = 0; i < created; i++) { pthread_join(th[i], NULL); }

    if (st == SB_OK) {
        out->nthreads = n;
        out->min_rate = 1e300;
        for (u32 i = 0; i < n; i++) {
            if (args[i].status != SB_OK) { st = args[i].status; }
            if (args[i].ns == 0) { st = SB_ERR_RANGE; continue; }
            f64 rate = (f64)args[i].units / ((f64)args[i].ns / 1e9);
            out->units         += (f64)args[i].units;
            out->units_per_sec += rate;
            if (rate < out->min_rate) { out->min_rate = rate; }
            if (rate > out->max_rate) { out->max_rate = rate; }
        }
    }
    SB_FREE(args);
    SB_FREE(th);
    return st;
}
