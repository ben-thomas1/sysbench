#include "branch/pmu.h"

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* kperf is a private framework and needs root. Events are looked up by name
 * in the kpep database (/usr/share/kpep/<cpu>.plist) through kperfdata, so
 * event numbers are not hard-coded per chip. Any failure leaves the caller
 * without counters. */

#define KPERF_PATH     "/System/Library/PrivateFrameworks/kperf.framework/kperf"
#define KPERFDATA_PATH "/System/Library/PrivateFrameworks/kperfdata.framework/kperfdata"
#define KPC_MAX        32
#define NEVENTS        3

typedef struct kpep_db     kpep_db;
typedef struct kpep_config kpep_config;
typedef struct kpep_event  kpep_event;

static struct {
    bool  loaded;
    void *kperf;
    void *kperfdata;
    int  (*kpc_set_counting)(u32 classes);
    int  (*kpc_set_thread_counting)(u32 classes);
    int  (*kpc_set_config)(u32 classes, u64 *config);
    int  (*kpc_get_thread_counters)(u32 tid, u32 count, u64 *buf);
    int  (*kpc_force_all_ctrs_set)(int val);
    int  (*kpc_force_all_ctrs_get)(int *val);
    int  (*kpep_db_create)(const char *name, kpep_db **db);
    void (*kpep_db_free)(kpep_db *db);
    int  (*kpep_db_event)(kpep_db *db, const char *name, kpep_event **ev);
    int  (*kpep_config_create)(kpep_db *db, kpep_config **cfg);
    void (*kpep_config_free)(kpep_config *cfg);
    int  (*kpep_config_force_counters)(kpep_config *cfg);
    int  (*kpep_config_add_event)(kpep_config *cfg, kpep_event **ev, u32 flag, u32 *err);
    int  (*kpep_config_kpc_classes)(kpep_config *cfg, u32 *classes);
    int  (*kpep_config_kpc_count)(kpep_config *cfg, size_t *count);
    int  (*kpep_config_kpc_map)(kpep_config *cfg, size_t *buf, size_t size);
    int  (*kpep_config_kpc)(kpep_config *cfg, u64 *buf, size_t size);
} k;

#define LOAD(lib, fn) do {                                         \
        k.fn = (typeof(k.fn))dlsym(k.lib, #fn);                    \
        if (k.fn == NULL) { return false; }                        \
    } while (0)

static bool load(void) {
    if (k.loaded) { return true; }
    k.kperf     = dlopen(KPERF_PATH, RTLD_LAZY);
    k.kperfdata = dlopen(KPERFDATA_PATH, RTLD_LAZY);
    if (k.kperf == NULL || k.kperfdata == NULL) { return false; }
    LOAD(kperf, kpc_set_counting);
    LOAD(kperf, kpc_set_thread_counting);
    LOAD(kperf, kpc_set_config);
    LOAD(kperf, kpc_get_thread_counters);
    LOAD(kperf, kpc_force_all_ctrs_set);
    LOAD(kperf, kpc_force_all_ctrs_get);
    LOAD(kperfdata, kpep_db_create);
    LOAD(kperfdata, kpep_db_free);
    LOAD(kperfdata, kpep_db_event);
    LOAD(kperfdata, kpep_config_create);
    LOAD(kperfdata, kpep_config_free);
    LOAD(kperfdata, kpep_config_force_counters);
    LOAD(kperfdata, kpep_config_add_event);
    LOAD(kperfdata, kpep_config_kpc_classes);
    LOAD(kperfdata, kpep_config_kpc_count);
    LOAD(kperfdata, kpep_config_kpc_map);
    LOAD(kperfdata, kpep_config_kpc);
    k.loaded = true;
    return true;
}

static sb_status_e fail(sb_branch_pmu *p, const char *why) {
    snprintf(p->reason, sizeof(p->reason), "%s", why);
    if (p->cfg != NULL) { k.kpep_config_free(p->cfg); }
    if (p->db != NULL)  { k.kpep_db_free(p->db); }
    p->cfg = NULL;
    p->db  = NULL;
    return SB_ERR_UNSUPPORTED;
}

sb_status_e sb_branch_pmu_init(sb_branch_pmu *p) {
    memset(p, 0, sizeof(*p));
    if (geteuid() != 0) {
        snprintf(p->reason, sizeof(p->reason), "macOS kperf needs root");
        return SB_ERR_UNSUPPORTED;
    }
    if (!load()) {
        snprintf(p->reason, sizeof(p->reason), "kperf/kperfdata framework not loadable");
        return SB_ERR_UNSUPPORTED;
    }

    kpep_db     *db  = NULL;
    kpep_config *cfg = NULL;
    if (k.kpep_db_create(NULL, &db) != 0) { return fail(p, "kpep database for this CPU not found"); }
    p->db = db;
    if (k.kpep_config_create(db, &cfg) != 0) { return fail(p, "kpep_config_create failed"); }
    p->cfg = cfg;
    if (k.kpep_config_force_counters(cfg) != 0) { return fail(p, "kpep_config_force_counters failed"); }

    /* Order matters: index 0 cycles, 1 branches, 2 misses (see map[]). */
    static const char *const names[NEVENTS] = { "FIXED_CYCLES", "INST_BRANCH", "BRANCH_MISPRED_NONSPEC" };
    for (u32 i = 0; i < NEVENTS; i++) {
        kpep_event *ev = NULL;
        if (k.kpep_db_event(db, names[i], &ev) != 0 || ev == NULL) { return fail(p, "kpep event missing"); }
        if (k.kpep_config_add_event(cfg, &ev, 0, NULL) != 0) { return fail(p, "kpep_config_add_event failed"); }
    }

    size_t nregs = 0;
    u64    regs[KPC_MAX] = {0};
    if (k.kpep_config_kpc_classes(cfg, &p->classes) != 0 ||
        k.kpep_config_kpc_count(cfg, &nregs) != 0 ||
        k.kpep_config_kpc_map(cfg, p->map, sizeof(p->map)) != 0 ||
        k.kpep_config_kpc(cfg, regs, sizeof(regs)) != 0) {
        return fail(p, "kpep config export failed");
    }
    for (u32 i = 0; i < NEVENTS; i++) {
        if (p->map[i] >= KPC_MAX) { return fail(p, "kpep counter map out of range"); }
    }

    if (k.kpc_force_all_ctrs_get(&p->force_prev) != 0) { p->force_prev = 0; }
    if (k.kpc_force_all_ctrs_set(1) != 0) { return fail(p, "kpc_force_all_ctrs_set failed"); }
    if (nregs > 0 && k.kpc_set_config(p->classes, regs) != 0) {
        k.kpc_force_all_ctrs_set(p->force_prev);
        return fail(p, "kpc_set_config failed");
    }
    if (k.kpc_set_counting(p->classes) != 0 || k.kpc_set_thread_counting(p->classes) != 0) {
        k.kpc_force_all_ctrs_set(p->force_prev);
        return fail(p, "kpc_set_counting failed");
    }
    p->has_cycles = true;
    p->name       = "kperf";
    return SB_OK;
}

static sb_status_e snap(u64 out[NEVENTS], const sb_branch_pmu *p) {
    u64 buf[KPC_MAX] = {0};
    if (k.kpc_get_thread_counters(0, KPC_MAX, buf) != 0) { return SB_ERR_SYS; }
    for (u32 i = 0; i < NEVENTS; i++) { out[i] = buf[p->map[i]]; }
    return SB_OK;
}

sb_status_e sb_branch_pmu_start(sb_branch_pmu *p) {
    return snap(p->start, p);
}

sb_status_e sb_branch_pmu_stop(sb_branch_pmu *p, sb_branch_pmu_counts *out) {
    u64 end[NEVENTS];
    sb_status_e s = snap(end, p);
    if (s != SB_OK) { return s; }
    out->cycles     = end[0] - p->start[0];
    out->branches   = end[1] - p->start[1];
    out->misses     = end[2] - p->start[2];
    out->has_cycles = true;
    return SB_OK;
}

void sb_branch_pmu_free(sb_branch_pmu *p) {
    if (p->name != NULL) {
        k.kpc_set_thread_counting(0);
        k.kpc_set_counting(0);
        k.kpc_force_all_ctrs_set(p->force_prev);
    }
    if (p->cfg != NULL) { k.kpep_config_free(p->cfg); }
    if (p->db != NULL)  { k.kpep_db_free(p->db); }
    p->cfg  = NULL;
    p->db   = NULL;
    p->name = NULL;
}
