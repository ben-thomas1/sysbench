#include "cpu/cpu.h"
#include "cpu/kern.h"
#include "cpu/pmu.h"
#include "core/platform.h"
#include "core/report.h"
#include "core/thread.h"
#include "core/timer.h"

#include <pthread.h>
#include <stdio.h>

#define WINDOW_NS     250'000'000ULL  /* timed window per compute run */
#define IPC_WINDOW_NS 200'000'000ULL  /* timed window per ops-per-cycle kernel */
#define CHUNK_INSNS   16'000'000ULL   /* instructions per work call: ~1 ms on a P-core */

/* --- Compute throughput: every kernel on every thread configuration --- */

typedef struct {
    char      title[64];   /* group header */
    char      tag[16];     /* test-name suffix, e.g. "1T", "10P", "4E", "14T" */
    sb_core_e place;
    u32       nthreads;
} run_cfg;

static u64 chunk_for(u32 insns) {
    u64 c = CHUNK_INSNS / (insns ? insns : 1);
    return c ? c : 1;
}

static sb_status_e run_kernel(const sb_cpu_kernel *k, const run_cfg *c, f64 *out) {
    sb_par_cfg cfg = {
        .nthreads  = c->nthreads,
        .place     = c->place,
        .window_ns = WINDOW_NS,
        .chunk     = chunk_for(k->insns),
        .fn        = k->fn,
    };
    sb_par_result r;
    sb_status_e s = sb_par_run(&cfg, &r);
    if (s != SB_OK) { return s; }
    if (r.units_per_sec <= 0) { return SB_ERR_RANGE; }
    *out = r.units_per_sec * k->ops / 1e9;
    return SB_OK;
}

static void run_group(const sb_cpu_kernel *ks, u32 nk, const run_cfg *c) {
    sb_report_group(c->title);
    for (u32 i = 0; i < nk; i++) {
        char name[64];
        snprintf(name, sizeof(name), "%s %s", ks[i].name, c->tag);
        if (ks[i].fn == NULL) {
            sb_report_skip(name, ks[i].skip);
            continue;
        }
        f64 v = 0;
        sb_status_e s = run_kernel(&ks[i], c, &v);
        if (s != SB_OK) {
            sb_report_error(name, s);
            continue;
        }
        sb_report_value(name, v, ks[i].unit, SB_KIND_PEAK);
    }
}

static u32 build_configs(const sb_platform *p, run_cfg *out) {
    bool hybrid = p->nperf > 0 && p->neff > 0;
    u32  n      = 0;
    out[n] = (run_cfg){ .place = SB_CORE_PERF, .nthreads = 1 };
    snprintf(out[n].title, sizeof(out[n].title), "%s", hybrid ? "Single thread, P-core" : "Single thread");
    snprintf(out[n].tag, sizeof(out[n].tag), "1T");
    n++;
    if (p->ncpu > 1) {
        out[n] = (run_cfg){ .place = SB_CORE_ANY, .nthreads = p->ncpu };
        snprintf(out[n].title, sizeof(out[n].title), "All cores, %u threads", p->ncpu);
        snprintf(out[n].tag, sizeof(out[n].tag), "%uT", p->ncpu);
        n++;
    }
    if (hybrid) {
        out[n] = (run_cfg){ .place = SB_CORE_PERF, .nthreads = p->nperf };
        snprintf(out[n].title, sizeof(out[n].title), "P-cores, %u threads", p->nperf);
        snprintf(out[n].tag, sizeof(out[n].tag), "%uP", p->nperf);
        n++;
        out[n] = (run_cfg){ .place = SB_CORE_EFF, .nthreads = p->neff };
#if defined(__APPLE__)
        snprintf(out[n].title, sizeof(out[n].title), "E-cores, %u threads, BACKGROUND QoS", p->neff);
#else
        snprintf(out[n].title, sizeof(out[n].title), "E-cores, %u threads", p->neff);
#endif
        snprintf(out[n].tag, sizeof(out[n].tag), "%uE", p->neff);
        n++;
    }
    return n;
}

/* --- Ops per cycle: one P-core thread, clock from a dependent add chain
 * (assumed 1 cycle per add) or from the cycle counter where available. --- */

typedef struct {
    u64 iters;
    u64 ns;
    u64 cycles;   /* 0 without counters */
    u64 insns;
} ipc_meas;

typedef struct {
    sb_cpu_ipc  k;
    sb_cpu_pmu  pmu;
    bool        has_pmu;
    ipc_meas    dep;
    ipc_meas    ind;
    ipc_meas    lat;
    ipc_meas    tput;
    sb_status_e status;
} ipc_ctx;

static void ipc_measure(ipc_ctx *c, sb_work_fn fn, u32 n, ipc_meas *m) {
    u64 chunk = chunk_for(n);
    u64 iters = 0;
    if (c->has_pmu) { sb_cpu_pmu_start(&c->pmu); }
    u64 t0 = sb_timer_now_ns();
    u64 dt = 0;
    do {
        iters += fn(NULL, 0, chunk);
        dt = sb_timer_now_ns() - t0;
    } while (dt < IPC_WINDOW_NS);
    *m = (ipc_meas){ .iters = iters, .ns = dt };
    if (c->has_pmu && sb_cpu_pmu_stop(&c->pmu, &m->cycles, &m->insns) != SB_OK) {
        c->has_pmu = false;
        m->cycles  = 0;
    }
}

/* Runs on its own thread so placing it does not change the main thread. */
static void *ipc_thread(void *arg) {
    ipc_ctx *c = arg;
    c->status  = sb_thread_place_self(SB_CORE_PERF, 0);
    if (c->status == SB_ERR_SYS) { c->status = SB_OK; } /* placement is best effort */
    c->has_pmu = sb_cpu_pmu_init(&c->pmu) == SB_OK;

    u64 t0 = sb_timer_now_ns();
    while (sb_timer_now_ns() - t0 < SB_WARMUP_NS) { (void)c->k.dep_add(NULL, 0, chunk_for(c->k.dep_add_n)); }

    ipc_measure(c, c->k.dep_add, c->k.dep_add_n, &c->dep);
    ipc_measure(c, c->k.ind_add, c->k.ind_add_n, &c->ind);
    ipc_measure(c, c->k.fma_lat, c->k.fma_lat_n, &c->lat);
    ipc_measure(c, c->k.fma_tput, c->k.fma_tput_n, &c->tput);
    sb_cpu_pmu_free(&c->pmu);
    return NULL;
}

static void run_ipc(void) {
    ipc_ctx c = { 0 };
    sb_cpu_ipc_get(&c.k);
    pthread_t th;
    if (pthread_create(&th, NULL, ipc_thread, &c) != 0) {
        sb_report_error("Clock: dependent add chain", SB_ERR_SYS);
        return;
    }
    pthread_join(th, NULL);
    if (c.status != SB_OK || c.dep.ns == 0) {
        sb_report_error("Clock: dependent add chain", c.status != SB_OK ? c.status : SB_ERR_RANGE);
        return;
    }

    f64 ghz = (f64)c.dep.iters * c.k.dep_add_n / (f64)c.dep.ns;
    sb_kind_e kind = c.has_pmu ? SB_KIND_MEASURED : SB_KIND_ESTIMATE;
    char name[64];

    sb_report_group("Ops per cycle, single thread (P-core)");
    sb_report_value("Clock: dependent add chain", ghz, "GHz", SB_KIND_ESTIMATE);
    if (c.has_pmu) {
        sb_report_value("Clock: cycle counter", (f64)c.dep.cycles / (f64)c.dep.ns, "GHz", SB_KIND_MEASURED);
    }

    /* Cycles spent in a measurement: counter when present, else time x clock. */
    const ipc_meas *ms[] = { &c.ind, &c.lat, &c.tput };
    f64 cyc[3];
    for (u32 i = 0; i < 3; i++) {
        cyc[i] = c.has_pmu ? (f64)ms[i]->cycles : (f64)ms[i]->ns * ghz;
        if (cyc[i] <= 0) {
            sb_report_error("Ops per cycle", SB_ERR_RANGE);
            return;
        }
    }

    snprintf(name, sizeof(name), "Integer add, %u chains", c.k.ind_add_chains);
    sb_report_value(name, (f64)c.ind.iters * c.k.ind_add_n / cyc[0], "add/cycle", kind);
    snprintf(name, sizeof(name), "FP32 %s latency", c.k.fma_name);
    sb_report_value(name, cyc[1] / ((f64)c.lat.iters * c.k.fma_lat_n), "cycles", kind);
    snprintf(name, sizeof(name), "FP32 %s, %u chains", c.k.fma_name, c.k.fma_chains);
    sb_report_value(name, (f64)c.tput.iters * c.k.fma_tput_n / cyc[2], "insn/cycle", kind);
    if (c.has_pmu) {
        snprintf(name, sizeof(name), "IPC: integer add, %u chains", c.k.ind_add_chains);
        sb_report_value(name, (f64)c.ind.insns / cyc[0], "insn/cycle", SB_KIND_MEASURED);
        snprintf(name, sizeof(name), "IPC: FP32 %s, %u chains", c.k.fma_name, c.k.fma_chains);
        sb_report_value(name, (f64)c.tput.insns / cyc[2], "insn/cycle", SB_KIND_MEASURED);
    }
}

static sb_status_e cpu_run(void) {
    const sb_platform *p = sb_platform_get();
    sb_cpu_kernel ks[SB_CPU_MAX_KERNELS];
    u32 nk = sb_cpu_kernels_get(ks);
    run_cfg cfgs[4];
    u32 nc = build_configs(p, cfgs);

    if (p->nperf > 0 && p->neff > 0) {
        sb_report_info("CPU: %s, %u logical CPUs (%uP + %uE)", p->cpu[0] ? p->cpu : p->arch, p->ncpu, p->nperf,
                       p->neff);
    } else {
        sb_report_info("CPU: %s, %u logical CPUs", p->cpu[0] ? p->cpu : p->arch, p->ncpu);
    }
    sb_report_info("SIMD: %s", sb_cpu_simd_name());
    sb_report_info("Kernels: %s.", sb_cpu_kernel_note());
    sb_report_info("Multiply-add = 2 ops; dot product = 2 ops per 8-bit MAC. Scalar integer is one row:");
    sb_report_info("  the scalar multiplier is 64-bit and narrower C types use the same instruction.");
    sb_report_info("Runs are time-based: %llu ms warmup, %llu ms window, total = sum of per-thread rates.",
                   SB_WARMUP_NS / 1'000'000ULL, WINDOW_NS / 1'000'000ULL);
#if defined(__APPLE__)
    sb_report_info("Placement: QoS classes, P = USER_INTERACTIVE, E = BACKGROUND; the scheduler picks cores.");
    sb_report_info("E-core rows: macOS runs BACKGROUND threads on the E cluster at a reduced clock");
    sb_report_info("  (~1 GHz measured, 2.6 GHz max) shared with system daemons, so they are not the");
    sb_report_info("  E-core peak. macOS cannot run the E-cores alone at full clock; the all-core");
    sb_report_info("  rows include them at full clock.");
#else
    sb_report_info("Placement: P/E groups pin one thread per CPU of that type; all-core threads are unpinned.");
#endif
    sb_report_info("Ops per cycle: clock from a dependent register add chain (assumed 1 cycle/add).");
    sb_cpu_pmu pmu;
    if (sb_cpu_pmu_init(&pmu) == SB_OK) {
        sb_report_info("  Cycle-based rows use perf_event user-mode cycle/instruction counts (measured).");
    } else {
        sb_report_info("  No unprivileged cycle counter: cycle-based rows are estimates from that clock.");
    }
    sb_cpu_pmu_free(&pmu);
    if (p->freq_max_hz > 0) { sb_report_info("OS-reported max clock: %.2f GHz", (f64)p->freq_max_hz / 1e9); }
    sb_report_info("Not measured here: matrix units (SME, AMX); see the matrix section.");

    for (u32 i = 0; i < nc; i++) { run_group(ks, nk, &cfgs[i]); }
    run_ipc();
    return SB_OK;
}

const sb_section sb_section_cpu = {
    .name       = "cpu",
    .title      = "CPU Compute Throughput",
    .help       = "  Peak multiply-add throughput per data type (scalar FP64/FP32/INT64,\n"
                  "  SIMD FP64/FP32/FP16/INT32/INT16/INT8 and int8 dot product where the\n"
                  "  ISA has it). Kernels are inline-asm loops with enough independent\n"
                  "  accumulator chains to saturate the pipes; multiply-add = 2 ops.\n"
                  "  Runs: one thread on a P-core, all cores, and on hybrid CPUs the P and\n"
                  "  E cores separately (macOS: QoS classes; Linux: affinity). Time-based:\n"
                  "  each thread runs for a fixed window and the per-thread rates are summed.\n"
                  "  Ops per cycle: clock from a dependent add chain (estimate), or cycle and\n"
                  "  instruction counters via Linux perf_event when permitted (measured).\n"
                  "  Backends: NEON (FP16/DotProd detected at runtime); SSE2, AVX2 or\n"
                  "  AVX-512 chosen at build time from -march.\n",
    .run        = cpu_run,
    .repeatable = true,
};
