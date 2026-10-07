#include "bench.h"
#include "cpu_bench.h"
#include "sysinfo.h"
#include "sysinfo_platform.h"
#include "timer.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#ifdef __linux__
#include <sched.h>
#endif

#define IPC_ITERS   500000000ULL

static double measure_ipc_dependent(double freq_ghz) {
    uint64_t a = 1;
    uint64_t t0 = timer_ns();
    for (uint64_t i = 0; i < IPC_ITERS; i++) {
        a += 1;
        __asm__ volatile("" : "+r"(a));
    }
    uint64_t t1 = timer_ns();
    double elapsed_s = (double)(t1 - t0) / 1e9;
    double cycles = elapsed_s * freq_ghz * 1e9;
    return (double)IPC_ITERS / cycles;
}

static double measure_ipc_independent(double freq_ghz) {
    uint64_t a0=1, a1=2, a2=3, a3=4, a4=5;
    uint64_t a5=6, a6=7, a7=8, a8=9, a9=10;
    uint64_t t0 = timer_ns();
    for (uint64_t i = 0; i < IPC_ITERS; i++) {
        a0 += 1; a1 += 2; a2 += 3; a3 += 4; a4 += 5;
        a5 += 6; a6 += 7; a7 += 8; a8 += 9; a9 += 10;
        __asm__ volatile("" : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3), "+r"(a4),
                              "+r"(a5), "+r"(a6), "+r"(a7), "+r"(a8), "+r"(a9));
    }
    uint64_t t1 = timer_ns();
    double elapsed_s = (double)(t1 - t0) / 1e9;
    double cycles = elapsed_s * freq_ghz * 1e9;
    return (double)IPC_ITERS * 10.0 / cycles;
}

static double measure_ipc_fp_dependent(double freq_ghz) {
    float a = 1.0f;
    const float k1 = 0.9999999f, k2 = 0.0000001f;
    uint64_t t0 = timer_ns();
    for (uint64_t i = 0; i < IPC_ITERS; i++) {
        a = a * k1 + k2;
        __asm__ volatile("" : FP_REG(a));
    }
    uint64_t t1 = timer_ns();
    double elapsed_s = (double)(t1 - t0) / 1e9;
    double cycles = elapsed_s * freq_ghz * 1e9;
    return (double)IPC_ITERS / cycles;
}

static double measure_ipc_fp_independent(double freq_ghz) {
    float a0=1.0f, a1=1.01f, a2=1.02f, a3=1.03f, a4=1.04f;
    float a5=1.05f, a6=1.06f, a7=1.07f, a8=1.08f, a9=1.09f;
    const float k1 = 0.9999999f, k2 = 0.0000001f;
    uint64_t t0 = timer_ns();
    for (uint64_t i = 0; i < IPC_ITERS; i++) {
        a0=a0*k1+k2; a1=a1*k1+k2; a2=a2*k1+k2; a3=a3*k1+k2; a4=a4*k1+k2;
        a5=a5*k1+k2; a6=a6*k1+k2; a7=a7*k1+k2; a8=a8*k1+k2; a9=a9*k1+k2;
        __asm__ volatile("" : FP_REG(a0),FP_REG(a1),FP_REG(a2),FP_REG(a3),FP_REG(a4),
                              FP_REG(a5),FP_REG(a6),FP_REG(a7),FP_REG(a8),FP_REG(a9));
    }
    uint64_t t1 = timer_ns();
    double elapsed_s = (double)(t1 - t0) / 1e9;
    double cycles = elapsed_s * freq_ghz * 1e9;
    return (double)IPC_ITERS * 10.0 / cycles;
}

/* --- Calibration --- */

/* Time the calibration directly: SIMD GOPS includes a variable lane count. */
static uint64_t calibrate_iters(bench_fn fn, uint64_t min_iters) {
    if (!fn) return 0;
    uint64_t cal = 1000000;  /* 1M calibration burst */
    uint64_t t0 = timer_ns();
    double gops = fn(cal);
    uint64_t elapsed = timer_ns() - t0;
    if (gops <= 0 || elapsed == 0) return min_iters;
    double cal_time_s = (double)elapsed / 1e9;
    uint64_t target = (uint64_t)(0.5 / cal_time_s * cal);
    if (target < min_iters) target = min_iters;
    return target;
}

/* --- Multicore (per-benchmark) --- */

struct mcore_single_arg {
    int core_id;
    bench_fn fn;
    uint64_t iters;
    int ok;
};

static void *mcore_single_worker(void *arg) {
    struct mcore_single_arg *a = (struct mcore_single_arg *)arg;

#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(a->core_id, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0)
        return NULL;
#endif

    a->ok = a->fn(a->iters) > 0;
    return NULL;
}

/* One mode per group, with total work divided by a common wall interval.
 * Equal work per worker includes the time waiting for slower cores to finish. */
static double measure_multicore(bench_fn fn, unsigned lanes, uint64_t iters,
                                const int *core_ids, int nc) {
    if (!fn || nc < 1) return 0;
    struct mcore_single_arg args[64];
    pthread_t threads[64];
    int created = 0, ok = 1;
    for (int t = 0; t < nc; t++)
        args[t] = (struct mcore_single_arg){core_ids[t], fn, iters, 0};
    uint64_t t0 = timer_ns();
    for (int t = 0; t < nc; t++) {
        if (pthread_create(&threads[t], NULL, mcore_single_worker, &args[t]) != 0)
            break;
        created++;
    }
    for (int t = 0; t < created; t++) {
        pthread_join(threads[t], NULL);
        if (!args[t].ok) ok = 0;
    }
    uint64_t elapsed = timer_ns() - t0;
    if (!ok || created != nc || elapsed == 0) return 0;
    return (double)iters * BENCH_NCHAINS * 2.0 * lanes * nc / elapsed;
}

/* --- CPU info --- */

static void print_cpu_info(const struct cpu_info *ci) {
    if (ci->name[0])
        printf("  CPU: %s\n", ci->name);
    printf("  Arch: %s\n", ci->arch);
    printf("  SIMD: %s\n", cpu_simd_name);
    if (ci->cores_perf > 0 && ci->cores_eff > 0)
        printf("  Cores: %d (%dP + %dE)\n",
               ci->cores_total, ci->cores_perf, ci->cores_eff);
    else
        printf("  Logical CPUs: %d\n", ci->cores_total);
}

/* --- Entry point --- */

void bench_cpu(void) {
    struct cpu_info ci;
    query_cpu_info(&ci);
    int ncores = ci.cores_total;
    int core_ids[64], nc = 0;
#ifdef __linux__
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
        for (int cpu = 0; cpu < CPU_SETSIZE && nc < 64; cpu++)
            if (CPU_ISSET(cpu, &allowed)) core_ids[nc++] = cpu;
    }
#else
    for (int cpu = 0; cpu < ncores && nc < 64; cpu++) core_ids[nc++] = cpu;
#endif

    printf("=== CPU Compute Throughput ===\n");
    print_cpu_info(&ci);
    printf("  Multicore: %d workers, equal work per worker, wall-clock throughput\n", nc);
    if (ncores > 64) printf("  Worker count capped at 64\n");

    /* Print header, then each row as its SC + MC benchmarks complete */
    printf("\n%-18s %10s %10s %10s %10s %10s\n",
           "Test", "1c Scalar", "1c SIMD", "Nc Scalar", "Nc SIMD", "Unit");
    printf("%-18s %10s %10s %10s %10s %10s\n",
           "----", "---------", "-------", "---------", "-------", "----");
    fflush(stdout);

    struct ratio_row {
        const char *name;
        double simd_vs_scalar;
        double mc_scalar_vs_sc_scalar;
        double mc_simd_vs_sc_simd;
        double mc_simd_vs_sc_scalar;
    } ratios[32];
    int ratio_count = 0;

    for (int i = 0; i < cpu_bench_count; i++) {
        uint64_t sc_iters = calibrate_iters(cpu_benches[i].scalar, 10000000);
        uint64_t sv_iters = calibrate_iters(cpu_benches[i].simd, 10000000);
        double sc_s = cpu_benches[i].scalar ? cpu_benches[i].scalar(sc_iters) : 0;
        double sc_v = cpu_benches[i].simd ? cpu_benches[i].simd(sv_iters) : 0;

        /* Run this benchmark on all cores */
        double mc_s = measure_multicore(cpu_benches[i].scalar, 1, sc_iters / 2,
                                         core_ids, nc);
        double mc_v = measure_multicore(cpu_benches[i].simd, cpu_benches[i].simd_lanes,
                                         sv_iters / 2, core_ids, nc);

        /* Format: handle NULL scalar and/or simd */
        char s_sc[16], s_sv[16], s_mc[16], s_mv[16];
        if (sc_s > 0) snprintf(s_sc, sizeof(s_sc), "%10.2f", sc_s);
        else          snprintf(s_sc, sizeof(s_sc), "%10s", "n/a");
        if (sc_v > 0) snprintf(s_sv, sizeof(s_sv), "%10.2f", sc_v);
        else          snprintf(s_sv, sizeof(s_sv), "%10s", "n/a");
        if (mc_s > 0) snprintf(s_mc, sizeof(s_mc), "%10.2f", mc_s);
        else          snprintf(s_mc, sizeof(s_mc), "%10s", "n/a");
        if (mc_v > 0) snprintf(s_mv, sizeof(s_mv), "%10.2f", mc_v);
        else          snprintf(s_mv, sizeof(s_mv), "%10s", "n/a");

        printf("%-18s %s %s %s %s %10s\n",
               cpu_benches[i].name, s_sc, s_sv, s_mc, s_mv,
               cpu_benches[i].unit);
        fflush(stdout);

        if (ratio_count < (int)(sizeof(ratios) / sizeof(ratios[0]))) {
            ratios[ratio_count].name = cpu_benches[i].name;
            ratios[ratio_count].simd_vs_scalar =
                (sc_s > 0 && sc_v > 0) ? sc_v / sc_s : 0;
            ratios[ratio_count].mc_scalar_vs_sc_scalar =
                (sc_s > 0 && mc_s > 0) ? mc_s / sc_s : 0;
            ratios[ratio_count].mc_simd_vs_sc_simd =
                (sc_v > 0 && mc_v > 0) ? mc_v / sc_v : 0;
            ratios[ratio_count].mc_simd_vs_sc_scalar =
                (sc_s > 0 && mc_v > 0) ? mc_v / sc_s : 0;
            ratio_count++;
        }
    }

    printf("\n--- CPU Scaling Ratios ---\n");
    printf("%-18s %18s %18s %18s %18s\n",
           "Test", "1c SIMD/Scalar", "Nc Scalar/1c Scalar",
           "Nc SIMD/1c SIMD", "Nc SIMD/1c Scalar");
    printf("%-18s %18s %18s %18s %18s\n",
           "----", "---------------", "-------------------",
           "---------------", "-------------------");
    for (int i = 0; i < ratio_count; i++) {
        char r0[24], r1[24], r2[24], r3[24];
        if (ratios[i].simd_vs_scalar > 0)
            snprintf(r0, sizeof(r0), "%17.1fx", ratios[i].simd_vs_scalar);
        else
            snprintf(r0, sizeof(r0), "%18s", "n/a");
        if (ratios[i].mc_scalar_vs_sc_scalar > 0)
            snprintf(r1, sizeof(r1), "%17.1fx", ratios[i].mc_scalar_vs_sc_scalar);
        else
            snprintf(r1, sizeof(r1), "%18s", "n/a");
        if (ratios[i].mc_simd_vs_sc_simd > 0)
            snprintf(r2, sizeof(r2), "%17.1fx", ratios[i].mc_simd_vs_sc_simd);
        else
            snprintf(r2, sizeof(r2), "%18s", "n/a");
        if (ratios[i].mc_simd_vs_sc_scalar > 0)
            snprintf(r3, sizeof(r3), "%17.1fx", ratios[i].mc_simd_vs_sc_scalar);
        else
            snprintf(r3, sizeof(r3), "%18s", "n/a");

        printf("%-18s %18s %18s %18s %18s\n",
               ratios[i].name, r0, r1, r2, r3);
        fflush(stdout);
    }

    /* IPC */
    double freq = get_cpu_freq_ghz();
    printf("\n--- IPC Estimate (at %.2f GHz) ---\n", freq);
    printf("  Reference clock only; no hardware cycle/instruction counters.\n");
    if (!ci.freq_hz)
        printf("  Clock calibrated from dependent ADD, assuming one cycle per ADD.\n");
    else
        printf("  Reported maximum clock may differ from the clock during measurement.\n");
    printf("%-24s %10s %10s\n", "", "INT ADD", "FP32 FMA");
    printf("%-24s %10s %10s\n", "", "-------", "--------");

    double ipc_dep_int = measure_ipc_dependent(freq);
    double ipc_dep_fp = measure_ipc_fp_dependent(freq);
    printf("%-24s %10.2f %10.2f\n", "Dependent chain", ipc_dep_int, ipc_dep_fp);
    fflush(stdout);

    double ipc_ind_int = measure_ipc_independent(freq);
    double ipc_ind_fp = measure_ipc_fp_independent(freq);
    printf("%-24s %10.2f %10.2f\n", "Independent ops", ipc_ind_int, ipc_ind_fp);
    fflush(stdout);

    double ratio_int = ipc_dep_int > 0 ? ipc_ind_int / ipc_dep_int : 0;
    double ratio_fp  = ipc_dep_fp  > 0 ? ipc_ind_fp  / ipc_dep_fp  : 0;
    printf("%-24s %9.1fx %9.1fx\n", "Parallelism (ind/dep)", ratio_int, ratio_fp);
    fflush(stdout);

}
