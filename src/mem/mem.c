#include "bench.h"
#include "sysinfo.h"
#include "timer.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <unistd.h>

/* Platform bandwidth functions — implemented in mem_arm.c / mem_x86.c */
extern double measure_bandwidth_gbps(size_t size_bytes);
extern double measure_store_bandwidth_cached_gbps(size_t size_bytes);
extern double measure_store_bandwidth_streaming_gbps(size_t size_bytes);

static double measure_latency_ns_stride(size_t size_bytes, size_t stride) {
    size_t count = size_bytes / stride;
    if (count < 2) count = 2;

    void *raw;
    if (posix_memalign(&raw, 4096, count * stride) != 0)
        return -1;
    memset(raw, 0, count * stride);
    void **buf = (void **)raw;

    size_t *indices = malloc(count * sizeof(size_t));
    if (!indices) {
        free(raw);
        return -1;
    }
    for (size_t i = 0; i < count; i++) indices[i] = i;
    shuffle_indices(indices, count);

    for (size_t i = 0; i < count - 1; i++) {
        void **src = (void **)((char *)buf + indices[i] * stride);
        void **dst = (void **)((char *)buf + indices[i + 1] * stride);
        *src = (void *)dst;
    }
    void **last = (void **)((char *)buf + indices[count - 1] * stride);
    void **first = (void **)((char *)buf + indices[0] * stride);
    *last = (void *)first;
    free(indices);

    void **p = first;
    /* Warmup: run for at least 50ms to ensure CPU frequency ramp-up
       (Intel pstate needs sustained load to reach turbo) */
    {
        uint64_t tw0 = timer_ns();
        while (timer_ns() - tw0 < 50000000ULL) {
            for (size_t i = 0; i < 1000; i++) {
                p = (void **)*p;
                __asm__ volatile("" : "+r"(p));
            }
        }
    }

    /* Calibrate: run a short burst, then scale to ~200ms target */
    size_t iters = 1000;
    uint64_t tc0 = timer_ns();
    for (size_t i = 0; i < iters; i++) {
        p = (void **)*p;
        __asm__ volatile("" : "+r"(p));
    }
    uint64_t tc1 = timer_ns();
    double ns_per_iter = (double)(tc1 - tc0) / iters;
    if (ns_per_iter <= 0) {
        free(raw);
        return -1;
    }
    iters = (size_t)(200000000.0 / ns_per_iter);
    if (iters < 1000) iters = 1000;

    uint64_t t0 = timer_ns();
    for (size_t i = 0; i < iters; i++) {
        p = (void **)*p;
        __asm__ volatile("" : "+r"(p));
    }
    uint64_t t1 = timer_ns();

    free(raw);
    if (t1 <= t0) return -1;
    return (double)(t1 - t0) / iters;
}

static double measure_latency_ns(size_t size_bytes) {
    return measure_latency_ns_stride(size_bytes, 64);
}

/* --- Atomics --- */

static double measure_atomic_uncontended(void) {
    _Atomic uint64_t __attribute__((aligned(64))) counter = 0;

    size_t iters = 10000;
    uint64_t tc0 = timer_ns();
    for (size_t i = 0; i < iters; i++)
        atomic_fetch_add(&counter, 1);
    uint64_t tc1 = timer_ns();
    double ns_per = (double)(tc1 - tc0) / iters;
    if (ns_per <= 0) return -1;
    iters = (size_t)(200000000.0 / ns_per);
    if (iters < 10000) iters = 10000;

    uint64_t t0 = timer_ns();
    for (size_t i = 0; i < iters; i++)
        atomic_fetch_add(&counter, 1);
    uint64_t t1 = timer_ns();
    if (t1 <= t0) return -1;
    return (double)(t1 - t0) / iters;
}

struct atomic_arg {
    _Atomic uint64_t *counter;
    size_t iters;
};

static void *atomic_contended_worker(void *arg) {
    struct atomic_arg *a = (struct atomic_arg *)arg;
    for (size_t i = 0; i < a->iters; i++)
        atomic_fetch_add(a->counter, 1);
    return NULL;
}

static double measure_atomic_contended(int nthreads) {
    _Atomic uint64_t __attribute__((aligned(64))) counter = 0;
    size_t per_thread = 1000000;

    struct atomic_arg args[64];
    pthread_t threads[64];
    int nt = nthreads > 64 ? 64 : nthreads;

    uint64_t t0 = timer_ns();
    for (int t = 0; t < nt; t++) {
        args[t].counter = &counter;
        args[t].iters = per_thread;
        if (pthread_create(&threads[t], NULL, atomic_contended_worker, &args[t]) != 0) {
            for (int j = 0; j < t; j++)
                pthread_join(threads[j], NULL);
            return -1;
        }
    }
    for (int t = 0; t < nt; t++)
        pthread_join(threads[t], NULL);
    uint64_t t1 = timer_ns();

    if (t1 <= t0) return -1;
    return (double)(t1 - t0) / (per_thread * nt);
}

/* --- Malloc throughput --- */

static double measure_malloc_throughput(size_t alloc_size) {
    size_t iters = 100000;
    uint64_t tc0 = timer_ns();
    for (size_t i = 0; i < iters; i++) {
        void *p = malloc(alloc_size);
        if (!p) return -1;
        __asm__ volatile("" : "+r"(p));
        free(p);
    }
    uint64_t tc1 = timer_ns();
    double ns_per = (double)(tc1 - tc0) / iters;
    if (ns_per <= 0) return -1;
    iters = (size_t)(2000000000.0 / ns_per);
    if (iters < 100000) iters = 100000;

    uint64_t t0 = timer_ns();
    for (size_t i = 0; i < iters; i++) {
        void *p = malloc(alloc_size);
        if (!p) return -1;
        __asm__ volatile("" : "+r"(p));
        free(p);
    }
    uint64_t t1 = timer_ns();
    if (t1 <= t0) return -1;
    return (double)iters / ((double)(t1 - t0) / 1e9) / 1e6;
}

/* --- Entry point --- */

void bench_memory(void) {
    size_t sizes[] = {
        4*1024, 8*1024, 16*1024, 32*1024, 64*1024,
        128*1024, 256*1024, 512*1024,
        1*1024*1024, 2*1024*1024, 4*1024*1024,
        8*1024*1024, 16*1024*1024, 32*1024*1024,
        64*1024*1024, 128*1024*1024, 256*1024*1024, 512*1024*1024,
        1024UL*1024*1024
    };
    size_t nsizes = sizeof(sizes) / sizeof(sizes[0]);

    struct mem_info mi;
    query_mem_info(&mi);
    struct cpu_info ci;
    query_cpu_info(&ci);

    printf("=== Memory Latency & Bandwidth ===\n");
    if (mi.total_bytes > 0)
        printf("  Total: %zu GiB\n", (size_t)(mi.total_bytes / (1024UL * 1024 * 1024)));
    printf("  Page size: %zu bytes\n", mi.page_size);

    char cache[128] = "";
    int pos = 0;
    if (ci.l1d_bytes > 0)
        pos += snprintf(cache + pos, sizeof(cache) - pos,
                        "L1d: %s", fmt_size(ci.l1d_bytes));
    if (ci.l2_bytes > 0)
        pos += snprintf(cache + pos, sizeof(cache) - pos,
                        "  L2: %s", fmt_size(ci.l2_bytes));
    if (ci.l3_bytes > 0)
        pos += snprintf(cache + pos, sizeof(cache) - pos,
                        "  L3: %s", fmt_size(ci.l3_bytes));
    if (pos > 0)
        printf("  Cache: %s\n", cache);
    printf("%-12s %12s %12s\n", "Size", "Latency", "Bandwidth");
    printf("%-12s %12s %12s\n", "----", "-------", "---------");
    for (size_t i = 0; i < nsizes; i++) {
        double lat = measure_latency_ns(sizes[i]);
        double bw = measure_bandwidth_gbps(sizes[i]);
        if (lat > 0 && bw > 0)
            printf("%-12s %9.2f ns %8.2f GB/s\n", fmt_size(sizes[i]), lat, bw);
        else
            printf("%-12s %12s %12s\n", fmt_size(sizes[i]), "error", "error");
        fflush(stdout);
    }

    /* --- Store Bandwidth --- */
    printf("\n--- Store Bandwidth ---\n");
    printf("%-12s %12s %12s\n", "Size", "Cached", "Streaming");
    printf("%-12s %12s %12s\n", "----", "------", "---------");
    size_t store_sizes[] = {
        32*1024, 256*1024, 4*1024*1024, 64*1024*1024, 512*1024*1024
    };
    for (size_t i = 0; i < sizeof(store_sizes)/sizeof(store_sizes[0]); i++) {
        double cached = measure_store_bandwidth_cached_gbps(store_sizes[i]);
        double streaming = measure_store_bandwidth_streaming_gbps(store_sizes[i]);
        char cached_s[24], streaming_s[24];
        if (cached > 0) snprintf(cached_s, sizeof(cached_s), "%8.2f GB/s", cached);
        else            snprintf(cached_s, sizeof(cached_s), "%12s", "error");
        if (streaming > 0) snprintf(streaming_s, sizeof(streaming_s), "%8.2f GB/s", streaming);
        else               snprintf(streaming_s, sizeof(streaming_s), "%12s", "n/a");
        printf("%-12s %12s %12s\n", fmt_size(store_sizes[i]), cached_s, streaming_s);
        fflush(stdout);
    }

    /* --- Atomics Contention --- */
    printf("\n--- Atomics ---\n");
    double uncontended = measure_atomic_uncontended();
    if (uncontended > 0)
        printf("%-24s %8.2f ns/op\n", "Uncontended", uncontended);
    else
        printf("%-24s %12s\n", "Uncontended", "error");
    fflush(stdout);

    int contention_levels[] = {2, 4, 8};
    for (int c = 0; c < 3; c++) {
        double ct = measure_atomic_contended(contention_levels[c]);
        char label[32];
        snprintf(label, sizeof(label), "Contended (%d threads)", contention_levels[c]);
        if (ct > 0)
            printf("%-24s %8.2f ns/op\n", label, ct);
        else
            printf("%-24s %12s\n", label, "error");
        fflush(stdout);
    }

    /* --- Allocation Throughput --- */
    printf("\n--- Allocation ---\n");
    size_t alloc_sizes[] = {64, 256, 4096};
    for (size_t i = 0; i < 3; i++) {
        double mops = measure_malloc_throughput(alloc_sizes[i]);
        if (mops > 0)
            printf("malloc/free %-5zuB %10.2f Mops/s\n", alloc_sizes[i], mops);
        else
            printf("malloc/free %-5zuB %10s\n", alloc_sizes[i], "error");
        fflush(stdout);
    }
}
