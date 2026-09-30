#include "timer.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <arm_neon.h>

/* --- NEON read bandwidth: 8 x 128-bit accumulators --- */

#define BW_READ_LOOP(buf, count, v0,v1,v2,v3,v4,v5,v6,v7) \
    do {                                                \
    for (size_t i = 0; i < count; i += 16) {               \
        v0 = vaddq_u64(v0, vld1q_u64(&buf[i]));            \
        v1 = vaddq_u64(v1, vld1q_u64(&buf[i+2]));          \
        v2 = vaddq_u64(v2, vld1q_u64(&buf[i+4]));          \
        v3 = vaddq_u64(v3, vld1q_u64(&buf[i+6]));          \
        v4 = vaddq_u64(v4, vld1q_u64(&buf[i+8]));          \
        v5 = vaddq_u64(v5, vld1q_u64(&buf[i+10]));         \
        v6 = vaddq_u64(v6, vld1q_u64(&buf[i+12]));         \
        v7 = vaddq_u64(v7, vld1q_u64(&buf[i+14]));         \
    }                                                       \
    __asm__ volatile("" : "+w"(v0),"+w"(v1),"+w"(v2),"+w"(v3), \
                          "+w"(v4),"+w"(v5),"+w"(v6),"+w"(v7) :: "memory"); \
    } while (0)

double measure_bandwidth_gbps(size_t size_bytes) {
    size_t alloc = size_bytes;
    if (alloc < 512) alloc = 512;
    alloc = alloc & ~127ULL;  /* round down to 128B loop stride */

    void *raw;
    if (posix_memalign(&raw, 4096, alloc) != 0)
        return -1;
    memset(raw, 0xAB, alloc);

    const uint64_t *buf = (const uint64_t *)raw;
    size_t count = alloc / sizeof(uint64_t);
    uint64x2_t v0 = vdupq_n_u64(0), v1 = vdupq_n_u64(0);
    uint64x2_t v2 = vdupq_n_u64(0), v3 = vdupq_n_u64(0);
    uint64x2_t v4 = vdupq_n_u64(0), v5 = vdupq_n_u64(0);
    uint64x2_t v6 = vdupq_n_u64(0), v7 = vdupq_n_u64(0);

    /* Warmup */
    BW_READ_LOOP(buf, count, v0,v1,v2,v3,v4,v5,v6,v7);

    /* Calibrate */
    size_t passes = 1;
    {
        uint64_t t0 = timer_ns();
        BW_READ_LOOP(buf, count, v0,v1,v2,v3,v4,v5,v6,v7);
        uint64_t elapsed = timer_ns() - t0;
        if (elapsed > 0) passes = (size_t)(200000000ULL / elapsed) + 1;
        if (passes < 2) passes = 2;
    }

    uint64_t t0 = timer_ns();
    for (size_t p = 0; p < passes; p++)
        BW_READ_LOOP(buf, count, v0,v1,v2,v3,v4,v5,v6,v7);
    uint64_t t1 = timer_ns();

    double elapsed_s = (double)(t1 - t0) / 1e9;
    free(raw);
    if (elapsed_s <= 0) return -1;
    return (double)alloc * passes / elapsed_s / 1e9;
}

#undef BW_READ_LOOP

/* --- NEON cached store bandwidth: 8 x 128-bit stores --- */

#define BW_STORE_CACHED_LOOP(buf, count, sv0,sv1,sv2,sv3,sv4,sv5,sv6,sv7) \
    do {                                                               \
    for (size_t i = 0; i < count; i += 16) {                              \
        vst1q_u64(&buf[i],    sv0); vst1q_u64(&buf[i+2],  sv1);           \
        vst1q_u64(&buf[i+4],  sv2); vst1q_u64(&buf[i+6],  sv3);           \
        vst1q_u64(&buf[i+8],  sv4); vst1q_u64(&buf[i+10], sv5);           \
        vst1q_u64(&buf[i+12], sv6); vst1q_u64(&buf[i+14], sv7);           \
    }                                                                      \
    __asm__ volatile("dsb st" ::: "memory"); \
    } while (0)

double measure_store_bandwidth_cached_gbps(size_t size_bytes) {
    size_t alloc = size_bytes;
    if (alloc < 512) alloc = 512;
    alloc = alloc & ~127ULL;

    void *raw;
    if (posix_memalign(&raw, 4096, alloc) != 0)
        return -1;
    memset(raw, 0, alloc);

    uint64_t *buf = (uint64_t *)raw;
    size_t count = alloc / sizeof(uint64_t);
    uint64x2_t sv0 = vdupq_n_u64(1), sv1 = vdupq_n_u64(2);
    uint64x2_t sv2 = vdupq_n_u64(3), sv3 = vdupq_n_u64(4);
    uint64x2_t sv4 = vdupq_n_u64(5), sv5 = vdupq_n_u64(6);
    uint64x2_t sv6 = vdupq_n_u64(7), sv7 = vdupq_n_u64(8);

    BW_STORE_CACHED_LOOP(buf, count, sv0,sv1,sv2,sv3,sv4,sv5,sv6,sv7);

    size_t passes = 1;
    {
        uint64_t t0 = timer_ns();
        BW_STORE_CACHED_LOOP(buf, count, sv0,sv1,sv2,sv3,sv4,sv5,sv6,sv7);
        uint64_t elapsed = timer_ns() - t0;
        if (elapsed > 0) passes = (size_t)(200000000ULL / elapsed) + 1;
        if (passes < 2) passes = 2;
    }

    uint64_t t0 = timer_ns();
    for (size_t p = 0; p < passes; p++)
        BW_STORE_CACHED_LOOP(buf, count, sv0,sv1,sv2,sv3,sv4,sv5,sv6,sv7);
    uint64_t t1 = timer_ns();

    double elapsed_s = (double)(t1 - t0) / 1e9;
    free(raw);
    if (elapsed_s <= 0) return -1;
    return (double)alloc * passes / elapsed_s / 1e9;
}

#undef BW_STORE_CACHED_LOOP

/* --- NEON streaming store bandwidth: 8 x 128-bit non-temporal stores ---
 *
 * Uses STNP as a streaming-store hint and a DSB ST before timing stops.
 */

#define BW_STORE_STREAM_LOOP(buf, count, sv0,sv1,sv2,sv3,sv4,sv5,sv6,sv7) \
    do {                                                               \
    for (size_t i = 0; i < count; i += 16) {                       \
        __asm__ volatile(                                            \
            "stnp %q[s0], %q[s1], [%[b]]\n\t"                      \
            "stnp %q[s2], %q[s3], [%[b], #32]\n\t"                 \
            "stnp %q[s4], %q[s5], [%[b], #64]\n\t"                 \
            "stnp %q[s6], %q[s7], [%[b], #96]\n\t"                 \
            :                                                        \
            : [b]"r"(&buf[i]),                                      \
              [s0]"w"(sv0), [s1]"w"(sv1),                           \
              [s2]"w"(sv2), [s3]"w"(sv3),                           \
              [s4]"w"(sv4), [s5]"w"(sv5),                           \
              [s6]"w"(sv6), [s7]"w"(sv7)                            \
            : "memory"                                               \
        );                                                           \
    }                                                                \
    __asm__ volatile("dsb st" ::: "memory"); \
    } while (0)

double measure_store_bandwidth_streaming_gbps(size_t size_bytes) {
    size_t alloc = size_bytes;
    if (alloc < 512) alloc = 512;
    alloc = alloc & ~127ULL;

    void *raw;
    if (posix_memalign(&raw, 4096, alloc) != 0)
        return -1;
    memset(raw, 0, alloc);

    uint64_t *buf = (uint64_t *)raw;
    size_t count = alloc / sizeof(uint64_t);
    uint64x2_t sv0 = vdupq_n_u64(1), sv1 = vdupq_n_u64(2);
    uint64x2_t sv2 = vdupq_n_u64(3), sv3 = vdupq_n_u64(4);
    uint64x2_t sv4 = vdupq_n_u64(5), sv5 = vdupq_n_u64(6);
    uint64x2_t sv6 = vdupq_n_u64(7), sv7 = vdupq_n_u64(8);

    /* Warmup */
    BW_STORE_STREAM_LOOP(buf, count, sv0,sv1,sv2,sv3,sv4,sv5,sv6,sv7);

    /* Calibrate */
    size_t passes = 1;
    {
        uint64_t t0 = timer_ns();
        BW_STORE_STREAM_LOOP(buf, count, sv0,sv1,sv2,sv3,sv4,sv5,sv6,sv7);
        uint64_t elapsed = timer_ns() - t0;
        if (elapsed > 0) passes = (size_t)(200000000ULL / elapsed) + 1;
        if (passes < 2) passes = 2;
    }

    uint64_t t0 = timer_ns();
    for (size_t p = 0; p < passes; p++)
        BW_STORE_STREAM_LOOP(buf, count, sv0,sv1,sv2,sv3,sv4,sv5,sv6,sv7);
    uint64_t t1 = timer_ns();

    double elapsed_s = (double)(t1 - t0) / 1e9;
    free(raw);
    if (elapsed_s <= 0) return -1;
    return (double)alloc * passes / elapsed_s / 1e9;
}

#undef BW_STORE_STREAM_LOOP
