#include "timer.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h>

#if defined(__AVX2__)

/* --- AVX2 read bandwidth: 8 x 256-bit accumulators --- */

#define BW_READ_LOOP(buf, count, v0,v1,v2,v3,v4,v5,v6,v7) \
    do {                                                \
    for (size_t i = 0; i < count; i += 8) {                \
        v0 = _mm256_add_epi64(v0, _mm256_load_si256(&buf[i]));   \
        v1 = _mm256_add_epi64(v1, _mm256_load_si256(&buf[i+1])); \
        v2 = _mm256_add_epi64(v2, _mm256_load_si256(&buf[i+2])); \
        v3 = _mm256_add_epi64(v3, _mm256_load_si256(&buf[i+3])); \
        v4 = _mm256_add_epi64(v4, _mm256_load_si256(&buf[i+4])); \
        v5 = _mm256_add_epi64(v5, _mm256_load_si256(&buf[i+5])); \
        v6 = _mm256_add_epi64(v6, _mm256_load_si256(&buf[i+6])); \
        v7 = _mm256_add_epi64(v7, _mm256_load_si256(&buf[i+7])); \
    }                                                       \
    __asm__ volatile("" : "+x"(v0),"+x"(v1),"+x"(v2),"+x"(v3), \
                          "+x"(v4),"+x"(v5),"+x"(v6),"+x"(v7) :: "memory"); \
    } while (0)

double measure_bandwidth_gbps(size_t size_bytes) {
    size_t alloc = size_bytes;
    if (alloc < 512) alloc = 512;
    alloc = alloc & ~255ULL;  /* round down to 256B loop stride */

    void *raw;
    if (posix_memalign(&raw, 4096, alloc) != 0)
        return -1;
    memset(raw, 0xAB, alloc);

    const __m256i *buf = (const __m256i *)raw;
    size_t count = alloc / sizeof(__m256i);
    __m256i v0 = _mm256_setzero_si256(), v1 = _mm256_setzero_si256();
    __m256i v2 = _mm256_setzero_si256(), v3 = _mm256_setzero_si256();
    __m256i v4 = _mm256_setzero_si256(), v5 = _mm256_setzero_si256();
    __m256i v6 = _mm256_setzero_si256(), v7 = _mm256_setzero_si256();

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

/* --- AVX2 store bandwidth: 8 x 256-bit stores --- */

#define BW_STORE_LOOP(buf, count, sv0,sv1,sv2,sv3,sv4,sv5,sv6,sv7) \
    do {                                                        \
    for (size_t i = 0; i < count; i += 8) {                        \
        _mm256_store_si256(&buf[i],   sv0); _mm256_store_si256(&buf[i+1], sv1); \
        _mm256_store_si256(&buf[i+2], sv2); _mm256_store_si256(&buf[i+3], sv3); \
        _mm256_store_si256(&buf[i+4], sv4); _mm256_store_si256(&buf[i+5], sv5); \
        _mm256_store_si256(&buf[i+6], sv6); _mm256_store_si256(&buf[i+7], sv7); \
    }                                                               \
    __asm__ volatile("" ::: "memory"); \
    _mm_sfence(); \
    } while (0)

double measure_store_bandwidth_cached_gbps(size_t size_bytes) {
    size_t alloc = size_bytes;
    if (alloc < 512) alloc = 512;
    alloc = alloc & ~255ULL;

    void *raw;
    if (posix_memalign(&raw, 4096, alloc) != 0)
        return -1;
    memset(raw, 0, alloc);

    __m256i *buf = (__m256i *)raw;
    size_t count = alloc / sizeof(__m256i);
    __m256i sv0 = _mm256_set1_epi64x(1), sv1 = _mm256_set1_epi64x(2);
    __m256i sv2 = _mm256_set1_epi64x(3), sv3 = _mm256_set1_epi64x(4);
    __m256i sv4 = _mm256_set1_epi64x(5), sv5 = _mm256_set1_epi64x(6);
    __m256i sv6 = _mm256_set1_epi64x(7), sv7 = _mm256_set1_epi64x(8);

    /* Warmup */
    BW_STORE_LOOP(buf, count, sv0,sv1,sv2,sv3,sv4,sv5,sv6,sv7);

    /* Calibrate */
    size_t passes = 1;
    {
        uint64_t t0 = timer_ns();
        BW_STORE_LOOP(buf, count, sv0,sv1,sv2,sv3,sv4,sv5,sv6,sv7);
        uint64_t elapsed = timer_ns() - t0;
        if (elapsed > 0) passes = (size_t)(200000000ULL / elapsed) + 1;
        if (passes < 2) passes = 2;
    }

    uint64_t t0 = timer_ns();
    for (size_t p = 0; p < passes; p++)
        BW_STORE_LOOP(buf, count, sv0,sv1,sv2,sv3,sv4,sv5,sv6,sv7);
    uint64_t t1 = timer_ns();

    double elapsed_s = (double)(t1 - t0) / 1e9;
    free(raw);
    if (elapsed_s <= 0) return -1;
    return (double)alloc * passes / elapsed_s / 1e9;
}

#undef BW_STORE_LOOP

/* --- AVX2 streaming store bandwidth: 8 x 256-bit non-temporal stores --- */

#define BW_STORE_STREAM_LOOP(buf, count, sv0,sv1,sv2,sv3,sv4,sv5,sv6,sv7) \
    do {                                                               \
    for (size_t i = 0; i < count; i += 8) {                               \
        _mm256_stream_si256(&buf[i],   sv0); _mm256_stream_si256(&buf[i+1], sv1); \
        _mm256_stream_si256(&buf[i+2], sv2); _mm256_stream_si256(&buf[i+3], sv3); \
        _mm256_stream_si256(&buf[i+4], sv4); _mm256_stream_si256(&buf[i+5], sv5); \
        _mm256_stream_si256(&buf[i+6], sv6); _mm256_stream_si256(&buf[i+7], sv7); \
    }                                                                      \
    _mm_sfence(); \
    } while (0)

double measure_store_bandwidth_streaming_gbps(size_t size_bytes) {
    size_t alloc = size_bytes;
    if (alloc < 512) alloc = 512;
    alloc = alloc & ~255ULL;

    void *raw;
    if (posix_memalign(&raw, 4096, alloc) != 0)
        return -1;
    memset(raw, 0, alloc);

    __m256i *buf = (__m256i *)raw;
    size_t count = alloc / sizeof(__m256i);
    __m256i sv0 = _mm256_set1_epi64x(1), sv1 = _mm256_set1_epi64x(2);
    __m256i sv2 = _mm256_set1_epi64x(3), sv3 = _mm256_set1_epi64x(4);
    __m256i sv4 = _mm256_set1_epi64x(5), sv5 = _mm256_set1_epi64x(6);
    __m256i sv6 = _mm256_set1_epi64x(7), sv7 = _mm256_set1_epi64x(8);

    BW_STORE_STREAM_LOOP(buf, count, sv0,sv1,sv2,sv3,sv4,sv5,sv6,sv7);

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

#else /* SSE2 / no AVX2 — scalar fallback */

#define BW_READ_LOOP(buf, count, v0,v1,v2,v3,v4,v5,v6,v7) \
    do {                                                \
    for (size_t i = 0; i < count; i += 8) {                \
        v0 += buf[i];   v1 += buf[i+1]; v2 += buf[i+2]; v3 += buf[i+3]; \
        v4 += buf[i+4]; v5 += buf[i+5]; v6 += buf[i+6]; v7 += buf[i+7]; \
    }                                                       \
    __asm__ volatile("" : "+r"(v0),"+r"(v1),"+r"(v2),"+r"(v3), \
                          "+r"(v4),"+r"(v5),"+r"(v6),"+r"(v7) :: "memory"); \
    } while (0)

double measure_bandwidth_gbps(size_t size_bytes) {
    size_t alloc = size_bytes;
    if (alloc < 512) alloc = 512;
    alloc = alloc & ~63ULL;

    void *raw;
    if (posix_memalign(&raw, 4096, alloc) != 0)
        return -1;
    memset(raw, 0xAB, alloc);

    const uint64_t *buf = (const uint64_t *)raw;
    size_t count = alloc / sizeof(uint64_t);
    uint64_t v0=0, v1=0, v2=0, v3=0, v4=0, v5=0, v6=0, v7=0;

    BW_READ_LOOP(buf, count, v0,v1,v2,v3,v4,v5,v6,v7);

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

#define BW_STORE_LOOP(buf, count) \
    do { \
    for (size_t i = 0; i < count; i += 8) { \
        buf[i]=1; buf[i+1]=2; buf[i+2]=3; buf[i+3]=4; \
        buf[i+4]=5; buf[i+5]=6; buf[i+6]=7; buf[i+7]=8; \
    } \
    __asm__ volatile("" ::: "memory"); \
    _mm_sfence(); \
    } while (0)

double measure_store_bandwidth_cached_gbps(size_t size_bytes) {
    size_t alloc = size_bytes;
    if (alloc < 512) alloc = 512;
    alloc = alloc & ~63ULL;

    void *raw;
    if (posix_memalign(&raw, 4096, alloc) != 0)
        return -1;
    memset(raw, 0, alloc);

    uint64_t *buf = (uint64_t *)raw;
    size_t count = alloc / sizeof(uint64_t);

    BW_STORE_LOOP(buf, count);

    size_t passes = 1;
    {
        uint64_t t0 = timer_ns();
        BW_STORE_LOOP(buf, count);
        uint64_t elapsed = timer_ns() - t0;
        if (elapsed > 0) passes = (size_t)(200000000ULL / elapsed) + 1;
        if (passes < 2) passes = 2;
    }

    uint64_t t0 = timer_ns();
    for (size_t p = 0; p < passes; p++)
        BW_STORE_LOOP(buf, count);
    uint64_t t1 = timer_ns();

    double elapsed_s = (double)(t1 - t0) / 1e9;
    free(raw);
    if (elapsed_s <= 0) return -1;
    return (double)alloc * passes / elapsed_s / 1e9;
}

#undef BW_STORE_LOOP

double measure_store_bandwidth_streaming_gbps(size_t size_bytes) {
    (void)size_bytes;
    return -1;
}

#endif
