/* x86-64 bandwidth kernels. The SIMD tier is fixed at compile time by -march
 * (AVX-512 > AVX2 > SSE2), like the cpu module. */
#include "mem/mem_internal.h"

#include <immintrin.h>

#if defined(__AVX512F__)
typedef __m512i vec;
#define VW            64u
#define VZERO()       _mm512_setzero_si512()
#define VSET(x)       _mm512_set1_epi64(x)
#define VLOAD(p)      _mm512_load_si512((const void *)(p))
#define VADD(a, b)    _mm512_add_epi64(a, b)
#define VSTORE(p, v)  _mm512_store_si512((void *)(p), v)
#define VSTREAM(p, v) _mm512_stream_si512((void *)(p), v)
const char *const mem_kern_isa = "AVX-512";
#elif defined(__AVX2__)
typedef __m256i vec;
#define VW            32u
#define VZERO()       _mm256_setzero_si256()
#define VSET(x)       _mm256_set1_epi64x(x)
#define VLOAD(p)      _mm256_load_si256((const __m256i *)(const void *)(p))
#define VADD(a, b)    _mm256_add_epi64(a, b)
#define VSTORE(p, v)  _mm256_store_si256((__m256i *)(void *)(p), v)
#define VSTREAM(p, v) _mm256_stream_si256((__m256i *)(void *)(p), v)
const char *const mem_kern_isa = "AVX2";
#else
typedef __m128i vec;
#define VW            16u
#define VZERO()       _mm_setzero_si128()
#define VSET(x)       _mm_set1_epi64x(x)
#define VLOAD(p)      _mm_load_si128((const __m128i *)(const void *)(p))
#define VADD(a, b)    _mm_add_epi64(a, b)
#define VSTORE(p, v)  _mm_store_si128((__m128i *)(void *)(p), v)
#define VSTREAM(p, v) _mm_stream_si128((__m128i *)(void *)(p), v)
const char *const mem_kern_isa = "SSE2";
#endif

static_assert(MEM_KERN_GRAIN % (8 * VW) == 0, "kernel step must divide the grain");

/* The empty asm with a "+x" operand makes the stored value opaque each
 * iteration, so the compiler can't turn the loop into memset. */
#define STORE_PATTERN 0x0123456789abcdefLL

/* 8 vector loads per iteration into 8 independent add chains. */
void mem_kern_read(const void *buf, size_t bytes, u64 passes) {
    const u8 *b = buf;
    vec a0 = VZERO(), a1 = VZERO(), a2 = VZERO(), a3 = VZERO();
    vec a4 = VZERO(), a5 = VZERO(), a6 = VZERO(), a7 = VZERO();
    for (u64 p = 0; p < passes; p++) {
        for (size_t i = 0; i < bytes; i += 8 * VW) {
            a0 = VADD(a0, VLOAD(b + i));
            a1 = VADD(a1, VLOAD(b + i + VW));
            a2 = VADD(a2, VLOAD(b + i + 2 * VW));
            a3 = VADD(a3, VLOAD(b + i + 3 * VW));
            a4 = VADD(a4, VLOAD(b + i + 4 * VW));
            a5 = VADD(a5, VLOAD(b + i + 5 * VW));
            a6 = VADD(a6, VLOAD(b + i + 6 * VW));
            a7 = VADD(a7, VLOAD(b + i + 7 * VW));
        }
        __asm__ volatile("" : "+x"(a0), "+x"(a1), "+x"(a2), "+x"(a3),
                              "+x"(a4), "+x"(a5), "+x"(a6), "+x"(a7) :: "memory");
    }
}

void mem_kern_store128(void *buf, size_t bytes, u64 passes) {
    u8 *b = buf;
    vec v = VSET(STORE_PATTERN);
    for (u64 p = 0; p < passes; p++) {
        for (size_t i = 0; i < bytes; i += 128) {
            __asm__ volatile("" : "+x"(v));
            for (u32 k = 0; k < 128; k += VW) { VSTORE(b + i + k, v); }
        }
        __asm__ volatile("" ::: "memory");
    }
}

void mem_kern_store256(void *buf, size_t bytes, u64 passes) {
    u8 *b = buf;
    vec v = VSET(STORE_PATTERN);
    for (u64 p = 0; p < passes; p++) {
        for (size_t i = 0; i < bytes; i += 256) {
            __asm__ volatile("" : "+x"(v));
            for (u32 k = 0; k < 256; k += VW) { VSTORE(b + i + k, v); }
        }
        __asm__ volatile("" ::: "memory");
    }
}

/* movntdq / vmovntdq: write-combining stores that bypass the caches. */
void mem_kern_store_nt(void *buf, size_t bytes, u64 passes) {
    u8 *b = buf;
    vec v = VSET(STORE_PATTERN);
    for (u64 p = 0; p < passes; p++) {
        for (size_t i = 0; i < bytes; i += 256) {
            __asm__ volatile("" : "+x"(v));
            for (u32 k = 0; k < 256; k += VW) { VSTREAM(b + i + k, v); }
        }
        __asm__ volatile("" ::: "memory");
    }
}

/* mfence drains the store buffer and the write-combining buffers. */
void mem_kern_drain(void) { _mm_mfence(); }
