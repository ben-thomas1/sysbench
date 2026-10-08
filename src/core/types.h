#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef int8_t   i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float    f32;
typedef double   f64;

/* Allocation hooks; override before including any sb header to reroute. */
#ifndef SB_MALLOC
#define SB_MALLOC(n)     malloc(n)
#endif
#ifndef SB_REALLOC
#define SB_REALLOC(p, n) realloc(p, n)
#endif
#ifndef SB_FREE
#define SB_FREE(p)       free(p)
#endif

/* Aligned allocation (direct I/O buffers, cache-line-sized objects). `align`
 * must be a power of two and a multiple of sizeof(void *). Release with
 * SB_ALIGNED_FREE, never SB_FREE, so the pair can be overridden together. */
#ifndef SB_ALIGNED_ALLOC
static inline void *sb_aligned_alloc_default_(size_t align, size_t n) {
    void *p = NULL;
    return posix_memalign(&p, align, n) == 0 ? p : NULL;
}
#define SB_ALIGNED_ALLOC(align, n) sb_aligned_alloc_default_(align, n)
#endif
#ifndef SB_ALIGNED_FREE
#define SB_ALIGNED_FREE(p) free(p)
#endif

#define SB_ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))
