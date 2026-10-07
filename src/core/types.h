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

#define SB_ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))
