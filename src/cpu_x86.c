#include "cpu_bench.h"

/*
 * Compile-time SIMD tier selection via -march=native.
 * Exactly one of the tier files (cpu_x86_sse2/avx2/avx512.c) will be active,
 * determined by #ifdef guards. We reference the correct one here.
 */

#if defined(__AVX512F__)

extern const struct cpu_bench cpu_benches_avx512[];
extern const int cpu_bench_count_avx512;
const struct cpu_bench *cpu_benches = cpu_benches_avx512;
int cpu_bench_count;
const char *cpu_simd_name = "AVX-512";

__attribute__((constructor))
static void cpu_x86_init(void) {
    cpu_bench_count = cpu_bench_count_avx512;
}

#elif defined(__AVX2__)

extern const struct cpu_bench cpu_benches_avx2[];
extern const int cpu_bench_count_avx2;
const struct cpu_bench *cpu_benches = cpu_benches_avx2;
int cpu_bench_count;
const char *cpu_simd_name = "AVX2";

__attribute__((constructor))
static void cpu_x86_init(void) {
    cpu_bench_count = cpu_bench_count_avx2;
}

#else

extern const struct cpu_bench cpu_benches_sse2[];
extern const int cpu_bench_count_sse2;
const struct cpu_bench *cpu_benches = cpu_benches_sse2;
int cpu_bench_count;
const char *cpu_simd_name = "SSE2";

__attribute__((constructor))
static void cpu_x86_init(void) {
    cpu_bench_count = cpu_bench_count_sse2;
}

#endif
