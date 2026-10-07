#include "bench.h"
#include "timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef HAS_BLAS

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

static double bench_sgemm(int N) {
    float *A = malloc((size_t)N * N * sizeof(float));
    float *B = malloc((size_t)N * N * sizeof(float));
    float *C = calloc((size_t)N * N, sizeof(float));
    if (!A || !B || !C) { free(A); free(B); free(C); return -1; }

    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / (float)RAND_MAX;
        B[i] = (float)rand() / (float)RAND_MAX;
    }

    /* Warmup */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);

    /* Calibrate to ~2s */
    uint64_t tc0 = timer_ns();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
    uint64_t tc1 = timer_ns();
    double ns_per = (double)(tc1 - tc0);
    if (ns_per <= 0) { free(A); free(B); free(C); return -1; }
    size_t passes = (size_t)(2000000000.0 / ns_per);
    if (passes < 2) passes = 2;

    uint64_t t0 = timer_ns();
    for (size_t p = 0; p < passes; p++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
    uint64_t t1 = timer_ns();

    double elapsed_s = (double)(t1 - t0) / 1e9;
    double gflops = 2.0 * (double)N * N * N * passes / elapsed_s / 1e9;

    free(A);
    free(B);
    free(C);
    return gflops;
}

void bench_matrix(void) {
    printf("=== Matrix Multiply ===\n");
#ifdef __APPLE__
    printf("  CPU library: Accelerate (library-selected kernels)\n");
#else
    printf("  CPU library: OpenBLAS\n");
#endif

    int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
    int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));

    printf("\n  CPU SGEMM\n");
    printf("  %-14s %10s\n", "Size", "GFLOPS");
    printf("  %-14s %10s\n", "----", "------");

    for (int i = 0; i < nsizes; i++) {
        int N = sizes[i];
        double gflops = bench_sgemm(N);
        char label[32];
        snprintf(label, sizeof(label), "%dx%d", N, N);
        if (gflops > 0) printf("  %-14s %10.2f\n", label, gflops);
        else printf("  %-14s %10s\n", label, "error");
        fflush(stdout);
    }
}

#else

void bench_matrix(void) {
    printf("=== Matrix Multiply ===\n");
    printf("  No BLAS library found, skipping\n");
}

#endif
