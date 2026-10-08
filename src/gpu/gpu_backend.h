#pragma once

/* Interface between the backend-neutral driver (gpu.c) and the Metal
 * (gpu_metal.m) / Vulkan (gpu_vulkan.c) backends. The driver owns test sizes,
 * calibration, timing policy and result names, so both backends report the
 * same rows; a backend only builds resources and times command buffers.
 *
 * The kernel constants below are mirrored in the Metal source string and in
 * the shaders/ sources. Change them together. */

#include "core/status.h"
#include "core/types.h"

#define SB_GPU_TG            256   /* threads per threadgroup / workgroup (all grid kernels) */

/* FMA: float2 / half2 x 32 independent chains, loop body unrolled 4x.
 * Flops per thread per loop iteration = 32 * 4 * 2 lanes * 2. */
#define SB_GPU_FMA_CHAINS    32
#define SB_GPU_FMA_UNROLL    4
#define SB_GPU_FMA_OPS_ITER  (SB_GPU_FMA_CHAINS * SB_GPU_FMA_UNROLL * 2 * 2)

/* INT32: 16 chains of a = a * b + c, unrolled 2x; b and c are runtime values. */
#define SB_GPU_INT_CHAINS    16
#define SB_GPU_INT_UNROLL    2
#define SB_GPU_INT_OPS_ITER  (SB_GPU_INT_CHAINS * SB_GPU_INT_UNROLL * 2)

/* Threadgroup memory: float4 array of TG_N (+ TG_LOADS * TG_STEP padding)
 * elements; per iteration each thread does TG_LOADS lane-contiguous float4
 * loads whose SIMD footprint is aligned to TG_STEP elements. */
#define SB_GPU_TG_N          1024
#define SB_GPU_TG_LOADS      16
#define SB_GPU_TG_STEP       32
#define SB_GPU_TG_BYTES_ITER (SB_GPU_TG_LOADS * 16)

#define SB_GPU_LAT_STRIDE    128   /* bytes between pointer-chase nodes */
#define SB_GPU_TEX_CACHED    64    /* side of the cache-resident RGBA8 texture */
#define SB_GPU_TEX_ROWS      256   /* rows walked per thread in the streaming texture test */

typedef enum {
    SB_GPU_K_FP32 = 0,
    SB_GPU_K_FP16,
    SB_GPU_K_INT32,
    SB_GPU_K_BW_DEVICE,   /* coalesced read of a device-local buffer */
    SB_GPU_K_BW_HOST,     /* same kernel, host-visible system memory (discrete GPUs) */
    SB_GPU_K_LATENCY,     /* one thread chasing a random cycle */
    SB_GPU_K_TGMEM,
    SB_GPU_K_TEX_CACHED,  /* bilinear, small texture: filter rate */
    SB_GPU_K_TEX_STREAM,  /* bilinear, large texture read once: texture path from DRAM */
} sb_gpu_kernel_e;

/* Device facts the driver needs to size and label tests. */
typedef struct {
    char name[256];
    char api[64];
    bool is_cpu;          /* software implementation (llvmpipe): shrink all work */
    bool unified;         /* no host-visible memory distinct from device-local memory */
    bool fp16;            /* half arithmetic available */
    bool gpu_timer;       /* GPU timestamps; false = host wall time around submit+wait */
    u64  max_buffer;      /* largest storage buffer the backend can bind */
    u32  max_tex_dim;     /* largest 2D texture side */
    char notes[3][192];   /* extra info lines, printed after the device line ("" = none) */
} sb_gpu_caps;

typedef struct {
    sb_gpu_kernel_e kernel;
    u32 threads;          /* grid size, multiple of SB_GPU_TG (1 for LATENCY) */
    u64 bytes;            /* BW_*: buffer size; LATENCY: chain size */
    u32 tex_dim;          /* TEX_STREAM: texture side */
    const u32 *chain;     /* LATENCY: bytes / 4 words to upload */
} sb_gpu_test;

typedef struct sb_gpu_dev sb_gpu_dev;

/* Opens the device and fills caps. Reports device-selection info lines. */
sb_status_e sb_gpu_open(sb_gpu_dev **out, sb_gpu_caps *caps);
void        sb_gpu_close(sb_gpu_dev *d);

/* Builds the pipeline and resources for one test (replacing any previous one). */
sb_status_e sb_gpu_prepare(sb_gpu_dev *d, const sb_gpu_test *t);

/* Encodes `ndispatch` back-to-back dispatches of the prepared test with
 * runtime parameter `n` (loop iterations / chase steps) in ONE command
 * buffer, waits, and returns its GPU execution time in ns. */
sb_status_e sb_gpu_run(sb_gpu_dev *d, u32 n, u32 ndispatch, f64 *out_ns);

/* Frees the prepared test's resources. */
void        sb_gpu_release(sb_gpu_dev *d);
