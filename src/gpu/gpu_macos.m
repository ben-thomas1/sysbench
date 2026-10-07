#include "bench.h"

#ifdef __APPLE__

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "timer.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BW_TOTAL_BYTES (256ULL * 1024 * 1024)
#define LAT_BUF_SIZE   (64ULL * 1024 * 1024)
#define LAT_STRIDE     16  /* elements (64 bytes) per node */

static int complete_command(id<MTLCommandBuffer> cmd) {
    if (!cmd) return 0;
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "  Metal command failed: %s\n",
                [[cmd.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

/* 10 independent accumulator chains per thread, each with different initial
   values (from buf[tid]) to prevent the Metal compiler from merging chains.
   Each chain: 1 FMA/iter = 2 FLOPS. Total: 10 × 4096 × 2 = 81920 FLOPS/thread. */
static NSString *const fma_shader = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void fma_bench(device float *buf [[buffer(0)]],\n"
    "                      uint tid [[thread_position_in_grid]]) {\n"
    "    float s = buf[tid];\n"
    "    float a0=s, a1=s+0.1f, a2=s+0.2f, a3=s+0.3f, a4=s+0.4f,\n"
    "          a5=s+0.5f, a6=s+0.6f, a7=s+0.7f, a8=s+0.8f, a9=s+0.9f;\n"
    "    float b = 0.9999f, c = 0.0001f;\n"
    "    for (int i = 0; i < 4096; i++) {\n"
    "        a0 = a0*b+c; a1 = a1*b+c; a2 = a2*b+c;\n"
    "        a3 = a3*b+c; a4 = a4*b+c; a5 = a5*b+c;\n"
    "        a6 = a6*b+c; a7 = a7*b+c; a8 = a8*b+c;\n"
    "        a9 = a9*b+c;\n"
    "    }\n"
    "    buf[tid] = (a0+a1+a2+a3+a4+a5+a6+a7+a8+a9) * 0.1f;\n"
    "}\n";

static NSString *const int_shader = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void int_bench(device uint *buf [[buffer(0)]],\n"
    "                      uint tid [[thread_position_in_grid]]) {\n"
    "    uint s = buf[tid];\n"
    "    uint a0=s, a1=s+1u, a2=s+2u, a3=s+3u, a4=s+4u,\n"
    "         a5=s+5u, a6=s+6u, a7=s+7u, a8=s+8u, a9=s+9u;\n"
    "    uint b = 0x4F6CDD1Du, c = 0x7F4A7C15u;\n"
    "    for (int i = 0; i < 4096; i++) {\n"
    "        a0 = a0*b+c; a1 = a1*b+c; a2 = a2*b+c;\n"
    "        a3 = a3*b+c; a4 = a4*b+c; a5 = a5*b+c;\n"
    "        a6 = a6*b+c; a7 = a7*b+c; a8 = a8*b+c;\n"
    "        a9 = a9*b+c;\n"
    "    }\n"
    "    buf[tid] = a0+a1+a2+a3+a4+a5+a6+a7+a8+a9;\n"
    "}\n";

/* GPU memory bandwidth: read a large buffer, sum it */
static NSString *const bw_shader = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void bw_bench(device const float4 *src [[buffer(0)]],\n"
    "                     device float *dst [[buffer(1)]],\n"
    "                     uint tid [[thread_position_in_grid]]) {\n"
    "    float4 sum = float4(0.0f);\n"
    "    uint base = tid * 256;\n"
    "    for (uint i = 0; i < 256; i++) {\n"
    "        sum += src[base + i];\n"
    "    }\n"
    "    dst[tid] = sum.x + sum.y + sum.z + sum.w;\n"
    "}\n";

/* Pointer-chase latency shader: single thread, dependent loads */
static NSString *const latency_shader = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void latency_bench(device const uint *data [[buffer(0)]],\n"
    "                          device uint *result [[buffer(1)]],\n"
    "                          constant uint &num_chases [[buffer(2)]]) {\n"
    "    uint p = 0;\n"
    "    for (uint i = 0; i < num_chases; i++) {\n"
    "        p = data[p];\n"
    "    }\n"
    "    result[0] = p;\n"
    "}\n";

/* FP16 (half precision) compute shader */
static NSString *const fp16_shader = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void fp16_bench(device half *buf [[buffer(0)]],\n"
    "                       uint tid [[thread_position_in_grid]]) {\n"
    "    half s = buf[tid];\n"
    "    half a0=s, a1=s+0.1h, a2=s+0.2h, a3=s+0.3h, a4=s+0.4h,\n"
    "         a5=s+0.5h, a6=s+0.6h, a7=s+0.7h, a8=s+0.8h, a9=s+0.9h;\n"
    "    half b = 0.999h, c = 0.001h;\n"
    "    for (int i = 0; i < 4096; i++) {\n"
    "        a0 = a0*b+c; a1 = a1*b+c; a2 = a2*b+c;\n"
    "        a3 = a3*b+c; a4 = a4*b+c; a5 = a5*b+c;\n"
    "        a6 = a6*b+c; a7 = a7*b+c; a8 = a8*b+c;\n"
    "        a9 = a9*b+c;\n"
    "    }\n"
    "    buf[tid] = (a0+a1+a2+a3+a4+a5+a6+a7+a8+a9) * 0.1f;\n"
    "}\n";

/* Shared memory (threadgroup) bandwidth shader */
static NSString *const shared_mem_shader = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void shared_mem_bench(device float *buf [[buffer(0)]],\n"
    "                             uint tid [[thread_position_in_grid]],\n"
    "                             uint lid [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float smem[256];\n"
    "    smem[lid] = buf[tid];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    float sum = 0.0f;\n"
    "    for (int i = 0; i < 256; i++) {\n"
    "        sum += smem[(lid + i) & 255];\n"
    "    }\n"
    "    buf[tid] = sum / 256.0f;\n"
    "}\n";

/* Texture sampling shader */
static NSString *const tex_shader = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void tex_bench(texture2d<float, access::sample> tex [[texture(0)]],\n"
    "                      device float *out [[buffer(0)]],\n"
    "                      uint tid [[thread_position_in_grid]]) {\n"
    "    constexpr sampler smp(coord::normalized, filter::linear,\n"
    "                          address::repeat);\n"
    "    float sum = 0.0f;\n"
    "    float u = float(tid) * 0.0001f;\n"
    "    for (int i = 0; i < 4096; i++) {\n"
    "        float4 c = tex.sample(smp, float2(u, float(i) * 0.0001f));\n"
    "        sum += c.x;\n"
    "    }\n"
    "    out[tid] = sum;\n"
    "}\n";

/* Fill a uint32_t buffer with shuffled pointer-chase indices.
   Nodes are spaced `stride` elements apart, forming a single cycle. */
static int fill_chase_indices(uint32_t *buf, size_t total_elements, size_t stride) {
    size_t count = total_elements / stride;

    memset(buf, 0, total_elements * sizeof(uint32_t));

    size_t *indices = malloc(count * sizeof(size_t));
    if (!indices) return -1;
    for (size_t i = 0; i < count; i++) indices[i] = i;

    shuffle_indices(indices, count);

    /* Link nodes into a cycle */
    for (size_t i = 0; i < count - 1; i++)
        buf[indices[i] * stride] = (uint32_t)(indices[i + 1] * stride);
    buf[indices[count - 1] * stride] = (uint32_t)(indices[0] * stride);

    free(indices);
    return 0;
}

/* --- Compute benchmark (ALU-bound) --- */

static double run_gpu_bench(id<MTLDevice> device, NSString *source,
                            NSString *funcName, size_t num_threads) {
    NSError *error = nil;

    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                 options:nil
                                                   error:&error];
    if (!library) {
        fprintf(stderr, "  Metal shader compile error: %s\n",
                [[error localizedDescription] UTF8String]);
        return -1.0;
    }

    id<MTLFunction> function = [library newFunctionWithName:funcName];
    if (!function) {
        fprintf(stderr, "  Metal function '%s' not found\n",
                [funcName UTF8String]);
        return -1.0;
    }

    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        fprintf(stderr, "  Metal pipeline error: %s\n",
                [[error localizedDescription] UTF8String]);
        return -1.0;
    }

    size_t buf_size = num_threads * sizeof(float);
    id<MTLBuffer> buffer = [device newBufferWithLength:buf_size
                                              options:MTLResourceStorageModeShared];
    float *ptr = (float *)[buffer contents];
    if (!ptr) return -1;
    if ([funcName isEqualToString:@"int_bench"]) {
        uint32_t *ints = buffer.contents;
        for (size_t i = 0; i < num_threads; i++) ints[i] = (uint32_t)i + 1;
    } else {
        for (size_t i = 0; i < num_threads; i++) ptr[i] = 1.0f + (float)(i % 256) * 0.0001f;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* Warmup */
    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -1.0;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buffer offset:0 atIndex:0];
        NSUInteger tpg = pipeline.maxTotalThreadsPerThreadgroup;
        if (tpg > num_threads) tpg = num_threads;
        [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        if (!complete_command(cmd)) return -1.0;
    }

    /* Timed run */
    uint64_t t0 = timer_ns();
    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -1.0;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buffer offset:0 atIndex:0];
        NSUInteger tpg = pipeline.maxTotalThreadsPerThreadgroup;
        if (tpg > num_threads) tpg = num_threads;
        [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        if (!complete_command(cmd)) return -1.0;
    }
    uint64_t t1 = timer_ns();

    double elapsed_s = (double)(t1 - t0) / 1e9;
    /* 10 chains × 4096 iters × 2 ops (mul+add) per thread */
    double ops = (double)num_threads * 10.0 * 4096.0 * 2.0;
    return ops / elapsed_s / 1e9;
}

/* --- Bandwidth benchmark (parameterized by storage mode) --- */

static double run_gpu_bw(id<MTLDevice> device, MTLResourceOptions storage) {
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:bw_shader
                                                 options:nil
                                                   error:&error];
    if (!library) return -1.0;
    id<MTLFunction> function = [library newFunctionWithName:@"bw_bench"];
    if (!function) return -1.0;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) return -1.0;

    /* Each thread reads 256 float4 = 4096 bytes.
       Use enough threads to read 256 MiB total. */
    size_t total_bytes = BW_TOTAL_BYTES;
    size_t bytes_per_thread = 256 * sizeof(float) * 4; /* 256 float4 = 4096 bytes */
    size_t num_threads = total_bytes / bytes_per_thread;

    int is_private = (storage & MTLResourceStorageModePrivate) != 0;

    id<MTLBuffer> src;
    if (is_private) {
        /* Allocate private buffer, fill via staging + blit */
        src = [device newBufferWithLength:total_bytes
                                  options:MTLResourceStorageModePrivate];
        id<MTLBuffer> staging = [device newBufferWithLength:total_bytes
                                                   options:MTLResourceStorageModeShared];
        float *p = (float *)[staging contents];
        if (!src || !p) return -1;
        for (size_t i = 0; i < total_bytes / sizeof(float); i++) p[i] = 1.0f;

        id<MTLCommandQueue> blit_queue = [device newCommandQueue];
        id<MTLCommandBuffer> blit_cmd = [blit_queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [blit_cmd blitCommandEncoder];
        if (!blit) return -1.0;
        [blit copyFromBuffer:staging sourceOffset:0
                    toBuffer:src destinationOffset:0
                        size:total_bytes];
        [blit endEncoding];
        if (!complete_command(blit_cmd)) return -1.0;
    } else {
        src = [device newBufferWithLength:total_bytes
                                  options:MTLResourceStorageModeShared];
        float *p = (float *)[src contents];
        if (!p) return -1;
        for (size_t i = 0; i < total_bytes / sizeof(float); i++) p[i] = 1.0f;
    }

    id<MTLBuffer> dst = [device newBufferWithLength:num_threads * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    if (!dst) return -1;

    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* Warmup */
    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -1.0;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
        NSUInteger tpg = pipeline.maxTotalThreadsPerThreadgroup;
        if (tpg > num_threads) tpg = num_threads;
        [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        if (!complete_command(cmd)) return -1.0;
    }

    /* Run multiple passes to get stable measurement */
    int passes = 10;
    uint64_t t0 = timer_ns();
    for (int r = 0; r < passes; r++) {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -1.0;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
        NSUInteger tpg = pipeline.maxTotalThreadsPerThreadgroup;
        if (tpg > num_threads) tpg = num_threads;
        [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        if (!complete_command(cmd)) return -1.0;
    }
    uint64_t t1 = timer_ns();

    double elapsed_s = (double)(t1 - t0) / 1e9;
    return (double)total_bytes * passes / elapsed_s / 1e9;
}

/* --- Latency benchmark (pointer chase, single thread) --- */

static double run_gpu_latency(id<MTLDevice> device, MTLResourceOptions storage) {
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:latency_shader
                                                 options:nil
                                                   error:&error];
    if (!library) {
        fprintf(stderr, "  Metal latency shader compile error: %s\n",
                [[error localizedDescription] UTF8String]);
        return -1.0;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"latency_bench"];
    if (!function) return -1.0;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) return -1.0;

    size_t data_bytes = LAT_BUF_SIZE;
    size_t total_elements = data_bytes / sizeof(uint32_t);
    int is_private = (storage & MTLResourceStorageModePrivate) != 0;

    /* Prepare chase indices */
    uint32_t *chase_data = malloc(data_bytes);
    if (!chase_data) return -1;
    if (fill_chase_indices(chase_data, total_elements, LAT_STRIDE) < 0) {
        free(chase_data);
        return -1;
    }

    id<MTLBuffer> data_buf;
    if (is_private) {
        data_buf = [device newBufferWithLength:data_bytes
                                      options:MTLResourceStorageModePrivate];
        id<MTLBuffer> staging = [device newBufferWithBytes:chase_data
                                                   length:data_bytes
                                                  options:MTLResourceStorageModeShared];
        free(chase_data);
        chase_data = NULL;
        if (!data_buf || !staging) return -1;
        id<MTLCommandQueue> blit_queue = [device newCommandQueue];
        id<MTLCommandBuffer> blit_cmd = [blit_queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [blit_cmd blitCommandEncoder];
        if (!blit) return -1.0;
        [blit copyFromBuffer:staging sourceOffset:0
                    toBuffer:data_buf destinationOffset:0
                        size:data_bytes];
        [blit endEncoding];
        if (!complete_command(blit_cmd)) return -1.0;
    } else {
        data_buf = [device newBufferWithBytes:chase_data
                                       length:data_bytes
                                      options:MTLResourceStorageModeShared];
    }
    free(chase_data);

    id<MTLBuffer> result_buf = [device newBufferWithLength:sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];
    if (!data_buf || !result_buf) return -1;

    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* Warmup with small chase count */
    uint32_t warmup_chases = 1000;
    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -1.0;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:data_buf offset:0 atIndex:0];
        [enc setBuffer:result_buf offset:0 atIndex:1];
        [enc setBytes:&warmup_chases length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        if (!complete_command(cmd)) return -1.0;
    }

    /* Calibrate: time 10000 chases */
    uint32_t cal_chases = 10000;
    uint64_t tc0, tc1;
    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -1.0;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:data_buf offset:0 atIndex:0];
        [enc setBuffer:result_buf offset:0 atIndex:1];
        [enc setBytes:&cal_chases length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        tc0 = timer_ns();
        if (!complete_command(cmd)) return -1.0;
        tc1 = timer_ns();
    }

    /* Scale to ~500ms for accurate measurement */
    uint64_t cal_ns = tc1 - tc0;
    uint32_t num_chases = cal_chases;
    if (cal_ns > 0) {
        num_chases = (uint32_t)((double)cal_chases * 500000000.0 / (double)cal_ns);
        if (num_chases < 10000) num_chases = 10000;
        if (num_chases > 100000000) num_chases = 100000000;
    }

    /* Timed run */
    uint64_t t0, t1;
    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -1.0;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:data_buf offset:0 atIndex:0];
        [enc setBuffer:result_buf offset:0 atIndex:1];
        [enc setBytes:&num_chases length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        t0 = timer_ns();
        if (!complete_command(cmd)) return -1.0;
        t1 = timer_ns();
    }

    double elapsed_ns = (double)(t1 - t0);
    if (num_chases > 0)
        return elapsed_ns / (double)num_chases;
    return -1.0;
}

/* --- FP16 benchmark (half-precision buffer) --- */

static double run_gpu_fp16_bench(id<MTLDevice> device, size_t num_threads) {
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:fp16_shader
                                                 options:nil error:&error];
    if (!library) return -1.0;
    id<MTLFunction> function = [library newFunctionWithName:@"fp16_bench"];
    if (!function) return -1.0;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) return -1.0;

    /* Half-precision buffer (2 bytes per element) */
    size_t buf_size = num_threads * sizeof(uint16_t);
    id<MTLBuffer> buffer = [device newBufferWithLength:buf_size
                                              options:MTLResourceStorageModeShared];
    uint16_t *ptr = (uint16_t *)[buffer contents];
    if (!ptr) return -1;
    for (size_t i = 0; i < num_threads; i++) ptr[i] = 0x3C00; /* 1.0 in FP16 */

    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* Warmup */
    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -1.0;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buffer offset:0 atIndex:0];
        NSUInteger tpg = pipeline.maxTotalThreadsPerThreadgroup;
        if (tpg > num_threads) tpg = num_threads;
        [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        if (!complete_command(cmd)) return -1.0;
    }

    uint64_t t0 = timer_ns();
    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -1.0;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buffer offset:0 atIndex:0];
        NSUInteger tpg = pipeline.maxTotalThreadsPerThreadgroup;
        if (tpg > num_threads) tpg = num_threads;
        [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        if (!complete_command(cmd)) return -1.0;
    }
    uint64_t t1 = timer_ns();

    double elapsed_s = (double)(t1 - t0) / 1e9;
    double ops = (double)num_threads * 10.0 * 4096.0 * 2.0;
    return ops / elapsed_s / 1e9;
}

/* --- Shared memory bandwidth --- */

static double run_gpu_shared_mem(id<MTLDevice> device) {
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:shared_mem_shader
                                                 options:nil error:&error];
    if (!library) return -1.0;
    id<MTLFunction> function = [library newFunctionWithName:@"shared_mem_bench"];
    if (!function) return -1.0;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) return -1.0;

    size_t num_threads = 1 << 20; /* 1M threads */
    size_t buf_size = num_threads * sizeof(float);
    id<MTLBuffer> buffer = [device newBufferWithLength:buf_size
                                              options:MTLResourceStorageModeShared];
    float *ptr = (float *)[buffer contents];
    if (!ptr || pipeline.maxTotalThreadsPerThreadgroup < 256) return -1;
    for (size_t i = 0; i < num_threads; i++) ptr[i] = 1.0f;

    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* Warmup */
    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -1.0;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buffer offset:0 atIndex:0];
        [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        if (!complete_command(cmd)) return -1.0;
    }

    int passes = 10;
    uint64_t t0 = timer_ns();
    for (int r = 0; r < passes; r++) {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -1.0;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buffer offset:0 atIndex:0];
        [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        if (!complete_command(cmd)) return -1.0;
    }
    uint64_t t1 = timer_ns();

    double elapsed_s = (double)(t1 - t0) / 1e9;
    /* Each thread reads 256 floats from shared memory = 1024 bytes */
    double bytes = (double)num_threads * 256.0 * sizeof(float) * passes;
    return bytes / elapsed_s / 1e9;
}

/* --- Texture sampling throughput --- */

static double run_gpu_texture(id<MTLDevice> device) {
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:tex_shader
                                                 options:nil error:&error];
    if (!library) return -1.0;
    id<MTLFunction> function = [library newFunctionWithName:@"tex_bench"];
    if (!function) return -1.0;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) return -1.0;

    /* Create a 1024x1024 RGBA8 texture */
    MTLTextureDescriptor *desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                          width:1024 height:1024
                                                      mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [device newTextureWithDescriptor:desc];

    /* Fill texture with data */
    uint8_t *texdata = malloc(1024 * 1024 * 4);
    if (!texture || !texdata) { free(texdata); return -1; }
    memset(texdata, 0x80, 1024 * 1024 * 4);
    [texture replaceRegion:MTLRegionMake2D(0, 0, 1024, 1024)
               mipmapLevel:0
                 withBytes:texdata
               bytesPerRow:1024 * 4];
    free(texdata);

    size_t num_threads = 1 << 20; /* 1M threads */
    id<MTLBuffer> out_buf = [device newBufferWithLength:num_threads * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!out_buf) return -1;

    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* Warmup */
    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -1.0;
        [enc setComputePipelineState:pipeline];
        [enc setTexture:texture atIndex:0];
        [enc setBuffer:out_buf offset:0 atIndex:0];
        NSUInteger tpg = pipeline.maxTotalThreadsPerThreadgroup;
        if (tpg > num_threads) tpg = num_threads;
        [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        if (!complete_command(cmd)) return -1.0;
    }

    uint64_t t0 = timer_ns();
    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -1.0;
        [enc setComputePipelineState:pipeline];
        [enc setTexture:texture atIndex:0];
        [enc setBuffer:out_buf offset:0 atIndex:0];
        NSUInteger tpg = pipeline.maxTotalThreadsPerThreadgroup;
        if (tpg > num_threads) tpg = num_threads;
        [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        if (!complete_command(cmd)) return -1.0;
    }
    uint64_t t1 = timer_ns();

    double elapsed_s = (double)(t1 - t0) / 1e9;
    /* Each thread does 4096 texture samples */
    double texels = (double)num_threads * 4096.0;
    return texels / elapsed_s / 1e9;  /* Gtexels/s */
}

/* --- Entry point --- */

void bench_gpu(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("=== GPU Compute Throughput (Metal) ===\n");
            printf("  Metal not available\n");
            return;
        }

        printf("=== GPU Compute Throughput (Metal) ===\n");
        printf("  Device: %s\n", [[device name] UTF8String]);
        uint64_t vram = [device recommendedMaxWorkingSetSize];
        if (vram > 0)
            printf("  Recommended GPU working set: %llu MiB\n", vram / (1024 * 1024));
        printf("  Private and shared buffers use unified memory on Apple Silicon.\n");
        printf("%-20s %14s\n", "Test", "Throughput");
        printf("%-20s %14s\n", "----", "----------");

        size_t num_threads = 1 << 24; /* 16M threads */

        double fp32 = run_gpu_bench(device, fma_shader, @"fma_bench", num_threads);
        if (fp32 > 0) printf("%-20s %10.2f GFLOPS\n", "FP32", fp32);
        else          printf("%-20s %14s\n", "FP32", "error");
        fflush(stdout);

        double fp16 = run_gpu_fp16_bench(device, num_threads);
        if (fp16 > 0) printf("%-20s %10.2f GFLOPS\n", "FP16", fp16);
        else          printf("%-20s %14s\n", "FP16", "error");
        fflush(stdout);

        double int32 = run_gpu_bench(device, int_shader, @"int_bench", num_threads);
        if (int32 > 0) printf("%-20s %10.2f GINTOPS\n", "INT32", int32);
        else           printf("%-20s %14s\n", "INT32", "error");
        fflush(stdout);

        double vram_bw = run_gpu_bw(device, MTLResourceStorageModePrivate);
        if (vram_bw > 0) printf("%-20s %10.2f GB/s\n", "VRAM BW", vram_bw);
        else             printf("%-20s %14s\n", "VRAM BW", "error");
        fflush(stdout);

        double vram_lat = run_gpu_latency(device, MTLResourceStorageModePrivate);
        if (vram_lat > 0) printf("%-20s %10.2f ns\n", "VRAM Latency", vram_lat);
        else              printf("%-20s %14s\n", "VRAM Latency", "error");
        fflush(stdout);

        double host_bw = run_gpu_bw(device, MTLResourceStorageModeShared);
        if (host_bw > 0) printf("%-20s %10.2f GB/s\n", "Host BW", host_bw);
        else             printf("%-20s %14s\n", "Host BW", "error");
        fflush(stdout);

        double host_lat = run_gpu_latency(device, MTLResourceStorageModeShared);
        if (host_lat > 0) printf("%-20s %10.2f ns\n", "Host Latency", host_lat);
        else              printf("%-20s %14s\n", "Host Latency", "error");
        fflush(stdout);

        double smem = run_gpu_shared_mem(device);
        if (smem > 0) printf("%-20s %10.2f GB/s\n", "Shared Mem BW", smem);
        else          printf("%-20s %14s\n", "Shared Mem BW", "error");
        fflush(stdout);

        double tex = run_gpu_texture(device);
        if (tex > 0) printf("%-20s %10.2f Gtex/s\n", "Texture Sample", tex);
        else         printf("%-20s %14s\n", "Texture Sample", "error");
        fflush(stdout);
    }
}

#else

#include <stdio.h>

void bench_gpu(void) {
    printf("=== GPU Compute Throughput (Metal) ===\n");
    printf("  Metal not available (non-Apple platform)\n");
}

#endif
