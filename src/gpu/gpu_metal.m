/* Metal backend for the GPU section (see gpu_backend.h). Timing uses the
 * command buffer's GPUStartTime/GPUEndTime. */
#include "gpu/gpu_backend.h"
#include "core/report.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdlib.h>
#include <string.h>

/* Kernel constants must match gpu_backend.h and the shaders/ sources. Runtime
 * parameters arrive in `prm`: n = loop count / element count, a = texture
 * side, b and c = FMA/IMAD operands (floats as bit patterns), so the compiler
 * cannot fold the arithmetic. */
static NSString *const kernel_src = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct prm { uint n; uint a; uint b; uint c; };\n"
    "#define R4(M, x) M(x##0) M(x##1) M(x##2) M(x##3)\n"
    "#define R16(M) R4(M, a0) R4(M, a1) R4(M, a2) R4(M, a3)\n"
    "#define R32(M) R16(M) R4(M, a4) R4(M, a5) R4(M, a6) R4(M, a7)\n"
    "#define DECL(v) vt v = s; s += d;\n"
    "#define FMA(v)  v = fma(v, b, c);\n"
    "#define SUM(v)  r += v;\n"
    "#define FMA_KERNEL(NAME, ST, VT)                                                   \\\n"
    "kernel void NAME(device float *buf [[buffer(0)]], constant prm &p [[buffer(1)]],   \\\n"
    "                 uint tid [[thread_position_in_grid]]) {                           \\\n"
    "    typedef ST st; typedef VT vt;                                                  \\\n"
    "    vt s = vt(st(buf[tid]));                                                       \\\n"
    "    vt b = vt(st(as_type<float>(p.b))), c = vt(st(as_type<float>(p.c)));           \\\n"
    "    vt d = vt(st(0.001f));                                                         \\\n"
    "    R32(DECL)                                                                      \\\n"
    "    for (uint i = 0; i < p.n; i++) { R32(FMA) R32(FMA) R32(FMA) R32(FMA) }         \\\n"
    "    vt r = vt(st(0)); R32(SUM)                                                     \\\n"
    "    buf[tid] = float(r.x) + float(r.y);                                            \\\n"
    "}\n"
    "FMA_KERNEL(fma_f32, float, float2)\n"
    "FMA_KERNEL(fma_f16, half, half2)\n"
    "\n"
    "#define IDECL(v) uint v = s; s += 1u;\n"
    "#define IMAD(v)  v = v * b + c;\n"
    "#define IXOR(v)  r ^= v;\n"
    "kernel void int32(device uint *buf [[buffer(0)]], constant prm &p [[buffer(1)]],\n"
    "                  uint tid [[thread_position_in_grid]]) {\n"
    "    uint s = buf[tid], b = p.b, c = p.c;\n"
    "    R16(IDECL)\n"
    "    for (uint i = 0; i < p.n; i++) { R16(IMAD) R16(IMAD) }\n"
    "    uint r = 0; R16(IXOR)\n"
    "    buf[tid] = r;\n"
    "}\n"
    "\n"
    "kernel void bw(device const float4 *src [[buffer(0)]], device float *dst [[buffer(1)]],\n"
    "               constant prm &p [[buffer(2)]], uint tid [[thread_position_in_grid]],\n"
    "               uint nt [[threads_per_grid]]) {\n"
    "    float4 s = 0;\n"
    "    for (uint i = tid; i < p.n; i += nt) { s += src[i]; }\n"
    "    dst[tid] = s.x + s.y + s.z + s.w;\n"
    "}\n"
    "\n"
    "kernel void chase(device const uint *d [[buffer(0)]], device uint *r [[buffer(1)]],\n"
    "                  constant prm &p [[buffer(2)]]) {\n"
    "    uint x = r[0];\n"
    "    for (uint i = 0; i < p.n; i++) { x = d[x]; }\n"
    "    r[0] = x;\n"
    "}\n"
    "\n"
    "#define TGL(j) s += sm[b + j * 32u];\n"
    "kernel void tgmem(device float4 *buf [[buffer(0)]], constant prm &p [[buffer(1)]],\n"
    "                  uint tid [[thread_position_in_grid]], uint lid [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float4 sm[1024 + 16 * 32];\n"
    "    for (uint i = lid; i < 1024 + 16 * 32; i += 256) { sm[i] = buf[(tid & ~255u) + (i & 255u)]; }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    float4 s = 0;\n"
    "    for (uint i = 0; i < p.n; i++) {\n"
    "        uint b = (lid + i * 32u) & 1023u;\n"
    "        TGL(0) TGL(1) TGL(2) TGL(3) TGL(4) TGL(5) TGL(6) TGL(7)\n"
    "        TGL(8) TGL(9) TGL(10) TGL(11) TGL(12) TGL(13) TGL(14) TGL(15)\n"
    "    }\n"
    "    buf[tid] = s;\n"
    "}\n"
    "\n"
    "kernel void tex(texture2d<float, access::sample> t [[texture(0)]], device float *out [[buffer(0)]],\n"
    "                constant prm &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {\n"
    "    constexpr sampler smp(coord::normalized, filter::linear, address::repeat);\n"
    "    uint  w   = p.a;\n"
    "    float inv = 1.0f / float(w);\n"
    "    float u   = float(tid % w + 1u) * inv;\n"
    "    float v   = float(((tid / w) * 2u * p.n) % w + 1u) * inv;\n"
    "    float4 a0 = 0, a1 = 0;\n"
    "    for (uint i = 0; i < p.n; i++) {\n"
    "        a0 += t.sample(smp, float2(u, v));\n"
    "        a1 += t.sample(smp, float2(u, v + inv));\n"
    "        v  += 2.0f * inv;\n"
    "    }\n"
    "    float4 a = a0 + a1;\n"
    "    out[tid] = a.x + a.y + a.z + a.w;\n"
    "}\n";

typedef struct {
    u32 n;
    u32 a;
    u32 b;
    u32 c;
} prm;

struct sb_gpu_dev {
    id<MTLDevice>               dev;
    id<MTLCommandQueue>         queue;
    id<MTLLibrary>              lib;
    /* prepared test */
    sb_gpu_kernel_e             kernel;
    id<MTLComputePipelineState> pso;
    id<MTLBuffer>               b0;
    id<MTLBuffer>               b1;
    id<MTLTexture>              tex;
    u32                         threads;
    u32                         tex_dim;
};

/* One device at a time; static storage keeps ARC-managed fields zeroed. */
static struct sb_gpu_dev g_dev;

static u32 f32_bits(f32 f) {
    u32 u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

static sb_status_e wait_cb(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    return cb.status == MTLCommandBufferStatusCompleted ? SB_OK : SB_ERR_IO;
}

/* SB_GPU_DEVICE=<index|name substring> selects among MTLCopyAllDevices(). */
static id<MTLDevice> pick_device(void) {
    const char *env = getenv("SB_GPU_DEVICE");
    NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
    if (all.count > 1) {
        for (NSUInteger i = 0; i < all.count; i++) {
            sb_report_info("Metal device %lu: %s", (unsigned long)i, all[i].name.UTF8String);
        }
    }
    if (env == NULL || env[0] == '\0') { return MTLCreateSystemDefaultDevice(); }
    char *end = NULL;
    unsigned long idx = strtoul(env, &end, 10);
    if (end != env && *end == '\0') {
        return idx < all.count ? all[idx] : nil;
    }
    for (id<MTLDevice> d in all) {
        if (strstr(d.name.UTF8String, env) != NULL) { return d; }
    }
    return nil;
}

sb_status_e sb_gpu_open(sb_gpu_dev **out, sb_gpu_caps *caps) {
    @autoreleasepool {
        sb_gpu_dev *d = &g_dev;
        d->dev = pick_device();
        if (d->dev == nil) {
            if (getenv("SB_GPU_DEVICE") != NULL) { sb_report_info("SB_GPU_DEVICE matches no Metal device"); }
            return SB_ERR_NOTFOUND;
        }
        d->queue = [d->dev newCommandQueue];
        NSError *err = nil;
        MTLCompileOptions *opt = [MTLCompileOptions new];
        d->lib = [d->dev newLibraryWithSource:kernel_src options:opt error:&err];
        if (d->queue == nil || d->lib == nil) {
            if (err != nil) { sb_report_info("Metal compile error: %s", err.localizedDescription.UTF8String); }
            sb_gpu_close(d);
            return SB_ERR_SYS;
        }

        memset(caps, 0, sizeof(*caps));
        snprintf(caps->name, sizeof(caps->name), "%s", d->dev.name.UTF8String);
        snprintf(caps->api, sizeof(caps->api), "Metal");
        caps->unified     = d->dev.hasUnifiedMemory;
        caps->fp16        = true;
        caps->gpu_timer   = true;
        caps->max_buffer  = (u64)d->dev.maxBufferLength;
        caps->max_tex_dim = [d->dev supportsFamily:MTLGPUFamilyApple3] ? 16384 : 8192;
        snprintf(caps->notes[0], sizeof(caps->notes[0]), "Recommended GPU working set: %llu MiB",
                 (unsigned long long)(d->dev.recommendedMaxWorkingSetSize >> 20));
        if (caps->unified) {
            snprintf(caps->notes[1], sizeof(caps->notes[1]),
                     "Unified memory: private and shared storage are the same DRAM; device-local uses private storage");
        }
        *out = d;
        return SB_OK;
    }
}

void sb_gpu_close(sb_gpu_dev *d) {
    if (d == NULL) { return; }
    sb_gpu_release(d);
    d->lib   = nil;
    d->queue = nil;
    d->dev   = nil;
}

void sb_gpu_release(sb_gpu_dev *d) {
    d->pso = nil;
    d->b0  = nil;
    d->b1  = nil;
    d->tex = nil;
}

static id<MTLBuffer> shared_buf(sb_gpu_dev *d, u64 bytes) {
    return [d->dev newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
}

/* Private buffer filled on the GPU (contents only need to be defined). */
static id<MTLBuffer> private_buf(sb_gpu_dev *d, u64 bytes) {
    id<MTLBuffer> b = [d->dev newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModePrivate];
    if (b == nil) { return nil; }
    id<MTLCommandBuffer>      cb   = [d->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit fillBuffer:b range:NSMakeRange(0, (NSUInteger)bytes) value:0x3F];
    [blit endEncoding];
    return wait_cb(cb) == SB_OK ? b : nil;
}

/* Private RGBA8 texture of incompressible bytes, uploaded through a blit.
 * Apple GPUs losslessly compress private textures: on the M4 Pro a linear
 * byte pattern streams at ~260 GB/s and zeros at ~450 GB/s, above DRAM
 * bandwidth, so random content is required to measure the DRAM path. */
static id<MTLTexture> make_texture(sb_gpu_dev *d, u32 dim) {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                    width:dim
                                                                                   height:dim
                                                                                mipmapped:NO];
    desc.usage       = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModePrivate;
    id<MTLTexture> t = [d->dev newTextureWithDescriptor:desc];
    u64 row   = (u64)dim * 4;
    u64 bytes = row * dim;
    id<MTLBuffer> staging = shared_buf(d, bytes);
    if (t == nil || staging == nil) { return nil; }
    u32 *p = staging.contents;
    u64  x = 0x243F6A8885A308D3ULL;
    for (u64 i = 0; i < bytes / 4; i++) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        p[i] = (u32)(x >> 32);
    }
    id<MTLCommandBuffer>      cb   = [d->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:staging
             sourceOffset:0
        sourceBytesPerRow:(NSUInteger)row
      sourceBytesPerImage:(NSUInteger)bytes
               sourceSize:MTLSizeMake(dim, dim, 1)
                toTexture:t
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    return wait_cb(cb) == SB_OK ? t : nil;
}

static NSString *kernel_name(sb_gpu_kernel_e k) {
    switch (k) {
    case SB_GPU_K_FP32:       return @"fma_f32";
    case SB_GPU_K_FP16:       return @"fma_f16";
    case SB_GPU_K_INT32:      return @"int32";
    case SB_GPU_K_BW_DEVICE:
    case SB_GPU_K_BW_HOST:    return @"bw";
    case SB_GPU_K_LATENCY:    return @"chase";
    case SB_GPU_K_TGMEM:      return @"tgmem";
    case SB_GPU_K_TEX_CACHED:
    case SB_GPU_K_TEX_STREAM: return @"tex";
    }
    return nil;
}

sb_status_e sb_gpu_prepare(sb_gpu_dev *d, const sb_gpu_test *t) {
    @autoreleasepool {
        sb_gpu_release(d);
        id<MTLFunction> fn = [d->lib newFunctionWithName:kernel_name(t->kernel)];
        NSError *err = nil;
        d->pso = fn != nil ? [d->dev newComputePipelineStateWithFunction:fn error:&err] : nil;
        if (d->pso == nil) { return SB_ERR_SYS; }
        u32 tg = t->kernel == SB_GPU_K_LATENCY ? 1 : SB_GPU_TG;
        if (d->pso.maxTotalThreadsPerThreadgroup < tg) { return SB_ERR_UNSUPPORTED; }
        d->kernel  = t->kernel;
        d->threads = t->threads;
        d->tex_dim = t->tex_dim;

        switch (t->kernel) {
        case SB_GPU_K_FP32:
        case SB_GPU_K_FP16:
        case SB_GPU_K_INT32:
            d->b0 = shared_buf(d, (u64)t->threads * 4);
            if (d->b0 == nil) { return SB_ERR_NOMEM; }
            if (t->kernel == SB_GPU_K_INT32) {
                u32 *p = d->b0.contents;
                for (u32 i = 0; i < t->threads; i++) { p[i] = i * 2654435761U; }
            } else {
                f32 *p = d->b0.contents;
                for (u32 i = 0; i < t->threads; i++) { p[i] = 0.5f + (f32)(i & 255) * (1.0f / 512.0f); }
            }
            break;
        case SB_GPU_K_BW_DEVICE:
        case SB_GPU_K_BW_HOST:
            d->b0 = t->kernel == SB_GPU_K_BW_DEVICE ? private_buf(d, t->bytes) : shared_buf(d, t->bytes);
            d->b1 = shared_buf(d, (u64)t->threads * 4);
            if (d->b0 == nil || d->b1 == nil) { return SB_ERR_NOMEM; }
            if (t->kernel == SB_GPU_K_BW_HOST) { memset(d->b0.contents, 0x3F, (size_t)t->bytes); }
            break;
        case SB_GPU_K_LATENCY:
            d->b0 = [d->dev newBufferWithBytes:t->chain length:(NSUInteger)t->bytes
                                       options:MTLResourceStorageModeShared];
            d->b1 = shared_buf(d, 4);    /* chase position, carried across dispatches */
            if (d->b0 == nil || d->b1 == nil) { return SB_ERR_NOMEM; }
            *(u32 *)d->b1.contents = 0;
            break;
        case SB_GPU_K_TGMEM:
            d->b0 = shared_buf(d, (u64)t->threads * 16);
            if (d->b0 == nil) { return SB_ERR_NOMEM; }
            break;
        case SB_GPU_K_TEX_CACHED:
        case SB_GPU_K_TEX_STREAM:
            d->tex = make_texture(d, t->tex_dim);
            d->b0  = shared_buf(d, (u64)t->threads * 4);
            if (d->tex == nil || d->b0 == nil) { return SB_ERR_NOMEM; }
            break;
        }
        return SB_OK;
    }
}

sb_status_e sb_gpu_run(sb_gpu_dev *d, u32 n, u32 ndispatch, f64 *out_ns) {
    @autoreleasepool {
        prm p = { .n = n, .a = d->tex_dim };
        if (d->kernel == SB_GPU_K_FP16) {
            p.b = f32_bits(0.999f);
            p.c = f32_bits(0.001f);
        } else if (d->kernel == SB_GPU_K_INT32) {
            p.b = 0x4F6CDD1DU;
            p.c = 0x7F4A7C15U;
        } else {
            p.b = f32_bits(0.9999f);
            p.c = f32_bits(0.0001f);
        }

        id<MTLCommandBuffer>         cb  = [d->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (cb == nil || enc == nil) { return SB_ERR_SYS; }
        [enc setComputePipelineState:d->pso];
        NSUInteger pidx = 1;
        switch (d->kernel) {
        case SB_GPU_K_BW_DEVICE:
        case SB_GPU_K_BW_HOST:
        case SB_GPU_K_LATENCY:
            [enc setBuffer:d->b0 offset:0 atIndex:0];
            [enc setBuffer:d->b1 offset:0 atIndex:1];
            pidx = 2;
            break;
        case SB_GPU_K_TEX_CACHED:
        case SB_GPU_K_TEX_STREAM:
            [enc setTexture:d->tex atIndex:0];
            [enc setBuffer:d->b0 offset:0 atIndex:0];
            break;
        case SB_GPU_K_FP32:
        case SB_GPU_K_FP16:
        case SB_GPU_K_INT32:
        case SB_GPU_K_TGMEM:
            [enc setBuffer:d->b0 offset:0 atIndex:0];
            break;
        }
        [enc setBytes:&p length:sizeof(p) atIndex:pidx];
        MTLSize grid = MTLSizeMake(d->threads, 1, 1);
        MTLSize tg   = MTLSizeMake(d->kernel == SB_GPU_K_LATENCY ? 1 : SB_GPU_TG, 1, 1);
        for (u32 i = 0; i < ndispatch; i++) {
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        }
        [enc endEncoding];
        sb_status_e s = wait_cb(cb);
        if (s != SB_OK) { return s; }
        f64 sec = cb.GPUEndTime - cb.GPUStartTime;
        if (sec <= 0) { return SB_ERR_RANGE; }
        *out_ns = sec * 1e9;
        return SB_OK;
    }
}
