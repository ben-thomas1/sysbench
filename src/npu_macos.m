#include "bench.h"

#ifdef __APPLE__

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include "timer.h"
#include "npu_model_info.h"

#include <stdio.h>
#include <stdlib.h>

#define TARGET_NS 2000000000ULL /* 2 seconds */

struct bench_result {
    double tops;
    double inf_per_sec;
    double ms_per_inf;
    size_t passes;
};

static int predict(MLModel *model, id<MLFeatureProvider> input) {
    @autoreleasepool {
        NSError *error = nil;
        id<MLFeatureProvider> output = [model predictionFromFeatures:input error:&error];
        if (!output) {
            fprintf(stderr, "  Core ML prediction failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 0;
        }
        return 1;
    }
}

static struct bench_result run_inference_bench(NSURL *compiled_url,
                                               MLComputeUnits units) {
    struct bench_result res = {-1.0, 0, 0, 0};
    @autoreleasepool {
        NSError *error = nil;
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = units;

        MLModel *model = [MLModel modelWithContentsOfURL:compiled_url
                                           configuration:config
                                                   error:&error];
        if (!model) {
            fprintf(stderr, "  Failed to load model: %s\n",
                    [[error localizedDescription] UTF8String]);
            return res;
        }

        /* Create input: [3, 256, 256] (C, H, W) — no batch dim */
        NSArray<NSNumber *> *shape = @[@(NPU_IN_CH), @(NPU_SPATIAL), @(NPU_SPATIAL)];
        MLMultiArray *input_arr =
            [[MLMultiArray alloc] initWithShape:shape
                                      dataType:MLMultiArrayDataTypeFloat32
                                         error:&error];
        if (!input_arr) {
            fprintf(stderr, "  Failed to create input array: %s\n",
                    [[error localizedDescription] UTF8String]);
            return res;
        }
        /* Fill with small values */
        float *ptr = (float *)input_arr.dataPointer;
        NSInteger count = NPU_IN_CH * NPU_SPATIAL * NPU_SPATIAL;
        for (NSInteger i = 0; i < count; i++)
            ptr[i] = 0.001f * (float)(i % 256);

        MLDictionaryFeatureProvider *provider =
            [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{@"input": input_arr}
                             error:&error];
        if (!provider) return res;

        /* Warmup (first calls include JIT compilation) */
        for (int w = 0; w < 3; w++)
            if (!predict(model, provider)) return res;

        /* Calibrate */
        uint64_t tc0 = timer_ns();
        if (!predict(model, provider)) return res;
        uint64_t tc1 = timer_ns();
        uint64_t one_pass = tc1 - tc0;
        size_t passes = 1;
        if (one_pass > 0) passes = (size_t)(TARGET_NS / one_pass) + 1;
        if (passes < 2) passes = 2;

        /* Timed run */
        uint64_t t0 = timer_ns();
        for (size_t p = 0; p < passes; p++)
            if (!predict(model, provider)) return res;
        uint64_t t1 = timer_ns();

        double elapsed_s = (double)(t1 - t0) / 1e9;
        if (elapsed_s <= 0) return res;

        res.tops = (double)NPU_OPS_PER_INFERENCE * passes / elapsed_s / 1e12;
        res.inf_per_sec = passes / elapsed_s;
        res.ms_per_inf = elapsed_s / passes * 1e3;
        res.passes = passes;
        return res;
    }
}

void bench_npu(void) {
    @autoreleasepool {
        printf("=== NPU Compute Throughput (Core ML / ANE) ===\n");

        /* Find model file */
        NSString *model_path = nil;
        NSArray *candidates = @[
            @"models/npu_bench.mlmodel",
            @"models/npu_bench.mlpackage",
        ];
        NSFileManager *fm = [NSFileManager defaultManager];
        for (NSString *path in candidates) {
            if ([fm fileExistsAtPath:path]) {
                model_path = path;
                break;
            }
        }
        if (!model_path) {
            printf("  Model not found, skipping\n");
            printf("  (generate with: python3 scripts/gen_npu_model.py models/)\n");
            return;
        }

        /* Compile .mlmodel -> .mlmodelc at runtime */
        NSError *error = nil;
        NSURL *url = [NSURL fileURLWithPath:model_path];
        NSURL *compiled = [MLModel compileModelAtURL:url error:&error];
        if (!compiled) {
            printf("  Failed to compile model: %s\n",
                   [[error localizedDescription] UTF8String]);
            return;
        }

        printf("  Model: %dx Conv2d(3x3) @ %dx%d, %dch\n",
               NPU_NUM_LAYERS, NPU_SPATIAL, NPU_SPATIAL, NPU_MID_CHANNELS);
        printf("  Compute units are allowed devices; Core ML may use CPU fallback.\n");
        printf("  TOPS counts model multiply/add operations; execution precision is framework-selected.\n");
        printf("%-24s %10s %10s %12s\n", "Test", "TOPS", "Latency", "Throughput");
        printf("%-24s %10s %10s %12s\n", "----", "----", "-------", "----------");

        /* ANE benchmark */
        struct bench_result ane = run_inference_bench(compiled,
            MLComputeUnitsCPUAndNeuralEngine);
        if (ane.tops > 0)
            printf("%-24s %10.2f %7.2f ms %8.1f inf/s\n",
                   "CPU + Neural Engine", ane.tops, ane.ms_per_inf, ane.inf_per_sec);
        else printf("%-24s %10s\n", "CPU + Neural Engine", "error");
        fflush(stdout);

        /* GPU benchmark */
        struct bench_result gpu = run_inference_bench(compiled,
            MLComputeUnitsCPUAndGPU);
        if (gpu.tops > 0)
            printf("%-24s %10.2f %7.2f ms %8.1f inf/s\n",
                   "CPU + GPU", gpu.tops, gpu.ms_per_inf, gpu.inf_per_sec);
        else printf("%-24s %10s\n", "CPU + GPU", "error");
        fflush(stdout);

        /* CPU-only baseline (uses AMX on Apple Silicon) */
        struct bench_result cpu = run_inference_bench(compiled,
            MLComputeUnitsCPUOnly);
        if (cpu.tops > 0)
            printf("%-24s %10.2f %7.2f ms %8.1f inf/s\n",
                   "CPU only", cpu.tops, cpu.ms_per_inf, cpu.inf_per_sec);
        else printf("%-24s %10s\n", "CPU only", "error");
        fflush(stdout);

        /* Clean up compiled model */
        [[NSFileManager defaultManager] removeItemAtURL:compiled error:nil];
    }
}

#else

void bench_npu(void) {
    printf("=== NPU Compute Throughput ===\n");
    printf("  Not available (requires Apple Silicon + Core ML)\n");
}

#endif
