/* NPU section, macOS: Core ML inference of the generated convolution model
 * under three compute-unit settings, with MLComputePlan placement. */
#include "npu/npu.h"
#include "core/report.h"
#include "core/stats.h"
#include "core/timer.h"
#include "npu_model_info.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <stdio.h>
#include <string.h>

#define WINDOW_NS     2'000'000'000ULL  /* timed window per compute-unit setting */
#define MIN_WARMUP    3                 /* first predictions include specialisation */
#define MIN_INFER     5
#define MAX_INFER     1024
#define ASYNC_WAIT_NS (120 * NSEC_PER_SEC)

typedef struct {
    MLComputeUnits units;
    const char    *title;   /* group header */
    const char    *tag;     /* row-name prefix, stable across passes */
} unit_mode;

static const unit_mode modes[] = {
    { MLComputeUnitsCPUAndNeuralEngine, "CPU + Neural Engine", "CPU+ANE" },
    { MLComputeUnitsCPUAndGPU,          "CPU + GPU",           "CPU+GPU" },
    { MLComputeUnitsCPUOnly,            "CPU only",            "CPU"     },
};

typedef struct {
    f64 tops;         /* model ops / time */
    f64 inf_per_sec;  /* inferences / total timed wall time */
    f64 ms_median;    /* median single-inference latency */
    u32 n;
} infer_result;

static const char *model_paths[] = {
    "models/npu_bench.mlpackage",   /* ML Program, FP16 (gen_npu_model.py default) */
    "models/npu_bench.mlmodel",     /* NeuralNetwork (gen_npu_model.py --neuralnetwork) */
};

static void report_ns_error(const char *what, NSError *err) {
    sb_report_info("%s: %s", what, err ? err.localizedDescription.UTF8String : "unknown error");
}

/* --- Model compile -------------------------------------------------------- */

static sb_status_e compile_model(NSString *path, NSURL **out) {
    __block NSURL   *url = nil;
    __block NSError *err = nil;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [MLModel compileModelAtURL:[NSURL fileURLWithPath:path]
             completionHandler:^(NSURL *u, NSError *e) {
                 url = u;
                 err = e;
                 dispatch_semaphore_signal(sem);
             }];
    if (dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (i64)ASYNC_WAIT_NS)) != 0) {
        return SB_ERR_TIMEOUT;
    }
    if (url == nil) {
        report_ns_error("Model compile failed", err);
        return SB_ERR_INVALID;
    }
    *out = url;
    return SB_OK;
}

/* --- Placement (MLComputePlan, macOS 14.4+) -------------------------------- */

typedef struct {
    u32 ane;
    u32 gpu;
    u32 cpu;
    u32 total;        /* layers / operations that have a device assignment */
    bool program;     /* ML Program (vs NeuralNetwork) */
} placement;

API_AVAILABLE(macos(14.4))
static void count_device(id<MLComputeDeviceProtocol> d, placement *p) {
    p->total++;
    if ([(id)d isKindOfClass:[MLNeuralEngineComputeDevice class]]) { p->ane++; }
    else if ([(id)d isKindOfClass:[MLGPUComputeDevice class]])     { p->gpu++; }
    else if ([(id)d isKindOfClass:[MLCPUComputeDevice class]])     { p->cpu++; }
}

API_AVAILABLE(macos(14.4))
static sb_status_e plan_placement(NSURL *compiled, MLComputeUnits units, placement *out) {
    MLModelConfiguration *cfg = [MLModelConfiguration new];
    cfg.computeUnits = units;

    __block placement   p   = { 0 };
    __block sb_status_e s   = SB_OK;
    __block NSError    *err = nil;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [MLComputePlan loadContentsOfURL:compiled configuration:cfg
                   completionHandler:^(MLComputePlan *plan, NSError *e) {
        if (plan == nil) {
            err = e;
            s   = SB_ERR_INVALID;
            dispatch_semaphore_signal(sem);
            return;
        }
        MLModelStructureNeuralNetwork *nn   = plan.modelStructure.neuralNetwork;
        MLModelStructureProgram       *prog = plan.modelStructure.program;
        if (nn != nil) {
            for (MLModelStructureNeuralNetworkLayer *l in nn.layers) {
                MLComputePlanDeviceUsage *u = [plan computeDeviceUsageForNeuralNetworkLayer:l];
                if (u != nil) { count_device(u.preferredComputeDevice, &p); }
            }
        } else if (prog != nil) {
            p.program = true;
            MLModelStructureProgramFunction *f = prog.functions[@"main"];
            for (MLModelStructureProgramOperation *op in f.block.operations) {
                /* Constants have no device usage. */
                MLComputePlanDeviceUsage *u = [plan computeDeviceUsageForMLProgramOperation:op];
                if (u != nil) { count_device(u.preferredComputeDevice, &p); }
            }
        } else {
            s = SB_ERR_UNSUPPORTED;
        }
        dispatch_semaphore_signal(sem);
    }];
    if (dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (i64)ASYNC_WAIT_NS)) != 0) {
        return SB_ERR_TIMEOUT;
    }
    if (s != SB_OK) {
        if (err != nil) { report_ns_error("MLComputePlan failed", err); }
        return s;
    }
    *out = p;
    return SB_OK;
}

static void report_placement(NSURL *compiled) {
    if (@available(macOS 14.4, *)) {
        sb_report_info("Placement (MLComputePlan, preferred device per layer or operation):");
        const char *format = NULL;
        for (size_t i = 0; i < SB_ARRAY_LEN(modes); i++) {
            placement   p;
            sb_status_e s = plan_placement(compiled, modes[i].units, &p);
            if (s != SB_OK) {
                sb_report_info("  %-8s unavailable (%s)", modes[i].tag, sb_status_str(s));
                continue;
            }
            sb_report_info("  %-8s ANE %u, GPU %u, CPU %u of %u %s", modes[i].tag, p.ane, p.gpu, p.cpu,
                           p.total, p.program ? "ops" : "layers");
            format = p.program ? "ML Program (ops include ReLU)" : "NeuralNetwork (ReLU not listed separately)";
            if (modes[i].units != MLComputeUnitsCPUOnly && p.cpu > 0) {
                sb_report_info("  WARNING: %u %s fall back to the CPU under %s", p.cpu,
                               p.program ? "ops" : "layers", modes[i].tag);
            }
        }
        if (format != NULL) { sb_report_info("Model format: %s", format); }
    } else {
        sb_report_info("Placement: needs macOS 14.4 (MLComputePlan); Core ML may use CPU fallback");
    }
}

/* --- Inference ------------------------------------------------------------ */

static const char *dtype_str(MLMultiArrayDataType t) {
    switch (t) {
    case MLMultiArrayDataTypeFloat16: return "FP16";
    case MLMultiArrayDataTypeFloat32: return "FP32";
    case MLMultiArrayDataTypeDouble:  return "FP64";
    default:                          return "other";
    }
}

/* An array with the feature's shape and type, filled with small values. */
static MLMultiArray *make_array(MLFeatureDescription *d, NSError **err) {
    MLMultiArrayConstraint *c   = d.multiArrayConstraint;
    MLMultiArray           *arr = [[MLMultiArray alloc] initWithShape:c.shape dataType:c.dataType error:err];
    if (arr == nil) { return nil; }
    NSInteger n = arr.count;
    NSInteger esz = c.dataType == MLMultiArrayDataTypeFloat16 ? 2 : c.dataType == MLMultiArrayDataTypeDouble ? 8 : 4;
    [arr getMutableBytesWithHandler:^(void *bytes, NSInteger size, NSArray<NSNumber *> *strides) {
        (void)strides;
        /* Values are arbitrary: fill linearly, bounded by the buffer size. */
        NSInteger cnt = size / esz < n ? size / esz : n;
        for (NSInteger i = 0; i < cnt; i++) {
            f32 v = 0.001f * (f32)(i % 256);
            if (c.dataType == MLMultiArrayDataTypeFloat16)      { ((_Float16 *)bytes)[i] = (_Float16)v; }
            else if (c.dataType == MLMultiArrayDataTypeFloat32) { ((f32 *)bytes)[i] = v; }
            else if (c.dataType == MLMultiArrayDataTypeDouble)  { ((f64 *)bytes)[i] = (f64)v; }
        }
    }];
    return arr;
}

/* One synchronous prediction; reports the failure reason unless quiet. */
static bool predict(MLModel *model, id<MLFeatureProvider> in, MLPredictionOptions *opt, bool quiet) {
    @autoreleasepool {
        NSError *err = nil;
        id<MLFeatureProvider> outp = [model predictionFromFeatures:in options:opt error:&err];
        if (outp == nil && !quiet) { report_ns_error("Prediction failed", err); }
        return outp != nil;
    }
}

static sb_status_e run_mode(NSURL *compiled, const unit_mode *m, infer_result *out) {
    @autoreleasepool {
        NSError              *err = nil;
        MLModelConfiguration *cfg = [MLModelConfiguration new];
        cfg.computeUnits = m->units;
        MLModel *model = [MLModel modelWithContentsOfURL:compiled configuration:cfg error:&err];
        if (model == nil) {
            report_ns_error("Model load failed", err);
            return SB_ERR_INVALID;
        }

        MLModelDescription   *desc = model.modelDescription;
        MLFeatureDescription *in_d = desc.inputDescriptionsByName.allValues.firstObject;
        MLFeatureDescription *ot_d = desc.outputDescriptionsByName.allValues.firstObject;
        if (in_d == nil || ot_d == nil || in_d.type != MLFeatureTypeMultiArray ||
            ot_d.type != MLFeatureTypeMultiArray) {
            return SB_ERR_INVALID;
        }
        MLMultiArray *in_arr = make_array(in_d, &err);
        MLMultiArray *ot_arr = make_array(ot_d, &err);
        if (in_arr == nil || ot_arr == nil) { return SB_ERR_NOMEM; }
        MLDictionaryFeatureProvider *in = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{ in_d.name: in_arr } error:&err];
        if (in == nil) { return SB_ERR_INVALID; }

        /* Preallocated output: no per-inference output allocation. Fall back to
         * framework-allocated output if the engine rejects the backing. */
        MLPredictionOptions *opt = [MLPredictionOptions new];
        opt.outputBackings = @{ ot_d.name: ot_arr };
        if (!predict(model, in, opt, true)) {
            sb_report_info("%s: output backing rejected; Core ML allocates the output", m->tag);
            opt = [MLPredictionOptions new];
            if (!predict(model, in, opt, false)) { return SB_ERR_IO; }
        }

        u32 warm = 0;
        u64 w0   = sb_timer_now_ns();
        while (warm < MIN_WARMUP || sb_timer_now_ns() - w0 < SB_WARMUP_NS) {
            if (!predict(model, in, opt, false)) { return SB_ERR_IO; }
            warm++;
        }

        f64 lat[MAX_INFER];
        u32 n  = 0;
        u64 t0 = sb_timer_now_ns();
        u64 t1 = t0;
        while (n < MAX_INFER && (n < MIN_INFER || t1 - t0 < WINDOW_NS)) {
            u64 a = sb_timer_now_ns();
            if (!predict(model, in, opt, false)) { return SB_ERR_IO; }
            t1       = sb_timer_now_ns();
            lat[n++] = (f64)(t1 - a);
        }
        if (t1 <= t0) { return SB_ERR_RANGE; }

        sb_stats    st;
        sb_status_e s = sb_stats_compute(lat, n, &st);
        if (s != SB_OK) { return s; }
        f64 secs         = (f64)(t1 - t0) / 1e9;
        out->n           = n;
        out->inf_per_sec = (f64)n / secs;
        out->tops        = (f64)NPU_OPS_PER_INFERENCE * out->inf_per_sec / 1e12;
        out->ms_median   = st.median / 1e6;

        return SB_OK;
    }
}

/* Input/output types, read from a CPU-only load (no device compilation). */
static void report_io(NSURL *compiled) {
    MLModelConfiguration *cfg = [MLModelConfiguration new];
    cfg.computeUnits = MLComputeUnitsCPUOnly;
    MLModel *model = [MLModel modelWithContentsOfURL:compiled configuration:cfg error:nil];
    MLFeatureDescription *in_d = model.modelDescription.inputDescriptionsByName.allValues.firstObject;
    MLFeatureDescription *ot_d = model.modelDescription.outputDescriptionsByName.allValues.firstObject;
    if (in_d.multiArrayConstraint == nil || ot_d.multiArrayConstraint == nil) { return; }
    sb_report_info("I/O: %s input, %s output; timed inferences include Core ML I/O handling",
                   dtype_str(in_d.multiArrayConstraint.dataType), dtype_str(ot_d.multiArrayConstraint.dataType));
}

static sb_status_e npu_run(void) {
    @autoreleasepool {
        NSFileManager *fm   = [NSFileManager defaultManager];
        NSString      *path = nil;
        for (size_t i = 0; i < SB_ARRAY_LEN(model_paths); i++) {
            NSString *p = @(model_paths[i]);
            if ([fm fileExistsAtPath:p]) {
                path = p;
                break;
            }
        }
        if (path == nil) {
            sb_report_info("Generate the model from the repo root: uv run tools/gen_npu_model.py models/");
            sb_report_skip("Core ML inference", "model not found (models/npu_bench.mlmodel)");
            return SB_OK;
        }

        NSURL      *compiled = nil;
        sb_status_e s        = compile_model(path, &compiled);
        if (s != SB_OK) {
            sb_report_error("Core ML inference", s);
            return SB_OK;
        }

        sb_report_info("Model: %s, %dx Conv2d 3x3 @ %dx%d, %d channels, %.1f G model ops",
                       path.UTF8String, NPU_NUM_LAYERS, NPU_SPATIAL, NPU_SPATIAL, NPU_MID_CHANNELS,
                       (f64)NPU_OPS_PER_INFERENCE / 1e9);
        sb_report_info("TOPS are effective: model ops (multiply + add = 2) / time, not executed");
        sb_report_info("hardware ops. Core ML may execute fewer ops (e.g. Winograd 3x3 convolution):");
        sb_report_info("CPU+GPU can exceed the GPU FMA peak. Precision is framework-selected.");
        report_io(compiled);
        report_placement(compiled);

        for (size_t i = 0; i < SB_ARRAY_LEN(modes); i++) {
            const unit_mode *m = &modes[i];
            char             test[64];
            sb_report_group(m->title);

            infer_result r;
            s = run_mode(compiled, m, &r);
            snprintf(test, sizeof(test), "%s effective", m->tag);
            if (s != SB_OK) {
                sb_report_error(test, s);
                continue;
            }
            sb_report_value(test, r.tops, "TOPS", SB_KIND_EFFECTIVE);
            snprintf(test, sizeof(test), "%s latency (median)", m->tag);
            sb_report_value(test, r.ms_median, "ms", SB_KIND_MEASURED);
            snprintf(test, sizeof(test), "%s throughput", m->tag);
            sb_report_value(test, r.inf_per_sec, "inf/s", SB_KIND_MEASURED);
        }

        [fm removeItemAtURL:compiled error:nil];
        return SB_OK;
    }
}

const sb_section sb_section_npu = {
    .name       = "npu",
    .title      = "NPU Compute Throughput",
    .help       = SB_NPU_HELP,
    .run        = npu_run,
    .repeatable = true,
};
