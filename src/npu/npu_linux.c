/* NPU section, Linux: OpenVINO C API inference of the generated MatMul model
 * on NPU / GPU (when listed) and CPU. Builds without OpenVINO as a skip. */
#include "npu/npu.h"
#include "core/report.h"
#include "core/stats.h"
#include "core/timer.h"

#include <stdio.h>
#include <string.h>

#ifdef HAS_OPENVINO

#include <openvino/c/openvino.h>

/* Must match tools/gen_openvino_model.py. */
#define MM_SIZE        512
#define MM_LAYERS      10
#define MM_OPS         ((u64)MM_LAYERS * 2ULL * MM_SIZE * MM_SIZE * MM_SIZE)
#define MODEL_XML      "models/npu_bench.xml"
#define MODEL_BIN      "models/npu_bench.bin"

#define WINDOW_NS      2'000'000'000ULL
#define MIN_WARMUP     3
#define MIN_INFER      5
#define MAX_INFER      4096

typedef struct {
    const char *device;   /* OpenVINO device name */
    bool        always;   /* run even if not listed (CPU) */
} ov_target;

static const ov_target targets[] = {
    { "NPU", false },
    { "GPU", false },
    { "CPU", true  },
};

typedef struct {
    f64 tops;
    f64 inf_per_sec;
    f64 ms_median;
} infer_result;

static const char *ov_err(ov_status_e st) {
    const char *info = ov_get_error_info(st);
    return info != NULL ? info : "unknown error";
}

/* Time synchronous inferences on one device. `lat` holds MAX_INFER samples. */
static sb_status_e time_infer(ov_infer_request_t *req, f64 *lat, infer_result *out) {
    u32 warm = 0;
    u64 w0   = sb_timer_now_ns();
    while (warm < MIN_WARMUP || sb_timer_now_ns() - w0 < SB_WARMUP_NS) {
        if (ov_infer_request_infer(req) != OK) { return SB_ERR_IO; }
        warm++;
    }

    u32 n  = 0;
    u64 t0 = sb_timer_now_ns();
    u64 t1 = t0;
    while (n < MAX_INFER && (n < MIN_INFER || t1 - t0 < WINDOW_NS)) {
        u64 a = sb_timer_now_ns();
        if (ov_infer_request_infer(req) != OK) { return SB_ERR_IO; }
        t1       = sb_timer_now_ns();
        lat[n++] = (f64)(t1 - a);
    }
    if (t1 <= t0) { return SB_ERR_RANGE; }

    sb_stats    st;
    sb_status_e s = sb_stats_compute(lat, n, &st);
    if (s != SB_OK) { return s; }
    out->inf_per_sec = (f64)n / ((f64)(t1 - t0) / 1e9);
    out->tops        = (f64)MM_OPS * out->inf_per_sec / 1e12;
    out->ms_median   = st.median / 1e6;
    return SB_OK;
}

/* Compile for `device`, set a filled input tensor, time it. Errors that come
 * from OpenVINO are printed as an info line with the device name. */
static sb_status_e run_device(ov_core_t *core, const ov_model_t *model, const char *device,
                              f64 *lat, infer_result *out) {
    sb_status_e             s        = SB_OK;
    ov_status_e             st       = OK;
    const char             *what     = NULL;
    ov_output_const_port_t *port     = NULL;
    ov_shape_t              shape    = { 0 };
    ov_tensor_t            *tensor   = NULL;
    ov_compiled_model_t    *compiled = NULL;
    ov_infer_request_t     *req      = NULL;
    char                   *prec     = NULL;
    f32                    *data     = NULL;
    ov_element_type_e       type     = F32;

    st = ov_model_const_input(model, &port);
    if (st != OK) { what = "model input"; goto ov_fail; }
    st = ov_const_port_get_shape(port, &shape);
    if (st != OK) { what = "input shape"; goto ov_fail; }
    st = ov_port_get_element_type(port, &type);
    if (st != OK) { what = "input type"; goto ov_fail; }
    if (type != F32 || shape.rank != 2 || shape.dims[0] != MM_SIZE || shape.dims[1] != MM_SIZE) {
        sb_report_info("%s: expected an f32 [%d,%d] input; regenerate with gen_openvino_model.py",
                       device, MM_SIZE, MM_SIZE);
        s = SB_ERR_INVALID;
        goto cleanup;
    }

    st = ov_tensor_create(F32, shape, &tensor);
    if (st != OK) { what = "input tensor"; goto ov_fail; }
    st = ov_tensor_data(tensor, (void **)&data);
    if (st != OK || data == NULL) { what = "tensor data"; goto ov_fail; }
    for (u32 i = 0; i < MM_SIZE * MM_SIZE; i++) { data[i] = 0.001f * (f32)(i % 1024); }

    /* One property = two variadic arguments (key, value). */
    st = ov_core_compile_model(core, model, device, 2, &compiled,
                               ov_property_key_hint_performance_mode, "LATENCY");
    if (st != OK) { what = "compile"; goto ov_fail; }
    if (ov_compiled_model_get_property(compiled, ov_property_key_hint_inference_precision, &prec) == OK &&
        prec != NULL) {
        sb_report_info("%s: inference precision %s", device, prec);
    }

    st = ov_compiled_model_create_infer_request(compiled, &req);
    if (st != OK) { what = "infer request"; goto ov_fail; }
    st = ov_infer_request_set_input_tensor(req, tensor);
    if (st != OK) { what = "set input tensor"; goto ov_fail; }

    s = time_infer(req, lat, out);
    goto cleanup;

ov_fail:
    sb_report_info("%s: %s failed: %s", device, what, ov_err(st));
    s = SB_ERR_IO;

cleanup:
    if (prec != NULL)     { ov_free(prec); }
    if (req != NULL)      { ov_infer_request_free(req); }
    if (compiled != NULL) { ov_compiled_model_free(compiled); }
    if (tensor != NULL)   { ov_tensor_free(tensor); }
    if (port != NULL)     { ov_output_const_port_free(port); }
    ov_shape_free(&shape);
    return s;
}

/* Prints the device list; sets listed[i] for each entry of targets[]. */
static void report_devices(ov_core_t *core, bool *listed) {
    ov_available_devices_t devs = { 0 };
    ov_status_e            st   = ov_core_get_available_devices(core, &devs);
    if (st != OK) {
        sb_report_info("OpenVINO devices: unavailable (%s)", ov_err(st));
        return;
    }
    char   line[256] = "OpenVINO devices:";
    size_t len       = strlen(line);
    for (size_t i = 0; i < devs.size; i++) {
        const char *name = devs.devices[i];
        if (name == NULL) { continue; }
        int w = snprintf(line + len, sizeof(line) - len, " %s", name);
        if (w > 0 && (size_t)w < sizeof(line) - len) { len += (size_t)w; }
        /* Multi-device systems list "GPU.0", "GPU.1": match the prefix. */
        for (size_t t = 0; t < SB_ARRAY_LEN(targets); t++) {
            size_t n = strlen(targets[t].device);
            if (strncmp(name, targets[t].device, n) == 0) { listed[t] = true; }
        }
    }
    if (devs.size == 0) { snprintf(line + len, sizeof(line) - len, " none"); }
    sb_report_info("%s", line);
    ov_available_devices_free(&devs);

    for (size_t t = 0; t < SB_ARRAY_LEN(targets); t++) {
        char *full = NULL;
        if (!listed[t]) { continue; }
        if (ov_core_get_property(core, targets[t].device, ov_property_key_device_full_name, &full) == OK &&
            full != NULL) {
            sb_report_info("  %s: %s", targets[t].device, full);
            ov_free(full);
        }
    }
}

static sb_status_e npu_run(void) {
    FILE *f = fopen(MODEL_XML, "r");
    if (f == NULL) {
        sb_report_info("Generate the model from the repo root: uv run tools/gen_openvino_model.py models/");
        sb_report_skip("OpenVINO inference", "model not found (" MODEL_XML ")");
        return SB_OK;
    }
    fclose(f);

    ov_core_t   *core  = NULL;
    ov_model_t  *model = NULL;
    f64         *lat   = NULL;
    ov_version_t ver   = { 0 };
    ov_status_e  st    = ov_core_create(&core);
    if (st != OK) {
        sb_report_info("ov_core_create: %s", ov_err(st));
        sb_report_error("OpenVINO inference", SB_ERR_UNSUPPORTED);
        return SB_OK;
    }
    st = ov_core_read_model(core, MODEL_XML, MODEL_BIN, &model);
    if (st != OK) {
        sb_report_info("ov_core_read_model: %s", ov_err(st));
        sb_report_error("OpenVINO inference", SB_ERR_INVALID);
        goto cleanup;
    }
    lat = SB_MALLOC(MAX_INFER * sizeof(f64));
    if (lat == NULL) {
        sb_report_error("OpenVINO inference", SB_ERR_NOMEM);
        goto cleanup;
    }

    if (ov_get_openvino_version(&ver) == OK) {
        sb_report_info("OpenVINO %s", ver.buildNumber != NULL ? ver.buildNumber : "?");
        ov_version_free(&ver);
    }
    sb_report_info("Model: %dx MatMul %dx%d (f32), %.2f G model ops per inference",
                   MM_LAYERS, MM_SIZE, MM_SIZE, (f64)MM_OPS / 1e9);
    sb_report_info("TOPS are effective: model ops (multiply + add = 2) / time; precision is");
    sb_report_info("plugin-selected. Times include the plugin's input/output transfer.");

    bool listed[SB_ARRAY_LEN(targets)] = { false };
    report_devices(core, listed);

    for (size_t t = 0; t < SB_ARRAY_LEN(targets); t++) {
        const char *dev = targets[t].device;
        char        test[64];
        sb_report_group(dev);
        snprintf(test, sizeof(test), "%s effective", dev);
        if (!listed[t] && !targets[t].always) {
            sb_report_skip(test, "device not listed by OpenVINO");
            continue;
        }
        infer_result r;
        sb_status_e  s = run_device(core, model, dev, lat, &r);
        if (s != SB_OK) {
            sb_report_error(test, s);
            continue;
        }
        sb_report_value(test, r.tops, "TOPS", SB_KIND_EFFECTIVE);
        snprintf(test, sizeof(test), "%s latency (median)", dev);
        sb_report_value(test, r.ms_median, "ms", SB_KIND_MEASURED);
        snprintf(test, sizeof(test), "%s throughput", dev);
        sb_report_value(test, r.inf_per_sec, "inf/s", SB_KIND_MEASURED);
    }

cleanup:
    SB_FREE(lat);
    if (model != NULL) { ov_model_free(model); }
    ov_core_free(core);
    return SB_OK;
}

#else

static sb_status_e npu_run(void) {
    sb_report_skip("OpenVINO inference", "built without OpenVINO (see README)");
    return SB_OK;
}

#endif

const sb_section sb_section_npu = {
    .name       = "npu",
    .title      = "NPU Compute Throughput",
    .help       = SB_NPU_HELP,
    .run        = npu_run,
    .repeatable = true,
};
