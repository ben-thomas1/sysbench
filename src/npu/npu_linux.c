#include "bench.h"
#include "timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HAS_OPENVINO

#include <openvino/c/openvino.h>

#define NPU_MATMUL_SIZE 512
#define NPU_NUM_LAYERS 10
#define NPU_INPUT_ELEMS (NPU_MATMUL_SIZE * NPU_MATMUL_SIZE)
#define NPU_OPS_PER_INFERENCE (NPU_NUM_LAYERS * 2ULL * NPU_MATMUL_SIZE * NPU_MATMUL_SIZE * NPU_MATMUL_SIZE)
#define TARGET_NS 2000000000ULL

static const char *status_info(ov_status_e status) {
    const char *info = ov_get_error_info(status);
    return info ? info : "unknown error";
}

static double run_inference_bench(ov_core_t *core, ov_model_t *model,
                                  const char *device, size_t *out_passes,
                                  char *err, size_t err_len) {
    ov_compiled_model_t *compiled = NULL;
    ov_infer_request_t *request = NULL;
    ov_tensor_t *input_tensor = NULL;
    ov_output_const_port_t *input_port = NULL;
    ov_shape_t shape = {0};
    double result = -1.0;
    ov_status_e status = OK;

    if (err && err_len > 0) err[0] = '\0';

    status = ov_model_const_input(model, &input_port);
    if (status != OK) {
        if (err) snprintf(err, err_len, "model input: %s", status_info(status));
        goto cleanup;
    }

    status = ov_const_port_get_shape(input_port, &shape);
    if (status != OK) {
        if (err) snprintf(err, err_len, "input shape: %s", status_info(status));
        goto cleanup;
    }

    ov_element_type_e input_type;
    status = ov_port_get_element_type(input_port, &input_type);
    if (status != OK || input_type != F32 || shape.rank != 2 ||
        shape.dims[0] != NPU_MATMUL_SIZE || shape.dims[1] != NPU_MATMUL_SIZE) {
        if (err) snprintf(err, err_len,
                          "expected float32 [512,512]; regenerate with gen_openvino_model.py");
        goto cleanup;
    }

    status = ov_tensor_create(F32, shape, &input_tensor);
    if (status != OK) {
        if (err) snprintf(err, err_len, "input tensor: %s", status_info(status));
        goto cleanup;
    }

    /* Fill input */
    float *data = NULL;
    status = ov_tensor_data(input_tensor, (void **)&data);
    if (status != OK || !data) {
        if (err) snprintf(err, err_len, "tensor data: %s", status_info(status));
        goto cleanup;
    }
    for (int i = 0; i < NPU_INPUT_ELEMS; i++)
        data[i] = 0.001f * (float)i;

    status = ov_core_compile_model(core, model, device, 0, &compiled);
    if (status != OK) {
        if (err) snprintf(err, err_len, "compile on %s: %s", device, status_info(status));
        goto cleanup;
    }

    status = ov_compiled_model_create_infer_request(compiled, &request);
    if (status != OK) {
        if (err) snprintf(err, err_len, "infer request: %s", status_info(status));
        goto cleanup;
    }

    status = ov_infer_request_set_input_tensor(request, input_tensor);
    if (status != OK) {
        if (err) snprintf(err, err_len, "set input tensor: %s", status_info(status));
        goto cleanup;
    }

    /* Warmup */
    for (int w = 0; w < 3; w++) {
        status = ov_infer_request_infer(request);
        if (status != OK) {
            if (err) snprintf(err, err_len, "warmup infer: %s", status_info(status));
            goto cleanup;
        }
    }

    /* Calibrate */
    uint64_t tc0 = timer_ns();
    status = ov_infer_request_infer(request);
    if (status != OK) {
        if (err) snprintf(err, err_len, "calibration infer: %s", status_info(status));
        goto cleanup;
    }
    uint64_t tc1 = timer_ns();
    uint64_t one_pass = tc1 - tc0;
    size_t passes = 1;
    if (one_pass > 0) passes = (size_t)(TARGET_NS / one_pass) + 1;
    if (passes < 2) passes = 2;

    /* Timed run */
    uint64_t t0 = timer_ns();
    for (size_t p = 0; p < passes; p++) {
        status = ov_infer_request_infer(request);
        if (status != OK) {
            if (err) snprintf(err, err_len, "timed infer: %s", status_info(status));
            goto cleanup;
        }
    }
    uint64_t t1 = timer_ns();

    if (out_passes) *out_passes = passes;
    double elapsed_s = (double)(t1 - t0) / 1e9;
    if (elapsed_s > 0)
        result = (double)NPU_OPS_PER_INFERENCE * passes / elapsed_s / 1e12;

cleanup:
    if (request)      ov_infer_request_free(request);
    if (compiled)     ov_compiled_model_free(compiled);
    if (input_tensor) ov_tensor_free(input_tensor);
    if (input_port)   ov_output_const_port_free(input_port);
    ov_shape_free(&shape);
    return result;
}

static int print_available_devices(ov_core_t *core) {
    ov_available_devices_t devices = {0};
    int has_npu = 0;

    ov_status_e status = ov_core_get_available_devices(core, &devices);
    if (status != OK) {
        printf("  Available OpenVINO devices: unavailable (%s)\n", status_info(status));
        return 0;
    }

    printf("  Available OpenVINO devices:");
    if (devices.size == 0) {
        printf(" none");
    }
    for (size_t i = 0; i < devices.size; i++) {
        const char *name = devices.devices[i];
        printf(" %s", name);
        if (name && strncmp(name, "NPU", 3) == 0)
            has_npu = 1;
    }
    printf("\n");

    ov_available_devices_free(&devices);
    return has_npu;
}

void bench_npu(void) {
    printf("=== NPU Compute Throughput (OpenVINO) ===\n");

    /* Check model files exist */
    FILE *f = fopen("models/npu_bench.xml", "r");
    if (!f) {
        printf("  Model not found, skipping\n");
        printf("  (generate with: python3 tools/gen_openvino_model.py models/)\n");
        return;
    }
    fclose(f);

    ov_core_t *core = NULL;
    ov_model_t *model = NULL;

    if (ov_core_create(&core) != OK) {
        printf("  Failed to create OpenVINO core\n");
        return;
    }

    if (ov_core_read_model(core, "models/npu_bench.xml", "models/npu_bench.bin",
                           &model) != OK) {
        printf("  Failed to load model\n");
        ov_core_free(core);
        return;
    }

    printf("  Model: %dx matmul %dx%d\n", NPU_NUM_LAYERS,
           NPU_MATMUL_SIZE, NPU_MATMUL_SIZE);
    printf("  TOPS counts model multiply/add operations; execution precision is framework-selected.\n");
    int has_npu = print_available_devices(core);
    printf("%-20s %14s\n", "Test", "Throughput");
    printf("%-20s %14s\n", "----", "----------");

    /* Try NPU device */
    size_t npu_passes = 0;
    char npu_error[256];
    double npu_tops = has_npu ? run_inference_bench(core, model, "NPU", &npu_passes,
                                                    npu_error, sizeof(npu_error)) : -1.0;
    if (npu_tops > 0) {
        printf("%-20s %10.2f TOPS\n", "NPU", npu_tops);
    } else {
        printf("%-20s %14s\n", "NPU", "not available");
        if (!has_npu)
            printf("  NPU detail: OpenVINO did not list an NPU device\n");
        else if (npu_error[0] != '\0')
            printf("  NPU detail: %s\n", npu_error);
    }
    fflush(stdout);

    /* CPU baseline */
    size_t cpu_passes = 0;
    char cpu_error[256];
    double cpu_tops = run_inference_bench(core, model, "CPU", &cpu_passes,
                                          cpu_error, sizeof(cpu_error));
    if (cpu_tops > 0)
        printf("%-20s %10.2f TOPS\n", "CPU only", cpu_tops);
    else
        printf("%-20s %14s\n  CPU detail: %s\n", "CPU only", "error", cpu_error);
    fflush(stdout);

    if (npu_tops > 0 && cpu_tops > 0) {
        double speedup = npu_tops / cpu_tops;
        printf("%-20s %10.1fx\n", "NPU speedup", speedup);
    }
    if (npu_tops > 0 && npu_passes > 0) {
        double ms_per = (double)NPU_OPS_PER_INFERENCE / (npu_tops * 1e12) * 1e3;
        printf("%-20s %10.2f ms/inference\n", "NPU latency", ms_per);
    }
    fflush(stdout);

    ov_model_free(model);
    ov_core_free(core);
}

#else

void bench_npu(void) {
    printf("=== NPU Compute Throughput ===\n");
    printf("  Not available (requires OpenVINO runtime)\n");
}

#endif
