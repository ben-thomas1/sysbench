/* Legacy section adapter: the old module prints its own output.
 * Replace this file when the module is rewritten on the core API. */
#include "npu/npu.h"
#include "bench.h"

static sb_status_e run(void) {
    bench_npu();
    return SB_OK;
}

const sb_section sb_section_npu = {
    .name       = "npu",
    .title      = "NPU Compute Throughput",
    .help       =
                  "  macOS: ten convolutions, 256x256 spatial size, 256 intermediate channels.\n"
                  "    uv run tools/gen_npu_model.py models/   (coremltools + numpy)\n"
                  "    CPU + Neural Engine, CPU + GPU, and CPU-only compute-unit modes.\n"
                  "    Core ML may use CPU fallback within the first two modes.\n"
                  "  Linux: ten 512x512 matrix multiplications, via optional OpenVINO C API.\n"
                  "    uv run tools/gen_openvino_model.py models/   (numpy)\n"
                  "    NPU when available, plus a CPU baseline.\n"
                  "  Models are different across OSes; TOPS is not directly comparable.\n"
                  "  Execution precision is framework-selected. Run from the repo root.\n",
    .run        = run,
    .repeatable = false,
};
