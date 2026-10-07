/* Legacy section adapter: the old module prints its own output.
 * Replace this file when the module is rewritten on the core API. */
#include "gpu/gpu.h"
#include "bench.h"

static sb_status_e run(void) {
    bench_gpu();
    return SB_OK;
}

const sb_section sb_section_gpu = {
    .name       = "gpu",
    .title      = "GPU Compute Throughput",
    .help       =
                  "  Metal on macOS; Vulkan 1.1 on Linux.\n"
                  "  FP32/FP16/INT32 arithmetic, private/device-local and host-visible\n"
                  "  buffer bandwidth, dependent pointer-chase latency, shared-memory\n"
                  "  bandwidth, and bilinear sampling of a 1024x1024 RGBA8 texture.\n"
                  "  FP16 on Vulkan requires float16 arithmetic and 16-bit buffer storage.\n"
                  "  Timings include command submission and waiting for completion.\n"
                  "  Integrated GPUs may use the same physical RAM for both buffer types.\n",
    .run        = run,
    .repeatable = false,
};
