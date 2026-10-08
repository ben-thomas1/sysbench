#pragma once

#include "core/section.h"

extern const sb_section sb_section_npu;

/* Shared by npu_macos.m and npu_linux.c. */
#define SB_NPU_HELP                                                                    \
    "  macOS: Core ML, ten 3x3 convolutions at 256x256 with 256 channels, run\n"       \
    "    synchronously under CPU + Neural Engine, CPU + GPU and CPU only.\n"           \
    "    Placement per layer is printed from MLComputePlan (macOS 14.4+), so CPU\n"    \
    "    fallback is visible. Model: uv run tools/gen_npu_model.py models/\n"          \
    "  Linux: OpenVINO C API, ten 512x512 matrix multiplications on NPU (when\n"       \
    "    listed) and CPU. Model: uv run tools/gen_openvino_model.py models/\n"         \
    "  TOPS are effective (model ops / time), not hardware ops; the two OSes run\n"    \
    "  different models. Latency is the median of single synchronous inferences.\n"   \
    "  Run from the repo root.\n"
