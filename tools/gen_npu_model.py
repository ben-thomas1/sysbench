#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["coremltools==9.0", "numpy"]
# ///
"""Generate the macOS Core ML convolution benchmark model.

Requires coremltools and numpy. Linux uses gen_openvino_model.py.

Usage: uv run tools/gen_npu_model.py [--neuralnetwork] <output_dir>

Creates a Conv2D-heavy model designed to saturate NPU hardware:
  10x Conv2d (3x3) layers with ReLU activations, 3 -> 256 -> ... -> 3 channels

Formats:
  default          ML Program, FP16 weights, compute and I/O, macOS 13+
                   -> npu_bench.mlpackage. Input/output: [1, 3, 256, 256] FP16.
  --neuralnetwork  legacy NeuralNetwork (spec v4) -> npu_bench.mlmodel.
                   Input/output: [3, 256, 256] FP64 multi-arrays (the builder's
                   default array type); Core ML converts to the device precision.
The bench prefers npu_bench.mlpackage and falls back to npu_bench.mlmodel.

Both formats have the same operation count (npu_model_info.h).
"""

import os
import sys

# Model parameters
NUM_LAYERS = 10
SPATIAL = 256
IN_CH = 3
MID_CH = 256
KERNEL = 3


def compute_flops():
    """Compute total FLOPs per inference."""
    flops = 0
    channels = [IN_CH] + [MID_CH] * (NUM_LAYERS - 1) + [IN_CH]
    for i in range(NUM_LAYERS):
        c_in, c_out = channels[i], channels[i + 1]
        # Conv2d: 2 * K*K * C_in * C_out * H * W
        flops += 2 * KERNEL * KERNEL * c_in * c_out * SPATIAL * SPATIAL
    return flops


def gen_coreml(output_dir):
    """Generate the legacy NeuralNetwork model (NeuralNetworkBuilder)."""
    try:
        import coremltools as ct
        from coremltools.models.neural_network import NeuralNetworkBuilder
        import coremltools.models.datatypes as datatypes
        import numpy as np
    except ImportError as e:
        print(f"  Skipping Core ML: {e}")
        return False

    rng = np.random.RandomState(42)

    # NeuralNetworkBuilder uses (C, H, W) shapes
    builder = NeuralNetworkBuilder(
        input_features=[("input", datatypes.Array(IN_CH, SPATIAL, SPATIAL))],
        output_features=[("output", datatypes.Array(IN_CH, SPATIAL, SPATIAL))],
    )

    channels = [IN_CH] + [MID_CH] * (NUM_LAYERS - 1) + [IN_CH]
    for i in range(NUM_LAYERS):
        c_in, c_out = channels[i], channels[i + 1]
        inp = "input" if i == 0 else f"relu_{i - 1}"
        conv_out = f"conv_{i}"
        # NeuralNetworkBuilder expects (H, W, C_in, C_out).
        scale = np.sqrt(2.0 / (c_in * KERNEL * KERNEL))
        W = (rng.randn(KERNEL, KERNEL, c_in, c_out) * scale).astype(np.float32)

        builder.add_convolution(
            name=f"conv_{i}",
            kernel_channels=c_in,
            output_channels=c_out,
            height=KERNEL,
            width=KERNEL,
            stride_height=1,
            stride_width=1,
            border_mode="same",
            groups=1,
            W=W,
            b=None,
            has_bias=False,
            input_name=inp,
            output_name=conv_out if i < NUM_LAYERS - 1 else "output",
        )

        if i < NUM_LAYERS - 1:
            builder.add_activation(
                name=f"relu_{i}",
                non_linearity="RELU",
                input_name=conv_out,
                output_name=f"relu_{i}",
            )
        print(f"  Layer {i + 1}/{NUM_LAYERS}: Conv2d({c_in}→{c_out}, 3x3)")

    # Select the model schema version; this does not select execution precision.
    builder.spec.specificationVersion = 4

    model = ct.models.MLModel(builder.spec, skip_model_load=True)

    mlmodel_path = os.path.join(output_dir, "npu_bench.mlmodel")
    model.save(mlmodel_path)
    size_mb = os.path.getsize(mlmodel_path) / (1024 * 1024)
    print(f"  Saved: {mlmodel_path} ({size_mb:.1f} MiB)")
    return True


def gen_mlprogram(output_dir):
    """Generate an FP16 ML Program (.mlpackage) with the same layers."""
    try:
        import coremltools as ct
        import numpy as np
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
    except ImportError as e:
        print(f"  Skipping Core ML: {e}")
        return False

    rng = np.random.RandomState(42)
    channels = [IN_CH] + [MID_CH] * (NUM_LAYERS - 1) + [IN_CH]
    weights = []
    for i in range(NUM_LAYERS):
        c_in, c_out = channels[i], channels[i + 1]
        scale = np.sqrt(2.0 / (c_in * KERNEL * KERNEL))
        # MIL conv weights are (C_out, C_in, H, W).
        weights.append((rng.randn(c_out, c_in, KERNEL, KERNEL) * scale).astype(np.float16))

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, IN_CH, SPATIAL, SPATIAL), dtype=types.fp16)],
        opset_version=ct.target.macOS13,
    )
    def prog(input):
        x = input
        for i in range(NUM_LAYERS):
            last = i == NUM_LAYERS - 1
            x = mb.conv(x=x, weight=weights[i], pad_type="same",
                        name="output" if last else f"conv_{i}")
            if not last:
                x = mb.relu(x=x, name=f"relu_{i}")
            print(f"  Layer {i + 1}/{NUM_LAYERS}: Conv2d({channels[i]}→{channels[i + 1]}, 3x3)")
        return x

    model = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS13,
    )
    path = os.path.join(output_dir, "npu_bench.mlpackage")
    model.save(path)
    print(f"  Saved: {path}")
    return True


def write_header(output_dir, flops):
    """Write C header with benchmark constants."""
    path = os.path.join(output_dir, "npu_model_info.h")
    with open(path, "w") as f:
        f.write("/* Auto-generated by gen_npu_model.py */\n")
        f.write("#ifndef NPU_MODEL_INFO_H\n")
        f.write("#define NPU_MODEL_INFO_H\n\n")
        f.write(f"#define NPU_NUM_LAYERS {NUM_LAYERS}\n")
        f.write(f"#define NPU_SPATIAL {SPATIAL}\n")
        f.write(f"#define NPU_IN_CH {IN_CH}\n")
        f.write(f"#define NPU_MID_CHANNELS {MID_CH}\n")
        f.write(f"#define NPU_OPS_PER_INFERENCE {flops}ULL\n\n")
        f.write("#endif\n")
    print(f"  Saved: {path}")


def main():
    args = sys.argv[1:]
    mlprogram = "--neuralnetwork" not in args
    args = [a for a in args if a != "--neuralnetwork"]
    if len(args) != 1:
        print(f"Usage: {sys.argv[0]} [--neuralnetwork] <output_dir>", file=sys.stderr)
        sys.exit(1)

    output_dir = args[0]
    os.makedirs(output_dir, exist_ok=True)

    flops = compute_flops()
    print(f"Generating NPU benchmark model...")
    print(f"  Architecture: {NUM_LAYERS}x Conv2d(3x3) @ {SPATIAL}x{SPATIAL}, {MID_CH}ch")
    print(f"  FLOPs/inference: {flops:,}")
    print()

    print("--- Core ML (Apple ANE) ---")
    if not (gen_mlprogram(output_dir) if mlprogram else gen_coreml(output_dir)):
        sys.exit(1)
    print()

    write_header(output_dir, flops)
    print("Done.")


if __name__ == "__main__":
    main()
