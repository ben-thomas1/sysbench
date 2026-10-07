#!/usr/bin/env python3
"""Generate an OpenVINO IR model (.xml + .bin) for NPU benchmarking.

No dependencies beyond Python stdlib + numpy.
Creates a chain of MatMul layers for the Linux benchmark. The Core ML
benchmark uses a different convolution model; their TOPS are not comparable.

Usage: python3 gen_openvino_model.py <output_dir>
"""

import os
import sys
import xml.etree.ElementTree as ET

SIZE = 512
NUM_LAYERS = 10
OPS_PER_INFERENCE = NUM_LAYERS * 2 * SIZE * SIZE * SIZE


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_dir>", file=sys.stderr)
        sys.exit(1)

    output_dir = sys.argv[1]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating OpenVINO NPU benchmark model ({NUM_LAYERS}x matmul {SIZE}x{SIZE})...")

    import numpy as np
    rng = np.random.RandomState(42)
    scale = 1.0 / (SIZE ** 0.5)

    # Build weight binary blob
    bin_data = bytearray()
    weight_offsets = []  # (offset, size) for each layer's weights

    for i in range(NUM_LAYERS):
        W = (rng.randn(SIZE, SIZE) * scale).astype("<f4")
        offset = len(bin_data)
        w_bytes = W.tobytes()
        bin_data.extend(w_bytes)
        weight_offsets.append((offset, len(w_bytes)))
        print(f"  Layer {i + 1}/{NUM_LAYERS}")

    # Build OpenVINO IR XML (version 11)
    net = ET.Element("net", name="npu_bench", version="11")
    layers = ET.SubElement(net, "layers")
    edges = ET.SubElement(net, "edges")

    layer_id = 0

    # Input parameter: a full matrix so the measured work matches the
    # benchmark's 2*N^3 MatMul operation count.
    param = ET.SubElement(layers, "layer", id=str(layer_id), name="input", type="Parameter", version="opset1")
    data = ET.SubElement(param, "data", shape=f"{SIZE},{SIZE}", element_type="f32")
    out_port = ET.SubElement(param, "output")
    ET.SubElement(out_port, "port", id="0", precision="FP32").text = ""
    # Add dimension info
    port = out_port.find("port")
    ET.SubElement(port, "dim").text = str(SIZE)
    ET.SubElement(port, "dim").text = str(SIZE)
    input_layer_id = layer_id
    layer_id += 1

    prev_layer_id = input_layer_id
    prev_port = "0"

    for i in range(NUM_LAYERS):
        offset, size = weight_offsets[i]

        # Constant node for weights
        const_id = layer_id
        const = ET.SubElement(layers, "layer", id=str(const_id), name=f"weights_{i}", type="Const", version="opset1")
        ET.SubElement(const, "data",
                      element_type="f32",
                      shape=f"{SIZE},{SIZE}",
                      offset=str(offset),
                      size=str(size))
        const_out = ET.SubElement(const, "output")
        const_port = ET.SubElement(const_out, "port", id="0", precision="FP32")
        ET.SubElement(const_port, "dim").text = str(SIZE)
        ET.SubElement(const_port, "dim").text = str(SIZE)
        layer_id += 1

        # MatMul node
        matmul_id = layer_id
        matmul = ET.SubElement(layers, "layer", id=str(matmul_id), name=f"matmul_{i}", type="MatMul", version="opset1")
        ET.SubElement(matmul, "data", transpose_a="false", transpose_b="false")
        matmul_in = ET.SubElement(matmul, "input")
        # Input port 0: from previous layer
        p0 = ET.SubElement(matmul_in, "port", id="0", precision="FP32")
        ET.SubElement(p0, "dim").text = str(SIZE)
        ET.SubElement(p0, "dim").text = str(SIZE)
        # Input port 1: weights
        p1 = ET.SubElement(matmul_in, "port", id="1", precision="FP32")
        ET.SubElement(p1, "dim").text = str(SIZE)
        ET.SubElement(p1, "dim").text = str(SIZE)
        # Output port
        matmul_out = ET.SubElement(matmul, "output")
        p_out = ET.SubElement(matmul_out, "port", id="2", precision="FP32")
        ET.SubElement(p_out, "dim").text = str(SIZE)
        ET.SubElement(p_out, "dim").text = str(SIZE)
        layer_id += 1

        # Edges: prev -> matmul port 0, const -> matmul port 1
        ET.SubElement(edges, "edge",
                      **{"from-layer": str(prev_layer_id), "from-port": prev_port,
                         "to-layer": str(matmul_id), "to-port": "0"})
        ET.SubElement(edges, "edge",
                      **{"from-layer": str(const_id), "from-port": "0",
                         "to-layer": str(matmul_id), "to-port": "1"})

        prev_layer_id = matmul_id
        prev_port = "2"

    # Result node
    result_id = layer_id
    result = ET.SubElement(layers, "layer", id=str(result_id), name="output", type="Result", version="opset1")
    result_in = ET.SubElement(result, "input")
    rp = ET.SubElement(result_in, "port", id="0", precision="FP32")
    ET.SubElement(rp, "dim").text = str(SIZE)
    ET.SubElement(rp, "dim").text = str(SIZE)
    ET.SubElement(edges, "edge",
                  **{"from-layer": str(prev_layer_id), "from-port": prev_port,
                     "to-layer": str(result_id), "to-port": "0"})

    # Write files
    xml_path = os.path.join(output_dir, "npu_bench.xml")
    bin_path = os.path.join(output_dir, "npu_bench.bin")

    tree = ET.ElementTree(net)
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="unicode", xml_declaration=True)

    with open(bin_path, "wb") as f:
        f.write(bin_data)

    xml_size = os.path.getsize(xml_path) / 1024
    bin_size = os.path.getsize(bin_path) / (1024 * 1024)
    print(f"  Saved: {xml_path} ({xml_size:.1f} KiB)")
    print(f"  Saved: {bin_path} ({bin_size:.1f} MiB)")
    print(f"Done. Ops/inference: {OPS_PER_INFERENCE:,}")


if __name__ == "__main__":
    main()
