# bench

A system microbenchmark suite for macOS on Apple Silicon and Linux on x86-64.
It measures CPU arithmetic, branch prediction, GPU compute, memory, storage,
process and thread overhead, TCP loopback, matrix multiplication, and ML inference.
Written in C23, with Objective-C for Metal and Core ML.

## Build

Build on the machine you want to measure. The default `-march=native` selects
that machine's instructions; the binary is not portable to older CPUs.
A current Clang with C23 support is required (Apple Clang 21 is verified).

### macOS

Install Xcode Command Line Tools, then build:

```sh
xcode-select --install
make -j
```

Metal, Foundation, Core ML, and Accelerate are supplied by macOS.
The Core ML Neural Engine mode requires macOS 13 or newer.

### Arch Linux (Intel)

```sh
sudo pacman -S --needed base-devel clang python shaderc vulkan-headers vulkan-icd-loader vulkan-intel
make -j
```

`shaderc` supplies `glslc`, which compiles the Vulkan shaders during the build.
For other GPUs, install the corresponding Vulkan driver instead of `vulkan-intel`.
A Vulkan 1.1 device is required for GPU tests; FP16 also requires shader arithmetic
and 16-bit storage-buffer support.

OpenBLAS is optional:

```sh
sudo pacman -S --needed openblas
make clean
make -j
```

The build detects OpenBLAS with `pkg-config`. Install a C23-capable compiler
before building on older distributions. Rebuild after changing compiler flags
or installing optional libraries.

## Run

Run from the repository root so generated NPU models can be found:

```sh
./bench
./bench --only cpu,gpu
./bench --only branch
./bench --skip disk,npu
./bench --help
```

The full suite normally takes about one to two minutes on recent laptops.
Run on AC power with other demanding applications idle when comparing results.
Record the commit, compiler, OS, power mode, and library/driver versions.

The disk section creates a uniquely named 2 GiB file under `build/`, writes
several GiB during measurement, and removes the file on normal completion.
It measures the filesystem containing the repository. An interrupted run can
leave its temporary file in `build/`; `make clean` removes build artifacts.

| Section | Selection | Measurement |
|---------|-----------|-------------|
| CPU | `cpu` | Scalar and SIMD multiply/add throughput; single-thread and multicore runs |
| Branch prediction | `branch` | Forced branches over predictable and random outcome streams |
| GPU | `gpu` | FP32/FP16/INT32, buffer and shared-memory bandwidth, dependent-load latency, texture sampling |
| Memory | `mem` | Latency and read bandwidth from 4 KiB to 1 GiB; stores, atomics, allocation |
| Storage | `disk` | Sequential I/O, random 4 KiB I/O, concurrent readers, write + fsync latency |
| System | `sys` | getuid, pipe process handoff, thread create/join |
| Network | `net` | TCP loopback throughput and one-byte round-trip latency |
| Matrix | `matrix` | FP32 SGEMM via Accelerate or OpenBLAS |
| NPU | `npu` | Synchronous inference through Core ML or OpenVINO |

Missing optional BLAS libraries, NPU runtimes/models, or GPU devices are reported
and skipped. Benchmarks report errors when a measurement fails.

## NPU models

Models and Python environments are generated locally and ignored by Git.
Use the generator for your OS. The macOS and Linux workloads are different;
their reported TOPS should not be compared directly.

### macOS: Core ML

The generator is verified with Python 3.12 and coremltools 9.0. This example
uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to select that
Python version; newer Python versions may lack native coremltools wheels.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python coremltools==9.0 numpy
.venv/bin/python scripts/gen_npu_model.py models/
make -j
./bench --only npu
```

This writes `models/npu_bench.mlmodel` and `models/npu_model_info.h`.
The model has ten 3×3 convolutions at 256×256 spatial resolution, with 256
intermediate channels. The generated constants override the tracked fallback
header. Core ML compiles the model at runtime.

The rows select CPU + Neural Engine, CPU + GPU, and CPU only. Core ML chooses
where individual operations execute, so the first two rows allow CPU fallback.

### Linux: OpenVINO

Install an OpenVINO SDK with C headers and `libopenvino_c`. Intel NPU use also
requires a compatible kernel, NPU driver/firmware, and the OpenVINO NPU plugin.
Installing the Python package alone does not configure the C build or NPU driver.

If the SDK supplies an `openvino.pc` file, expose it through `PKG_CONFIG_PATH`.
For an SDK without that file, pass its actual include and library paths:

```sh
make clean
make -j OPENVINO_CFLAGS='-I/path/to/openvino/runtime/include' \
  OPENVINO_LIBS='-L/path/to/openvino/runtime/lib/intel64 -Wl,-rpath,/path/to/openvino/runtime/lib/intel64 -lopenvino_c'
```

Generate the Linux model using NumPy only:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install numpy
.venv/bin/python scripts/gen_openvino_model.py models/
./bench --only npu
```

This writes `models/npu_bench.xml` and `models/npu_bench.bin`: ten 512×512 matrix
multiplications. The program lists the OpenVINO devices, tries NPU when present,
and runs a CPU baseline. Use the supplied model parameters so the operation
count matches the workload.

## Interpreting results

- Arithmetic rates count source-level operations: multiply + add is two
  operations. Instructions, SIMD widths, compiler transformations, and inference
  precision differ between backends. These are workload measurements, not
  verified hardware peak rates.
- CPU multicore rows divide total work by wall time, including thread startup
  and completion. Workers get equal iteration counts, so slower cores affect
  completion time. Linux workers use the process's allowed CPU set, up to 64;
  macOS placement is controlled by the scheduler. Single-thread runs are not pinned.
- IPC is an estimate normalized to a reported maximum clock, or an ADD-loop
  clock calibration that assumes one cycle per dependent ADD. It is not a
  hardware instruction/cycle counter measurement.
- Branch penalties are estimates from extra elapsed time. Linux uses hardware
  counters when available for the whole measurement; otherwise the table
  shows assumed miss rates. Counts include loop and unconditional branches.
- Memory bandwidth is measured by one thread. The pointer chase uses a fixed
  64-byte stride; cache-line sizes and cache topology differ across machines.
  macOS cache metadata describes the performance cluster; Linux describes CPU 0.
- GPU times include host submission and completion overhead. “VRAM” identifies
  private/device-local buffers; integrated GPUs can share physical RAM with
  host-visible buffers. Shared-memory and texture rates describe these kernels,
  including their arithmetic and cache behavior.
- Storage bypasses the OS page cache with `F_NOCACHE` or `O_DIRECT`.
  Drive/controller caches may still participate. “Burst” writes have one final
  fsync; the other random-write row syncs every write. macOS `fsync` does not
  provide the stronger drive-flush guarantee of `F_FULLFSYNC`.
- Pipe handoff includes pipe syscalls and scheduling. TCP loopback measures
  the local OS stack, not the network interface.

[FEATURES.md](FEATURES.md) lists possible additions.
