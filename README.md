# bench

A system microbenchmark suite for macOS on Apple Silicon and Linux on x86-64.
It measures CPU arithmetic, branch prediction, GPU compute, memory, storage,
process and thread overhead, TCP loopback, matrix multiplication, and ML inference.
Written in C23, with Objective-C for Metal and Core ML.

## Build

Build on the machine you want to measure. The default `-march=native` selects
that machine's instructions; the binary is not portable to older CPUs.
A current Clang with C23 support is required (Apple Clang 21 and clang 19 are
verified). The build uses Meson and Ninja; `make` targets wrap them:

| Command | Result |
|---------|--------|
| `make release` | optimized build in `build/release/`, linked as `./bench` — use this for numbers |
| `make` | debug build with sanitizers and strict warnings, linked as `./bench-debug` |
| `make smoke` | x86-64 Linux build and run inside Docker (build/run check only) |
| `make clean` | remove `build/` and the links |

Pass `-Dmarch=<cpu>` to `meson setup` to target something other than `native`.

### macOS

Install Xcode Command Line Tools and Meson, then build:

```sh
xcode-select --install
brew install meson ninja
make release
```

Metal, Foundation, Core ML, and Accelerate are supplied by macOS.
The Core ML Neural Engine mode requires macOS 13 or newer.

### Arch Linux (Intel)

```sh
sudo pacman -S --needed base-devel clang meson ninja python shaderc vulkan-headers vulkan-icd-loader vulkan-intel
make release
```

`shaderc` supplies `glslc`, which compiles the Vulkan shaders during the build.
For other GPUs, install the corresponding Vulkan driver instead of `vulkan-intel`.
A Vulkan 1.1 device is required for GPU tests; FP16 also requires shader arithmetic
and 16-bit storage-buffer support.

OpenBLAS is optional:

```sh
sudo pacman -S --needed openblas
make clean
make release
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
./bench --repeat 5      # each section 5 times; median/min/max
./bench --help
```

Values tagged `[peak]` are hardware ceilings, `[effective]` are nominal work
divided by time, and `[estimate]` depend on stated assumptions.

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
| System | `sys` | getuid and open+close cost, pipe handoff (blocking and busy-poll), shared cache-line handoff, thread create/join |
| Network | `net` | TCP loopback and Unix socketpair throughput and one-byte round-trip latency (blocking and busy-poll) |
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
.venv/bin/python tools/gen_npu_model.py models/
make clean
make release
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
meson setup build/release --buildtype=release -Db_lto=true -Db_pie=true \
  -Dopenvino_include=/path/to/openvino/runtime/include \
  -Dopenvino_libdir=/path/to/openvino/runtime/lib/intel64
make release
```

Generate the Linux model using NumPy only:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install numpy
.venv/bin/python tools/gen_openvino_model.py models/
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
- `sys`: getuid is the bare kernel-trap cost; open+close of `/dev/null` is a
  file path that endpoint-security software hooks, so a large gap between the
  two points at such software (the section lists active macOS endpoint-security
  extensions). A handoff is half a one-byte ping-pong round trip. "Blocking"
  pipe handoffs run between two processes and include putting the reader to
  sleep and waking it; "busy-poll" spins on `O_NONBLOCK` reads, leaving pipe
  transport and syscall cost; the shared cache line row is two threads
  bouncing an atomic, the cross-core floor under any IPC. Ping-pong peers are
  pinned to two CPUs that are not SMT siblings on Linux and use QoS
  USER_INTERACTIVE on macOS. Thread create+join covers creation, the first
  schedule, exit and join of an empty thread.
- `net` measures the local OS stack, not a network interface: TCP over
  127.0.0.1, with an `AF_UNIX` socketpair as a non-network reference.
  Throughput streams 1 MiB writes with default socket buffers (Linux
  autotuning stays on). RTT rows are one-byte ping-pongs with `TCP_NODELAY`;
  blocking minus busy-poll is the sleep/wakeup cost. Network-filter software
  adds to TCP even when busy-polling; on macOS the section lists active
  network-filter extensions. A busy-poll TCP RTT far above the Unix one points
  at such a filter.

[FEATURES.md](FEATURES.md) lists possible additions.
