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
A Vulkan 1.1 device is required for GPU tests; FP16 also requires `shaderFloat16`
(VK_KHR_shader_float16_int8). The section prefers a discrete GPU over an integrated
one and prints the device it chose; set `SB_GPU_DEVICE=<index|name substring>` to
pick another (the index is the position in the printed device list).

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

The disk section creates a uniquely named 2 GiB file under `build/` and
unlinks it immediately, so even an interrupted run leaves nothing behind. It
writes about 3.2 GiB per pass, needs 3 GiB free, and takes about 15 s. It
measures the filesystem containing the repository.

| Section | Selection | Measurement |
|---------|-----------|-------------|
| CPU | `cpu` | Peak scalar and SIMD multiply-add throughput per type; 1 thread, all cores, P/E cores; ops per cycle |
| Branch prediction | `branch` | One asm branch over predictable and random outcome streams; mispredict penalty by direction |
| GPU | `gpu` | FP32/FP16/INT32, buffer and shared-memory bandwidth, dependent-load latency, texture sampling |
| Memory | `mem` | Latency 4 KiB to 1 GiB, TLB walk cost, read bandwidth (1 thread and multithreaded DRAM), memcpy, stores, atomics, allocation |
| Storage | `disk` | Sequential and random 4 KiB direct I/O, 1–32 threads, durable vs non-durable sync latency |
| System | `sys` | getuid and open+close cost, pipe handoff (blocking and busy-poll), shared cache-line handoff, thread create/join |
| Network | `net` | TCP loopback and Unix socketpair throughput and one-byte round-trip latency (blocking and busy-poll) |
| Matrix | `matrix` | SGEMM/DGEMM library throughput via Accelerate or OpenBLAS, result-checked |
| NPU | `npu` | Synchronous inference through Core ML (with placement) or OpenVINO |

Missing optional BLAS libraries, NPU runtimes/models, or GPU devices are reported
and skipped. Benchmarks report errors when a measurement fails.

## NPU models

Models and Python environments are generated locally and ignored by Git.
Use the generator for your OS. The macOS and Linux workloads are different;
their reported TOPS should not be compared directly.

### macOS: Core ML

The generator is verified with Python 3.12 and coremltools 9.0. Its inline
script metadata pins both, so [uv](https://docs.astral.sh/uv/getting-started/installation/)
can run it directly (newer Python versions may lack native coremltools wheels):

```sh
uv run tools/gen_npu_model.py models/
make clean
make release
./bench --only npu
```

This writes `models/npu_bench.mlpackage` (an FP16 ML Program, FP16 input and
output) and `models/npu_model_info.h`. `--neuralnetwork` writes the legacy
`models/npu_bench.mlmodel` (NeuralNetwork, FP64 input and output) instead; the
bench uses the `.mlpackage` when present and falls back to the `.mlmodel`.
Both have ten 3×3 convolutions at 256×256 spatial resolution with 256
intermediate channels and the same operation count, written to the generated
header (which overrides the tracked fallback header). Core ML compiles the
model at runtime.

The rows select CPU + Neural Engine, CPU + GPU, and CPU only. Core ML chooses
where individual operations execute, so the first two settings allow CPU
fallback. On macOS 14.4 and later the section prints the planned device of
every layer or operation (MLComputePlan) as counts, so fallback is visible.
TOPS are *effective*: model operations divided by time. Core ML can execute
fewer operations than the model nominally contains (for example Winograd 3×3
convolution), so CPU + GPU can exceed the GPU's FMA peak. Latency is the median
of single synchronous predictions and includes Core ML's input/output handling.

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
multiplications. The program lists the OpenVINO devices, runs NPU and GPU when
listed, and always runs a CPU baseline, each compiled with the `LATENCY`
performance hint. TOPS are effective (model operations divided by time), and
the plugin selects the precision (printed per device). Use the supplied model
parameters so the operation count matches the workload.

## Interpreting results

- Arithmetic rates count a multiply-add as two operations (an int8 dot product
  counts two per 8-bit multiply-accumulate). Instructions, SIMD widths and
  inference precision differ between backends. Rows tagged `[peak]` come from
  kernels built to saturate one unit; `[estimate]` rows depend on a stated
  assumption; `[effective]` rows count nominal work.
- CPU `[peak]` rows are inline-asm loops with enough independent accumulator
  chains to hide the multiply-add latency (20 on AArch64, 12 or 16 on x86), so the
  instruction mix does not depend on compiler flags. The row label names the
  instruction. Scalar integer is a single 64-bit row: narrower C integer types
  use the same scalar multiplier. The x86 SIMD tier (SSE2, AVX2, AVX-512) is
  fixed at build time by `-march`; SSE2 and integer lanes have no fused
  multiply-add, so those rows time a multiply and an add per step.
- CPU multicore rows are time-based: every thread runs the kernel for the same
  window after a warmup, and the total is the sum of per-thread rates. On hybrid
  CPUs the P-cores and E-cores are also measured separately. macOS has no
  pinning, so P = USER_INTERACTIVE and E = BACKGROUND QoS. macOS runs BACKGROUND
  threads at a reduced E-cluster clock and shares those cores with system daemons,
  so the macOS E rows understate the E-core peak. The all-core rows include the
  E-cores at full clock. Linux pins P/E threads by affinity; all-core threads are
  unpinned.
- "Ops per cycle" divides by a clock measured with a dependent integer add chain,
  which assumes one cycle per add (an estimate). On Linux, when perf_event is
  permitted, user-mode cycle and instruction counters replace that clock, and
  real IPC is reported as measured. macOS counters need root.
- Branch: both outcomes of the measured branch run the same instructions and
  one taken branch, so time above the always/never-taken mean is mispredict
  cost. Each iteration retires 3 branches (measured, path, loop back-edge).
  Miss counts are measured with Linux perf_event (user space; partial or
  multiplexed reads are rejected) or macOS kperf when run as root. Otherwise
  random streams are assumed to miss once per minority outcome, the other
  counts are implied from the taken-miss penalty, and both are tagged
  `[estimate]`. Penalties are split by direction: missing a branch that is
  usually taken costs more on Apple M4 (~21 vs ~14 cycles), which is why the
  50% stream lands in between. Cycles come from the PMU when counted,
  otherwise from a dependent add-chain clock estimate. Periodic patterns may
  or may not be learned depending on the core and even the code layout; read
  their miss counts rather than assuming either.
- Memory latency is a pointer chase through one random cycle with a stride of
  the detected cache line (128 B on Apple Silicon, 64 B on x86). Large sizes
  also miss the TLB; the TLB rows touch one line per page with data that stays
  in cache, and "Page walk" (an estimate) is how much of each large-size
  latency is address translation. Linux runs use base pages (THP disabled for
  the test buffer). macOS cache metadata describes the performance cluster;
  Linux describes CPU 0.
- Read bandwidth rows are kernels built to saturate the load path (`[peak]`):
  one thread over a size sweep, then DRAM with several threads, each streaming
  its own slice for a fixed time window. memcpy rows count bytes read plus bytes
  written (twice the bytes copied); write-allocate traffic is not counted.
- Store rows are the better of two loop shapes per size. Some cores (Apple
  M-series) only skip reading a line before overwriting it for certain loop
  shapes, so the slower shape is shown at DRAM size too. "Non-temporal" is a
  hint on AArch64 (`stnp`) and a cache-bypassing store on x86 (`movnt*`).
- Atomic latency rows are dependent chains on one cache line; throughput rows
  are independent adds spread over 8 lines. Contended rows put every thread on
  one line: "aggregate" is total increments per second, "per thread" is the
  average time each thread waits per increment (threads / aggregate).
- Allocation rows are libc malloc+free pairs (the same-size fast path) and a
  held batch of mixed sizes freed in random order, in ns per pair.
- GPU rows are timed with GPU timestamps (Metal `GPUStartTime`/`GPUEndTime`,
  Vulkan timestamp queries) over command buffers of ~100 ms containing several
  dispatches; the fastest of three is shown. Submission overhead is excluded.
  The same kernels run on Metal and Vulkan and produce the same rows.
- GPU compute `[peak]` rows use 32 independent FMA chains per thread on
  float2/half2, which saturates the ALUs (M4 Pro: ~8.0 TFLOPS of 8.08
  theoretical at 1578 MHz). Apple GPUs run FP16 at the FP32 rate; GPUs with
  packed FP16 report about twice the FP32 rate. INT32 is a 32-bit multiply-add
  with runtime operands; integer multiply is quarter rate on Apple GPUs.
- GPU memory bandwidth is a coalesced grid-stride read of a 1 GiB buffer. The
  host-visible row appears only on discrete GPUs, where it measures system
  memory across the bus; on unified memory (Apple, integrated GPUs) it is
  skipped because it is the same DRAM as device-local memory.
- GPU latency is one thread chasing a random cycle of nodes 128 B apart. Small
  sizes show the GPU caches; the largest sizes include TLB misses.
- Threadgroup (shared) memory is the best case: lane-contiguous float4 reads
  whose 32-lane footprint is aligned to 512 B, with no bank conflicts. Real
  kernels often get much less. On the M4 Pro, the same read shifted off that
  alignment by one element drops about 3×, and a 32-bit lane stride of 2/4/32
  words drops 2×/3.5×/20×.
- Texture “cache-resident” is the bilinear filtering rate with one RGBA8 texel
  per thread over a 16 KiB texture (texture unit throughput). “Streaming”
  samples a 256 MiB texture once, so it measures the texture path from DRAM in
  GB/s and is comparable with the buffer bandwidth row. The texture holds random
  (incompressible) data: GPUs that compress textures losslessly read compressible
  content faster than DRAM bandwidth (M4 Pro: ~450 GB/s for an all-zero texture).
- Storage bypasses the OS page cache with `F_NOCACHE` or `O_DIRECT`.
  Drive/controller caches may still participate. “N threads” rows are N threads
  that each keep one synchronous I/O in flight, not asynchronous queue depth;
  a single QD1 stream underestimates sequential read (use the 4-thread row).
  Rows marked “durable” include a drive cache flush in the timing:
  `F_FULLFSYNC` on macOS, `fdatasync` on Linux. On macOS, `fsync` and
  `F_BARRIERFSYNC` only reach the drive's volatile cache; those rows are marked
  NOT durable (about 30× and 3× faster than a real flush on an M4 Pro). Random-write rates
  depend on the drive's internal state and vary by ±30% between runs.
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
- Matrix GFLOPS are library throughput (2n³ / time, median of short batches
  after a warmup), not a hardware peak. The library chooses kernels and
  threads: Accelerate uses the SME matrix unit on M4-class CPUs, OpenBLAS
  prints its build configuration and thread count (`OPENBLAS_NUM_THREADS`
  overrides it). The single-thread rows show threading overhead at small
  sizes. Each result is checked against an FP64 reference outside the timed
  region; a wrong result replaces the number with `WRONG RESULT`.

[FEATURES.md](FEATURES.md) lists possible additions.
