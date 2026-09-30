#include "bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const char *fmt_size(size_t bytes) {
    static char buf[32];
    if (bytes >= 1024UL * 1024 * 1024)
        snprintf(buf, sizeof(buf), "%zu GiB", bytes / (1024UL * 1024 * 1024));
    else if (bytes >= 1024 * 1024)
        snprintf(buf, sizeof(buf), "%zu MiB", bytes / (1024 * 1024));
    else
        snprintf(buf, sizeof(buf), "%zu KiB", bytes / 1024);
    return buf;
}

/* --- Test registry --- */

struct bench_entry {
    const char *name;
    void (*run)(void);
};

static const struct bench_entry all_benchmarks[] = {
    {"cpu",    bench_cpu},
    {"branch", bench_branch},
    {"gpu",    bench_gpu},
    {"mem",    bench_memory},
    {"disk",   bench_disk},
    {"sys",    bench_sys},
    {"net",    bench_net},
    {"matrix", bench_matrix},
    {"npu",    bench_npu},
};

#define NUM_BENCHMARKS (sizeof(all_benchmarks) / sizeof(all_benchmarks[0]))

static int find_benchmark(const char *name) {
    for (int i = 0; i < (int)NUM_BENCHMARKS; i++)
        if (strcmp(name, all_benchmarks[i].name) == 0)
            return i;
    return -1;
}

/* Parse comma-separated list of benchmark names into a selection bitmask */
static int parse_list(const char *list, int selected[NUM_BENCHMARKS]) {
    char buf[256];
    size_t len = strlen(list);
    if (len == 0 || len >= sizeof(buf) || list[0] == ',' || list[len - 1] == ',' ||
        strstr(list, ",,")) {
        fprintf(stderr, "Invalid benchmark list: expected comma-separated names\n");
        return -1;
    }
    memcpy(buf, list, len + 1);

    char *tok = strtok(buf, ",");
    while (tok) {
        int idx = find_benchmark(tok);
        if (idx < 0) {
            fprintf(stderr, "Unknown benchmark: %s\n", tok);
            return -1;
        }
        selected[idx] = 1;
        tok = strtok(NULL, ",");
    }
    return 0;
}

static void usage(void) {
    fprintf(stderr,
        "Usage: bench [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  --only <list>  Run only the listed benchmarks (comma-separated)\n"
        "  --skip <list>  Skip the listed benchmarks (comma-separated)\n"
        "  --help         Show detailed help\n"
        "\n"
        "Benchmarks: cpu, branch, gpu, mem, disk, sys, net, matrix, npu\n"
        "\n"
        "Examples:\n"
        "  bench                        Run all benchmarks\n"
        "  bench --only cpu,gpu         Run only CPU and GPU\n"
        "  bench --skip disk,npu        Run all except disk and NPU\n"
    );
}

static void detailed_help(void) {
    printf(
        "bench - System Benchmark Suite\n"
        "\n"
        "Usage: bench [--only <list> | --skip <list>] [--help]\n"
        "Names: cpu, branch, gpu, mem, disk, sys, net, matrix, npu\n"
        "Lists must contain comma-separated names, without empty entries.\n"
        "\n"
        "CPU (cpu)\n"
        "  FP64/FP32 and INT64/INT32/INT16/INT8 multiply-add throughput.\n"
        "  Single-thread scalar/SIMD and separate multicore runs. Multiply + add\n"
        "  counts as two operations. Unsupported SIMD types print n/a.\n"
        "  Backends: NEON, SSE2, AVX2, AVX-512, selected at build time.\n"
        "  Multicore: equal work per worker, total work / wall time, up to 64 workers.\n"
        "  Linux respects allowed CPU IDs; macOS placement is scheduler-controlled.\n"
        "  IPC estimates use a reference clock, not hardware cycle counters.\n"
        "\n"
        "Branch (branch)\n"
        "  Forced conditional branches over a 1 MiB outcome stream.\n"
        "  Predictable, periodic, and random-biased patterns; best of five trials.\n"
        "  Penalty estimates divide extra time by measured or assumed miss rates.\n"
        "  Linux perf counters may be unavailable or multiplexed.\n"
        "\n"
        "GPU (gpu)\n"
        "  Metal on macOS; Vulkan 1.1 on Linux.\n"
        "  FP32/FP16/INT32 arithmetic, private/device-local and host-visible\n"
        "  buffer bandwidth, dependent pointer-chase latency, shared-memory\n"
        "  bandwidth, and bilinear sampling of a 1024x1024 RGBA8 texture.\n"
        "  FP16 on Vulkan requires float16 arithmetic and 16-bit buffer storage.\n"
        "  Timings include command submission and waiting for completion.\n"
        "  Integrated GPUs may use the same physical RAM for both buffer types.\n"
        "\n"
        "Memory (mem)\n"
        "  Shuffled pointer chase with 64-byte stride, 4 KiB to 1 GiB.\n"
        "  Single-thread read and cached/streaming store bandwidth.\n"
        "  Atomics with 1, 2, 4, and 8 threads; malloc/free at 64, 256, 4096 bytes.\n"
        "  Streaming stores use architecture-specific hints and completion fences.\n"
        "\n"
        "Storage (disk)\n"
        "  Unique 2 GiB temporary file under ./build/ on the current filesystem.\n"
        "  F_NOCACHE (macOS) or O_DIRECT (Linux) bypasses OS page caching.\n"
        "  Sequential 1 MiB I/O; random 4 KiB reads and writes; 1/2/4/8/16 readers.\n"
        "  Burst writes sync once at the end; the fsync row syncs every write.\n"
        "  Drive caches may participate. macOS fsync is not F_FULLFSYNC.\n"
        "  Several GiB are written. The file is removed on normal completion.\n"
        "\n"
        "System (sys)\n"
        "  getuid call time, pipe process handoff (half a round trip), and\n"
        "  pthread create+join time. Handoff includes pipe and scheduler costs.\n"
        "\n"
        "Network (net)\n"
        "  TCP loopback: 2 GiB bulk transfer and one-byte TCP_NODELAY ping-pong.\n"
        "  Measures the local OS stack, not NIC throughput.\n"
        "\n"
        "Matrix (matrix)\n"
        "  SGEMM: square FP32 matrices from 64x64 through 4096x4096.\n"
        "  Accelerate on macOS, optional OpenBLAS on Linux.\n"
        "\n"
        "NPU (npu)\n"
        "  macOS: ten convolutions, 256x256 spatial size, 256 intermediate channels.\n"
        "    python3 scripts/gen_npu_model.py models/   (coremltools + numpy)\n"
        "    CPU + Neural Engine, CPU + GPU, and CPU-only compute-unit modes.\n"
        "    Core ML may use CPU fallback within the first two modes.\n"
        "  Linux: ten 512x512 matrix multiplications, via optional OpenVINO C API.\n"
        "    python3 scripts/gen_openvino_model.py models/   (numpy)\n"
        "    NPU when available, plus a CPU baseline.\n"
        "  Models are different across OSes; TOPS is not directly comparable.\n"
        "  Execution precision is framework-selected. Run from the repo root.\n"
        "\n"
        "See README.md for dependencies and measurement limitations.\n"
    );
}

int main(int argc, char **argv) {
    int selected[NUM_BENCHMARKS];
    int have_only = 0, have_skip = 0;

    /* Default: all enabled */
    for (int i = 0; i < (int)NUM_BENCHMARKS; i++)
        selected[i] = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--only") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "--only requires a comma-separated list\n");
                usage();
                return 1;
            }
            if (have_skip || have_only) {
                fprintf(stderr, "Use --only once, without --skip\n");
                return 1;
            }
            have_only = 1;
            /* Start with none, enable listed */
            for (int j = 0; j < (int)NUM_BENCHMARKS; j++)
                selected[j] = 0;
            if (parse_list(argv[i], selected) < 0) {
                usage();
                return 1;
            }
        } else if (strcmp(argv[i], "--skip") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "--skip requires a comma-separated list\n");
                usage();
                return 1;
            }
            if (have_only || have_skip) {
                fprintf(stderr, "Use --skip once, without --only\n");
                return 1;
            }
            have_skip = 1;
            /* Start with all, then parse_list sets to 1 — invert after */
            int to_skip[NUM_BENCHMARKS];
            memset(to_skip, 0, sizeof(to_skip));
            if (parse_list(argv[i], to_skip) < 0) {
                usage();
                return 1;
            }
            for (int j = 0; j < (int)NUM_BENCHMARKS; j++)
                if (to_skip[j]) selected[j] = 0;
        } else if (strcmp(argv[i], "--help") == 0) {
            detailed_help();
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage();
            return 1;
        }
    }

    setvbuf(stdout, NULL, _IOLBF, 0);
    srand((unsigned)time(NULL));

    int first = 1;
    for (int i = 0; i < (int)NUM_BENCHMARKS; i++) {
        if (!selected[i]) continue;
        if (!first) printf("\n");
        all_benchmarks[i].run();
        first = 0;
    }

    return 0;
}
