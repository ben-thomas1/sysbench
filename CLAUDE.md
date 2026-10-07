# bench — system benchmark suite

Microbenchmarks for macOS on Apple Silicon (primary dev machine: M4 Pro) and Linux on x86-64.
C23, plus Objective-C for Metal and Core ML. House style is `~/.agents/C.md` (bstd style) with
the project prefix `sb_` / `SB_`.

## Status (2026-10)

The project is mid-rewrite. The goal is numbers that can be trusted.

- **Phase 0 (done):** the Meson build, the core API in `src/core/`, one directory per section, and `sys` rewritten on the
  core API as the worked example.
- **Phase 1 (in progress):** each remaining section is rewritten on the core API to fix its measurement defects. Each
  section is one subagent in its own worktree and branch (`phase1/<section>`), reviewed before it merges to master.
- **Phase 2 (planned):** Tier 2 roofline (arithmetic intensity × working set) and Tier 3 real-world kernels (naive and
  optimised), reported against the Tier 1 peaks.

**Ground truth:** `/Users/bt386671/Code/projects/sysbench-validation/FINDINGS.md`. It is scratch, never committed, and
must be read before touching a section. Probes go in that directory; raw results go in `results/agent-<section>/`.

## Layout

```
sysbench/
├── meson.build, meson.options   build definition (one meson.build per src/ dir)
├── Makefile                     helpers: make (debug) · make release · make smoke · make clean
├── docker/                      Ubuntu 24.04 amd64 smoke image + script
├── tools/                       spv2h.py, NPU model generators (uv run tools/...)
├── models/                      generated NPU models (gitignored)
└── src/
    ├── main.c                   CLI, section list, repeat loop
    ├── core/                    shared API (headers next to .c files)
    │   ├── types.h              i8..u64, f32, f64, SB_MALLOC/SB_REALLOC/SB_FREE, SB_ARRAY_LEN
    │   ├── status.h/.c          sb_status_e (SB_OK = 0, shared failure categories)
    │   ├── timer.h/.c           sb_timer_now_ns(), sb_timer_spin(), SB_WARMUP_NS
    │   ├── platform.h/.c        sb_platform_get(): OS, CPU, hybrid core ids, cache line, caches, memory
    │   │   platform_{macos,linux}.c
    │   ├── thread.h/.c          sb_par_run(): time-based multithreaded runner; core-type placement
    │   ├── report.h/.c          sb_report_*: rows with kind tags; --repeat aggregation
    │   ├── stats.h/.c           median/min/max
    │   └── section.h            sb_section descriptor
    ├── sys/                     rewritten (reference implementation)
    ├── cpu/ branch/ mem/ gpu/ disk/ net/ matrix/ npu/
    │                            legacy code; <name>_section.c adapts it to sb_section
    └── legacy/                  helpers used only by legacy modules; delete when empty
```

## Build

```
make            debug: -Og -g3, ASan+UBSan, strict warnings + -Werror on new code -> ./bench-debug
make release    -O3 -march=native -ffast-math LTO PIE -> ./bench   (ALL reported numbers come from this)
make smoke      Docker x86-64: debug build, AVX2 build+run, SSE2 build+run, AVX-512 compile
```
- Meson options: `-Dmarch=` (default native), `-Dblas=`, `-Dopenvino=`, `-Dopenvino_include=`, `-Dopenvino_libdir=`.
- Strict warnings (`strict_args`) apply to `core/`, `main.c` and rewritten modules. Legacy modules build with
  `warning_level=2`. When you rewrite a module, add `c_args: strict_args` to its `meson.build`.
- macOS can't use `-fcf-protection` or the `-z` linker flags, so it uses `-mbranch-protection=standard` instead.

## Writing a section on the core API

Use `src/sys/sys.c` as the template.
1. The module directory owns everything: `<name>.h` declares `extern const sb_section sb_section_<name>;`, the sources
   define it, and `meson.build` appends to `module_libs`. Don't edit `main.c` or other modules.
2. `run()` returns `sb_status_e`. Report results with `sb_report_value(test, value, unit, kind)`, failures with
   `sb_report_error`, unsupported cases with `sb_report_skip`, context with `sb_report_info` (printed once), and tables
   with `sb_report_group`. Never `printf` results directly; `--repeat` depends on the report API.
3. Set `.repeatable = true`. Test names must be stable across passes.
4. Choose the right kind: `SB_KIND_PEAK` (kernel built to saturate a unit), `MEASURED`, `EFFECTIVE` (nominal work /
   time), `ESTIMATE` (depends on an assumption; say which in an info line).
5. Delete `<name>_section.c` and drop `inc_legacy` / `legacy_lib` from the module's `meson.build` once nothing uses them.

## Measurement rules (from FINDINGS §13)

1. **Verify kernels in disassembly.** Release flags (`-ffast-math`, LTO) can change them. Every MR includes the
   hot-loop disassembly.
2. Warm up ≥ 200 ms before timing (`SB_WARMUP_NS`; `sb_par_run` does it for you).
3. Multithreaded = time-based (`sb_par_run`), never equal work per thread.
4. On hybrid CPUs, report per core type: macOS uses QoS (USER_INTERACTIVE → P, BACKGROUND → E); Linux uses affinity.
5. Read hardware facts from `sb_platform_get()` (cache line is 128 B on Apple Silicon); never hard-code them.
6. Label what is measured: peak vs measured vs effective vs estimate; durable vs non-durable sync; wakeup vs transport.
7. The default run must stay quick (whole suite ~1–2 min). Longer studies go behind `--repeat` or a dedicated flag.

## Working rules for agents

- **Timing runs take the machine-wide lock**, so parallel agents don't disturb each other's numbers:
  `lockf -k /Users/bt386671/Code/projects/sysbench-validation/.bench.lock ./bench --only <section>`.
  The same applies to `make smoke` and probe runs. Hold it for at most a few minutes at a time.
- Commit only on your own branch (`phase1/<section>`). Never push and never touch master.
- MR contents: before/after numbers vs FINDINGS, hot-loop disassembly, `make smoke` output, debug build clean, and an
  explanation of any remaining gap.
- Don't edit FINDINGS.md. Put new probe results in `results/agent-<section>/`.

## Environment caveats (dev Mac)

- Endpoint-security and network-filter software (Zscaler, Defender, CrowdStrike, ...) is installed, so network results
  and fork/exec/open costs are not hardware ground truth. Validate those on the Linux machines.
- No sudo in the agent sandbox. Root-only measurements (kperf, powermetrics) go to the user as commands to run.
- Docker uses Ubuntu, because TLS interception breaks Arch mirrors.
