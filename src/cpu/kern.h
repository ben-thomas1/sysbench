#pragma once

#include "core/status.h"
#include "core/thread.h"
#include "core/types.h"

/* Architecture-specific compute kernels (kern_arm.c / kern_x86.c).
 *
 * Every kernel is one inline-asm statement: it loads its operands,
 * initialises N independent accumulator chains, then runs `chunk` loop
 * iterations. The loop body is fixed text, so the instruction mix is exactly
 * what the source says regardless of compiler flags (verify in disassembly).
 * Work unit = one loop iteration. */

#define SB_CPU_MAX_KERNELS 16

typedef struct {
    const char *name;     /* row label, e.g. "FP32 NEON fmla.4s" */
    const char *unit;     /* "GFLOPS" or "GOPS" */
    sb_work_fn  fn;       /* NULL: unavailable on this build/CPU, see `skip` */
    u32         insns;    /* instructions per iteration (loop overhead excluded) */
    f64         ops;      /* arithmetic ops per iteration; multiply-add = 2 */
    const char *skip;     /* reason when fn == NULL */
} sb_cpu_kernel;

/* Kernels for the ops-per-cycle estimate. */
typedef struct {
    sb_work_fn  dep_add;        /* one dependent integer add chain */
    u32         dep_add_n;      /* adds per iteration */
    sb_work_fn  ind_add;        /* independent integer add chains */
    u32         ind_add_n;
    u32         ind_add_chains;
    sb_work_fn  fma_lat;        /* one dependent FP32 multiply-add chain */
    u32         fma_lat_n;      /* multiply-add steps per iteration */
    sb_work_fn  fma_tput;       /* FP32 SIMD peak kernel */
    u32         fma_tput_n;     /* instructions per iteration */
    u32         fma_chains;
    const char *fma_name;       /* instruction(s) of one step, e.g. "fmla.4s" */
} sb_cpu_ipc;

/* Fill `out` (capacity SB_CPU_MAX_KERNELS) with this build's kernels in
 * report order; returns the count. Runtime CPU features are checked here. */
u32 sb_cpu_kernels_get(sb_cpu_kernel *out);

void sb_cpu_ipc_get(sb_cpu_ipc *out);

/* Short description of the SIMD tier and kernel shape for the info lines. */
const char *sb_cpu_simd_name(void);
const char *sb_cpu_kernel_note(void);
