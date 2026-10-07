#pragma once

#include <stddef.h>

const char *fmt_size(size_t bytes);

void bench_cpu(void);
void bench_branch(void);
void bench_gpu(void);
void bench_memory(void);
void bench_disk(void);
void bench_net(void);
void bench_matrix(void);
void bench_npu(void);
