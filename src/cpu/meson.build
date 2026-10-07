cpu_srcs = ['cpu.c', 'cpu_section.c']
if cpu == 'aarch64'
  cpu_srcs += 'cpu_arm.c'
else
  cpu_srcs += ['cpu_x86.c', 'cpu_x86_sse2.c', 'cpu_x86_avx2.c', 'cpu_x86_avx512.c']
endif
module_libs += static_library('sb_cpu', cpu_srcs,
  include_directories: [inc_src, inc_legacy],
  link_with: legacy_lib,
  dependencies: threads,
)
