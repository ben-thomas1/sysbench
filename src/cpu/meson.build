cpu_srcs = ['cpu.c', 'pmu.c', cpu == 'aarch64' ? 'kern_arm.c' : 'kern_x86.c']
module_libs += static_library('sb_cpu', cpu_srcs,
  include_directories: inc_src,
  c_args: strict_args,
  link_with: core_lib,
  dependencies: threads,
)
