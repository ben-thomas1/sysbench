mem_srcs = ['mem.c', 'mem_lat.c', 'mem_bw.c', 'mem_atomic.c', 'mem_alloc.c',
            cpu == 'aarch64' ? 'mem_kern_arm.c' : 'mem_kern_x86.c']
module_libs += static_library('sb_mem', mem_srcs,
  include_directories: inc_src,
  c_args: strict_args,
  link_with: core_lib,
  dependencies: threads,
)
