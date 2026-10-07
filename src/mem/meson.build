module_libs += static_library('sb_mem',
  'mem.c', 'mem_section.c', cpu == 'aarch64' ? 'mem_arm.c' : 'mem_x86.c',
  include_directories: [inc_src, inc_legacy],
  link_with: legacy_lib,
  dependencies: threads,
)
