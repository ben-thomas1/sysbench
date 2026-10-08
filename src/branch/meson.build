module_libs += static_library('sb_branch',
  'branch.c', 'kernel.c', host == 'darwin' ? 'pmu_macos.c' : 'pmu_linux.c',
  include_directories: inc_src,
  c_args: strict_args,
  link_with: core_lib,
  dependencies: threads,
)
