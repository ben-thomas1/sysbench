core_lib = static_library('sb_core',
  'status.c', 'timer.c', 'platform.c', 'platform_' + (host == 'darwin' ? 'macos' : 'linux') + '.c',
  'thread.c', 'report.c', 'stats.c',
  include_directories: inc_src,
  c_args: strict_args,
  dependencies: threads,
)
