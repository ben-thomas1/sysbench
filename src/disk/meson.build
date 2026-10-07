module_libs += static_library('sb_disk', 'disk.c', 'disk_section.c',
  include_directories: [inc_src, inc_legacy],
  link_with: legacy_lib,
  dependencies: threads,
)
