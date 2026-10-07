# Shared code for modules not yet rewritten on the core API.
legacy_lib = static_library('sb_legacy',
  'legacy.c', 'sysinfo.c', 'sysinfo_' + (host == 'darwin' ? 'macos' : 'linux') + '.c',
  include_directories: [inc_src, inc_legacy],
)
