module_libs += static_library('sb_matrix', 'matrix.c', 'matrix_section.c',
  include_directories: [inc_src, inc_legacy],
  c_args: has_blas ? ['-DHAS_BLAS'] : [],
  link_with: legacy_lib,
  dependencies: has_blas ? [blas_dep] : [],
)
