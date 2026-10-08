module_libs += static_library('sb_matrix', 'matrix.c',
  include_directories: inc_src,
  c_args: strict_args + (has_blas ? ['-DHAS_BLAS'] : []),
  link_with: core_lib,
  dependencies: has_blas ? [blas_dep] : [],
)
