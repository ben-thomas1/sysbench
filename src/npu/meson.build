if host == 'darwin'
  # models/npu_model_info.h (generated, gitignored) overrides fallback/npu_model_info.h.
  npu_inc = [include_directories('fallback'), inc_src]
  if import('fs').is_file(meson.project_source_root() / 'models/npu_model_info.h')
    npu_inc = [include_directories('../../models')] + npu_inc
  endif
  module_libs += static_library('sb_npu', 'npu_macos.m',
    include_directories: npu_inc,
    objc_args: ['-fobjc-arc'] + strict_args,
    link_with: core_lib,
    dependencies: coreml_dep,
  )
else
  ov_args = []
  ov_deps = []
  ov_inc  = get_option('openvino_include')
  ov_lib  = get_option('openvino_libdir')
  if ov_inc != '' and ov_lib != ''
    ov_c = cc.find_library('openvino_c', dirs: [ov_lib], required: get_option('openvino'))
    if ov_c.found()
      ov_args += ['-DHAS_OPENVINO', '-I' + ov_inc]
      ov_deps += [ov_c, declare_dependency(link_args: ['-Wl,-rpath,' + ov_lib])]
    endif
  else
    ov_pc = dependency('openvino', required: get_option('openvino'))
    if ov_pc.found()
      ov_args += '-DHAS_OPENVINO'
      ov_deps += [ov_pc, cc.find_library('openvino_c', required: true)]
    endif
  endif
  module_libs += static_library('sb_npu', 'npu_linux.c',
    include_directories: inc_src,
    c_args: strict_args + ov_args,
    link_with: core_lib,
    dependencies: ov_deps,
  )
endif
