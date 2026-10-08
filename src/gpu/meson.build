if host == 'darwin'
  module_libs += static_library('sb_gpu', 'gpu.c', 'gpu_metal.m',
    include_directories: inc_src,
    c_args: strict_args,
    objc_args: strict_args + ['-fobjc-arc'],
    link_with: core_lib,
    dependencies: metal_dep,
  )
else
  vulkan = dependency('vulkan')
  glslc  = find_program('glslc')
  python = find_program('python3')
  spv_headers = []
  foreach sh : ['fma_f32', 'fma_f16', 'int32', 'bandwidth', 'chase', 'tgmem', 'texture']
    spv = custom_target(sh + '_spv',
      input: 'shaders' / sh + '.comp',
      output: sh + '.spv',
      depend_files: files('shaders/fma_common.glsl'),
      command: [glslc, '--target-env=vulkan1.1', '-O', '-o', '@OUTPUT@', '@INPUT@'],
    )
    spv_headers += custom_target(sh + '_h',
      input: spv,
      output: sh + '.h',
      command: [python, meson.project_source_root() / 'tools/spv2h.py', '@INPUT@', '@OUTPUT@'],
    )
  endforeach
  module_libs += static_library('sb_gpu', 'gpu.c', 'gpu_vulkan.c', spv_headers,
    include_directories: inc_src,
    c_args: strict_args,
    link_with: core_lib,
    dependencies: vulkan,
  )
endif
