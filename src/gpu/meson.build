if host == 'darwin'
  module_libs += static_library('sb_gpu', 'gpu_macos.m', 'gpu_section.c',
    include_directories: [inc_src, inc_legacy],
    objc_args: ['-fobjc-arc'],
    link_with: legacy_lib,
    dependencies: metal_dep,
  )
else
  vulkan = dependency('vulkan')
  glslc  = find_program('glslc')
  python = find_program('python3')
  spv_headers = []
  foreach sh : ['fma_bench', 'int_bench', 'bw_bench', 'latency_chase', 'fma_fp16_bench', 'shared_mem', 'tex_sample']
    env_args = sh == 'fma_fp16_bench' ? ['--target-env=vulkan1.1'] : []
    spv = custom_target(sh + '_spv',
      input: 'shaders' / sh + '.comp',
      output: sh + '.spv',
      command: [glslc] + env_args + ['-o', '@OUTPUT@', '@INPUT@'],
    )
    spv_headers += custom_target(sh + '_h',
      input: spv,
      output: sh + '.h',
      command: [python, meson.project_source_root() / 'tools/spv2h.py', '@INPUT@', '@OUTPUT@'],
    )
  endforeach
  module_libs += static_library('sb_gpu', 'gpu_vulkan.c', 'gpu_section.c', spv_headers,
    include_directories: [inc_src, inc_legacy],
    link_with: legacy_lib,
    dependencies: vulkan,
  )
endif
