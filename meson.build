project('sysbench', 'c',
  version: '0.2.0',
  meson_version: '>=1.3',
  default_options: [
    'c_std=c2x',            # C23; clang >= 18 treats c2x as an alias. Meson < 1.4 has no 'c23'.
    'warning_level=2',
    'buildtype=debug',
    'b_ndebug=if-release',
  ],
)

host = host_machine.system()       # 'darwin' or 'linux'
cpu  = host_machine.cpu_family()   # 'aarch64' or 'x86_64'
if host not in ['darwin', 'linux']
  error('Unsupported OS: use macOS or Linux')
endif
if cpu not in ['aarch64', 'x86_64']
  error('Unsupported architecture: use arm64 or x86_64')
endif
if host == 'darwin'
  add_languages('objc', native: false)
endif

cc       = meson.get_compiler('c')
is_debug = get_option('optimization') in ['0', 'g']
march    = get_option('march')

# --- Flags (see CLAUDE.md "Build") -------------------------------------------
common_args = ['-march=' + march]
if march == 'native'
  common_args += '-mtune=native'
endif
if host == 'linux'
  common_args += '-D_GNU_SOURCE'
endif

link_args = []
if is_debug
  common_args += [
    '-g3', '-fno-omit-frame-pointer', '-ftrivial-auto-var-init=pattern',
    '-fstack-protector-all', '-DSB_DEBUG',
  ]
  sanitize = get_option('b_sanitize')
  if sanitize.contains('address')
    common_args += '-fsanitize-address-use-after-scope'
  endif
  if sanitize != 'none' and sanitize != ''
    common_args += '-fno-sanitize-recover=all'
  endif
else
  common_args += ['-ffast-math', '-fomit-frame-pointer', '-fno-stack-protector']
  if host == 'linux'
    common_args += '-fcf-protection=full'
    link_args   += ['-Wl,-z,relro,-z,now', '-Wl,-z,noexecstack']
  elif cpu == 'aarch64'
    common_args += '-mbranch-protection=standard'
  endif
endif
add_project_arguments(common_args, language: host == 'darwin' ? ['c', 'objc'] : ['c'])
add_project_link_arguments(link_args, language: host == 'darwin' ? ['c', 'objc'] : ['c'])

# Strict warnings for code written on the core API. Modules still in legacy
# form build with warning_level=2 only until they are rewritten.
strict_args = [
  '-Wshadow', '-Wconversion', '-Wsign-conversion', '-Wvla', '-Wcast-align',
  '-Wpointer-arith', '-Wwrite-strings', '-Wformat=2', '-Wnull-dereference',
  '-Wdouble-promotion', '-Wstrict-prototypes', '-Wmissing-prototypes',
  '-Wimplicit-fallthrough',
]
if is_debug
  strict_args += '-Werror'
endif

# --- Dependencies ------------------------------------------------------------
threads = dependency('threads')
inc_src    = include_directories('src')
inc_legacy = include_directories('src/legacy')

if host == 'darwin'
  metal_dep  = dependency('appleframeworks', modules: ['Metal', 'Foundation'])
  coreml_dep = dependency('appleframeworks', modules: ['CoreML', 'Foundation'])
  blas_dep   = dependency('appleframeworks', modules: ['Accelerate'])
  has_blas   = true
else
  blas_dep = dependency('openblas', required: get_option('blas'))
  has_blas = blas_dep.found()
endif

# --- Core + modules ----------------------------------------------------------
subdir('src/core')
subdir('src/legacy')

module_libs = []
foreach m : ['cpu', 'branch', 'gpu', 'mem', 'disk', 'sys', 'net', 'matrix', 'npu']
  subdir('src' / m)
endforeach

executable('bench', 'src/main.c',
  include_directories: inc_src,
  c_args: strict_args,
  link_whole: module_libs,
  link_with: [core_lib, legacy_lib],
  dependencies: threads,
  install: true,
)
