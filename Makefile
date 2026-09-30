CC       = clang
CFLAGS   = -O3 -std=c23 -Wall -Wextra -march=native -mtune=native -ffp-contract=fast -funroll-loops
CPPFLAGS += -Iinclude -MMD -MP
OBJCFLAGS = -fobjc-arc
TARGET   = bench
BUILDDIR = build
SPVDIR   = $(BUILDDIR)/spv

SRCS     = src/main.c src/cpu.c src/branch.c src/mem.c src/disk.c src/sysinfo.c src/sys.c src/net.c src/matrix.c
UNAME    := $(shell uname)
ARCH     := $(shell uname -m)

SHADERS  = shaders/fma_bench.comp shaders/int_bench.comp shaders/bw_bench.comp shaders/latency_chase.comp \
           shaders/fma_fp16_bench.comp shaders/shared_mem.comp shaders/tex_sample.comp
SPVS     = $(SHADERS:shaders/%.comp=$(SPVDIR)/%.spv)
SPV_HDRS = $(SPVS:.spv=.h)

ifneq (,$(filter arm64 aarch64,$(ARCH)))
  SRCS   += src/cpu_arm.c src/mem_arm.c
else ifneq (,$(filter x86_64,$(ARCH)))
  SRCS   += src/cpu_x86.c src/cpu_x86_sse2.c src/cpu_x86_avx2.c src/cpu_x86_avx512.c src/mem_x86.c
else
  $(error Unsupported architecture $(ARCH); use ARM64 or x86_64)
endif

ifeq ($(UNAME),Darwin)
  SRCS    += src/gpu_macos.m src/npu_macos.m src/sysinfo_macos.c
  LDLIBS += -framework Metal -framework Foundation -framework CoreML
  # BLAS: always available via Accelerate on macOS
  CPPFLAGS += -DHAS_BLAS
  LDLIBS += -framework Accelerate
else ifeq ($(UNAME),Linux)
  CPPFLAGS += -D_GNU_SOURCE
  SRCS    += src/gpu_vulkan.c src/npu_linux.c src/sysinfo_linux.c
  LDLIBS += -lvulkan
  CPPFLAGS += -I$(SPVDIR)
  # BLAS: optional via OpenBLAS on Linux
  HAS_OPENBLAS := $(shell pkg-config --exists openblas 2>/dev/null && echo 1 || echo 0)
  ifeq ($(HAS_OPENBLAS),1)
    CPPFLAGS += -DHAS_BLAS $(shell pkg-config --cflags openblas)
    LDLIBS += $(shell pkg-config --libs openblas)
  endif
  # NPU: optional via OpenVINO on Linux
  # The benchmark uses the C API. SDKs without a .pc file can set these.
  OPENVINO_CFLAGS ?= $(shell pkg-config --cflags openvino 2>/dev/null)
  OPENVINO_LIBS ?= $(shell pkg-config --libs openvino 2>/dev/null)
  ifneq ($(strip $(OPENVINO_LIBS)),)
    CPPFLAGS += -DHAS_OPENVINO $(OPENVINO_CFLAGS)
    LDLIBS += $(OPENVINO_LIBS) -lopenvino_c
  endif
else
  $(error Unsupported OS $(UNAME); use macOS or Linux)
endif

LDLIBS += -lpthread

OBJS     = $(patsubst src/%.c,$(BUILDDIR)/%.o,$(filter %.c,$(SRCS)))
OBJS    += $(patsubst src/%.m,$(BUILDDIR)/%.o,$(filter %.m,$(SRCS)))
DEPS     = $(OBJS:.o=.d)

.PHONY: all clean help

all: $(TARGET)

$(OBJS): Makefile

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILDDIR)/%.o: src/%.c | $(BUILDDIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

$(BUILDDIR)/%.o: src/%.m | $(BUILDDIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OBJCFLAGS) -c -o $@ $<

$(BUILDDIR)/gpu_vulkan.o: src/gpu_vulkan.c $(SPV_HDRS) | $(BUILDDIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

$(BUILDDIR)/npu_macos.o: src/npu_macos.m include/npu_model_info.h $(wildcard models/npu_model_info.h) | $(BUILDDIR)
	$(CC) -Imodels $(CPPFLAGS) $(CFLAGS) $(OBJCFLAGS) -c -o $@ $<

$(SPVDIR)/fma_fp16_bench.spv: shaders/fma_fp16_bench.comp | $(SPVDIR)
	glslc --target-env=vulkan1.1 -o $@ $<

$(SPVDIR)/%.spv: shaders/%.comp | $(SPVDIR)
	glslc -o $@ $<

$(SPVDIR)/%.h: $(SPVDIR)/%.spv scripts/spv2h.py
	python3 scripts/spv2h.py $< $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(SPVDIR):
	mkdir -p $(SPVDIR)

clean:
	rm -rf $(TARGET) $(BUILDDIR)

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all     Build bench (default)"
	@echo "  clean   Remove binary and build directory"
	@echo "  help    Show this help"
	@echo ""
	@echo "Run:"
	@echo "  ./bench                        Run all benchmarks"
	@echo "  ./bench --only cpu,gpu         Run only CPU and GPU"
	@echo "  ./bench --skip disk,npu        Skip disk and NPU"
	@echo "  ./bench --help                 Show detailed help"

-include $(DEPS)
