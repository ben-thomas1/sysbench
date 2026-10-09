# Helpers around Meson. Build directories live under build/.
#   make            debug build (sanitizers, strict warnings)  -> ./bench-debug
#   make release    optimized build used for all reported numbers -> ./bench
#   make smoke      x86-64 Linux build + run in Docker (AVX2, SSE2, AVX-512 compile)
#   make clean      remove build/ and the binary links

MESON    ?= meson
DOCKER   ?= docker
SMOKE_IMG = sysbench-smoke

# Replace make's built-in CC=cc, but keep environment/command-line overrides.
ifeq ($(origin CC), default)
CC = clang
endif
CC ?= clang
export CC

.PHONY: all debug release install smoke clean help

all: debug

build/debug/build.ninja:
	$(MESON) setup build/debug -Ddebug=true -Doptimization=g -Db_sanitize=address,undefined -Db_lundef=false

build/release/build.ninja:
	$(MESON) setup build/release --buildtype=release -Db_lto=true -Db_pie=true

debug: build/debug/build.ninja
	$(MESON) compile -C build/debug
	ln -sf build/debug/bench bench-debug

release: build/release/build.ninja
	$(MESON) compile -C build/release
	ln -sf build/release/bench bench

install: release
	$(MESON) install -C build/release

smoke:
	$(DOCKER) build --platform linux/amd64 -f docker/Dockerfile -t $(SMOKE_IMG) .
	$(DOCKER) run --rm --platform linux/amd64 $(SMOKE_IMG)

clean:
	rm -rf build bench bench-debug

help:
	@sed -n '2,6p' Makefile
