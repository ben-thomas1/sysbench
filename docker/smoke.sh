#!/bin/sh
# Run inside the smoke container (see docker/Dockerfile).
set -eu
export CC=clang-19
fail=0
step() { printf '\n##### %s\n' "$*"; }

step "debug build (sanitizers, strict warnings, -Werror; compile only)"
meson setup /tmp/b-debug -Ddebug=true -Doptimization=g -Db_sanitize=address,undefined -Db_lundef=false >/dev/null
meson compile -C /tmp/b-debug | grep -E 'error|warning' || true
[ -x /tmp/b-debug/bench ] || fail=1

step "release build, -march=native (AVX2 under emulation)"
meson setup /tmp/b-native --buildtype=release -Db_lto=true -Db_pie=true >/dev/null
meson compile -C /tmp/b-native >/dev/null || fail=1
/tmp/b-native/bench --only cpu,branch,gpu,mem,sys,matrix,npu || fail=1

step "release build, -march=x86-64 (SSE2 tier)"
meson setup /tmp/b-sse2 --buildtype=release -Dmarch=x86-64 >/dev/null
meson compile -C /tmp/b-sse2 >/dev/null || fail=1
/tmp/b-sse2/bench --only cpu,mem,sys || fail=1

step "release build, -march=x86-64-v4 (AVX-512 tier, compile only)"
meson setup /tmp/b-avx512 --buildtype=release -Dmarch=x86-64-v4 >/dev/null
meson compile -C /tmp/b-avx512 >/dev/null || fail=1

step "result"
if [ $fail -eq 0 ]; then echo "SMOKE OK"; else echo "SMOKE FAILED"; exit 1; fi
