# x86-64 Linux smoke test: proves the code builds and runs. Numbers from an
# emulated container are meaningless. Usage: make smoke
FROM ubuntu:24.04
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq clang-19 make pkg-config python3 glslc libvulkan-dev mesa-vulkan-drivers libopenblas-dev meson ninja-build >/dev/null
ENV CC=clang-19
WORKDIR /src
COPY . .
CMD ["sh", "docker/smoke.sh"]
