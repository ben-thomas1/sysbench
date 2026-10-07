#!/usr/bin/env python3
"""Convert a SPIR-V binary to a C header with a uint32_t array."""

import struct
import sys
import os


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.spv> <output.h>", file=sys.stderr)
        sys.exit(1)

    spv_path = sys.argv[1]
    hdr_path = sys.argv[2]

    with open(spv_path, "rb") as f:
        data = f.read()

    # SPIR-V is already a stream of uint32_t words
    words = struct.unpack(f"<{len(data) // 4}I", data)

    var_name = os.path.splitext(os.path.basename(spv_path))[0]
    guard = os.path.basename(hdr_path).replace(".", "_").upper()

    with open(hdr_path, "w") as f:
        f.write(f"/* Auto-generated from {os.path.basename(spv_path)} */\n")
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")
        f.write(f"#include <stdint.h>\n\n")
        f.write(f"static const uint32_t {var_name}_spv[] = {{\n")

        # 6 words per line
        for i in range(0, len(words), 6):
            chunk = words[i : i + 6]
            line = ", ".join(f"0x{w:08x}" for w in chunk)
            f.write(f"    {line},\n")

        f.write(f"}};\n\n")
        f.write(
            f"static const uint32_t {var_name}_spv_size = sizeof({var_name}_spv);\n\n"
        )
        f.write(f"#endif\n")

    print(f"  {hdr_path}: {len(words)} words ({len(words) * 4} bytes)")


if __name__ == "__main__":
    main()
