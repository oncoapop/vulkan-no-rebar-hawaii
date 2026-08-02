#!/usr/bin/env python3
"""Render a SPIR-V binary as the deterministic C header used by Colibri."""

import argparse
import hashlib
from pathlib import Path
import struct


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--source", required=True)
    parser.add_argument("--compiler", required=True)
    args = parser.parse_args()

    data = Path(args.input).read_bytes()
    if not data or len(data) % 4:
        raise SystemExit("SPIR-V input must contain a nonempty whole number of words")
    source_hash = hashlib.sha256(Path(args.source).read_bytes()).hexdigest()
    spv_hash = hashlib.sha256(data).hexdigest()
    words = struct.unpack(f"<{len(data) // 4}I", data)
    lines = [
        "/* Generated file; do not edit.",
        f" * source-sha256: {source_hash}",
        f" * spirv-sha256: {spv_hash}",
        f" * compiler: {args.compiler}",
        " */",
        "#ifndef COLIBRI_VULKAN_QT_FMT2_DOWN_SPV_H",
        "#define COLIBRI_VULKAN_QT_FMT2_DOWN_SPV_H",
        "",
        "#include <stdint.h>",
        "",
        "static const uint32_t coli_vulkan_qt_fmt2_down_spv[] = {",
    ]
    for offset in range(0, len(words), 8):
        chunk = ", ".join(f"0x{word:08x}u" for word in words[offset:offset + 8])
        lines.append(f"    {chunk},")
    lines.extend([
        "};",
        "static const uint32_t coli_vulkan_qt_fmt2_down_spv_size =",
        "    (uint32_t)sizeof(coli_vulkan_qt_fmt2_down_spv);",
        "",
        "#endif",
        "",
    ])
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
