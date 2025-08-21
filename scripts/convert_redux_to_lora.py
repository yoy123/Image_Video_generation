#!/usr/bin/env python3
"""Convert a Redux-style safetensors file (keys like redux_down/weight) into a LoRA-style
safetensors file by renaming keys (redux_ -> lora_). This is a best-effort helper to try
making the adapter compatible with diffusers' LoRA loader.

Usage:
    python scripts/convert_redux_to_lora.py /path/to/input.safetensors /path/to/output.safetensors

If output path is omitted the script writes a sibling file with '-lora.safetensors'.
"""
import sys
import os
from safetensors.torch import load_file, save_file


def main():
    if len(sys.argv) < 2:
        print("Usage: convert_redux_to_lora.py <input.safetensors> [output.safetensors]")
        sys.exit(2)
    inp = sys.argv[1]
    if not os.path.isfile(inp):
        print(f"Input file not found: {inp}")
        sys.exit(1)
    out = None
    if len(sys.argv) >= 3:
        out = sys.argv[2]
    else:
        base = os.path.splitext(inp)[0]
        out = base + "-lora.safetensors"

    print(f"Loading: {inp}")
    tensors = load_file(inp)
    print(f"Loaded {len(tensors)} tensors")

    mapped = {}
    for k, v in tensors.items():
        if k.startswith("redux_"):
            nk = k.replace("redux_", "lora_", 1)
        else:
            # also try to be helpful for other variants
            nk = k
        mapped[nk] = v
    print(f"Mapped {len(mapped)} keys; writing to: {out}")
    save_file(mapped, out)
    print("Done")


if __name__ == '__main__':
    main()
