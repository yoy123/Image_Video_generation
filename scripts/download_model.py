#!/usr/bin/env python3
"""
Quick helper to download a model file by URL into the local webui models folder.
Usage:
  python scripts/download_model.py <url> [--dest SUBDIR] [--filename NAME]

Defaults:
  - Destination base: external/stable-diffusion-webui/models/Stable-diffusion
  - Subdir can be overridden via --dest (relative to base)
  - If --filename is omitted, the name is inferred from the URL.

Also updates models/gguf_inventory.txt if the downloaded file ends with .gguf.
"""
import argparse
import os
import sys
from pathlib import Path
import urllib.request

BASE = Path(__file__).resolve().parents[1]
WEBUI_SD_DIR = BASE / "external" / "stable-diffusion-webui" / "models" / "Stable-diffusion"
INVENTORY = BASE / "models" / "gguf_inventory.txt"


def download(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {out_path} ...")
    urllib.request.urlretrieve(url, out_path)
    print("Done.")
    return out_path


def maybe_update_inventory(path: Path):
    if path.suffix.lower() == ".gguf":
        INVENTORY.parent.mkdir(parents=True, exist_ok=True)
        line = f"file: {path}\n"
        existing = ""
        if INVENTORY.exists():
            existing = INVENTORY.read_text()
            if line in existing:
                return
        with INVENTORY.open("a", encoding="utf-8") as f:
            f.write(line)
        print(f"Updated inventory: {INVENTORY}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="Direct URL to model file (.safetensors, .ckpt, .gguf, etc.)")
    ap.add_argument("--dest", default=".", help="Subfolder under Stable-diffusion/ to place the file")
    ap.add_argument("--filename", default=None, help="Override output filename")
    args = ap.parse_args()

    dest_dir = (WEBUI_SD_DIR / args.dest).resolve()
    fname = args.filename or os.path.basename(args.url.split("?")[0])
    out_path = dest_dir / fname

    path = download(args.url, out_path)
    maybe_update_inventory(path)
    print(f"Saved: {path}")


if __name__ == "__main__":
    sys.exit(main())
