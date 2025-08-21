#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

try:
    import torch
except Exception as e:
    torch = None

# Ensure project root on path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generate import generate_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="tmp_test_input.png", help="Input image path")
    ap.add_argument("--output", default="tmp_test_out.png", help="Output image path")
    ap.add_argument("--prompt", default="A cinematic portrait, high detail, soft light", help="Prompt")
    ap.add_argument("--model", default=os.environ.get("MODEL_ID_OR_PATH", ""), help="HF repo id or local path")
    ap.add_argument("--redux", default=os.environ.get("REDUX_REPO_ID", ""), help="Optional Redux adapter repo id")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--strength", type=float, default=0.6)
    args = ap.parse_args()

    if args.redux:
        os.environ["REDUX_REPO_ID"] = args.redux

    device = "cpu"
    if torch is not None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    if args.redux:
        print(f"Redux adapter: {args.redux}")

    out = generate_image(
        input_image_path=args.input,
        prompt=args.prompt,
        output_path=args.output,
        model_id=args.model,
        device=device,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        strength=args.strength,
        seed=42,
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    raise SystemExit(main())
