#!/usr/bin/env python3
"""Depth Anything V2 adapter for the Real2Sim hybrid backend.

The adapter writes a normalized PGM/PNG depth map that the built-in asset
generator can fuse with its geometry prior. The default model is the small
Transformers checkpoint so it can run on the Wuying 12 GB GPU.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, REPO_ROOT.as_posix())

from real2sim.image_io import write_gray


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Depth Anything V2 monocular depth for Real2Sim")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        default=os.environ.get("REAL2SIM_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf"),
        help="Hugging Face depth-estimation model id",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--invert", dest="invert", action="store_true", help="Invert relative depth before writing")
    parser.add_argument("--no-invert", dest="invert", action="store_false")
    parser.set_defaults(invert=True)
    args = parser.parse_args()

    depth = estimate_depth(args.image, args.model, args.device)
    if args.invert:
        depth = 1.0 - depth
    write_gray(args.output, depth)
    return 0


def estimate_depth(image_path: Path, model_id: str, device: str) -> np.ndarray:
    try:
        import torch
        from PIL import Image
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover - external runtime guard
        raise SystemExit(
            "Depth Anything requires torch, pillow, and transformers. "
            "Install them in the Real2Sim environment before enabling depth_estimation."
        ) from exc

    image = Image.open(image_path).convert("RGB")
    device_index = -1
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        device_index = 0

    pipe = pipeline(task="depth-estimation", model=model_id, device=device_index)
    result = pipe(image)
    raw_depth = result.get("predicted_depth")
    if raw_depth is not None:
        depth = raw_depth.detach().float().cpu().numpy()
        depth = np.squeeze(depth)
    else:
        depth = np.asarray(result["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return normalize_depth(depth)


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    lo = float(np.percentile(depth, 1.0))
    hi = float(np.percentile(depth, 99.0))
    if hi <= lo:
        return np.zeros_like(depth, dtype=np.float32)
    return np.clip((depth - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
