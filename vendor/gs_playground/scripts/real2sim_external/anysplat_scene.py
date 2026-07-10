#!/usr/bin/env python3
"""AnySplat scene reconstruction adapter for Real2Sim.

This adapter uses the official AnySplat model/API and exports a real 3DGS PLY
through AnySplat's `export_ply` helper. Video inputs can provide real nearby
frames through `--frames-dir`; single-image inputs are kept as a single view
instead of pretending that duplicated frames are independent observations.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, REPO_ROOT.as_posix())

from real2sim.contracts import write_json
from real2sim.image_io import write_gray


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AnySplat scene reconstruction for Real2Sim")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--source-preview", type=Path, default=None, help="Optional source-camera RGB render preview PNG")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Optional extracted video frames directory")
    parser.add_argument("--anysplat-root", type=Path, default=os.environ.get("REAL2SIM_ANYSPLAT_ROOT"))
    parser.add_argument("--model-id", default=os.environ.get("REAL2SIM_ANYSPLAT_MODEL_ID", "lhjiang/anysplat"))
    parser.add_argument("--max-views", type=int, default=int(os.environ.get("REAL2SIM_ANYSPLAT_MAX_VIEWS", "8")))
    parser.add_argument("--mesh-grid", type=int, default=int(os.environ.get("REAL2SIM_ANYSPLAT_MESH_GRID", "48")))
    parser.add_argument("--shift-and-scale", action="store_true", help="Normalize exported Gaussian positions with AnySplat exporter")
    args = parser.parse_args()

    if args.anysplat_root is None:
        raise SystemExit("AnySplat root is required via --anysplat-root or REAL2SIM_ANYSPLAT_ROOT")
    sys.path.insert(0, args.anysplat_root.as_posix())

    import torch
    from src.model.model.anysplat import AnySplat
    from src.model.ply_export import export_ply
    from src.utils.image import process_image
    from real2sim.builtin import write_depth_mesh_obj

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnySplat.from_pretrained(args.model_id).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    image_paths = collect_image_paths(args.image, args.frames_dir, max_views=args.max_views)
    images = torch.stack([process_image(path.as_posix()) for path in image_paths], dim=0).unsqueeze(0).to(device)
    b, views, _, height, width = images.shape
    with torch.inference_mode():
        gaussians, pose = model.inference((images + 1) * 0.5)

    export_ply(
        gaussians.means[0],
        gaussians.scales[0],
        gaussians.rotations[0],
        gaussians.harmonics[0],
        gaussians.opacities[0],
        args.ply,
        shift_and_scale=args.shift_and_scale,
        save_sh_dc_only=True,
    )

    extrinsic = pose["extrinsic"]
    intrinsic = pose["intrinsic"]
    near = torch.ones(b, views, device=device) * 0.1
    far = torch.ones(b, views, device=device) * 100.0
    with torch.inference_mode():
        rendered = model.decoder.forward(gaussians, extrinsic, intrinsic.float(), near, far, (height, width), "depth")

    preview_metrics = None
    if args.source_preview is not None:
        source_rgb_tensor = ((images[0, 0].detach().cpu() + 1.0) * 0.5).clamp(0.0, 1.0)
        rendered_rgb_tensor = rendered.color[0, 0].detach().cpu().clamp(0.0, 1.0)
        preview_metrics = save_source_preview(args.source_preview, rendered_rgb_tensor, source_rgb_tensor)

    depth = extract_first_depth_map(rendered.depth)
    normalized_depth = normalize_depth(depth)
    write_gray(args.depth, normalized_depth)

    intrinsics = denormalize_intrinsics(intrinsic[0, 0].detach().cpu().numpy(), width, height)
    scene_rgb = (((images[0, 0].detach().cpu().permute(1, 2, 0).numpy()) + 1.0) * 0.5).clip(0.0, 1.0)
    write_depth_mesh_obj(args.mesh, scene_rgb, normalized_depth, intrinsics, grid=max(8, int(args.mesh_grid)))

    write_json(
        args.cameras,
        {
            "backend": "anysplat",
            "model_id": args.model_id,
            "image_paths": [path.as_posix() for path in image_paths],
            "image_shape": [height, width],
            "intrinsics": intrinsics,
            "source_preview": args.source_preview.as_posix() if args.source_preview is not None else None,
            "source_preview_metrics": preview_metrics,
            "pred_context_pose": to_jsonable(pose),
        },
    )
    write_json(
        args.output_dir / "anysplat_metadata.json",
        {
            "backend": "anysplat",
            "model_id": args.model_id,
            "num_views": len(image_paths),
            "max_views": args.max_views,
            "mesh_grid": max(8, int(args.mesh_grid)),
            "shift_and_scale": args.shift_and_scale,
            "source_preview": args.source_preview.as_posix() if args.source_preview is not None else None,
            "source_preview_metrics": preview_metrics,
        },
    )
    return 0


def save_source_preview(output: Path, rendered_rgb, source_rgb) -> dict[str, float]:
    from PIL import Image

    rendered = tensor_rgb_to_numpy(rendered_rgb)
    source = tensor_rgb_to_numpy(source_rgb)
    mae = float(np.mean(np.abs(rendered.astype(np.float32) / 255.0 - source.astype(np.float32) / 255.0)))
    mse = float(np.mean((rendered.astype(np.float32) / 255.0 - source.astype(np.float32) / 255.0) ** 2))
    psnr = float(99.0 if mse <= 1e-12 else -10.0 * np.log10(mse))
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rendered).save(output)
    return {"mae": mae, "mse": mse, "psnr": psnr}


def tensor_rgb_to_numpy(rgb) -> np.ndarray:
    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise SystemExit(f"RGB preview tensor must have shape 3xHxW or HxWx3, got {arr.shape}")
    return (arr.clip(0.0, 1.0) * 255.0).round().astype(np.uint8)


def collect_image_paths(image: Path, frames_dir: Path | None, *, max_views: int) -> list[Path]:
    if frames_dir is not None and str(frames_dir) not in {"", ".", "None", "__none__"} and frames_dir.is_dir():
        frames = sorted(path for path in frames_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if frames:
            return sample_evenly(frames, max_views=max(1, max_views))
    return [image]


def sample_evenly(paths: list[Path], *, max_views: int) -> list[Path]:
    if len(paths) <= max_views:
        return paths
    if max_views == 1:
        return [paths[0]]
    indices = np.linspace(0, len(paths) - 1, num=max_views).round().astype(int)
    return [paths[int(idx)] for idx in sorted(set(indices))]


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    finite = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(finite)
    if not valid.any():
        return np.zeros_like(finite, dtype=np.float32)
    lo = float(np.quantile(finite[valid], 0.02))
    hi = float(np.quantile(finite[valid], 0.98))
    if hi <= lo:
        return np.zeros_like(finite, dtype=np.float32)
    return np.clip((finite - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def extract_first_depth_map(depth) -> np.ndarray:
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 5 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 4:
        arr = arr[0, 0]
    elif arr.ndim == 3:
        arr = arr[0]
    elif arr.ndim != 2:
        raise SystemExit(f"AnySplat decoder depth must be 2D, 3D, or 4D, got shape {arr.shape}")
    if arr.ndim != 2:
        raise SystemExit(f"AnySplat decoder depth did not reduce to a 2D map, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def denormalize_intrinsics(intrinsic: np.ndarray, width: int, height: int) -> dict[str, float | int]:
    intr = np.asarray(intrinsic, dtype=np.float32)
    return {
        "width": int(width),
        "height": int(height),
        "fx": float(intr[0, 0] * width),
        "fy": float(intr[1, 1] * height),
        "cx": float(intr[0, 2] * width),
        "cy": float(intr[1, 2] * height),
    }


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
