#!/usr/bin/env python3
"""Speedy-Splat pruning adapter for Real2Sim 3DGS assets.

This adapter bridges Real2Sim's exported PLY assets into the official
Speedy-Splat pruning implementation. It loads the input PLY with
Speedy-Splat's GaussianModel, builds a camera from the AnySplat camera JSON,
calls the official train.prune function, and writes the pruned PLY.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune a 3DGS PLY with official Speedy-Splat scoring")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True, help="Source image used to build the pruning camera")
    parser.add_argument("--cameras", type=Path, required=True, help="AnySplat cameras.json artifact")
    parser.add_argument("--keep-ratio", type=float, default=float(os.environ.get("REAL2SIM_SPEEDY_KEEP_RATIO", "0.85")))
    parser.add_argument("--keep-count", type=int, default=None)
    parser.add_argument("--speedy-root", type=Path, default=os.environ.get("REAL2SIM_SPEEDY_SPLAT_ROOT"))
    parser.add_argument("--background", choices=("black", "white"), default=os.environ.get("REAL2SIM_SPEEDY_BACKGROUND", "black"))
    parser.add_argument("--metadata", type=Path, default=None)
    args = parser.parse_args()

    if args.speedy_root is None:
        raise SystemExit("Speedy-Splat root is required via --speedy-root or REAL2SIM_SPEEDY_SPLAT_ROOT")
    if not args.input.is_file():
        raise SystemExit(f"input PLY not found: {args.input}")
    if not args.image.is_file():
        raise SystemExit(f"source image not found: {args.image}")
    if not args.cameras.is_file():
        raise SystemExit(f"camera JSON not found: {args.cameras}")

    sys.path.insert(0, args.speedy_root.as_posix())

    import torch
    from PIL import Image
    from plyfile import PlyData
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from train import prune as speedy_prune

    props = [prop.name for prop in PlyData.read(args.input)["vertex"].properties]
    sh_degree = infer_sh_degree(props)
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(args.input.as_posix())
    gaussians.spatial_lr_scale = 1.0
    gaussians.training_setup(speedy_training_args())
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0],), device="cuda")
    prune_policy = install_exact_count_pruning(torch, gaussians)

    camera_payload = load_json(args.cameras)
    intrinsics = require_intrinsics(camera_payload)
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    image_tensor = load_image_tensor(Image, torch, args.image, width, height)
    view_index = image_view_index(camera_payload, args.image)
    extrinsic = first_matrix(camera_payload.get("pred_context_pose", {}).get("extrinsic"), rows=4, cols=4, view_index=view_index)
    rotation, translation = speedy_camera_rt(extrinsic)
    camera = Camera(
        colmap_id=0,
        R=rotation,
        T=translation,
        FoVx=focal_to_fov(float(intrinsics["fx"]), width),
        FoVy=focal_to_fov(float(intrinsics["fy"]), height),
        image=image_tensor,
        gt_alpha_mask=None,
        image_name=args.image.stem,
        uid=0,
    )

    prune_ratio = prune_ratio_from_keep(args.keep_ratio, args.keep_count, int(gaussians.get_xyz.shape[0]))
    background = torch.tensor([1.0, 1.0, 1.0] if args.background == "white" else [0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    scene = SingleCameraScene([camera])
    pipe = SimpleNamespace(debug=False, convert_SHs_python=False, compute_cov3D_python=False)
    prune_metrics = run_speedy_prune(torch, speedy_prune, scene, gaussians, pipe, background, prune_ratio)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(args.output.as_posix())
    metadata_path = args.metadata or args.output.with_suffix(".speedy_splat.json")
    write_json(
        metadata_path,
        {
            "backend": "speedy_splat",
            "speedy_root": args.speedy_root.as_posix(),
            "input": args.input.as_posix(),
            "output": args.output.as_posix(),
            "image": args.image.as_posix(),
            "cameras": args.cameras.as_posix(),
            "sh_degree": sh_degree,
            "keep_ratio": args.keep_ratio,
            "keep_count": args.keep_count,
            "camera_view_index": view_index,
            "prune_ratio": prune_ratio,
            "input_gaussians": int(len(PlyData.read(args.input)["vertex"])),
            "output_gaussians": int(gaussians.get_xyz.shape[0]),
            "prune_metrics": prune_metrics,
            "prune_policy": prune_policy,
        },
    )
    return 0


class SingleCameraScene:
    def __init__(self, cameras: list[Any]) -> None:
        self._cameras = cameras

    def getTrainCameras(self) -> list[Any]:
        return self._cameras


def run_speedy_prune(
    torch_module: Any,
    speedy_prune: Any,
    scene: Any,
    gaussians: Any,
    pipe: Any,
    background: Any,
    prune_ratio: float,
) -> dict[str, Any]:
    try:
        metrics = speedy_prune(scene, gaussians, pipe, background, prune_ratio)
        synchronize_cuda(torch_module)
        return dict(metrics) if isinstance(metrics, dict) else {"status": "ok"}
    except RuntimeError as exc:
        trace = traceback.format_exc()
        if not is_cuda_event_timing_failure(exc, trace):
            raise
        synchronize_cuda(torch_module)
        return {
            "status": "ok_with_cuda_timing_warning",
            "timing_warning": first_line(str(exc)),
        }


def install_exact_count_pruning(torch_module: Any, gaussians: Any) -> str:
    def prune_gaussians(percent: float, import_score: Any) -> None:
        prune_gaussians_exact_count(torch_module, gaussians, percent, import_score)

    gaussians.prune_gaussians = prune_gaussians
    return "speedy_splat_exact_count_tie_safe"


def prune_gaussians_exact_count(torch_module: Any, gaussians: Any, percent: float, import_score: Any) -> None:
    scores = import_score.reshape(-1)
    total = int(scores.shape[0])
    prune_count = exact_prune_count(total, percent)
    prune_mask = torch_module.zeros((total,), dtype=torch_module.bool, device=scores.device)
    if prune_count > 0:
        try:
            order = torch_module.argsort(scores, stable=True)
        except TypeError:
            order = torch_module.argsort(scores)
        prune_mask[order[:prune_count]] = True
    gaussians.prune_points(prune_mask)


def exact_prune_count(total: int, percent: float) -> int:
    if total <= 1:
        return 0
    if percent <= 0.0:
        return 0
    return min(max(int(round(total * percent)), 0), total - 1)


def is_cuda_event_timing_failure(exc: RuntimeError, trace: str) -> bool:
    message = str(exc).lower()
    return "device not ready" in message and "elapsed_time" in trace


def synchronize_cuda(torch_module: Any) -> None:
    cuda = getattr(torch_module, "cuda", None)
    synchronize = getattr(cuda, "synchronize", None)
    if callable(synchronize):
        synchronize()


def first_line(value: str) -> str:
    return value.splitlines()[0] if value.splitlines() else value


def speedy_training_args() -> SimpleNamespace:
    return SimpleNamespace(
        percent_dense=0.01,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30_000,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.001,
    )


def infer_sh_degree(property_names: list[str]) -> int:
    f_rest = [name for name in property_names if name.startswith("f_rest_")]
    coeffs = (len(f_rest) + 3) / 3
    degree = math.sqrt(coeffs) - 1
    rounded = int(round(degree))
    if rounded < 0 or not math.isclose(degree, rounded, rel_tol=0.0, abs_tol=1e-6):
        raise SystemExit(f"cannot infer spherical harmonics degree from {len(f_rest)} f_rest properties")
    return rounded


def prune_ratio_from_keep(keep_ratio: float, keep_count: int | None, total: int) -> float:
    if total <= 0:
        raise SystemExit("input PLY has no gaussians")
    if keep_count is not None:
        if keep_count <= 0:
            raise SystemExit("--keep-count must be positive")
        keep_ratio = min(1.0, max(0.0, keep_count / total))
    if not 0.0 < keep_ratio <= 1.0:
        raise SystemExit("--keep-ratio must be in (0, 1]")
    return max(0.0, 1.0 - keep_ratio)


def require_intrinsics(payload: dict[str, Any]) -> dict[str, float]:
    intrinsics = payload.get("intrinsics")
    required = ("width", "height", "fx", "fy")
    if not isinstance(intrinsics, dict) or any(key not in intrinsics for key in required):
        raise SystemExit(f"camera JSON must contain intrinsics with {', '.join(required)}")
    return intrinsics


def focal_to_fov(focal: float, pixels: int) -> float:
    if focal <= 0:
        raise SystemExit("camera focal length must be positive")
    return 2.0 * math.atan(float(pixels) / (2.0 * focal))


def image_view_index(payload: dict[str, Any], image_path: Path) -> int:
    image_paths = payload.get("image_paths")
    if not isinstance(image_paths, list):
        return 0
    target = image_path.expanduser().resolve()
    for index, raw_path in enumerate(image_paths):
        candidate = Path(str(raw_path)).expanduser().resolve()
        if candidate == target:
            return index
    return 0


def first_matrix(value: Any, *, rows: int, cols: int, view_index: int = 0) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[1:] == (rows, cols):
        index = min(max(int(view_index), 0), arr.shape[0] - 1)
        arr = arr[index]
    if arr.shape != (rows, cols):
        raise SystemExit(f"expected matrix shape {(rows, cols)}, got {arr.shape}")
    return arr


def speedy_camera_rt(extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Speedy-Splat's Camera builds world_view_transform with R.T and T.
    return np.asarray(extrinsic[:3, :3], dtype=np.float32).T, np.asarray(extrinsic[:3, 3], dtype=np.float32)


def load_image_tensor(image_module: Any, torch_module: Any, path: Path, width: int, height: int):
    image = image_module.open(path).convert("RGB").resize((width, height))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch_module.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
