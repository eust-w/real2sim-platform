#!/usr/bin/env python3
"""Materialize movable extraction tasks into provisional Real2Sim object assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real2sim.builtin import write_3dgs_ply, write_box_obj
from real2sim.contracts import write_json
from real2sim.image_io import read_gray, read_rgb


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert movable extraction task crops/masks into provisional object assets")
    parser.add_argument("--package", type=Path, required=True, help="Real2Sim package root")
    parser.add_argument("--tasks", type=Path, default=None, help="movable_extraction_tasks.json; defaults under package")
    parser.add_argument("--output-objects-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-objects", type=int, default=3)
    args = parser.parse_args()

    result = materialize_movable_tasks(
        package_root=args.package,
        tasks_path=args.tasks or args.package / "artifacts" / "assets" / "movable_extraction_tasks.json",
        output_objects_json=args.output_objects_json,
        output_dir=args.output_dir,
        max_objects=max(1, int(args.max_objects)),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def materialize_movable_tasks(
    *,
    package_root: Path,
    tasks_path: Path,
    output_objects_json: Path,
    output_dir: Path,
    max_objects: int,
) -> dict[str, Any]:
    tasks_data = json.loads(tasks_path.read_text(encoding="utf-8"))
    tasks = tasks_data.get("tasks", []) if isinstance(tasks_data, dict) else []
    output_dir.mkdir(parents=True, exist_ok=True)
    objects: list[dict[str, Any]] = []
    for index, task in enumerate(tasks[:max_objects]):
        if not isinstance(task, dict):
            continue
        asset = materialize_task_asset(package_root=package_root, output_dir=output_dir, task=task, index=index)
        if asset is not None:
            objects.append(asset)
    payload = {
        "format": "gs-playground-real2sim-provisional-objects-v1",
        "objects": objects,
        "quality": {
            "object_count": len(objects),
            "source_task_count": len(tasks),
            "provisional": True,
        },
        "commercial_gap_notes": [
            "These assets are deterministic provisional billboards/boxes from extraction crops.",
            "Replace them with SAM/SAM3D object reconstructions for commercial visual fidelity.",
        ],
    }
    write_json(output_objects_json, payload)
    return {"objects_json": output_objects_json.as_posix(), **payload["quality"]}


def materialize_task_asset(*, package_root: Path, output_dir: Path, task: dict[str, Any], index: int) -> dict[str, Any] | None:
    crop_value = task.get("crop")
    mask_value = task.get("mask")
    if not crop_value or not mask_value:
        return None
    crop_path = package_root / str(crop_value)
    mask_path = package_root / str(mask_value)
    rgb = read_rgb(crop_path)
    mask = read_gray(mask_path)
    local_mask = local_crop_mask(mask, task.get("bbox_xyxy"), rgb.shape[:2])
    if int(local_mask.sum()) < 4:
        local_mask = np.ones(rgb.shape[:2], dtype=bool)
    points, colors = crop_billboard_points(rgb, local_mask, max_points=6000)
    if points.size == 0:
        return None
    label = best_prompt_label(task)
    name = f"{index:02d}_{safe_token(label)}"
    asset_dir = output_dir / name
    gaussian = asset_dir / "gaussian.ply"
    mesh = asset_dir / "mesh.obj"
    extent = object_extent_from_mask(local_mask, rgb.shape[:2], label=label)
    write_3dgs_ply(gaussian, points, colors, point_scale=0.01, opacity_logit=4.0)
    write_box_obj(mesh, extent)
    pose = pose_from_task_bbox(task.get("bbox_xyxy"))
    return {
        "name": name,
        "label": label,
        "gaussian_ply": gaussian.as_posix(),
        "mesh_obj": mesh.as_posix(),
        "pose": pose,
        "dynamic": True,
        "source_instance_id": str(task.get("id") or name),
        "provisional": True,
    }


def local_crop_mask(global_mask: np.ndarray, bbox: Any, crop_shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return np.ones(crop_shape, dtype=bool)
    x0, y0, x1, y1 = [int(value) for value in bbox]
    cropped = global_mask[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] > 0.5
    if cropped.shape != crop_shape:
        return np.ones(crop_shape, dtype=bool)
    return cropped


def crop_billboard_points(rgb: np.ndarray, mask: np.ndarray, *, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    if ys.size > max_points:
        indices = np.linspace(0, ys.size - 1, max_points, dtype=np.int64)
        ys = ys[indices]
        xs = xs[indices]
    h, w = rgb.shape[:2]
    x = ((xs.astype(np.float32) / max(w - 1, 1)) - 0.5) * 0.38
    z = (0.5 - (ys.astype(np.float32) / max(h - 1, 1))) * 0.38
    y = np.zeros_like(x)
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = rgb[ys, xs, :3].astype(np.float32)
    return points, colors


def object_extent_from_mask(mask: np.ndarray, shape: tuple[int, int], *, label: str) -> np.ndarray:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.array([0.2, 0.08, 0.2], dtype=np.float32)
    h, w = shape
    width = max((float(xs.max() - xs.min() + 1) / max(w, 1)) * 0.38, 0.06)
    height = max((float(ys.max() - ys.min() + 1) / max(h, 1)) * 0.38, 0.06)
    depth = 0.08
    if "bottle" in label or "cup" in label:
        depth = max(width * 0.55, 0.06)
    return np.array([width, depth, height], dtype=np.float32)


def pose_from_task_bbox(bbox: Any) -> dict[str, list[float]]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return {"position": [0.0, -0.25, 0.65], "quat_wxyz": [1.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}
    x0, y0, x1, y1 = [float(value) for value in bbox]
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    position = [round((cx - 224.0) / 224.0 * 0.9, 6), -0.35, round((448.0 - cy) / 448.0 * 0.65 + 0.25, 6)]
    return {"position": position, "quat_wxyz": [1.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}


def best_prompt_label(task: dict[str, Any]) -> str:
    prompts = task.get("prompt_candidates")
    if isinstance(prompts, list) and prompts:
        return str(prompts[0])
    return "movable_object"


def safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_") or "object"


if __name__ == "__main__":
    raise SystemExit(main())
