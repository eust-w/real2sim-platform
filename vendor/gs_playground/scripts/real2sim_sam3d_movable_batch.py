#!/usr/bin/env python3
"""Run SAM-3D over movable extraction tasks and emit production object assets."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real2sim.contracts import write_json
from real2sim.image_io import read_gray, read_rgb, write_gray, write_rgb


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert movable extraction tasks into production SAM-3D object assets")
    parser.add_argument("--package", type=Path, required=True, help="Real2Sim package root")
    parser.add_argument("--tasks", type=Path, default=None, help="movable_extraction_tasks.json; defaults under package")
    parser.add_argument("--output-objects-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--command", default=os.environ.get("REAL2SIM_SAM3D_COMMAND", ""))
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Write planned SAM-3D jobs without executing them")
    args = parser.parse_args()

    result = run_sam3d_movable_batch(
        package_root=args.package,
        tasks_path=args.tasks or args.package / "artifacts" / "assets" / "movable_extraction_tasks.json",
        output_objects_json=args.output_objects_json,
        output_dir=args.output_dir,
        command_template=args.command,
        max_objects=max(1, int(args.max_objects)),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def run_sam3d_movable_batch(
    *,
    package_root: Path,
    tasks_path: Path,
    output_objects_json: Path,
    output_dir: Path,
    command_template: str,
    max_objects: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    tasks_data = json.loads(tasks_path.read_text(encoding="utf-8"))
    tasks = tasks_data.get("tasks", []) if isinstance(tasks_data, dict) else []
    if not dry_run and not command_template.strip():
        raise SystemExit("SAM-3D command is required via --command or REAL2SIM_SAM3D_COMMAND")
    output_dir.mkdir(parents=True, exist_ok=True)
    objects: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for index, task in enumerate(tasks[:max_objects]):
        if not isinstance(task, dict):
            continue
        job = prepare_sam3d_task(package_root=package_root, output_dir=output_dir, task=task, index=index)
        jobs.append(job)
        if dry_run:
            continue
        command = render_command(command_template, job["context"])
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"SAM-3D command failed for {job['id']} with exit code {completed.returncode}")
        objects.append(production_object_from_job(job, package_root=package_root))
    payload = {
        "format": "gs-playground-real2sim-production-objects-v1",
        "objects": objects,
        "quality": {
            "object_count": len(objects),
            "source_task_count": len(tasks),
            "planned_job_count": len(jobs),
            "production": bool(objects) and len(objects) == len(jobs),
            "dry_run": bool(dry_run),
        },
        "sam3d_jobs": [
            {
                "id": job["id"],
                "label": job["label"],
                "instance_json": job["instance_json"],
                "image": job["image"],
                "mask": job["mask"],
                "output_dir": job["output_dir"],
            }
            for job in jobs
        ],
    }
    write_json(output_objects_json, payload)
    return {"objects_json": output_objects_json.as_posix(), **payload["quality"]}


def prepare_sam3d_task(*, package_root: Path, output_dir: Path, task: dict[str, Any], index: int) -> dict[str, Any]:
    task_id = str(task.get("id") or f"task_{index:03d}")
    label = best_prompt_label(task)
    name = f"{index:02d}_{safe_token(label)}"
    object_dir = output_dir / name
    object_dir.mkdir(parents=True, exist_ok=True)
    crop_path = package_root / str(task["crop"])
    mask_path = package_root / str(task["mask"])
    rgb = read_rgb(crop_path)
    mask = read_gray(mask_path)
    local_mask = local_crop_mask(mask, task.get("bbox_xyxy"), rgb.shape[:2])
    image_out = object_dir / "sam3d_image.ppm"
    mask_out = object_dir / "sam3d_mask.pgm"
    write_rgb(image_out, rgb)
    write_gray(mask_out, local_mask.astype(np.float32))
    instance_json = object_dir / "instance.json"
    write_json(
        instance_json,
        {
            "id": task_id,
            "label": label,
            "bbox_xyxy": task.get("bbox_xyxy"),
            "prompt_candidates": task.get("prompt_candidates"),
            "source_task": task,
        },
    )
    gaussian = object_dir / "gaussian.ply"
    mesh = object_dir / "mesh.obj"
    pose = object_dir / "pose.json"
    cameras = package_root / "artifacts" / "cameras_json.json"
    context = {
        "image": image_out,
        "mask": mask_out,
        "instance_json": instance_json,
        "object_dir": object_dir,
        "object_ply": gaussian,
        "object_mesh": mesh,
        "object_pose": pose,
        "cameras_json": cameras if cameras.is_file() else "",
    }
    return {
        "id": task_id,
        "name": name,
        "label": label,
        "image": image_out.as_posix(),
        "mask": mask_out.as_posix(),
        "instance_json": instance_json.as_posix(),
        "output_dir": object_dir.as_posix(),
        "gaussian": gaussian.as_posix(),
        "mesh": mesh.as_posix(),
        "pose": pose.as_posix(),
        "context": context,
    }


def render_command(template: str, context: dict[str, Any]) -> list[str]:
    rendered = template.format_map({key: str(value) for key, value in context.items()})
    return shlex.split(rendered)


def production_object_from_job(job: dict[str, Any], *, package_root: Path) -> dict[str, Any]:
    gaussian = require_file(Path(job["gaussian"]), "SAM-3D object gaussian")
    mesh = require_file(Path(job["mesh"]), "SAM-3D object mesh")
    pose_path = require_file(Path(job["pose"]), "SAM-3D object pose")
    pose = json.loads(pose_path.read_text(encoding="utf-8"))
    metadata = Path(job["output_dir"]) / "sam3d_metadata.json"
    return {
        "name": job["name"],
        "label": job["label"],
        "gaussian_ply": gaussian.as_posix(),
        "mesh_obj": mesh.as_posix(),
        "pose": pose,
        "dynamic": True,
        "source_instance_id": f"sam3d_{job['id']}",
        "provisional": False,
        "reconstruction_quality": {
            "reconstruction_source": "sam3d_objects",
            "commercial_asset_grade": True,
            "has_gaussian_asset": True,
            "has_mesh_asset": True,
            "has_pose": True,
            "sam3d_metadata": metadata.as_posix() if metadata.is_file() else None,
        },
    }


def require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing {label}: {path}")
    return path


def local_crop_mask(global_mask: np.ndarray, bbox: Any, crop_shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return np.ones(crop_shape, dtype=bool)
    x0, y0, x1, y1 = [int(value) for value in bbox]
    cropped = global_mask[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] > 0.5
    if cropped.shape != crop_shape:
        return np.ones(crop_shape, dtype=bool)
    return cropped


def best_prompt_label(task: dict[str, Any]) -> str:
    prompts = task.get("prompt_candidates")
    if isinstance(prompts, list) and prompts:
        return str(prompts[0])
    return "movable_object"


def safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_") or "object"


if __name__ == "__main__":
    raise SystemExit(main())
