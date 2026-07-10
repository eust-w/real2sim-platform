#!/usr/bin/env python3
"""Export a COLMAP pose-initialized training dataset from a capture report."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any


COLMAP_TEXT_FILES = ("cameras.txt", "images.txt", "points3D.txt")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export registered COLMAP frames as a pose-init 3DGS training dataset")
    parser.add_argument("--sfm-report", type=Path, required=True, help="COLMAP capture report JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output pose-init dataset directory")
    parser.add_argument("--allow-failed-sfm", action="store_true", help="Export even when sfm_report.passed is false")
    args = parser.parse_args()

    summary = export_pose_init_dataset(
        sfm_report=args.sfm_report,
        output_dir=args.output_dir,
        allow_failed_sfm=bool(args.allow_failed_sfm),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def export_pose_init_dataset(*, sfm_report: Path, output_dir: Path, allow_failed_sfm: bool = False) -> dict[str, Any]:
    report = read_json(sfm_report)
    if not report.get("passed") and not allow_failed_sfm:
        raise SystemExit("SFM report did not pass; rerun with --allow-failed-sfm to export anyway")

    frames_dir = Path(str(report.get("frames_dir", "")))
    if not frames_dir.is_dir():
        raise SystemExit(f"SFM report frames_dir does not exist: {frames_dir}")
    text_model = Path(str(report.get("best_model_text_dir", "")))
    if not text_model.is_dir():
        raise SystemExit(f"SFM report best_model_text_dir does not exist: {text_model}")

    camera_models = {int(item["camera_id"]): item for item in report.get("camera_models", [])}
    camera_poses = list(report.get("camera_poses", []))
    if not camera_models:
        raise SystemExit("SFM report contains no camera models")
    if not camera_poses:
        raise SystemExit("SFM report contains no registered camera poses")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    image_dir = output_dir / "images"
    colmap_text_dir = output_dir / "colmap_text"
    image_dir.mkdir(parents=True, exist_ok=True)
    colmap_text_dir.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    copied_images = 0
    for pose in sorted(camera_poses, key=lambda item: str(item.get("image_name", ""))):
        image_name = str(pose.get("image_name", ""))
        source_image = frames_dir / image_name
        if not source_image.is_file():
            raise SystemExit(f"registered COLMAP frame is missing: {source_image}")
        dest_image = image_dir / source_image.name
        shutil.copy2(source_image, dest_image)
        copied_images += 1

        camera_id = int(pose["camera_id"])
        camera = camera_models.get(camera_id)
        if camera is None:
            raise SystemExit(f"pose references unknown camera_id {camera_id}: {image_name}")
        intrinsics = camera_intrinsics(camera)
        colmap_c2w = colmap_camera_to_world(pose["qvec_wxyz"], pose["tvec"])
        nerfstudio_c2w = colmap_c2w_to_nerfstudio(colmap_c2w)
        frames.append(
            {
                "file_path": f"images/{dest_image.name}",
                "transform_matrix": round_matrix(nerfstudio_c2w),
                "colmap_transform_matrix": round_matrix(colmap_c2w),
                "fl_x": intrinsics["fl_x"],
                "fl_y": intrinsics["fl_y"],
                "cx": intrinsics["cx"],
                "cy": intrinsics["cy"],
                "w": intrinsics["w"],
                "h": intrinsics["h"],
                "camera_model": intrinsics["camera_model"],
                "real2sim_pose": {
                    "image_id": pose.get("image_id"),
                    "camera_id": camera_id,
                    "qvec_wxyz": pose.get("qvec_wxyz"),
                    "tvec": pose.get("tvec"),
                    "position_colmap_world": pose.get("position"),
                },
            }
        )

    sparse_ply = write_sparse_points_ply(text_model / "points3D.txt", output_dir / "colmap_sparse_points.ply")
    copied_colmap_files = copy_colmap_text_model(text_model, colmap_text_dir)
    transforms = {
        "format": "gs-playground-real2sim-pose-init-transforms-v1",
        "camera_convention": "nerfstudio/OpenGL camera-to-world; raw COLMAP camera-to-world is preserved per frame",
        "source_sfm_report": sfm_report.resolve().as_posix(),
        "frame_count": len(frames),
        "registered_image_count": int(report.get("registered_image_count", len(frames))),
        "sparse_point_count": int(report.get("sparse_point_count", 0) or 0),
        "camera_path": report.get("camera_path", {}),
        "sparse_bounds": report.get("sparse_bounds", {}),
        "database_stats": report.get("database_stats", {}),
        "frames": frames,
    }
    (output_dir / "transforms.json").write_text(json.dumps(transforms, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_readme(output_dir, transforms, sparse_ply=sparse_ply)

    summary = {
        "format": "gs-playground-real2sim-pose-init-dataset-v1",
        "root": output_dir.resolve().as_posix(),
        "transforms": "transforms.json",
        "images": "images",
        "image_count": copied_images,
        "colmap_text": "colmap_text",
        "colmap_text_files": copied_colmap_files,
        "sparse_points_ply": sparse_ply.name if sparse_ply is not None else None,
        "sparse_point_count": int(report.get("sparse_point_count", 0) or 0),
        "registered_ratio": report.get("registered_ratio"),
        "ready_for_training": bool(report.get("passed") and copied_images > 0 and sparse_ply is not None),
    }
    (output_dir / "pose_init_dataset.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def camera_intrinsics(camera: dict[str, Any]) -> dict[str, Any]:
    model = str(camera.get("model", "")).upper()
    width = int(camera["width"])
    height = int(camera["height"])
    params = [float(value) for value in camera.get("params", [])]
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        fx = fy = f
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif model in {"SIMPLE_RADIAL", "RADIAL"}:
        f, cx, cy = params[:3]
        fx = fy = f
    elif model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        fx, fy, cx, cy = params[:4]
    else:
        raise SystemExit(f"unsupported COLMAP camera model for pose-init export: {model}")
    return {
        "camera_model": model,
        "w": width,
        "h": height,
        "fl_x": round(float(fx), 6),
        "fl_y": round(float(fy), 6),
        "cx": round(float(cx), 6),
        "cy": round(float(cy), 6),
    }


def colmap_camera_to_world(qvec: list[float], tvec: list[float]) -> list[list[float]]:
    rotation = qvec_to_rotmat([float(value) for value in qvec])
    t = [float(value) for value in tvec]
    rt = transpose(rotation)
    center = [-sum(rt[row][col] * t[col] for col in range(3)) for row in range(3)]
    return [
        [rt[0][0], rt[0][1], rt[0][2], center[0]],
        [rt[1][0], rt[1][1], rt[1][2], center[1]],
        [rt[2][0], rt[2][1], rt[2][2], center[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def colmap_c2w_to_nerfstudio(matrix: list[list[float]]) -> list[list[float]]:
    converted = [list(row) for row in matrix]
    for row in range(3):
        converted[row][1] *= -1.0
        converted[row][2] *= -1.0
    return converted


def qvec_to_rotmat(qvec: list[float]) -> list[list[float]]:
    w, x, y, z = qvec
    return [
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ]


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[col][row] for col in range(3)] for row in range(3)]


def round_matrix(matrix: list[list[float]], digits: int = 8) -> list[list[float]]:
    return [[round(float(value), digits) for value in row] for row in matrix]


def write_sparse_points_ply(points_txt: Path, output: Path) -> Path | None:
    points: list[tuple[float, float, float, int, int, int]] = []
    if not points_txt.is_file():
        return None
    for line in points_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        x, y, z = (float(parts[index]) for index in range(1, 4))
        r, g, b = (clamp_color(float(parts[index])) for index in range(4, 7))
        points.append((x, y, z, r, g, b))
    if not points:
        return None
    with output.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for x, y, z, r, g, b in points:
            handle.write(f"{x:.8f} {y:.8f} {z:.8f} {r} {g} {b}\n")
    return output


def clamp_color(value: float) -> int:
    return max(0, min(255, int(round(value))))


def copy_colmap_text_model(text_model: Path, output: Path) -> list[str]:
    copied: list[str] = []
    for name in COLMAP_TEXT_FILES:
        source = text_model / name
        if source.is_file():
            shutil.copy2(source, output / name)
            copied.append(name)
    return copied


def write_readme(output_dir: Path, transforms: dict[str, Any], *, sparse_ply: Path | None) -> None:
    lines = [
        "# Real2Sim Pose-Init Dataset",
        "",
        "This directory is generated from a COLMAP capture report that passed the Real2Sim capture gate.",
        "It is intended as the bridge from fast AnySplat preview reconstruction to a pose-initialized 3DGS training path.",
        "",
        "Contents:",
        "",
        "- `images/`: registered video frames copied from the SfM frame set.",
        "- `transforms.json`: Nerfstudio/OpenGL-style camera-to-world transforms plus raw COLMAP pose metadata.",
        "- `colmap_text/`: the selected COLMAP text model (`cameras.txt`, `images.txt`, `points3D.txt`).",
        "- `colmap_sparse_points.ply`: colored sparse point cloud exported from `points3D.txt`.",
        "",
        "Summary:",
        "",
        f"- frames: {transforms.get('frame_count')}",
        f"- registered images: {transforms.get('registered_image_count')}",
        f"- sparse points: {transforms.get('sparse_point_count')}",
        f"- sparse PLY: {sparse_ply.name if sparse_ply is not None else 'missing'}",
        "",
        "A downstream trainer should initialize from these poses and sparse points instead of random camera/point initialization.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
