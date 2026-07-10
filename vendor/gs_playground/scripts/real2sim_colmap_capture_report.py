#!/usr/bin/env python3
"""Run a COLMAP SfM preflight report for Real2Sim video keyframes."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}


class ColmapCommandError(RuntimeError):
    def __init__(self, command: list[str], returncode: int) -> None:
        super().__init__(f"COLMAP command failed with exit code {returncode}: {' '.join(command[:3])}")
        self.command = command
        self.returncode = returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a COLMAP capture-quality report from selected frames")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory containing selected image frames")
    parser.add_argument("--work-dir", type=Path, required=True, help="COLMAP intermediate directory")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON report")
    parser.add_argument("--colmap-bin", default="colmap")
    parser.add_argument("--matcher", choices=["exhaustive", "sequential"], default="sequential")
    parser.add_argument("--sequential-overlap", type=int, default=10)
    parser.add_argument("--camera-model", default="SIMPLE_PINHOLE")
    parser.add_argument("--max-image-size", type=int, default=0)
    parser.add_argument("--sift-max-num-features", type=int, default=8192)
    parser.add_argument("--sift-peak-threshold", type=float, default=None)
    parser.add_argument("--feature-threads", type=int, default=1)
    parser.add_argument("--matcher-threads", type=int, default=1)
    parser.add_argument("--guided-matching", action="store_true")
    parser.add_argument("--mapper-random-seed", type=int, default=0)
    parser.add_argument("--min-registered-ratio", type=float, default=0.70)
    parser.add_argument("--min-sparse-points", type=int, default=500)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    report = run_colmap_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.require_pass and not report["passed"]:
        return 1
    return 0


def run_colmap_report(args: argparse.Namespace) -> dict[str, Any]:
    frames = collect_frames(args.frames_dir)
    if not frames:
        raise SystemExit(f"frames directory contains no supported images: {args.frames_dir}")
    colmap = resolve_colmap(args.colmap_bin)

    work = args.work_dir.resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    database = work / "database.db"
    sparse = work / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    text_model = work / "sparse_text"

    commands: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        feature_command = [
            colmap,
            "feature_extractor",
            "--database_path",
            database.as_posix(),
            "--image_path",
            args.frames_dir.resolve().as_posix(),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            str(args.camera_model).upper(),
        ]
        feature_gpu_option = colmap_option(colmap, "feature_extractor", ["--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu"])
        if feature_gpu_option is not None:
            feature_command.extend([feature_gpu_option, "0"])
        feature_threads_option = colmap_option(
            colmap,
            "feature_extractor",
            ["--FeatureExtraction.num_threads", "--SiftExtraction.num_threads"],
        )
        if feature_threads_option is not None and int(args.feature_threads) > 0:
            feature_command.extend([feature_threads_option, str(int(args.feature_threads))])
        if int(args.max_image_size) > 0:
            append_option_if_supported(
                colmap,
                "feature_extractor",
                feature_command,
                ["--SiftExtraction.max_image_size"],
                str(int(args.max_image_size)),
            )
        append_option_if_supported(
            colmap,
            "feature_extractor",
            feature_command,
            ["--SiftExtraction.max_num_features"],
            str(max(int(args.sift_max_num_features), 1)),
        )
        if args.sift_peak_threshold is not None:
            append_option_if_supported(
                colmap,
                "feature_extractor",
                feature_command,
                ["--SiftExtraction.peak_threshold"],
                str(float(args.sift_peak_threshold)),
            )
        run_logged(feature_command, commands)
        matcher_command = build_matcher_command(
            colmap,
            args.matcher,
            database,
            sequential_overlap=args.sequential_overlap,
            matcher_threads=int(args.matcher_threads),
            guided_matching=bool(args.guided_matching),
        )
        run_logged(matcher_command, commands)
        mapper_command = [
            colmap,
            "mapper",
            "--database_path",
            database.as_posix(),
            "--image_path",
            args.frames_dir.resolve().as_posix(),
            "--output_path",
            sparse.as_posix(),
        ]
        append_option_if_supported(
            colmap,
            "mapper",
            mapper_command,
            ["--Mapper.random_seed"],
            str(int(args.mapper_random_seed)),
        )
        run_logged(mapper_command, commands)
    except ColmapCommandError as exc:
        warnings.append(str(exc))

    model_dir, registered_images, sparse_points, model_candidates = select_best_model_dir(
        colmap,
        sparse,
        text_model,
        commands,
    )
    if model_dir is None:
        warnings.append("COLMAP mapper produced no sparse model")

    registered_ratio = len(registered_images) / max(float(len(frames)), 1.0)
    passed = registered_ratio >= args.min_registered_ratio and sparse_points >= args.min_sparse_points
    if registered_ratio < args.min_registered_ratio:
        warnings.append(
            f"registered image ratio {registered_ratio:.3f} < {float(args.min_registered_ratio):.3f}"
        )
    if sparse_points < args.min_sparse_points:
        warnings.append(f"sparse point count {sparse_points} < {int(args.min_sparse_points)}")
    database_stats = read_database_stats(database)
    geometry = read_best_model_geometry(text_model, work, warnings)

    return {
        "format": "gs-playground-real2sim-colmap-capture-report-v1",
        "frames_dir": args.frames_dir.resolve().as_posix(),
        "frame_count": len(frames),
        "frame_names": [path.name for path in frames],
        "colmap_bin": colmap,
        "matcher": args.matcher,
        "camera_model": str(args.camera_model).upper(),
        "max_image_size": int(args.max_image_size),
        "sift_max_num_features": int(args.sift_max_num_features),
        "sift_peak_threshold": float(args.sift_peak_threshold) if args.sift_peak_threshold is not None else None,
        "feature_threads": int(args.feature_threads),
        "matcher_threads": int(args.matcher_threads),
        "model_dir": model_dir.as_posix() if model_dir is not None else None,
        "model_candidates": model_candidates,
        "registered_image_count": len(registered_images),
        "registered_ratio": round(float(registered_ratio), 4),
        "registered_images": registered_images,
        "sparse_point_count": int(sparse_points),
        "best_model_text_dir": text_model.as_posix() if model_dir is not None else None,
        "camera_models": geometry["camera_models"],
        "camera_poses": geometry["camera_poses"],
        "camera_path": geometry["camera_path"],
        "sparse_bounds": geometry["sparse_bounds"],
        "coverage_preview": geometry["coverage_preview"],
        "database_stats": database_stats,
        "gates": {
            "min_registered_ratio": float(args.min_registered_ratio),
            "min_sparse_points": int(args.min_sparse_points),
        },
        "passed": bool(passed),
        "warnings": warnings,
        "recommendations": capture_recommendations(
            passed=bool(passed),
            registered_ratio=registered_ratio,
            sparse_points=sparse_points,
            database_stats=database_stats,
            warnings=warnings,
        ),
        "commands": commands,
    }


def collect_frames(frames_dir: Path) -> list[Path]:
    if not frames_dir.is_dir():
        raise SystemExit(f"frames directory not found: {frames_dir}")
    return sorted(path for path in frames_dir.iterdir() if is_frame_image(path))


def is_frame_image(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    name = path.name.lower()
    return "contact_sheet" not in name and "preview" not in name


def resolve_colmap(colmap_bin: str) -> str:
    path = shutil.which(colmap_bin)
    if path is not None:
        return path
    candidate = Path(colmap_bin)
    if candidate.is_file():
        return candidate.resolve().as_posix()
    raise SystemExit(f"COLMAP binary not found: {colmap_bin}")


def run_logged(command: list[str], commands: list[dict[str, Any]]) -> None:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    commands.append(
        {
            "argv": list(command),
            "returncode": completed.returncode,
            "output_tail": "\n".join(completed.stdout.splitlines()[-80:]),
        }
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        raise ColmapCommandError(command, completed.returncode)


def build_matcher_command(
    colmap: str,
    matcher: str,
    database: Path,
    *,
    sequential_overlap: int,
    matcher_threads: int,
    guided_matching: bool,
) -> list[str]:
    command_name = "sequential_matcher" if matcher == "sequential" else "exhaustive_matcher"
    command = [
        colmap,
        command_name,
        "--database_path",
        database.as_posix(),
    ]
    matcher_gpu_option = colmap_option(colmap, command_name, ["--FeatureMatching.use_gpu", "--SiftMatching.use_gpu"])
    if matcher_gpu_option is not None:
        command.extend([matcher_gpu_option, "0"])
    matcher_threads_option = colmap_option(
        colmap,
        command_name,
        ["--FeatureMatching.num_threads", "--SiftMatching.num_threads"],
    )
    if matcher_threads_option is not None and matcher_threads > 0:
        command.extend([matcher_threads_option, str(matcher_threads)])
    if guided_matching:
        append_option_if_supported(
            colmap,
            command_name,
            command,
            ["--FeatureMatching.guided_matching", "--SiftMatching.guided_matching"],
            "1",
        )
    append_cpu_bruteforce_option(colmap, command_name, command)
    if command_name == "sequential_matcher":
        overlap_option = colmap_option(colmap, command_name, ["--SequentialMatching.overlap"])
        if overlap_option is not None:
            command.extend([overlap_option, str(max(int(sequential_overlap), 1))])
    return command


def colmap_option(colmap: str, command: str, names: list[str]) -> str | None:
    completed = subprocess.run([colmap, command, "-h"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    help_text = completed.stdout
    for name in names:
        if name in help_text:
            return name
    return None


def append_cpu_bruteforce_option(colmap: str, command: str, argv: list[str]) -> None:
    option = colmap_option(colmap, command, ["--SiftMatching.cpu_brute_force_matcher"])
    if option is not None:
        argv.extend([option, "1"])


def append_option_if_supported(
    colmap: str,
    command: str,
    argv: list[str],
    option_names: list[str],
    value: str,
) -> None:
    option = colmap_option(colmap, command, option_names)
    if option is not None:
        argv.extend([option, value])


def select_best_model_dir(
    colmap: str,
    sparse_dir: Path,
    text_model: Path,
    commands: list[dict[str, Any]],
) -> tuple[Path | None, list[str], int, list[dict[str, Any]]]:
    candidates = sorted(path for path in sparse_dir.iterdir() if path.is_dir()) if sparse_dir.is_dir() else []
    if not candidates:
        return None, [], 0, []

    candidate_text_root = text_model.parent / f"{text_model.name}_candidates"
    if candidate_text_root.exists():
        shutil.rmtree(candidate_text_root)
    candidate_text_root.mkdir(parents=True, exist_ok=True)

    scored: list[tuple[int, int, Path, Path, list[str]]] = []
    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_text = candidate_text_root / candidate.name
        candidate_text.mkdir(parents=True, exist_ok=True)
        run_logged(
            [
                colmap,
                "model_converter",
                "--input_path",
                candidate.as_posix(),
                "--output_path",
                candidate_text.as_posix(),
                "--output_type",
                "TXT",
            ],
            commands,
        )
        registered = parse_registered_images(candidate_text / "images.txt")
        points = count_sparse_points(candidate_text / "points3D.txt")
        summaries.append(
            {
                "model_dir": candidate.as_posix(),
                "registered_image_count": len(registered),
                "sparse_point_count": points,
            }
        )
        scored.append((len(registered), points, candidate, candidate_text, registered))

    registered_count, points, model_dir, candidate_text, registered_images = max(
        scored,
        key=lambda item: (item[0], item[1]),
    )
    if text_model.exists():
        shutil.rmtree(text_model)
    shutil.copytree(candidate_text, text_model)
    return model_dir, registered_images, int(points), summaries


def parse_registered_images(path: Path) -> list[str]:
    if not path.is_file():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    data_lines = [line for line in lines if line and not line.startswith("#")]
    registered: list[str] = []
    for index in range(0, len(data_lines), 2):
        parts = data_lines[index].split(maxsplit=9)
        if len(parts) >= 10:
            registered.append(parts[9])
    return registered


def count_sparse_points(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(
        1
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip() and not line.startswith("#")
    )


def read_best_model_geometry(text_model: Path, work_dir: Path, warnings: list[str]) -> dict[str, Any]:
    camera_models = parse_camera_models(text_model / "cameras.txt")
    camera_poses = parse_camera_poses(text_model / "images.txt")
    sparse_points, sparse_bounds = parse_sparse_points(text_model / "points3D.txt")
    camera_path = summarize_camera_path(camera_poses)
    coverage_preview = None
    if camera_poses and sparse_points:
        try:
            coverage_preview = write_coverage_preview(
                camera_poses,
                sparse_points,
                work_dir / "sfm_capture_coverage.png",
            ).as_posix()
        except Exception as exc:
            warnings.append(f"failed to write COLMAP coverage preview: {exc}")
    return {
        "camera_models": camera_models,
        "camera_poses": camera_poses,
        "camera_path": camera_path,
        "sparse_bounds": sparse_bounds,
        "coverage_preview": coverage_preview,
    }


def parse_camera_models(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    cameras: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cameras.append(
            {
                "camera_id": int(parts[0]),
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(value) for value in parts[4:]],
            }
        )
    return cameras


def parse_camera_poses(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    data_lines = [line for line in lines if line and not line.startswith("#")]
    poses: list[dict[str, Any]] = []
    for index in range(0, len(data_lines), 2):
        parts = data_lines[index].split(maxsplit=9)
        if len(parts) < 10:
            continue
        qvec = [float(value) for value in parts[1:5]]
        tvec = [float(value) for value in parts[5:8]]
        position = camera_center_from_qt(qvec, tvec)
        poses.append(
            {
                "image_id": int(parts[0]),
                "image_name": parts[9],
                "camera_id": int(parts[8]),
                "qvec_wxyz": round_list(qvec),
                "tvec": round_list(tvec),
                "position": round_list(position),
            }
        )
    return sorted(poses, key=lambda item: item["image_name"])


def camera_center_from_qt(qvec: list[float], tvec: list[float]) -> list[float]:
    rotation = qvec_to_rotmat(qvec)
    return [
        -sum(rotation[row][col] * tvec[row] for row in range(3))
        for col in range(3)
    ]


def qvec_to_rotmat(qvec: list[float]) -> list[list[float]]:
    w, x, y, z = qvec
    return [
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ]


def parse_sparse_points(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        return [], {}
    points: list[dict[str, Any]] = []
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        color = [int(float(parts[4])), int(float(parts[5])), int(float(parts[6]))]
        for axis in range(3):
            mins[axis] = min(mins[axis], xyz[axis])
            maxs[axis] = max(maxs[axis], xyz[axis])
        if len(points) < 20000:
            points.append({"xyz": xyz, "rgb": color})
    if any(not math.isfinite(value) for value in mins + maxs):
        return points, {}
    center = [(mins[axis] + maxs[axis]) / 2.0 for axis in range(3)]
    extent = [maxs[axis] - mins[axis] for axis in range(3)]
    robust_mins = [percentile([point["xyz"][axis] for point in points], 0.02) for axis in range(3)]
    robust_maxs = [percentile([point["xyz"][axis] for point in points], 0.98) for axis in range(3)]
    return points, {
        "bounds_min": round_list(mins),
        "bounds_max": round_list(maxs),
        "bounds_center": round_list(center),
        "bounds_extent": round_list(extent),
        "robust_bounds_min": round_list(robust_mins),
        "robust_bounds_max": round_list(robust_maxs),
        "robust_bounds_extent": round_list([robust_maxs[axis] - robust_mins[axis] for axis in range(3)]),
        "preview_point_count": len(points),
    }


def summarize_camera_path(camera_poses: list[dict[str, Any]]) -> dict[str, Any]:
    if not camera_poses:
        return {}
    positions = [pose["position"] for pose in camera_poses]
    mins = [min(position[axis] for position in positions) for axis in range(3)]
    maxs = [max(position[axis] for position in positions) for axis in range(3)]
    path_length = sum(
        distance(positions[index - 1], positions[index])
        for index in range(1, len(positions))
    )
    return {
        "pose_count": len(camera_poses),
        "path_length": round(float(path_length), 6),
        "start_image": camera_poses[0]["image_name"],
        "end_image": camera_poses[-1]["image_name"],
        "bounds_min": round_list(mins),
        "bounds_max": round_list(maxs),
        "bounds_extent": round_list([maxs[axis] - mins[axis] for axis in range(3)]),
    }


def write_coverage_preview(camera_poses: list[dict[str, Any]], sparse_points: list[dict[str, Any]], output: Path) -> Path:
    all_xyz = [point["xyz"] for point in sparse_points] + [pose["position"] for pose in camera_poses]
    extents = [
        max(point[axis] for point in all_xyz) - min(point[axis] for point in all_xyz)
        for axis in range(3)
    ]
    axes = sorted(range(3), key=lambda axis: extents[axis], reverse=True)[:2]
    axis_x, axis_y = axes[0], axes[1]
    xs = [point[axis_x] for point in all_xyz]
    ys = [point[axis_y] for point in all_xyz]
    min_x, max_x = robust_axis_bounds(xs)
    min_y, max_y = robust_axis_bounds(ys)
    pad_x = max((max_x - min_x) * 0.08, 1e-6)
    pad_y = max((max_y - min_y) * 0.08, 1e-6)
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    width, height = 1200, 820
    margin = 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((margin, margin, width - margin, height - margin), outline=(210, 210, 210))

    def project(point: list[float]) -> tuple[int, int]:
        px = margin + (point[axis_x] - min_x) / max(max_x - min_x, 1e-9) * (width - 2 * margin)
        py = height - margin - (point[axis_y] - min_y) / max(max_y - min_y, 1e-9) * (height - 2 * margin)
        return clamp_int(px, 0, width - 1), clamp_int(py, 0, height - 1)

    for point in sparse_points:
        x, y = project(point["xyz"])
        rgb = point.get("rgb", [150, 150, 150])
        color = tuple(max(45, min(210, int(value))) for value in rgb)
        draw.point((x, y), fill=color)

    camera_xy = [project(pose["position"]) for pose in camera_poses]
    if len(camera_xy) >= 2:
        draw.line(camera_xy, fill=(28, 98, 201), width=3)
    for idx, (x, y) in enumerate(camera_xy):
        radius = 5 if idx not in {0, len(camera_xy) - 1} else 8
        color = (220, 48, 48) if idx == 0 else (34, 148, 83) if idx == len(camera_xy) - 1 else (28, 98, 201)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255))

    draw.text((margin, 22), "COLMAP capture coverage: sparse points + registered camera path", fill=(20, 20, 20))
    draw.text((margin, height - 44), f"Axes: {axis_name(axis_x)} / {axis_name(axis_y)}   Cameras: {len(camera_poses)}   Preview points: {len(sparse_points)}", fill=(70, 70, 70))
    if camera_xy:
        sx, sy = camera_xy[0]
        ex, ey = camera_xy[-1]
        draw.text((sx + 10, sy + 10), "start", fill=(140, 30, 30))
        draw.text((ex + 10, ey + 10), "end", fill=(20, 105, 55))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def axis_name(axis: int) -> str:
    return ("x", "y", "z")[axis]


def robust_axis_bounds(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = percentile(values, 0.02)
    high = percentile(values, 0.98)
    if high <= low:
        return min(values), max(values)
    return low, high


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(max(int(round((len(ordered) - 1) * fraction)), 0), len(ordered) - 1)
    return ordered[index]


def clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[axis] - b[axis]) ** 2 for axis in range(3)))


def round_list(values: list[float], digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def read_database_stats(database: Path) -> dict[str, Any]:
    if not database.is_file():
        return {}
    try:
        connection = sqlite3.connect(database.as_posix())
        try:
            image_count = table_count(connection, "images")
            keypoint_rows = connection.execute("SELECT rows FROM keypoints").fetchall()
            match_rows = connection.execute("SELECT rows FROM matches").fetchall()
            geometry_rows = connection.execute("SELECT rows FROM two_view_geometries").fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return {}

    keypoint_counts = [int(row[0]) for row in keypoint_rows]
    match_counts = [int(row[0]) for row in match_rows if int(row[0]) > 0]
    geometry_counts = [int(row[0]) for row in geometry_rows if int(row[0]) > 0]
    return {
        "image_count": image_count,
        "keypoint_total": int(sum(keypoint_counts)),
        "keypoint_mean": round(float(sum(keypoint_counts) / max(len(keypoint_counts), 1)), 2),
        "matched_pair_count": len(match_counts),
        "matched_pair_mean": round(float(sum(match_counts) / max(len(match_counts), 1)), 2) if match_counts else 0.0,
        "verified_pair_count": len(geometry_counts),
        "verified_pair_mean": round(float(sum(geometry_counts) / max(len(geometry_counts), 1)), 2) if geometry_counts else 0.0,
    }


def table_count(connection: sqlite3.Connection, table: str) -> int:
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def capture_recommendations(
    *,
    passed: bool,
    registered_ratio: float,
    sparse_points: int,
    database_stats: dict[str, Any],
    warnings: list[str],
) -> list[str]:
    if passed:
        return ["COLMAP registered enough frames for a pose-initialized reconstruction path."]
    recommendations = []
    warning_text = "\n".join(warnings).lower()
    verified_pairs = int(database_stats.get("verified_pair_count", 0) or 0)
    if "no sparse model" in warning_text or "mapper" in warning_text:
        recommendations.append("Capture slower video with stronger sideways parallax; avoid pure forward motion or mostly rotational sweeps.")
    if verified_pairs < 3:
        recommendations.append("Increase overlap between adjacent views and keep textured surfaces in frame so COLMAP can verify image pairs.")
    if registered_ratio < 0.7:
        recommendations.append("Collect 30-80 sharp frames around the same room volume before attempting commercial-grade reconstruction.")
    if sparse_points < 500:
        recommendations.append("Avoid motion blur, reflective glass, blank walls, and very close foreground occluders during capture.")
    return recommendations


if __name__ == "__main__":
    raise SystemExit(main())
