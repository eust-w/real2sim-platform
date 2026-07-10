#!/usr/bin/env python3
"""Create a capture acceptance report for high-fidelity Real2Sim reconstruction."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate whether a capture is ready for high-fidelity Real2Sim reconstruction")
    parser.add_argument("--sfm-report", type=Path, required=True, help="COLMAP capture report JSON")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--min-registered-ratio", type=float, default=0.70)
    parser.add_argument("--min-registered-images", type=int, default=20)
    parser.add_argument("--min-sparse-points", type=int, default=5000)
    parser.add_argument("--min-verified-pairs", type=int, default=40)
    parser.add_argument(
        "--min-camera-path-length",
        type=float,
        default=1.0,
        help="Minimum COLMAP camera path length for space-level capture acceptance",
    )
    parser.add_argument(
        "--min-camera-planar-coverage-ratio",
        type=float,
        default=0.12,
        help="Minimum second-largest camera extent / largest extent ratio; rejects nearly 1D camera tracks",
    )
    parser.add_argument(
        "--max-camera-step-ratio",
        type=float,
        default=0.55,
        help="Maximum largest camera step / full path length ratio; rejects discontinuous captures",
    )
    args = parser.parse_args()

    sfm = json.loads(args.sfm_report.read_text(encoding="utf-8"))
    report = build_acceptance_report(sfm, args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    return 0 if report["accepted"] else 2


def build_acceptance_report(sfm: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    stats = sfm.get("database_stats") if isinstance(sfm.get("database_stats"), dict) else {}
    camera_metrics = camera_space_metrics(sfm)
    metrics = {
        "frame_count": int(sfm.get("frame_count", 0) or 0),
        "registered_image_count": int(sfm.get("registered_image_count", 0) or 0),
        "registered_ratio": float(sfm.get("registered_ratio", 0.0) or 0.0),
        "sparse_point_count": int(sfm.get("sparse_point_count", 0) or 0),
        "keypoint_total": int(stats.get("keypoint_total", 0) or 0),
        "keypoint_mean": float(stats.get("keypoint_mean", 0.0) or 0.0),
        "matched_pair_count": int(stats.get("matched_pair_count", 0) or 0),
        "verified_pair_count": int(stats.get("verified_pair_count", 0) or 0),
        "verified_pair_mean": float(stats.get("verified_pair_mean", 0.0) or 0.0),
        **camera_metrics,
    }
    gates = {
        "enough_frames": gate(metrics["frame_count"] >= args.min_frames, metrics["frame_count"], args.min_frames),
        "enough_registered_images": gate(
            metrics["registered_image_count"] >= args.min_registered_images,
            metrics["registered_image_count"],
            args.min_registered_images,
        ),
        "registered_ratio": gate(
            metrics["registered_ratio"] >= args.min_registered_ratio,
            round(metrics["registered_ratio"], 4),
            args.min_registered_ratio,
        ),
        "sparse_points": gate(metrics["sparse_point_count"] >= args.min_sparse_points, metrics["sparse_point_count"], args.min_sparse_points),
        "verified_pairs": gate(metrics["verified_pair_count"] >= args.min_verified_pairs, metrics["verified_pair_count"], args.min_verified_pairs),
        "camera_path_length": gate(
            metrics["camera_path_length"] >= args.min_camera_path_length,
            round(metrics["camera_path_length"], 4),
            args.min_camera_path_length,
        ),
        "camera_planar_coverage": gate(
            metrics["camera_planar_coverage_ratio"] >= args.min_camera_planar_coverage_ratio,
            round(metrics["camera_planar_coverage_ratio"], 4),
            args.min_camera_planar_coverage_ratio,
        ),
        "camera_step_consistency": gate(
            metrics["camera_max_step_ratio"] is not None and metrics["camera_max_step_ratio"] <= args.max_camera_step_ratio,
            round(metrics["camera_max_step_ratio"], 4) if metrics["camera_max_step_ratio"] is not None else None,
            args.max_camera_step_ratio,
        ),
    }
    accepted = all(item["passed"] for item in gates.values())
    blockers = [name for name, item in gates.items() if not item["passed"]]
    return {
        "format": "gs-playground-real2sim-capture-acceptance-v1",
        "accepted": accepted,
        "target_quality": "polycam_realitycapture_like_space",
        "decision": "ready_for_pose_initialized_reconstruction" if accepted else "recapture_required_before_high_fidelity_reconstruction",
        "metrics": metrics,
        "gates": gates,
        "blockers": blockers,
        "sfm_warnings": sfm.get("warnings", []),
        "sfm_recommendations": sfm.get("recommendations", []),
        "capture_requirements": capture_requirements(blockers, metrics),
    }


def camera_space_metrics(sfm: dict[str, Any]) -> dict[str, Any]:
    path = sfm.get("camera_path") if isinstance(sfm.get("camera_path"), dict) else {}
    poses = camera_positions(sfm.get("camera_poses"))
    if poses:
        extents = axis_extents(poses)
        path_length = path_length_from_positions(poses)
        step_distances = [
            distance(poses[index - 1], poses[index])
            for index in range(1, len(poses))
        ]
        closure_distance = distance(poses[0], poses[-1]) if len(poses) >= 2 else 0.0
    else:
        extents = numeric_list(path.get("bounds_extent"), expected=3) or [0.0, 0.0, 0.0]
        path_length = float(path.get("path_length", 0.0) or 0.0)
        step_distances = []
        closure_distance = None

    sorted_extents = sorted((max(float(value), 0.0) for value in extents), reverse=True)
    dominant_extent = sorted_extents[0] if sorted_extents else 0.0
    secondary_extent = sorted_extents[1] if len(sorted_extents) >= 2 else 0.0
    planar_ratio = secondary_extent / dominant_extent if dominant_extent > 1e-9 else 0.0
    max_step = max(step_distances) if step_distances else None
    median_step = percentile(step_distances, 0.5) if step_distances else None
    max_step_ratio = max_step / path_length if max_step is not None and path_length > 1e-9 else None
    closure_ratio = closure_distance / path_length if closure_distance is not None and path_length > 1e-9 else None
    return {
        "camera_pose_count": int(path.get("pose_count", len(poses)) or len(poses)),
        "camera_path_length": round(float(path_length), 6),
        "camera_bounds_extent": round_list(extents),
        "camera_dominant_extent": round(float(dominant_extent), 6),
        "camera_secondary_extent": round(float(secondary_extent), 6),
        "camera_planar_coverage_ratio": round(float(planar_ratio), 6),
        "camera_max_step_distance": round(float(max_step), 6) if max_step is not None else None,
        "camera_median_step_distance": round(float(median_step), 6) if median_step is not None else None,
        "camera_max_step_ratio": round(float(max_step_ratio), 6) if max_step_ratio is not None else None,
        "camera_loop_closure_distance": round(float(closure_distance), 6) if closure_distance is not None else None,
        "camera_loop_closure_ratio": round(float(closure_ratio), 6) if closure_ratio is not None else None,
    }


def camera_positions(raw_poses: Any) -> list[list[float]]:
    positions: list[list[float]] = []
    if not isinstance(raw_poses, list):
        return positions
    for pose in raw_poses:
        if not isinstance(pose, dict):
            continue
        position = numeric_list(pose.get("position"), expected=3)
        if position is not None:
            positions.append(position)
    return positions


def numeric_list(value: Any, *, expected: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != expected:
        return None
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in values):
        return None
    return values


def axis_extents(positions: list[list[float]]) -> list[float]:
    return [
        max(position[axis] for position in positions) - min(position[axis] for position in positions)
        for axis in range(3)
    ]


def path_length_from_positions(positions: list[list[float]]) -> float:
    return sum(distance(positions[index - 1], positions[index]) for index in range(1, len(positions)))


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(max(int(round((len(ordered) - 1) * fraction)), 0), len(ordered) - 1)
    return ordered[index]


def round_list(values: list[float]) -> list[float]:
    return [round(float(value), 6) for value in values]


def gate(passed: bool, value: Any, threshold: int | float) -> dict[str, Any]:
    return {"passed": bool(passed), "value": value, "threshold": threshold}


def capture_requirements(blockers: list[str], metrics: dict[str, Any]) -> list[str]:
    requirements = [
        "Capture one room-sized volume with continuous overlap; do not jump between unrelated viewpoints.",
        "Move slowly with strong sideways parallax around desks, cabinets, floor, and wall features.",
        "Keep 30-80 sharp frames after blur filtering; avoid fast pans, glass-only regions, and blank walls.",
    ]
    if "registered_ratio" in blockers or "enough_registered_images" in blockers:
        requirements.append(
            f"Current registration is {metrics['registered_image_count']} images / {metrics['frame_count']} frames; recapture until COLMAP registers at least 70 percent."
        )
    if "sparse_points" in blockers:
        requirements.append(
            f"Current sparse points are {metrics['sparse_point_count']}; add textured overlap and lateral baseline until sparse points exceed 5000."
        )
    if "verified_pairs" in blockers:
        requirements.append(
            f"Current verified pairs are {metrics['verified_pair_count']}; increase adjacent overlap and reduce motion blur."
        )
    if "camera_path_length" in blockers:
        requirements.append(
            f"Current registered camera path length is {metrics['camera_path_length']}; walk a longer continuous path with sideways parallax around the room volume."
        )
    if "camera_planar_coverage" in blockers:
        requirements.append(
            f"Current camera planar coverage ratio is {metrics['camera_planar_coverage_ratio']}; avoid a single straight-line sweep and capture from at least two lateral directions."
        )
    if "camera_step_consistency" in blockers:
        requirements.append(
            f"Current camera max-step ratio is {metrics['camera_max_step_ratio']}; remove discontinuous jumps or recapture as one continuous overlapping trajectory."
        )
    return requirements


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# Real2Sim Capture Acceptance Report",
        "",
        f"Decision: `{report['decision']}`",
        f"Accepted: `{str(report['accepted']).lower()}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in metrics.items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Gates", "", "| Gate | Value | Threshold | Passed |", "| --- | ---: | ---: | --- |"])
    for name, item in report["gates"].items():
        lines.append(f"| `{name}` | {item['value']} | {item['threshold']} | `{str(item['passed']).lower()}` |")
    if report["blockers"]:
        lines.extend(["", "## Blockers", ""])
        for blocker in report["blockers"]:
            lines.append(f"- `{blocker}`")
    if report["capture_requirements"]:
        lines.extend(["", "## Capture Requirements", ""])
        for requirement in report["capture_requirements"]:
            lines.append(f"- {requirement}")
    if report["sfm_warnings"]:
        lines.extend(["", "## SfM Warnings", ""])
        for warning in report["sfm_warnings"]:
            lines.append(f"- {warning}")
    if report["sfm_recommendations"]:
        lines.extend(["", "## SfM Recommendations", ""])
        for recommendation in report["sfm_recommendations"]:
            lines.append(f"- {recommendation}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
