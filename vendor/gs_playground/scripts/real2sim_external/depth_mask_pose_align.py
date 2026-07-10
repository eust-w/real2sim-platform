#!/usr/bin/env python3
"""Depth/mask pose alignment for Real2Sim object assets.

This optional stage refines object pose and scale after AnySplat scene
reconstruction and SAM-3D object reconstruction. It uses the same image-space
mask that was passed to SAM-3D, the scene depth map, and camera intrinsics to
place each object in the reconstructed scene coordinate frame.
"""

from __future__ import annotations

import argparse
import itertools
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, REPO_ROOT.as_posix())

from real2sim.contracts import load_instances, read_json, write_json
from real2sim.image_io import read_gray

MIN_RENDER_AREA_RATIO = 0.2
MAX_RENDER_AREA_RATIO = 4.0
MIN_RENDER_MASK_IOU = 0.005
MAX_GAUSSIAN_ALIGNMENT_POINTS = 50000
ALIGNMENT_REFINEMENT_ITERATIONS = 3


def main() -> int:
    parser = argparse.ArgumentParser(description="Refine object poses from masks, depth, and camera intrinsics")
    parser.add_argument("--assets", type=Path, required=True, help="Input Real2Sim manifest JSON")
    parser.add_argument("--instances", type=Path, required=True, help="Instance JSON with mask paths")
    parser.add_argument("--depth", type=Path, required=True, help="Scene depth map")
    parser.add_argument("--cameras", type=Path, required=True, help="Camera metadata JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output aligned manifest JSON")
    parser.add_argument("--work-dir", type=Path, default=None, help="Root used for relative asset paths")
    parser.add_argument("--depth-min", type=float, default=0.4)
    parser.add_argument("--depth-max", type=float, default=2.4)
    parser.add_argument("--z-offset", type=float, default=0.55)
    args = parser.parse_args()

    work_dir = args.work_dir or args.assets.parent.parent
    manifest = read_json(args.assets)
    refined = refine_assets(
        manifest,
        instances_json=args.instances,
        depth_path=args.depth,
        cameras_path=args.cameras,
        work_dir=work_dir,
        depth_range=(args.depth_min, args.depth_max),
        z_offset=args.z_offset,
    )
    write_json(args.output, refined)
    return 0


def refine_assets(
    manifest: dict[str, Any],
    *,
    instances_json: Path,
    depth_path: Path,
    cameras_path: Path,
    work_dir: Path,
    depth_range: tuple[float, float],
    z_offset: float,
) -> dict[str, Any]:
    refined = dict(manifest)
    objects = [dict(obj) for obj in manifest.get("objects", [])]
    refined["objects"] = objects

    alignment_meta: dict[str, Any] = {
        "backend": "depth_mask_pose_align",
        "depth": relative_or_absolute(depth_path, work_dir),
        "cameras": relative_or_absolute(cameras_path, work_dir),
        "instances": relative_or_absolute(instances_json, work_dir),
        "aligned_objects": [],
        "skipped_objects": [],
    }

    if not depth_path.exists() or not cameras_path.exists() or not instances_json.exists():
        alignment_meta["status"] = "skipped_missing_global_artifact"
        return with_alignment_metadata(refined, alignment_meta)

    depth = read_gray(depth_path)
    intrinsics = load_intrinsics(cameras_path, depth.shape)
    instances = {instance.instance_id: instance for instance in load_instances(instances_json)}

    for obj in objects:
        name = str(obj.get("name", obj.get("label", "object")))
        instance_id = str(obj.get("source_instance_id") or name)
        instance = instances.get(instance_id)
        if instance is None or instance.mask_path is None or not instance.mask_path.exists():
            alignment_meta["skipped_objects"].append({"name": name, "reason": "missing_instance_mask"})
            continue

        mask = read_gray(instance.mask_path) > 0.5
        if mask.shape != depth.shape:
            mask = resize_mask_to_camera_shape(mask, depth.shape)
        pose = dict(obj.get("pose") or {})
        gaussian_path = resolve_optional_path(obj.get("gaussian_ply") or obj.get("ply"), work_dir)
        render_failures: list[dict[str, Any]] = []
        if gaussian_path is not None and gaussian_path.exists():
            aligned_pose, render_meta = align_pose_by_rendered_gaussian(
                gaussian_path,
                pose,
                target_mask=mask,
                background_depth=depth,
                intrinsics=intrinsics,
                depth_range=depth_range,
                z_offset=z_offset,
            )
            if aligned_pose is not None:
                pose.update(aligned_pose)
                obj["pose"] = pose
                alignment_entry = {
                    "name": name,
                    "source_instance_id": instance_id,
                    "mask": relative_or_absolute(instance.mask_path, work_dir),
                    **render_meta,
                }
                alignment_meta["aligned_objects"].append(alignment_entry)
                continue
            render_failures.append(render_meta)

        mesh_path = resolve_optional_path(obj.get("mesh_obj") or obj.get("mesh"), work_dir)
        if mesh_path is not None and mesh_path.exists():
            aligned_pose, render_meta = align_pose_by_rendered_mesh(
                mesh_path,
                pose,
                target_mask=mask,
                background_depth=depth,
                intrinsics=intrinsics,
                depth_range=depth_range,
                z_offset=z_offset,
            )
            if aligned_pose is not None:
                pose.update(aligned_pose)
                obj["pose"] = pose
                alignment_entry = {
                    "name": name,
                    "source_instance_id": instance_id,
                    "mask": relative_or_absolute(instance.mask_path, work_dir),
                    **render_meta,
                }
                alignment_meta["aligned_objects"].append(alignment_entry)
                continue
            render_failures.append(render_meta)

        aligned_pose, fallback_meta = align_pose_by_mask_depth(
            pose,
            target_mask=mask,
            background_depth=depth,
            intrinsics=intrinsics,
            depth_range=depth_range,
            z_offset=z_offset,
            render_failures=render_failures,
        )
        if aligned_pose is None:
            alignment_meta["skipped_objects"].append(
                {
                    "name": name,
                    "reason": "empty_mask_depth_overlap",
                    "render_failures": render_failures,
                    **fallback_meta,
                }
            )
            continue

        pose.update(aligned_pose)
        obj["pose"] = pose
        alignment_meta["aligned_objects"].append(
            {
                "name": name,
                "source_instance_id": instance_id,
                "mask": relative_or_absolute(instance.mask_path, work_dir),
                **fallback_meta,
            }
        )

    alignment_meta["quality_summary"] = summarize_alignment_quality(alignment_meta["aligned_objects"])
    alignment_meta["status"] = "ok"
    return with_alignment_metadata(refined, alignment_meta)


def align_pose_by_mask_depth(
    pose: dict[str, Any],
    *,
    target_mask: np.ndarray,
    background_depth: np.ndarray,
    intrinsics: dict[str, float],
    depth_range: tuple[float, float],
    z_offset: float,
    render_failures: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    points = mask_depth_points(target_mask, background_depth, intrinsics, depth_range=depth_range, z_offset=z_offset)
    meta: dict[str, Any] = {
        "render_source": "mask_depth_fallback",
        "target_mask_pixels": int(target_mask.sum()),
        "mask_depth_points": int(points.shape[0]),
        "render_failures": render_failures or [],
    }
    if points.size == 0:
        return None, meta

    position = np.median(points, axis=0)
    extent = np.maximum(points.max(axis=0) - points.min(axis=0), np.array([0.03, 0.03, 0.03], dtype=np.float32))
    output_pose = {
        "position": vector_json(position),
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "scale": vector_json(extent),
    }
    meta["quat_source"] = "identity_mask_depth_fallback"
    meta["scale_source"] = "mask_depth_extent"
    if render_failures:
        meta["fallback_reason"] = "rendered_asset_mask_empty"
    elif any(key in pose for key in ("gaussian_ply", "mesh_obj", "mesh")):
        meta["fallback_reason"] = "renderable_asset_missing"
    else:
        meta["fallback_reason"] = "no_renderable_asset"
    return output_pose, meta


def align_pose_by_rendered_mesh(
    mesh_path: Path,
    pose: dict[str, Any],
    *,
    target_mask: np.ndarray,
    background_depth: np.ndarray,
    intrinsics: dict[str, float],
    depth_range: tuple[float, float],
    z_offset: float = 0.55,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    vertices, faces = load_obj_mesh(mesh_path)
    if vertices.size == 0:
        return None, {"render_source": "mesh_obj", "target_mask_pixels": int(target_mask.sum()), "rendered_mask_pixels": 0}
    return align_pose_by_rendered_asset(
        lambda position, scale, rotation: render_mesh_arrays_depth_mask(vertices, faces, position, scale, rotation, intrinsics, target_mask.shape),
        pose,
        target_mask=target_mask,
        background_depth=background_depth,
        intrinsics=intrinsics,
        depth_range=depth_range,
        render_source="mesh_obj",
        local_points=vertices,
        z_offset=z_offset,
    )


def align_pose_by_rendered_gaussian(
    gaussian_path: Path,
    pose: dict[str, Any],
    *,
    target_mask: np.ndarray,
    background_depth: np.ndarray,
    intrinsics: dict[str, float],
    depth_range: tuple[float, float],
    z_offset: float = 0.55,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    points = load_gaussian_points(gaussian_path)
    if points.size == 0:
        return None, {"render_source": "gaussian_ply", "target_mask_pixels": int(target_mask.sum()), "rendered_mask_pixels": 0}
    original_point_count = int(points.shape[0])
    points = downsample_alignment_points(points, max_points=MAX_GAUSSIAN_ALIGNMENT_POINTS)
    canonicalization = load_gaussian_canonicalization_metadata(gaussian_path)
    aligned_pose, render_meta = align_pose_by_rendered_asset(
        lambda position, scale, rotation: render_gaussian_depth_mask(points, position, scale, rotation, intrinsics, target_mask.shape),
        pose,
        target_mask=target_mask,
        background_depth=background_depth,
        intrinsics=intrinsics,
        depth_range=depth_range,
        render_source="gaussian_ply",
        local_points=points,
        z_offset=z_offset,
    )
    render_meta["gaussian_points"] = original_point_count
    render_meta["gaussian_alignment_points"] = int(points.shape[0])
    if canonicalization is not None:
        render_meta["gaussian_canonicalization"] = canonicalization
    return aligned_pose, render_meta


def align_pose_by_rendered_asset(
    render_depth_mask,
    pose: dict[str, Any],
    *,
    target_mask: np.ndarray,
    background_depth: np.ndarray,
    intrinsics: dict[str, float],
    depth_range: tuple[float, float],
    render_source: str,
    local_points: np.ndarray | None = None,
    z_offset: float = 0.55,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    position = np.asarray(pose.get("position", [0.0, 1.0, 0.55]), dtype=np.float32)
    scale = np.asarray(pose.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32)
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    rotation = rotation_matrix_from_quat_wxyz(pose.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]))
    target_pixels = int(target_mask.sum())
    if target_pixels <= 0:
        return None, {"render_source": render_source, "target_mask_pixels": target_pixels, "rendered_mask_pixels": 0}

    target_depth = denormalized_depth_median(background_depth, target_mask, depth_range)
    target_seed = estimate_mask_depth_pose(target_mask, background_depth, intrinsics, depth_range=depth_range, z_offset=z_offset)
    attempts: list[dict[str, Any]] = [
        align_render_candidate(
            render_depth_mask,
            position,
            scale,
            rotation,
            target_mask=target_mask,
            target_depth=target_depth,
            intrinsics=intrinsics,
            label="pose",
        )
    ]

    candidate_search = "pose_only"
    if should_run_axis_search(attempts[0], target_mask.shape):
        for candidate in axis_alignment_candidates(rotation):
            seed_scale = candidate_extent_scale(local_points, target_seed["extent"], candidate["rotation"], scale)
            attempts.append(
                align_render_candidate(
                    render_depth_mask,
                    target_seed["position"],
                    seed_scale,
                    candidate["rotation"],
                    target_mask=target_mask,
                    target_depth=target_depth,
                    intrinsics=intrinsics,
                    label=candidate["label"],
                )
            )
        candidate_search = "target_seed_axis_search"

    best = best_alignment_attempt(attempts)
    if best is None or int(best.get("rerendered_mask_pixels", 0)) <= 0:
        return (
            None,
            {
                "render_source": render_source,
                "target_mask_pixels": target_pixels,
                "rendered_mask_pixels": int(attempts[0].get("rendered_mask_pixels", 0)) if attempts else 0,
                "candidate_count": len(attempts),
                "candidate_attempts_top": compact_attempts(attempts),
            },
        )
    rejection_reason = render_alignment_rejection_reason(best)
    if rejection_reason is not None:
        return (
            None,
            {
                "render_source": render_source,
                "target_mask_pixels": target_pixels,
                "rendered_mask_pixels": int(best.get("rendered_mask_pixels", 0)),
                "rerendered_mask_pixels": int(best.get("rerendered_mask_pixels", 0)),
                "mask_iou": round(float(best.get("mask_iou", 0.0)), 7),
                "area_ratio": round(float(best.get("area_ratio", 0.0)), 7),
                "centroid_error_px": round(float(best.get("centroid_error_px", 0.0)), 7),
                "alignment_score": round(float(best.get("alignment_score", 0.0)), 7),
                "refinement_iterations": int(best.get("refinement_iterations", 0)),
                "rejection_reason": rejection_reason,
                "candidate_count": len(attempts),
                "candidate_attempts_top": compact_attempts(attempts),
            },
        )

    output_pose = {
        "position": vector_json(best["position"]),
        "quat_wxyz": vector_json(quat_wxyz_from_rotation_matrix(best["rotation"])),
        "scale": vector_json(best["scale"]),
    }
    meta = {
        "render_source": render_source,
        "target_mask_pixels": target_pixels,
        "rendered_mask_pixels": int(best["rendered_mask_pixels"]),
        "rerendered_mask_pixels": int(best["rerendered_mask_pixels"]),
        "target_depth": round(float(target_depth), 7),
        "rendered_depth": round(float(best["rendered_depth"]), 7),
        "rerendered_depth": round(float(best["rerendered_depth"]), 7),
        "scale_factor": round(float(best["scale_factor"]), 7),
        "mask_iou": round(float(best["mask_iou"]), 7),
        "centroid_error_px": round(float(best["centroid_error_px"]), 7),
        "depth_abs_error": round(float(best["depth_abs_error"]), 7),
        "area_ratio": round(float(best["area_ratio"]), 7),
        "alignment_score": round(float(best["alignment_score"]), 7),
        "refinement_iterations": int(best.get("refinement_iterations", 0)),
        "candidate_search": candidate_search if len(attempts) > 1 else "pose_only",
        "candidate_label": best["label"],
        "candidate_count": len(attempts),
        "candidate_attempts_top": compact_attempts(attempts),
    }
    return output_pose, meta


def align_render_candidate(
    render_depth_mask,
    position: np.ndarray,
    scale: np.ndarray,
    rotation: np.ndarray,
    *,
    target_mask: np.ndarray,
    target_depth: float,
    intrinsics: dict[str, float],
    label: str,
) -> dict[str, Any]:
    position = np.asarray(position, dtype=np.float32).reshape(3).copy()
    scale = normalize_scale(scale)
    initial_scale = scale.copy()
    rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    rendered_mask, rendered_depth = render_depth_mask(position, scale, rotation)
    rendered_pixels = int(rendered_mask.sum())
    target_pixels = int(target_mask.sum())
    rendered_depth_median = float(np.median(rendered_depth[rendered_mask])) if rendered_pixels > 0 else 0.0
    attempt: dict[str, Any] = {
        "label": label,
        "position": position,
        "scale": scale,
        "rotation": rotation,
        "target_mask_pixels": target_pixels,
        "rendered_mask_pixels": rendered_pixels,
        "rerendered_mask_pixels": 0,
        "rendered_depth": rendered_depth_median,
        "rerendered_depth": 0.0,
        "scale_factor": 1.0,
        "refinement_iterations": 0,
        "mask_iou": 0.0,
        "centroid_error_px": float(max(target_mask.shape)),
        "depth_abs_error": float("inf"),
        "area_ratio": 0.0,
        "alignment_score": -float("inf"),
    }
    if rendered_pixels <= 0 or target_pixels <= 0:
        return attempt

    target_center_yx = mask_centroid(target_mask)
    scale_floor = np.array([1e-4, 1e-4, 1e-4], dtype=np.float32)
    rerendered_mask = rendered_mask
    rerendered_depth = rendered_depth
    completed_iterations = 0
    best_state: dict[str, Any] | None = None
    for _idx in range(ALIGNMENT_REFINEMENT_ITERATIONS):
        current_pixels = int(rerendered_mask.sum())
        if current_pixels <= 0:
            break
        current_depth = float(np.median(rerendered_depth[rerendered_mask]))
        scale_step = float(np.sqrt(target_pixels / float(current_pixels)))
        if not np.isfinite(scale_step) or scale_step <= 0:
            scale_step = 1.0
        scale = np.maximum(scale * scale_step, scale_floor)
        position[1] += target_depth - current_depth

        rendered_center_yx = mask_centroid(rerendered_mask)
        if target_center_yx is not None and rendered_center_yx is not None:
            dy = float(target_center_yx[0] - rendered_center_yx[0])
            dx = float(target_center_yx[1] - rendered_center_yx[1])
            fx = max(float(intrinsics["fx"]), 1e-6)
            fy = max(float(intrinsics["fy"]), 1e-6)
            position[0] += dx / fx * target_depth
            position[2] -= dy / fy * target_depth

        rerendered_mask, rerendered_depth = render_depth_mask(position, scale, rotation)
        completed_iterations += 1
        metrics = render_alignment_metrics(target_mask, rerendered_mask, rerendered_depth, target_depth)
        if best_state is None or metrics["alignment_score"] > best_state["metrics"]["alignment_score"]:
            best_state = {
                "position": position.copy(),
                "scale": scale.copy(),
                "rerendered_mask_pixels": int(rerendered_mask.sum()),
                "metrics": metrics,
                "refinement_iterations": completed_iterations,
            }

    if best_state is None:
        metrics = render_alignment_metrics(target_mask, rerendered_mask, rerendered_depth, target_depth)
        best_state = {
            "position": position.copy(),
            "scale": scale.copy(),
            "rerendered_mask_pixels": int(rerendered_mask.sum()),
            "metrics": metrics,
            "refinement_iterations": completed_iterations,
        }
    metrics = best_state["metrics"]
    initial_scale_values = np.maximum(initial_scale, scale_floor)
    best_scale = np.asarray(best_state["scale"], dtype=np.float32)
    total_scale = float(np.median(best_scale / initial_scale_values))
    attempt.update(
        {
            "position": best_state["position"],
            "scale": best_scale,
            "rerendered_mask_pixels": int(best_state["rerendered_mask_pixels"]),
            "rerendered_depth": metrics["rerendered_depth"],
            "scale_factor": total_scale if np.isfinite(total_scale) else 1.0,
            "refinement_iterations": int(best_state["refinement_iterations"]),
            **metrics,
        }
    )
    return attempt


def best_alignment_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [attempt for attempt in attempts if int(attempt.get("rerendered_mask_pixels", 0)) > 0]
    if not valid:
        return None
    return max(valid, key=lambda attempt: float(attempt.get("alignment_score", -float("inf"))))


def render_alignment_rejection_reason(attempt: dict[str, Any]) -> str | None:
    area_ratio = float(attempt.get("area_ratio", 0.0))
    if not np.isfinite(area_ratio) or area_ratio < MIN_RENDER_AREA_RATIO or area_ratio > MAX_RENDER_AREA_RATIO:
        return "rendered_area_ratio_out_of_range"
    mask_iou = float(attempt.get("mask_iou", 0.0))
    if not np.isfinite(mask_iou) or mask_iou < MIN_RENDER_MASK_IOU:
        return "rendered_mask_iou_too_low"
    return None


def render_alignment_metrics(
    target_mask: np.ndarray,
    rendered_mask: np.ndarray,
    rendered_depth: np.ndarray,
    target_depth: float,
) -> dict[str, float]:
    rendered_pixels = int(rendered_mask.sum())
    target_pixels = int(target_mask.sum())
    iou = mask_iou(target_mask, rendered_mask)
    centroid_error = centroid_distance(target_mask, rendered_mask)
    rerendered_depth = float(np.median(rendered_depth[rendered_mask])) if rendered_pixels > 0 else 0.0
    depth_abs_error = abs(float(target_depth) - rerendered_depth) if rendered_pixels > 0 else float("inf")
    area_ratio = float(rendered_pixels / max(target_pixels, 1))
    area_penalty = abs(np.log(max(area_ratio, 1e-6)))
    score = iou * 100.0 - centroid_error * 0.05 - depth_abs_error * 5.0 - area_penalty
    return {
        "mask_iou": float(iou),
        "centroid_error_px": float(centroid_error),
        "rerendered_depth": float(rerendered_depth),
        "depth_abs_error": float(depth_abs_error),
        "area_ratio": float(area_ratio),
        "alignment_score": float(score),
    }


def should_run_axis_search(initial_attempt: dict[str, Any], shape: tuple[int, int]) -> bool:
    if int(initial_attempt.get("rerendered_mask_pixels", 0)) <= 0:
        return True
    if float(initial_attempt.get("mask_iou", 0.0)) < 0.2:
        return True
    return float(initial_attempt.get("centroid_error_px", 0.0)) > max(shape) * 0.15


def estimate_mask_depth_pose(
    target_mask: np.ndarray,
    background_depth: np.ndarray,
    intrinsics: dict[str, float],
    *,
    depth_range: tuple[float, float],
    z_offset: float,
) -> dict[str, np.ndarray]:
    points = mask_depth_points(target_mask, background_depth, intrinsics, depth_range=depth_range, z_offset=z_offset)
    if points.size == 0:
        depth = denormalized_depth_median(background_depth, target_mask, depth_range)
        return {
            "position": np.asarray([0.0, depth, z_offset], dtype=np.float32),
            "extent": np.asarray([0.05, 0.05, 0.05], dtype=np.float32),
        }
    extent = np.maximum(points.max(axis=0) - points.min(axis=0), np.array([0.03, 0.03, 0.03], dtype=np.float32))
    return {"position": np.median(points, axis=0).astype(np.float32), "extent": extent.astype(np.float32)}


def candidate_extent_scale(
    local_points: np.ndarray | None,
    target_extent: np.ndarray,
    rotation: np.ndarray,
    fallback_scale: np.ndarray,
) -> np.ndarray:
    fallback = normalize_scale(fallback_scale)
    if local_points is None or local_points.size == 0:
        return fallback
    local_extent = np.maximum(local_points.max(axis=0) - local_points.min(axis=0), np.array([1e-4, 1e-4, 1e-4], dtype=np.float32))
    rotated_extent = np.maximum(np.abs(rotation) @ local_extent, np.array([1e-4, 1e-4, 1e-4], dtype=np.float32))
    ratios = np.asarray(target_extent, dtype=np.float32).reshape(3) / rotated_extent
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size == 0:
        return fallback
    uniform = float(np.median(ratios))
    if not np.isfinite(uniform) or uniform <= 0:
        return fallback
    return np.repeat(max(uniform, 1e-4), 3).astype(np.float32)


def axis_alignment_candidates(base_rotation: np.ndarray) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [{"label": "pose_rotation", "rotation": np.asarray(base_rotation, dtype=np.float32).reshape(3, 3)}]
    seen = {rotation_key(candidates[0]["rotation"])}
    axis_names = ("x", "y", "z")
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = np.zeros((3, 3), dtype=np.float32)
            for world_axis, local_axis in enumerate(perm):
                rotation[world_axis, local_axis] = signs[world_axis]
            if np.linalg.det(rotation) < 0.5:
                continue
            key = rotation_key(rotation)
            if key in seen:
                continue
            seen.add(key)
            label = "axis_" + "".join(axis_names[idx] for idx in perm) + "_" + "".join("p" if sign > 0 else "n" for sign in signs)
            candidates.append({"label": label, "rotation": rotation})
    return candidates


def load_gaussian_canonicalization_metadata(gaussian_path: Path) -> dict[str, Any] | None:
    metadata_path = gaussian_path.parent / "sam3d_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = read_json(metadata_path)
    except Exception:
        return {"status": "unreadable", "metadata": relative_or_absolute(metadata_path, gaussian_path.parent)}
    canonicalization = metadata.get("gaussian_canonicalization") if isinstance(metadata, dict) else None
    if not isinstance(canonicalization, dict):
        return None
    keep = {
        "enabled": canonicalization.get("enabled"),
        "status": canonicalization.get("status"),
        "changed": canonicalization.get("changed"),
        "method": canonicalization.get("method"),
        "gaussian_axes_to_mesh_axes": canonicalization.get("gaussian_axes_to_mesh_axes"),
        "mesh_axis_signs": canonicalization.get("mesh_axis_signs"),
    }
    return {key: value for key, value in keep.items() if value is not None}


def compact_attempts(attempts: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    ranked = sorted(attempts, key=lambda attempt: float(attempt.get("alignment_score", -float("inf"))), reverse=True)
    compact = []
    for attempt in ranked[:limit]:
        compact.append(
            {
                "label": attempt.get("label"),
                "rendered_mask_pixels": int(attempt.get("rendered_mask_pixels", 0)),
                "rerendered_mask_pixels": int(attempt.get("rerendered_mask_pixels", 0)),
                "mask_iou": round(float(attempt.get("mask_iou", 0.0)), 7),
                "centroid_error_px": round(float(attempt.get("centroid_error_px", 0.0)), 7),
                "depth_abs_error": round(float(attempt.get("depth_abs_error", 0.0)), 7)
                if np.isfinite(float(attempt.get("depth_abs_error", 0.0)))
                else None,
                "refinement_iterations": int(attempt.get("refinement_iterations", 0)),
                "alignment_score": round(float(attempt.get("alignment_score", -float("inf"))), 7)
                if np.isfinite(float(attempt.get("alignment_score", -float("inf"))))
                else None,
            }
        )
    return compact


def render_gaussian_depth_mask(
    points: np.ndarray,
    position: np.ndarray,
    scale: np.ndarray,
    rotation: np.ndarray | None,
    intrinsics: dict[str, float],
    shape: tuple[int, int],
    *,
    radius_pixels: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=np.float32)
    transformed = transform_local_points(points, position, scale, rotation)
    return render_points_depth_mask(transformed, intrinsics, shape, radius_pixels=radius_pixels)


def render_mesh_depth_mask(
    mesh_path: Path,
    position: np.ndarray,
    scale: np.ndarray,
    rotation: np.ndarray | None,
    intrinsics: dict[str, float],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = load_obj_mesh(mesh_path)
    return render_mesh_arrays_depth_mask(vertices, faces, position, scale, rotation, intrinsics, shape)


def render_mesh_arrays_depth_mask(
    vertices: np.ndarray,
    faces: list[tuple[int, int, int]],
    position: np.ndarray,
    scale: np.ndarray,
    rotation: np.ndarray | None,
    intrinsics: dict[str, float],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if vertices.size == 0:
        return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=np.float32)
    transformed = transform_local_points(vertices, position, scale, rotation)
    if not faces:
        return render_points_depth_mask(transformed, intrinsics, shape, radius_pixels=1)
    height, width = shape
    fx = max(float(intrinsics["fx"]), 1e-6)
    fy = max(float(intrinsics["fy"]), 1e-6)
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    depth = np.full(shape, np.inf, dtype=np.float32)
    pixels = np.zeros((transformed.shape[0], 2), dtype=np.float32)
    z = np.maximum(transformed[:, 1], 1e-6)
    pixels[:, 0] = transformed[:, 0] / z * fx + cx
    pixels[:, 1] = cy - (transformed[:, 2] / z * fy)
    for face in faces:
        rasterize_triangle(pixels[list(face)], z[list(face)], depth)
    mask = np.isfinite(depth)
    depth[~mask] = 0.0
    return mask, depth


def transform_local_points(
    points: np.ndarray,
    position: np.ndarray,
    scale: np.ndarray,
    rotation: np.ndarray | None,
) -> np.ndarray:
    local = np.asarray(points, dtype=np.float32) * normalize_scale(scale).reshape(1, 3)
    rot = np.eye(3, dtype=np.float32) if rotation is None else np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    return local @ rot.T + np.asarray(position, dtype=np.float32).reshape(1, 3)


def render_points_depth_mask(
    points: np.ndarray,
    intrinsics: dict[str, float],
    shape: tuple[int, int],
    *,
    radius_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    fx = max(float(intrinsics["fx"]), 1e-6)
    fy = max(float(intrinsics["fy"]), 1e-6)
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    depth = np.full(shape, np.inf, dtype=np.float32)
    z = np.maximum(points[:, 1], 1e-6)
    px = points[:, 0] / z * fx + cx
    py = cy - (points[:, 2] / z * fy)
    radius = max(0, int(radius_pixels))
    for x_value, y_value, z_value in zip(px, py, z):
        x_center = int(round(float(x_value)))
        y_center = int(round(float(y_value)))
        if x_center < -radius or x_center >= width + radius or y_center < -radius or y_center >= height + radius:
            continue
        for y in range(max(0, y_center - radius), min(height, y_center + radius + 1)):
            for x in range(max(0, x_center - radius), min(width, x_center + radius + 1)):
                if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                    depth[y, x] = min(depth[y, x], float(z_value))
    mask = np.isfinite(depth)
    depth[~mask] = 0.0
    return mask, depth


def rasterize_triangle(points: np.ndarray, z_values: np.ndarray, depth: np.ndarray) -> None:
    height, width = depth.shape
    min_x = max(0, int(np.floor(float(points[:, 0].min()))))
    max_x = min(width - 1, int(np.ceil(float(points[:, 0].max()))))
    min_y = max(0, int(np.floor(float(points[:, 1].min()))))
    max_y = min(height - 1, int(np.ceil(float(points[:, 1].max()))))
    if max_x < min_x or max_y < min_y:
        return
    a, b, c = points.astype(np.float32)
    denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(float(denom)) < 1e-8:
        return
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            px = x + 0.5
            py = y + 0.5
            w0 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / denom
            w1 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / denom
            w2 = 1.0 - w0 - w1
            if w0 >= -1e-5 and w1 >= -1e-5 and w2 >= -1e-5:
                z = float(w0 * z_values[0] + w1 * z_values[1] + w2 * z_values[2])
                if z < depth[y, x]:
                    depth[y, x] = z


def load_obj_mesh(path: Path) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                raw = []
                for token in line.split()[1:]:
                    index = token.split("/")[0]
                    if index:
                        raw.append(int(index) - 1)
                if len(raw) >= 3:
                    for idx in range(1, len(raw) - 1):
                        faces.append((raw[0], raw[idx], raw[idx + 1]))
    return np.asarray(vertices, dtype=np.float32), faces


PLY_SCALAR_FORMATS = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def load_gaussian_points(path: Path) -> np.ndarray:
    data = path.read_bytes()
    header_marker = b"end_header"
    marker_index = data.find(header_marker)
    if marker_index < 0:
        return np.zeros((0, 3), dtype=np.float32)
    body_start = data.find(b"\n", marker_index)
    if body_start < 0:
        return np.zeros((0, 3), dtype=np.float32)
    header = data[: body_start + 1].decode("ascii", errors="ignore").splitlines()
    body = data[body_start + 1 :]
    fmt = "ascii"
    properties: list[tuple[str, str]] = []
    vertex_count = 0
    current_element = ""
    for line in header:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0] == "format":
            fmt = parts[1]
        elif len(parts) >= 3 and parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif len(parts) >= 3 and parts[0] == "property" and current_element == "vertex":
            if parts[1] == "list":
                return np.zeros((0, 3), dtype=np.float32)
            properties.append((parts[-1], parts[1]))
    if vertex_count <= 0 or not properties:
        return np.zeros((0, 3), dtype=np.float32)
    names = [name for name, _kind in properties]
    if all(name in names for name in ("x", "y", "z")):
        indices = [names.index("x"), names.index("y"), names.index("z")]
    else:
        indices = [0, 1, 2]
    if len(properties) <= max(indices):
        return np.zeros((0, 3), dtype=np.float32)
    if fmt == "ascii":
        points = load_ascii_ply_points(body, vertex_count, indices)
    elif fmt in {"binary_little_endian", "binary_big_endian"}:
        points = load_binary_ply_points(body, vertex_count, properties, indices, little_endian=fmt == "binary_little_endian")
    else:
        points = []
    return np.asarray(points, dtype=np.float32)


def downsample_alignment_points(points: np.ndarray, *, max_points: int = MAX_GAUSSIAN_ALIGNMENT_POINTS) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = np.linspace(0, points.shape[0] - 1, int(max_points), dtype=np.int64)
    return points[indices]


def load_ascii_ply_points(body: bytes, vertex_count: int, indices: list[int]) -> list[tuple[float, float, float]]:
    points = []
    for line in body.decode("ascii", errors="ignore").splitlines():
        if len(points) >= vertex_count:
            break
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = [float(value) for value in stripped.split()]
        except ValueError:
            continue
        if len(row) > max(indices):
            points.append((row[indices[0]], row[indices[1]], row[indices[2]]))
    return points


def load_binary_ply_points(
    body: bytes,
    vertex_count: int,
    properties: list[tuple[str, str]],
    indices: list[int],
    *,
    little_endian: bool,
) -> list[tuple[float, float, float]]:
    try:
        row_format = ("<" if little_endian else ">") + "".join(PLY_SCALAR_FORMATS[kind] for _name, kind in properties)
    except KeyError:
        return []
    row_size = struct.calcsize(row_format)
    if row_size <= 0:
        return []
    count = min(vertex_count, len(body) // row_size)
    points = []
    for idx in range(count):
        row = struct.unpack_from(row_format, body, idx * row_size)
        points.append((float(row[indices[0]]), float(row[indices[1]]), float(row[indices[2]])))
    return points


def denormalized_depth_median(depth: np.ndarray, mask: np.ndarray, depth_range: tuple[float, float]) -> float:
    values = np.asarray(depth[mask], dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(sum(depth_range) * 0.5)
    depth_min, depth_max = depth_range
    return float(depth_min + np.median(np.clip(values, 0.0, 1.0)) * (depth_max - depth_min))


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return (float(ys.mean()), float(xs.mean()))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return float(intersection / union) if union > 0 else 0.0


def centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_center = mask_centroid(a)
    b_center = mask_centroid(b)
    if a_center is None or b_center is None:
        return float(max(a.shape))
    return float(np.hypot(a_center[0] - b_center[0], a_center[1] - b_center[1]))


def normalize_scale(scale: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    values = np.asarray(scale, dtype=np.float32).reshape(-1)
    if values.size == 0:
        values = np.ones(3, dtype=np.float32)
    if values.size == 1:
        values = np.repeat(values, 3)
    values = values[:3] if values.size >= 3 else np.pad(values, (0, 3 - values.size), constant_values=1.0)
    return np.maximum(values.astype(np.float32), np.array([1e-4, 1e-4, 1e-4], dtype=np.float32))


def rotation_matrix_from_quat_wxyz(raw_quat: Any) -> np.ndarray:
    quat = np.asarray(raw_quat, dtype=np.float32).reshape(-1)
    if quat.size < 4:
        quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    w, x, y, z = quat[:4]
    norm = float(np.linalg.norm([w, x, y, z]))
    if norm <= 1e-8:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = (w / norm, x / norm, y / norm, z / norm)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def quat_wxyz_from_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [
                0.25 * s,
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / s,
                    0.25 * s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                ],
                dtype=np.float64,
            )
        elif axis == 1:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    0.25 * s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                ],
                dtype=np.float64,
            )
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    quat = quat / norm
    if quat[0] < 0:
        quat = -quat
    return quat.astype(np.float32)


def rotation_key(rotation: np.ndarray) -> tuple[int, ...]:
    return tuple(int(round(float(value) * 1000.0)) for value in np.asarray(rotation).reshape(-1))


def summarize_alignment_quality(entries: list[dict[str, Any]]) -> dict[str, Any]:
    rendered_entries = [entry for entry in entries if entry.get("render_source") in {"gaussian_ply", "mesh_obj"}]
    fallback_entries = [entry for entry in entries if entry.get("render_source") == "mask_depth_fallback"]
    summary: dict[str, Any] = {
        "aligned_count": len(entries),
        "rendered_asset_count": len(rendered_entries),
        "mask_depth_fallback_count": len(fallback_entries),
    }
    for key in ("mask_iou", "centroid_error_px", "depth_abs_error", "area_ratio"):
        values = [float(entry[key]) for entry in rendered_entries if entry.get(key) is not None and np.isfinite(float(entry[key]))]
        if values:
            summary[f"mean_{key}"] = round(float(np.mean(values)), 7)
            summary[f"min_{key}"] = round(float(np.min(values)), 7)
            summary[f"max_{key}"] = round(float(np.max(values)), 7)
    return summary


def load_intrinsics(cameras_path: Path, shape: tuple[int, int]) -> dict[str, float]:
    height, width = shape
    default = {
        "width": float(width),
        "height": float(height),
        "fx": float(max(width, height) * 0.85),
        "fy": float(max(width, height) * 0.85),
        "cx": float(width) / 2.0,
        "cy": float(height) / 2.0,
    }
    raw = read_json(cameras_path)
    intrinsics = raw.get("intrinsics", raw) if isinstance(raw, dict) else raw
    if isinstance(intrinsics, dict):
        out = dict(default)
        for key in ("width", "height", "fx", "fy", "cx", "cy"):
            if intrinsics.get(key) is not None:
                out[key] = float(intrinsics[key])
        return out

    arr = np.asarray(intrinsics, dtype=np.float32)
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return default
    fx = float(arr[0, 0])
    fy = float(arr[1, 1])
    cx = float(arr[0, 2])
    cy = float(arr[1, 2])
    if max(abs(fx), abs(fy), abs(cx), abs(cy)) <= 4.0:
        fx *= width
        fy *= height
        cx *= width
        cy *= height
    return {"width": float(width), "height": float(height), "fx": fx, "fy": fy, "cx": cx, "cy": cy}


def mask_depth_points(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: dict[str, float],
    *,
    depth_range: tuple[float, float],
    z_offset: float,
) -> np.ndarray:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    values = np.asarray(depth[ys, xs], dtype=np.float32)
    valid = np.isfinite(values)
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float32)
    ys = ys[valid].astype(np.float32)
    xs = xs[valid].astype(np.float32)
    values = np.clip(values[valid], 0.0, 1.0)
    depth_min, depth_max = depth_range
    z_depth = depth_min + values * (depth_max - depth_min)
    fx = max(float(intrinsics["fx"]), 1e-6)
    fy = max(float(intrinsics["fy"]), 1e-6)
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    xw = (xs - cx) / fx * z_depth
    yw = z_depth
    zw = (cy - ys) / fy * z_depth + z_offset
    return np.stack([xw, yw, zw], axis=1).astype(np.float32)


def resize_mask_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    src_h, src_w = mask.shape
    y_idx = np.clip(np.round(np.linspace(0, src_h - 1, height)).astype(int), 0, src_h - 1)
    x_idx = np.clip(np.round(np.linspace(0, src_w - 1, width)).astype(int), 0, src_w - 1)
    return mask[y_idx][:, x_idx]


def resize_mask_to_camera_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Match AnySplat process_image: aspect-preserving cover resize, center crop."""
    if mask.shape == shape:
        return mask
    src_h, src_w = mask.shape
    dst_h, dst_w = shape
    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        return np.zeros(shape, dtype=bool)
    scale = max(dst_w / float(src_w), dst_h / float(src_h))
    resized_h = max(dst_h, int(round(src_h * scale)))
    resized_w = max(dst_w, int(round(src_w * scale)))
    resized = resize_mask_nearest(mask, (resized_h, resized_w))
    top = max(0, (resized_h - dst_h) // 2)
    left = max(0, (resized_w - dst_w) // 2)
    cropped = resized[top : top + dst_h, left : left + dst_w]
    if cropped.shape == shape:
        return cropped
    out = np.zeros(shape, dtype=cropped.dtype)
    out[: cropped.shape[0], : cropped.shape[1]] = cropped
    return out


def with_alignment_metadata(manifest: dict[str, Any], alignment_meta: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(manifest.get("metadata") or {})
    metadata["pose_alignment"] = alignment_meta
    manifest["metadata"] = metadata
    return manifest


def vector_json(values: np.ndarray) -> list[float]:
    return [round(float(value), 7) for value in np.asarray(values, dtype=np.float32).reshape(-1).tolist()]


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_optional_path(value: Any, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
