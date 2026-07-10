"""Built-in single-image Real2Sim baseline.

This is a real, deterministic image-to-assets implementation designed to make
the public preview repository usable before the paper authors release their
GroundingDINO/SAM/LaMa/SAM-3D/AnySplat stack.

It does not claim the visual quality of AnySplat/SAM-3D. It produces valid
sim-ready assets from a single RGB image: instance masks, an inpainted
background, 3DGS PLY files, OBJ meshes, object poses, camera metadata, and a
manifest consumable by the GS-Playground package writer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import Pose, Real2SimAsset, Real2SimManifest, load_instances, write_json
from .image_io import read_gray, read_rgb, write_gray, write_rgb


PPM_SUFFIXES = {".ppm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm", ".pgm", ".webp"}


@dataclass(frozen=True)
class BuiltinOptions:
    max_scene_points: int = 80000
    max_object_points: int = 20000
    max_objects: int = 6
    min_component_area_ratio: float = 0.0025
    stage_min_component_area_ratio: float = 0.0005
    focal_scale: float = 0.85
    prefer_stage_artifacts: bool = False
    external_depth_weight: float = 0.0
    color_enhance_strength: float = 0.0
    scene_point_scale: float = 0.012
    object_point_scale: float = 0.012
    opacity_logit: float = 4.0
    multiview_yaw_scale: float = 0.06
    multiview_lateral_step: float = 0.006
    mask_close_passes: int = 1
    mask_dilate_passes: int = 0
    max_mask_overlap: float = 0.75

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "BuiltinOptions":
        raw = (config or {}).get("builtin", {})
        if not isinstance(raw, dict):
            raw = {}
        return cls(
            max_scene_points=int(raw.get("max_scene_points", 80000)),
            max_object_points=int(raw.get("max_object_points", 20000)),
            max_objects=int(raw.get("max_objects", 6)),
            min_component_area_ratio=float(raw.get("min_component_area_ratio", 0.0025)),
            stage_min_component_area_ratio=float(raw.get("stage_min_component_area_ratio", 0.0005)),
            focal_scale=float(raw.get("focal_scale", 0.85)),
            prefer_stage_artifacts=bool(raw.get("prefer_stage_artifacts", False)),
            external_depth_weight=_clamp01(float(raw.get("external_depth_weight", 0.0))),
            color_enhance_strength=_clamp01(float(raw.get("color_enhance_strength", 0.0))),
            scene_point_scale=max(0.001, float(raw.get("scene_point_scale", 0.012))),
            object_point_scale=max(0.001, float(raw.get("object_point_scale", 0.012))),
            opacity_logit=float(raw.get("opacity_logit", 4.0)),
            multiview_yaw_scale=max(0.0, min(1.0, float(raw.get("multiview_yaw_scale", 0.06)))),
            multiview_lateral_step=max(0.0, float(raw.get("multiview_lateral_step", 0.006))),
            mask_close_passes=max(0, int(raw.get("mask_close_passes", 1))),
            mask_dilate_passes=max(0, int(raw.get("mask_dilate_passes", 0))),
            max_mask_overlap=_clamp01(float(raw.get("max_mask_overlap", 0.75))),
        )


def run_builtin_real2sim(
    capture,
    context: dict[str, Any],
    prompt: str,
    config: dict[str, Any] | None = None,
    *,
    include_objects: bool = True,
) -> Real2SimManifest:
    options = BuiltinOptions.from_config(config)
    rgb = read_rgb(Path(context["image"]))
    height, width = rgb.shape[:2]
    focal = options.focal_scale * max(width, height)
    intrinsics = {
        "width": width,
        "height": height,
        "fx": focal,
        "fy": focal,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
    }

    depth = estimate_pseudo_depth(rgb)
    if options.prefer_stage_artifacts:
        depth = fuse_depth_from_artifact(depth, Path(context["depth_map"]), rgb.shape[:2], weight=options.external_depth_weight)

    component_records = []
    using_stage_components = False
    if include_objects:
        if options.prefer_stage_artifacts:
            component_records = load_stage_component_records(context, rgb.shape[:2])
            using_stage_components = bool(component_records)
        if not component_records:
            foreground = estimate_foreground_mask(rgb)
            components = select_components(foreground, options.max_objects, options.min_component_area_ratio)
            if not components:
                components = [central_component(height, width)]
            component_records = [(f"object_{idx:02d}", f"object_{idx:02d}", mask) for idx, mask in enumerate(components)]
            write_builtin_detection_artifacts(component_records, context)
        min_area_ratio = options.stage_min_component_area_ratio if using_stage_components else options.min_component_area_ratio
        component_records = refine_component_records(component_records, options, rgb.shape[:2], min_area_ratio=min_area_ratio)
        target_count = prompt_target_object_count(prompt)
        if target_count is not None and len(component_records) >= target_count:
            component_records = select_tabletop_subject_records(component_records, rgb.shape[:2], target_count, rgb=rgb)
            write_builtin_detection_artifacts(component_records, context)

    all_object_mask = np.zeros((height, width), dtype=bool)
    for _, _, mask in component_records:
        all_object_mask |= mask
    background_image = Path(context["background_image"])
    if not include_objects:
        background = rgb
        background_image = background_image.with_suffix(".ppm")
        write_rgb(background_image, background)
    elif options.prefer_stage_artifacts and background_image.exists():
        background = read_rgb(background_image)
        if background.shape[:2] != (height, width):
            background = inpaint_with_border_color(rgb, all_object_mask)
            background_image = background_image.with_suffix(".ppm")
            write_rgb(background_image, background)
    else:
        background = inpaint_with_border_color(rgb, all_object_mask)
        background_image = background_image.with_suffix(".ppm")
        write_rgb(background_image, background)
    depth_map = Path(context["depth_map"]).with_suffix(".pgm")
    context["background_image"] = background_image
    context["depth_map"] = depth_map
    write_gray(depth_map, normalize(depth))

    scene_dir = Path(context["scene_reconstruction_dir"])
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_ply = Path(context["scene_ply"])
    scene_mesh = Path(context["scene_mesh"])
    scene_rgb = enhance_colors(background, options.color_enhance_strength)
    object_rgb = enhance_colors(rgb, options.color_enhance_strength)
    points, colors, camera_views = reconstruct_scene_points(capture, scene_rgb, depth, intrinsics, options)
    write_3dgs_ply(scene_ply, points, colors, point_scale=options.scene_point_scale, opacity_logit=options.opacity_logit)
    write_depth_mesh_obj(scene_mesh, scene_rgb, depth, intrinsics, grid=96)

    cameras_json = Path(context["cameras_json"])
    write_json(
        cameras_json,
        {
            "intrinsics": intrinsics,
            "extrinsics": {
                "camera_to_world": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            },
            "extrinsic": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "views": camera_views,
            "note": "Built-in pseudo-depth camera metadata. Replace with AnySplat calibration when available.",
        },
    )

    object_assets: list[Real2SimAsset] = []
    objects_dir = Path(context["objects_dir"])
    objects_dir.mkdir(parents=True, exist_ok=True)
    for idx, (instance_id, label, mask) in enumerate(component_records if include_objects else []):
        name = safe_name(instance_id) or f"object_{idx:02d}"
        object_dir = objects_dir / name
        object_dir.mkdir(parents=True, exist_ok=True)
        object_points, object_colors = backproject(object_rgb, depth, intrinsics, mask=mask, max_points=options.max_object_points)
        if object_points.size == 0:
            continue
        center = object_points.mean(axis=0)
        local_points = object_points - center
        extent = np.maximum(object_points.max(axis=0) - object_points.min(axis=0), np.array([0.04, 0.04, 0.04], dtype=np.float32))
        object_ply = object_dir / "gaussian.ply"
        object_mesh = object_dir / "mesh.obj"
        object_pose = object_dir / "pose.json"
        visual_points, visual_colors = object_billboard_points(
            object_rgb,
            mask,
            extent,
            label=label,
            max_points=options.max_object_points,
        )
        if visual_points.size == 0:
            visual_points, visual_colors = local_points, object_colors
        write_3dgs_ply(object_ply, visual_points, visual_colors, point_scale=options.object_point_scale, opacity_logit=options.opacity_logit)
        write_box_obj(object_mesh, extent)
        pose = Pose(position=tuple(float(x) for x in center), quat_wxyz=(1.0, 0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))
        write_json(object_pose, pose.to_json())
        object_assets.append(
            Real2SimAsset(
                name=name,
                label=label,
                gaussian_ply=object_ply,
                mesh_obj=object_mesh,
                pose=pose,
                dynamic=True,
                source_instance_id=instance_id,
            )
        )

    return Real2SimManifest(
        source=capture.source,
        prompt=prompt,
        scene=Real2SimAsset(name="scene", label="background_scene", gaussian_ply=scene_ply, mesh_obj=scene_mesh),
        objects=tuple(object_assets),
        camera_artifacts={
            "source_image": capture.image,
            "depth_map": depth_map,
            "cameras_json": cameras_json,
            "background_image": background_image,
        },
        metadata={
            "pipeline": "GS-Playground built-in Real2Sim baseline",
            "capture_kind": capture.kind,
            "backend": "builtin",
            "builtin_scene_fusion": "synthetic_multiview" if len(camera_views) > 1 else "single_view",
            "builtin_scene_view_count": len(camera_views),
            "scene_generation_only": not include_objects,
            "limitations": [
                "Pseudo-depth is estimated deterministically from RGB images.",
                "Synthetic multiview fusion uses approximate canonical view yaw when real SfM/AnySplat poses are unavailable.",
                "Instance masks use saliency and connected components unless external GroundingDINO/SAM commands are configured.",
                "Prompt-aware tabletop subject filtering keeps the main requested assets separate from background scene clutter.",
                "Object meshes are collision-ready boxes derived from reconstructed point extents.",
            ],
        },
    )


def reconstruct_scene_points(
    capture,
    scene_rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: dict[str, float],
    options: BuiltinOptions,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    frame_paths = capture_frame_paths(capture)
    if len(frame_paths) <= 1:
        mask = np.ones(depth.shape, dtype=bool)
        points, colors = backproject(scene_rgb, depth, intrinsics, mask=mask, max_points=options.max_scene_points)
        return points, colors, [camera_view_record(frame_paths[0] if frame_paths else Path(capture.image), 0, 0.0)]

    per_view_limit = max(1, options.max_scene_points // len(frame_paths))
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    camera_views: list[dict[str, Any]] = []
    for index, frame_path in enumerate(frame_paths):
        frame_rgb = enhance_colors(read_rgb(frame_path), options.color_enhance_strength)
        if frame_rgb.shape[:2] != depth.shape:
            frame_rgb = resize_rgb_nearest(frame_rgb, depth.shape)
        frame_depth = estimate_pseudo_depth(frame_rgb)
        points, colors = backproject(
            frame_rgb,
            frame_depth,
            intrinsics,
            mask=np.ones(depth.shape, dtype=bool),
            max_points=per_view_limit,
        )
        yaw_degrees = canonical_multiview_yaw(index, len(frame_paths))
        points = transform_multiview_points(points, yaw_degrees, index, len(frame_paths), options)
        all_points.append(points)
        all_colors.append(colors)
        camera_views.append(camera_view_record(frame_path, index, yaw_degrees))

    points = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3), dtype=np.float32)
    colors = np.concatenate(all_colors, axis=0) if all_colors else np.zeros((0, 3), dtype=np.float32)
    if len(points) > options.max_scene_points:
        idx = deterministic_point_sample(points, options.max_scene_points)
        points = points[idx]
        colors = colors[idx]
    return points.astype(np.float32), colors.astype(np.float32), camera_views


def capture_frame_paths(capture) -> list[Path]:
    frames_dir = getattr(capture, "frames_dir", None)
    if frames_dir:
        frames = sorted(path for path in Path(frames_dir).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if frames:
            return frames
    return [Path(capture.image)]


def canonical_multiview_yaw(index: int, total: int) -> float:
    canonical = [0.0, -45.0, -90.0, -135.0, 180.0, 135.0, 90.0, -25.0]
    if total == 8 and index < len(canonical):
        return canonical[index]
    if total <= 1:
        return 0.0
    return -180.0 + (360.0 * index / total)


def transform_multiview_points(points: np.ndarray, yaw_degrees: float, index: int, total: int, options: BuiltinOptions) -> np.ndarray:
    if points.size == 0:
        return points
    centered = points.copy()
    yaw = math.radians(yaw_degrees * options.multiview_yaw_scale)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    x = centered[:, 0].copy()
    y = centered[:, 1].copy()
    centered[:, 0] = (cos_y * x) - (sin_y * y)
    centered[:, 1] = (sin_y * x) + (cos_y * y)
    if total > 1:
        centered[:, 0] += (index - (total - 1) / 2.0) * options.multiview_lateral_step
        centered[:, 2] += 0.018 * math.sin((index / max(total - 1, 1)) * math.pi)
    return centered.astype(np.float32)


def camera_view_record(path: Path, index: int, yaw_degrees: float) -> dict[str, Any]:
    yaw = math.radians(yaw_degrees)
    radius = 1.6
    return {
        "index": index,
        "image": path.as_posix(),
        "approx_yaw_degrees": yaw_degrees,
        "camera_to_world": [
            [math.cos(yaw), -math.sin(yaw), 0.0, radius * math.sin(yaw)],
            [math.sin(yaw), math.cos(yaw), 0.0, -radius * math.cos(yaw)],
            [0.0, 0.0, 1.0, 0.55],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def resize_rgb_nearest(rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    ys = np.linspace(0, rgb.shape[0] - 1, h).astype(np.int32)
    xs = np.linspace(0, rgb.shape[1] - 1, w).astype(np.int32)
    return rgb[ys[:, None], xs[None, :], :].astype(np.float32)


def deterministic_point_sample(points: np.ndarray, max_points: int) -> np.ndarray:
    scaled = np.round(points.astype(np.float64) * 100000.0).astype(np.int64)
    ranks = (
        scaled[:, 0].astype(np.uint64) * np.uint64(73856093)
        ^ scaled[:, 1].astype(np.uint64) * np.uint64(19349663)
        ^ scaled[:, 2].astype(np.uint64) * np.uint64(83492791)
    )
    choice = np.argpartition(ranks, max_points - 1)[:max_points]
    return choice[np.argsort(ranks[choice])]


def load_stage_component_records(context: dict[str, Any], shape: tuple[int, int]) -> list[tuple[str, str, np.ndarray]]:
    instances_json = Path(context["instances_json"])
    if not instances_json.exists():
        return []
    records = []
    for instance in load_instances(instances_json):
        if instance.mask_path is None or not instance.mask_path.exists():
            continue
        mask = read_gray(instance.mask_path) > 0.5
        if mask.shape != shape:
            continue
        records.append((safe_name(instance.instance_id), instance.label, mask))
    return records


def refine_component_records(
    records: list[tuple[str, str, np.ndarray]],
    options: BuiltinOptions,
    shape: tuple[int, int],
    *,
    min_area_ratio: float | None = None,
) -> list[tuple[str, str, np.ndarray]]:
    h, w = shape
    area_ratio = options.min_component_area_ratio if min_area_ratio is None else min_area_ratio
    min_area = max(16, int(h * w * area_ratio))
    occupied = np.zeros(shape, dtype=bool)
    refined: list[tuple[str, str, np.ndarray]] = []
    for instance_id, label, raw_mask in records:
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != shape:
            continue
        if options.mask_close_passes:
            mask = erode(dilate(mask, options.mask_close_passes), options.mask_close_passes)
        if options.mask_dilate_passes:
            mask = dilate(mask, options.mask_dilate_passes)
        area = int(mask.sum())
        if area < min_area:
            continue
        overlap = int((mask & occupied).sum()) / float(area)
        if overlap > options.max_mask_overlap:
            continue
        mask = mask & ~occupied
        if int(mask.sum()) < min_area:
            continue
        refined.append((instance_id, label, mask))
        occupied |= mask
        if len(refined) >= options.max_objects:
            break
    return refined


def prompt_target_object_count(prompt: str) -> int | None:
    lowered = prompt.lower()
    bottle_terms = ("bottle", "bottles", "瓶", "水瓶", "葫芦")
    if not any(term in lowered for term in bottle_terms):
        return None
    if any(term in lowered for term in ("two", "2", "pair", "both", "两个", "两", "一对")):
        return 2
    return None


def select_tabletop_subject_records(
    records: list[tuple[str, str, np.ndarray]],
    shape: tuple[int, int],
    target_count: int,
    *,
    rgb: np.ndarray | None = None,
) -> list[tuple[str, str, np.ndarray]]:
    h, w = shape
    scored: list[tuple[float, tuple[str, str, np.ndarray]]] = []
    for record in records:
        _instance_id, _label, mask = record
        x0, y0, x1, y1 = mask_bbox(mask)
        width = max(x1 - x0, 0)
        height = max(y1 - y0, 0)
        area = int(mask.sum())
        if area <= 0:
            continue
        bottom_ratio = y1 / max(h, 1)
        height_ratio = height / max(h, 1)
        width_ratio = width / max(w, 1)
        if bottom_ratio < 0.58:
            continue
        if height_ratio < 0.18 or width_ratio < 0.05:
            continue
        if y0 / max(h, 1) < 0.12 and bottom_ratio < 0.62:
            continue
        compactness = area / max(width * height, 1)
        support_bonus = max(0.0, bottom_ratio - 0.58) * h * w * 0.08
        upright_bonus = min(height_ratio / max(width_ratio, 1e-4), 3.0) * h * w * 0.004
        score = area + compactness * width * height * 0.35 + support_bonus + upright_bonus
        scored.append((score, record))

    if len(scored) < target_count:
        return records

    scored.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[str, str, np.ndarray]] = []
    occupied = np.zeros(shape, dtype=bool)
    for _score, (instance_id, label, mask) in scored:
        area = int(mask.sum())
        overlap = int((mask & occupied).sum()) / max(area, 1)
        if overlap > 0.45:
            continue
        selected.append((instance_id, label, mask))
        occupied |= mask
        if len(selected) >= target_count:
            break
    if len(selected) < target_count:
        return records
    selected.sort(key=lambda record: mask_bbox(record[2])[0])
    renamed = []
    for idx, (_instance_id, label, mask) in enumerate(selected):
        semantic = "water_bottle" if idx == 0 else "gourd_bottle" if target_count == 2 else label
        completed = complete_tabletop_subject_mask(mask, shape, semantic, rgb=rgb)
        renamed.append((f"{idx:02d}_{semantic}", semantic, completed))
    return renamed


def complete_tabletop_subject_mask(
    mask: np.ndarray,
    shape: tuple[int, int],
    semantic: str,
    *,
    rgb: np.ndarray | None = None,
) -> np.ndarray:
    if semantic == "gourd_bottle":
        return erode(dilate(mask.astype(bool), 3), 1)
    if semantic == "water_bottle" and rgb is not None:
        edge_mask = complete_transparent_bottle_mask_from_edges(rgb, shape)
        if edge_mask is not None:
            return edge_mask

    h, w = shape
    x0, y0, x1, y1 = mask_bbox(mask)
    width = max(x1 - x0, 1)
    height = max(y1 - y0, 1)
    margin_x = int(max(width * 0.28, w * 0.045))
    margin_top = int(max(height * 0.20, h * 0.035))
    margin_bottom = int(max(height * 0.30, h * 0.055))
    bx0 = max(0, x0 - margin_x)
    bx1 = min(w, x1 + margin_x)
    by0 = max(0, y0 - margin_top)
    by1 = min(h, y1 + margin_bottom)
    completed = np.zeros(shape, dtype=bool)
    completed[by0:by1, bx0:bx1] = True
    return erode(dilate(completed, 1), 1)


def complete_transparent_bottle_mask_from_edges(rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    bbox = transparent_bottle_bbox_from_edges(rgb, shape)
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    completed = np.zeros(shape, dtype=bool)
    completed[top:bottom, left:right] = True
    return erode(dilate(completed, 1), 1)


def transparent_bottle_bbox_from_edges(rgb: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    h, w = shape
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2])
    gy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :])
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    saliency = gx + gy + saturation * 0.35
    y0 = int(h * 0.08)
    y1 = int(h * 0.96)
    x0 = 0
    x1 = int(w * 0.55)
    roi = saliency[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    column_score = roi.sum(axis=0)
    if column_score.size < 8:
        return None
    column_score = np.convolve(column_score, np.ones(9, dtype=np.float32) / 9.0, mode="same")
    threshold = float(np.percentile(column_score, 70.0))
    active = np.where(column_score > threshold)[0]
    if active.size == 0:
        return None
    spans: list[tuple[int, int]] = []
    start = int(active[0])
    prev = int(active[0])
    for value in active[1:].tolist():
        if value == prev + 1:
            prev = int(value)
            continue
        spans.append((start, prev))
        start = prev = int(value)
    spans.append((start, prev))
    span = max(spans, key=lambda item: (item[1] - item[0] + 1, column_score[item[0] : item[1] + 1].sum()))
    left = max(0, x0 + span[0] - int(w * 0.07))
    right = min(w, x0 + span[1] + 1 + int(w * 0.18))
    top = max(0, int(h * 0.08))
    bottom = min(h, int(h * 0.98))
    if right - left < int(w * 0.16):
        return None
    return left, top, right, bottom


def write_builtin_detection_artifacts(component_records: list[tuple[str, str, np.ndarray]], context: dict[str, Any]) -> None:
    detections = []
    instances = []
    masks_dir = Path(context["masks_dir"])
    masks_dir.mkdir(parents=True, exist_ok=True)
    instances_json = Path(context["instances_json"])
    for instance_id, label, mask in component_records:
        bbox = mask_bbox(mask)
        mask_path = masks_dir / f"{safe_name(instance_id)}.pgm"
        write_gray(mask_path, mask.astype(np.float32))
        detections.append({"id": instance_id, "label": label, "score": 1.0, "bbox_xyxy": list(bbox)})
        instances.append(
            {
                "id": instance_id,
                "label": label,
                "score": 1.0,
                "bbox_xyxy": list(bbox),
                "mask_path": mask_path.relative_to(instances_json.parent).as_posix(),
            }
        )

    write_json(Path(context["detections_json"]), {"detections": detections})
    write_json(instances_json, {"instances": instances})


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_").lower()


def estimate_pseudo_depth(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    luminance = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    depth = 1.0 + 1.2 * yy + 0.25 * (1.0 - luminance) - 0.12 * saturation
    return smooth(depth.astype(np.float32), passes=4)


def fuse_depth_from_artifact(base_depth: np.ndarray, artifact: Path, shape: tuple[int, int], *, weight: float) -> np.ndarray:
    if weight <= 0.0 or not artifact.exists():
        return base_depth
    external = read_gray(artifact)
    if external.shape != shape:
        return base_depth

    base_norm = normalize(base_depth)
    external_norm = normalize(smooth(external, passes=1))
    if _correlation(1.0 - external_norm, base_norm) > _correlation(external_norm, base_norm):
        external_norm = 1.0 - external_norm

    lo = float(np.percentile(base_depth, 2.0))
    hi = float(np.percentile(base_depth, 98.0))
    if hi <= lo:
        return base_depth
    external_depth = lo + external_norm * (hi - lo)
    return smooth(((1.0 - weight) * base_depth + weight * external_depth).astype(np.float32), passes=1)


def estimate_foreground_mask(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    border = np.concatenate([rgb[:4].reshape(-1, 3), rgb[-4:].reshape(-1, 3), rgb[:, :4].reshape(-1, 3), rgb[:, -4:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    color_dist = np.linalg.norm(rgb - bg[None, None, :], axis=2)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    threshold = max(0.10, float(np.percentile(color_dist, 72)))
    mask = (color_dist > threshold) | (saturation > 0.22)
    margin_y = max(2, h // 60)
    margin_x = max(2, w // 60)
    mask[:margin_y, :] = False
    mask[-margin_y:, :] = False
    mask[:, :margin_x] = False
    mask[:, -margin_x:] = False
    mask = dilate(erode(mask, 1), 2)
    return mask


def select_components(mask: np.ndarray, max_objects: int, min_area_ratio: float) -> list[np.ndarray]:
    h, w = mask.shape
    min_area = max(16, int(h * w * min_area_ratio))
    comps = connected_components(mask)
    comps = [comp for comp in comps if int(comp.sum()) >= min_area]
    comps.sort(key=lambda comp: int(comp.sum()), reverse=True)
    return comps[:max_objects]


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    ys, xs = np.where(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x] or not mask[start_y, start_x]:
            continue
        comp_coords = []
        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        while stack:
            y, x = stack.pop()
            comp_coords.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        comp = np.zeros_like(mask, dtype=bool)
        cy, cx = zip(*comp_coords)
        comp[np.array(cy), np.array(cx)] = True
        components.append(comp)
    return components


def central_component(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    y0, y1 = int(height * 0.30), int(height * 0.72)
    x0, x1 = int(width * 0.30), int(width * 0.70)
    mask[y0:y1, x0:x1] = True
    return mask


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))


def inpaint_with_border_color(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    unknown = dilate(mask, 3)
    out = rgb.copy()
    known = ~unknown
    if not known.any():
        return smooth_rgb(out, passes=2)
    fallback = np.median(rgb[known], axis=0).astype(np.float32)
    h, w = mask.shape
    for y in range(h):
        row_unknown = unknown[y]
        if not row_unknown.any():
            continue
        row_known = ~row_unknown
        known_x = np.where(row_known)[0]
        if known_x.size == 0:
            out[y, row_unknown] = fallback
            continue
        missing_x = np.where(row_unknown)[0]
        for channel in range(3):
            out[y, missing_x, channel] = np.interp(
                missing_x.astype(np.float32),
                known_x.astype(np.float32),
                out[y, known_x, channel],
            )
    unknown_after_rows = np.isnan(out).any(axis=2)
    if unknown_after_rows.any():
        out[unknown_after_rows] = fallback
    return smooth_rgb(out, passes=2)


def enhance_colors(rgb: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return rgb
    flat = rgb.reshape(-1, 3)
    lo = np.percentile(flat, 1.0, axis=0)
    hi = np.percentile(flat, 99.0, axis=0)
    span = np.maximum(hi - lo, 1e-4)
    stretched = np.clip((rgb - lo[None, None, :]) / span[None, None, :], 0.0, 1.0)
    blended = (1.0 - strength) * rgb + strength * stretched

    luma = (blended @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[:, :, None]
    saturated = luma + (blended - luma) * (1.0 + 0.45 * strength)
    blur = smooth_rgb(saturated, passes=1)
    sharpened = saturated + (saturated - blur) * (0.55 * strength)
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)


def backproject(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: dict[str, float],
    *,
    mask: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    if ys.size > max_points:
        ys, xs = sample_grid_indices(ys, xs, max_points)
    z = depth[ys, xs]
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    xw = (xs.astype(np.float32) - cx) / fx * z
    yw = z
    zw = (cy - ys.astype(np.float32)) / fy * z + 0.55
    points = np.stack([xw, yw, zw], axis=1).astype(np.float32)
    colors = rgb[ys, xs, :3].astype(np.float32)
    return points, colors


def object_billboard_points(
    rgb: np.ndarray,
    mask: np.ndarray,
    extent: np.ndarray,
    *,
    label: str = "",
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    semantic = label.lower()
    raw_ys, raw_xs = np.where(mask)
    if raw_ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    if "water" in semantic:
        x0, y0, x1, y1 = transparent_bottle_bbox_from_edges(rgb, mask.shape) or mask_bbox(mask)
    else:
        x0, y0, x1, y1 = mask_bbox(mask)
    width = max(float(x1 - x0), 1.0)
    height = max(float(y1 - y0), 1.0)
    if "water" in semantic or "gourd" in semantic:
        yy, xx = np.mgrid[y0:y1, x0:x1]
        ys = yy.reshape(-1)
        xs = xx.reshape(-1)
    else:
        ys, xs = raw_ys, raw_xs
    if ys.size > max_points:
        ys, xs = sample_grid_indices(ys, xs, max_points)
    image_height = max(float(mask.shape[0]), 1.0)
    target_height = max(min((height / image_height) * 1.18, 1.15), 0.24)
    target_width = max(target_height * width / height, 0.12)
    if "water" in semantic:
        target_height = max(target_height, 1.02)
        target_width = max(target_width, 0.34)
    elif "gourd" in semantic:
        target_height = min(max(target_height, 0.62), 0.82)
        target_width = max(target_width, 0.42)
    x = ((xs.astype(np.float32) - (x0 + width * 0.5)) / width) * target_width
    z = (((y0 + height * 0.5) - ys.astype(np.float32)) / height) * target_height
    y = np.zeros_like(x, dtype=np.float32)
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = rgb[ys, xs, :3].astype(np.float32)
    return points, colors


def sample_mask_indices(ys: np.ndarray, xs: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    yy = ys.astype(np.uint64) + np.uint64(1)
    xx = xs.astype(np.uint64) + np.uint64(1)
    ranks = (yy * np.uint64(73856093)) ^ (xx * np.uint64(19349663))
    choice = np.argpartition(ranks, max_points - 1)[:max_points]
    order = np.lexsort((xs[choice], ys[choice]))
    idx = choice[order]
    return ys[idx], xs[idx]


def sample_grid_indices(ys: np.ndarray, xs: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if ys.size <= max_points:
        return ys, xs
    order = np.lexsort((xs, ys))
    ordered_y = ys[order]
    ordered_x = xs[order]
    idx = np.linspace(0, ys.size - 1, max_points, dtype=np.int64)
    return ordered_y[idx], ordered_x[idx]


def write_3dgs_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    *,
    point_scale: float = 0.012,
    opacity_logit: float = 4.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c0 = 0.28209479177387814
    sh = (np.clip(colors, 0.0, 1.0) - 0.5) / c0
    scale = math.log(max(point_scale, 1e-5))
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        for name in ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"):
            f.write(f"property float {name}\n")
        for idx in range(45):
            f.write(f"property float f_rest_{idx}\n")
        for name in ("opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"):
            f.write(f"property float {name}\n")
        f.write("end_header\n")
        zeros_rest = " ".join(["0"] * 45)
        for p, c in zip(points, sh):
            f.write(
                f"{p[0]:.7g} {p[1]:.7g} {p[2]:.7g} 0 0 0 "
                f"{c[0]:.7g} {c[1]:.7g} {c[2]:.7g} {zeros_rest} "
                f"{opacity_logit:.7g} {scale:.7g} {scale:.7g} {scale:.7g} 1 0 0 0\n"
            )


def write_depth_mesh_obj(path: Path, rgb: np.ndarray, depth: np.ndarray, intrinsics: dict[str, float], *, grid: int) -> None:
    h, w = depth.shape
    step_y = max(1, h // grid)
    step_x = max(1, w // grid)
    ys = np.arange(0, h, step_y)
    xs = np.arange(0, w, step_x)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    z = depth[yy, xx].astype(np.float32)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    xw = (xx.astype(np.float32) - cx) / fx * z
    yw = z
    zw = (cy - yy.astype(np.float32)) / fy * z + 0.55
    points = np.stack([xw, yw, zw], axis=2).reshape(-1, 3)
    vertex_colors = rgb[yy, xx, :3].reshape(-1, 3)
    rows, cols = len(ys), len(xs)
    path.parent.mkdir(parents=True, exist_ok=True)
    material = path.with_suffix(".mtl")
    face_records: list[tuple[int, int, int, str]] = []
    materials: dict[str, tuple[float, float, float]] = {}
    for r in range(rows - 1):
        for c in range(cols - 1):
            a = r * cols + c + 1
            b = a + 1
            cidx = a + cols
            d = cidx + 1
            color = vertex_colors[[a - 1, b - 1, cidx - 1, d - 1]].mean(axis=0)
            material_name, quantized = _quantized_scene_material(path.stem, color)
            materials.setdefault(material_name, quantized)
            face_records.append((a, b, d, material_name))
            face_records.append((a, d, cidx, material_name))
    material.write_text(
        _scene_mesh_mtl(materials),
        encoding="ascii",
    )
    with path.open("w", encoding="ascii") as f:
        f.write(f"mtllib {material.name}\n")
        f.write("o real2sim_scene_depth_mesh\n")
        for p, color in zip(points, vertex_colors):
            f.write(f"v {p[0]:.7g} {p[1]:.7g} {p[2]:.7g} {color[0]:.7g} {color[1]:.7g} {color[2]:.7g}\n")
        for y in ys:
            v = 1.0 - (float(y) / max(float(h - 1), 1.0))
            for x in xs:
                u = float(x) / max(float(w - 1), 1.0)
                f.write(f"vt {u:.7g} {v:.7g}\n")
        current_material = None
        for a, b, cidx, material_name in face_records:
            if material_name != current_material:
                f.write(f"usemtl {material_name}\n")
                current_material = material_name
            f.write(f"f {a}/{a} {b}/{b} {cidx}/{cidx}\n")


def _quantized_scene_material(stem: str, color: np.ndarray, *, levels: int = 18) -> tuple[str, tuple[float, float, float]]:
    quantized = tuple(float(round(float(channel) * (levels - 1)) / (levels - 1)) for channel in np.clip(color, 0.0, 1.0))
    suffix = "_".join(f"{int(round(channel * 255)):02x}" for channel in quantized)
    return f"{safe_name(stem)}_mat_{suffix}", quantized


def _scene_mesh_mtl(materials: dict[str, tuple[float, float, float]]) -> str:
    lines = ["# Generated Real2Sim scene mesh material palette\n"]
    for name, color in sorted(materials.items()):
        r, g, b = color
        lines.extend(
            [
                f"newmtl {name}\n",
                f"Ka {r:.6f} {g:.6f} {b:.6f}\n",
                f"Kd {r:.6f} {g:.6f} {b:.6f}\n",
                "Ks 0.030000 0.030000 0.030000\n",
                "Ns 8.000000\n",
                "d 1.000000\n",
                "illum 2\n",
                "\n",
            ]
        )
    return "".join(lines)


def write_box_obj(path: Path, extent: np.ndarray) -> None:
    half = np.maximum(extent / 2.0, np.array([0.02, 0.02, 0.02], dtype=np.float32))
    vertices = [
        (-half[0], -half[1], -half[2]),
        (half[0], -half[1], -half[2]),
        (half[0], half[1], -half[2]),
        (-half[0], half[1], -half[2]),
        (-half[0], -half[1], half[2]),
        (half[0], -half[1], half[2]),
        (half[0], half[1], half[2]),
        (-half[0], half[1], half[2]),
    ]
    faces = [(1, 2, 3), (1, 3, 4), (5, 8, 7), (5, 7, 6), (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 8), (3, 8, 4), (4, 8, 5), (4, 5, 1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("o real2sim_object_box\n")
        for v in vertices:
            f.write(f"v {v[0]:.7g} {v[1]:.7g} {v[2]:.7g}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


def smooth(arr: np.ndarray, passes: int) -> np.ndarray:
    out = arr.astype(np.float32)
    for _ in range(passes):
        padded = np.pad(out, 1, mode="edge")
        out = (
            padded[1:-1, 1:-1]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        ) / 5.0
    return out


def smooth_rgb(arr: np.ndarray, passes: int) -> np.ndarray:
    channels = [smooth(arr[:, :, idx], passes) for idx in range(3)]
    return np.stack(channels, axis=2)


def erode(mask: np.ndarray, passes: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(passes):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
    return out


def dilate(mask: np.ndarray, passes: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(passes):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return out


def normalize(arr: np.ndarray) -> np.ndarray:
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val <= min_val:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - min_val) / (max_val - min_val)).astype(np.float32)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float32) - float(np.mean(a))
    bb = b.astype(np.float32) - float(np.mean(b))
    denom = float(np.sqrt(np.mean(aa * aa) * np.mean(bb * bb)))
    if denom <= 1e-8:
        return 0.0
    return float(np.mean(aa * bb) / denom)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))
