#!/usr/bin/env python3
"""Extract coarse room/scene structure cues from an OBJ mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze axis-aligned scene structure candidates in an OBJ mesh")
    parser.add_argument("--input-obj", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--up-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--normal-threshold", type=float, default=0.78)
    parser.add_argument("--support-normal-threshold", type=float, default=0.82)
    args = parser.parse_args()

    report = analyze_obj_structure(
        input_obj=args.input_obj,
        up_axis=args.up_axis,
        normal_threshold=max(min(float(args.normal_threshold), 1.0), 0.0),
        support_normal_threshold=max(min(float(args.support_normal_threshold), 1.0), 0.0),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def analyze_obj_structure(
    *,
    input_obj: Path,
    up_axis: str = "z",
    normal_threshold: float = 0.78,
    support_normal_threshold: float = 0.82,
) -> dict[str, Any]:
    vertices, faces = read_obj_triangles(input_obj)
    if len(vertices) < 3 or len(faces) < 1:
        raise SystemExit(f"OBJ has no usable triangle mesh: {input_obj}")
    points = vertices[faces]
    normals = triangle_normals(points)
    areas = triangle_areas(points)
    centroids = points.mean(axis=1)
    total_area = float(np.sum(areas))
    up_index = "xyz".index(up_axis)
    axis_reports = [
        axis_structure_report(
            axis=index,
            name=name,
            normals=normals,
            centroids=centroids,
            areas=areas,
            total_area=total_area,
            threshold=normal_threshold,
        )
        for index, name in enumerate("xyz")
    ]
    support = support_plane_candidates(
        up_axis=up_index,
        normals=normals,
        centroids=centroids,
        areas=areas,
        total_area=total_area,
        threshold=support_normal_threshold,
    )
    bounds = {
        "min": [float(value) for value in vertices.min(axis=0)],
        "max": [float(value) for value in vertices.max(axis=0)],
        "p05": [float(np.percentile(vertices[:, idx], 5)) for idx in range(3)],
        "p95": [float(np.percentile(vertices[:, idx], 95)) for idx in range(3)],
    }
    semantic_planes = semantic_plane_candidates(
        up_axis=up_index,
        normals=normals,
        centroids=centroids,
        areas=areas,
        bounds=bounds,
        total_area=total_area,
        threshold=normal_threshold,
    )
    semantic_planes = add_recovered_floor_candidate(
        semantic_planes,
        up_axis=up_index,
        normals=normals,
        centroids=centroids,
        areas=areas,
        bounds=bounds,
        total_area=total_area,
        threshold=support_normal_threshold,
    )
    structure_score = min(
        1.0,
        sum(float(axis["area_ratio"]) for axis in axis_reports) / 3.0
        + min(float(support["support_area_ratio"]) * 0.5, 0.25),
    )
    semantic_score = semantic_structure_score(semantic_planes)
    return {
        "format": "gs-playground-real2sim-obj-structure-report-v1",
        "input_obj": input_obj.resolve().as_posix(),
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "total_area": total_area,
        "up_axis": up_axis,
        "bounds": bounds,
        "axis_aligned_planes": axis_reports,
        "support_planes": support,
        "semantic_planes": semantic_planes,
        "structure_score": round(float(structure_score), 6),
        "commercial_readiness": {
            "structure_score": round(float(structure_score), 6),
            "semantic_structure_score": round(float(semantic_score), 6),
            "has_floor": any(plane["semantic_label"] == "floor" for plane in semantic_planes),
            "has_wall": any(plane["semantic_label"].startswith("wall_") for plane in semantic_planes),
            "has_support_surface": any(plane["semantic_label"] == "support_surface" for plane in semantic_planes),
            "remaining_blockers": [
                "semantic instance segmentation for movable assets",
                "watertight hole filling and room-boundary regularization",
                "collision/nav mesh generation from semantic planes",
                "physically meaningful material parameters",
            ],
        },
        "commercial_gap_notes": [
            "This report now adds coarse semantic structure candidates, but it is not full room understanding.",
            "Commercial-grade Real2Sim still needs instance semantics, watertight reconstruction, collision/nav mesh derivation, and physical material inference.",
        ],
    }


def axis_structure_report(
    *,
    axis: int,
    name: str,
    normals: np.ndarray,
    centroids: np.ndarray,
    areas: np.ndarray,
    total_area: float,
    threshold: float,
) -> dict[str, Any]:
    aligned = np.abs(normals[:, axis]) >= threshold
    area = float(np.sum(areas[aligned])) if np.any(aligned) else 0.0
    values = centroids[aligned, axis] if np.any(aligned) else np.asarray([], dtype=np.float64)
    return {
        "axis": name,
        "aligned_face_count": int(np.count_nonzero(aligned)),
        "area": area,
        "area_ratio": round((area / total_area) if total_area > 0 else 0.0, 6),
        "low_plane": plane_summary(values, areas[aligned], quantile=5),
        "high_plane": plane_summary(values, areas[aligned], quantile=95),
    }


def support_plane_candidates(
    *,
    up_axis: int,
    normals: np.ndarray,
    centroids: np.ndarray,
    areas: np.ndarray,
    total_area: float,
    threshold: float,
) -> dict[str, Any]:
    upward = np.abs(normals[:, up_axis]) >= threshold
    values = centroids[upward, up_axis] if np.any(upward) else np.asarray([], dtype=np.float64)
    support_areas = areas[upward] if np.any(upward) else np.asarray([], dtype=np.float64)
    low = plane_summary(values, support_areas, quantile=10)
    high = plane_summary(values, support_areas, quantile=75)
    support_area = float(np.sum(support_areas))
    return {
        "up_axis": "xyz"[up_axis],
        "support_face_count": int(np.count_nonzero(upward)),
        "support_area": support_area,
        "support_area_ratio": round((support_area / total_area) if total_area > 0 else 0.0, 6),
        "floor_like": low,
        "tabletop_or_shelf_like": high,
    }


def semantic_plane_candidates(
    *,
    up_axis: int,
    normals: np.ndarray,
    centroids: np.ndarray,
    areas: np.ndarray,
    bounds: dict[str, list[float]],
    total_area: float,
    threshold: float,
) -> list[dict[str, Any]]:
    planes: list[dict[str, Any]] = []
    for axis, name in enumerate("xyz"):
        axis_span = max(float(bounds["p95"][axis] - bounds["p05"][axis]), 1e-6)
        clusters = clustered_axis_planes(
            axis=axis,
            normals=normals,
            centroids=centroids,
            areas=areas,
            threshold=threshold,
            cluster_tolerance=max(axis_span * 0.035, 0.04),
        )
        for cluster in clusters:
            label = semantic_label_for_plane(axis=axis, up_axis=up_axis, coordinate=cluster["coordinate"], bounds=bounds)
            if label == "vertical_partition" and cluster["area_ratio"] >= 0.05:
                label = "wall_visible"
            if label == "support_surface" and cluster["area_ratio"] < 0.01:
                continue
            if label.startswith("wall_") and cluster["area_ratio"] < 0.015:
                continue
            planes.append(
                {
                    "semantic_label": label,
                    "axis": name,
                    "coordinate": round(float(cluster["coordinate"]), 6),
                    "area": cluster["area"],
                    "area_ratio": round(float(cluster["area_ratio"]), 6),
                    "face_count": cluster["face_count"],
                    "extent": cluster["extent"],
                    "collision_proxy_hint": collision_proxy_hint(
                        label=label,
                        axis=axis,
                        coordinate=float(cluster["coordinate"]),
                        extent=cluster["extent"],
                        bounds=bounds,
                    ),
                }
            )
    planes.sort(key=lambda item: (semantic_priority(item["semantic_label"]), -float(item["area_ratio"])))
    return planes[:16]


def add_recovered_floor_candidate(
    planes: list[dict[str, Any]],
    *,
    up_axis: int,
    normals: np.ndarray,
    centroids: np.ndarray,
    areas: np.ndarray,
    bounds: dict[str, list[float]],
    total_area: float,
    threshold: float,
) -> list[dict[str, Any]]:
    if any(plane["semantic_label"] == "floor" for plane in planes):
        return planes
    candidate = recovered_floor_candidate(
        up_axis=up_axis,
        normals=normals,
        centroids=centroids,
        areas=areas,
        bounds=bounds,
        total_area=total_area,
        threshold=threshold,
    )
    if candidate is None:
        return planes
    updated = [candidate, *planes]
    updated.sort(key=lambda item: (semantic_priority(item["semantic_label"]), -float(item["area_ratio"])))
    return updated[:16]


def recovered_floor_candidate(
    *,
    up_axis: int,
    normals: np.ndarray,
    centroids: np.ndarray,
    areas: np.ndarray,
    bounds: dict[str, list[float]],
    total_area: float,
    threshold: float,
) -> dict[str, Any] | None:
    upward = np.abs(normals[:, up_axis]) >= threshold
    if not np.any(upward):
        return None
    up_values = centroids[upward, up_axis]
    up_areas = areas[upward]
    low = float(bounds["p05"][up_axis])
    high = float(bounds["p95"][up_axis])
    span = max(high - low, 1e-6)
    coordinate = float(np.percentile(up_values, 10))
    if coordinate > low + span * 0.18:
        return None
    near = np.abs(up_values - coordinate) <= max(span * 0.025, 0.035)
    if not np.any(near):
        return None
    near_points = centroids[upward][near]
    near_area = float(np.sum(up_areas[near]))
    area_ratio = near_area / total_area if total_area > 0 else 0.0
    if area_ratio < 0.012 or near_points.shape[0] < 80:
        return None
    extent = {
        name: [float(np.percentile(near_points[:, idx], 5)), float(np.percentile(near_points[:, idx], 95))]
        for idx, name in enumerate("xyz")
    }
    return {
        "semantic_label": "floor",
        "axis": "xyz"[up_axis],
        "coordinate": round(coordinate, 6),
        "area": near_area,
        "area_ratio": round(float(area_ratio), 6),
        "face_count": int(near_points.shape[0]),
        "extent": extent,
        "recovery_method": "low_horizontal_support_quantile",
        "recovery_confidence": round(min(1.0, area_ratio / 0.05), 6),
        "collision_proxy_hint": collision_proxy_hint(
            label="floor",
            axis=up_axis,
            coordinate=coordinate,
            extent=extent,
            bounds=bounds,
        ),
    }


def clustered_axis_planes(
    *,
    axis: int,
    normals: np.ndarray,
    centroids: np.ndarray,
    areas: np.ndarray,
    threshold: float,
    cluster_tolerance: float,
) -> list[dict[str, Any]]:
    aligned = np.abs(normals[:, axis]) >= threshold
    if not np.any(aligned):
        return []
    aligned_centroids = centroids[aligned]
    aligned_areas = areas[aligned]
    order = np.argsort(aligned_centroids[:, axis])
    sorted_centroids = aligned_centroids[order]
    sorted_areas = aligned_areas[order]
    clusters: list[dict[str, Any]] = []
    start = 0
    for index in range(1, sorted_centroids.shape[0] + 1):
        boundary = index == sorted_centroids.shape[0]
        if not boundary:
            boundary = float(sorted_centroids[index, axis] - sorted_centroids[index - 1, axis]) > cluster_tolerance
        if boundary:
            points = sorted_centroids[start:index]
            weights = sorted_areas[start:index]
            area = float(np.sum(weights))
            if points.size > 0 and area > 0:
                clusters.append(
                    {
                        "coordinate": float(np.average(points[:, axis], weights=weights)),
                        "area": area,
                        "area_ratio": area / float(np.sum(areas)) if np.sum(areas) > 0 else 0.0,
                        "face_count": int(points.shape[0]),
                        "extent": {
                            name: [float(np.percentile(points[:, idx], 5)), float(np.percentile(points[:, idx], 95))]
                            for idx, name in enumerate("xyz")
                        },
                    }
                )
            start = index
    clusters.sort(key=lambda item: item["area"], reverse=True)
    return clusters[:8]


def semantic_label_for_plane(*, axis: int, up_axis: int, coordinate: float, bounds: dict[str, list[float]]) -> str:
    low = float(bounds["p05"][axis])
    high = float(bounds["p95"][axis])
    span = max(high - low, 1e-6)
    if axis == up_axis:
        if coordinate <= low + span * 0.12:
            return "floor"
        if coordinate >= high - span * 0.12:
            return "ceiling"
        return "support_surface"
    if coordinate <= low + span * 0.18:
        return f"wall_{'xyz'[axis]}_low"
    if coordinate >= high - span * 0.18:
        return f"wall_{'xyz'[axis]}_high"
    return "vertical_partition"


def collision_proxy_hint(
    *,
    label: str,
    axis: int,
    coordinate: float,
    extent: dict[str, list[float]],
    bounds: dict[str, list[float]],
) -> dict[str, Any]:
    center = [(float(extent[name][0]) + float(extent[name][1])) * 0.5 for name in "xyz"]
    size = [max(float(extent[name][1]) - float(extent[name][0]), 0.02) for name in "xyz"]
    center[axis] = coordinate
    size[axis] = max(float(bounds["p95"][axis] - bounds["p05"][axis]) * 0.01, 0.02)
    return {
        "type": "box",
        "label": label,
        "center": [round(value, 6) for value in center],
        "size": [round(value, 6) for value in size],
    }


def semantic_priority(label: str) -> int:
    if label == "floor":
        return 0
    if label.startswith("wall_"):
        return 1
    if label == "support_surface":
        return 2
    if label == "ceiling":
        return 3
    return 4


def semantic_structure_score(planes: list[dict[str, Any]]) -> float:
    has_floor = any(plane["semantic_label"] == "floor" for plane in planes)
    wall_count = sum(1 for plane in planes if plane["semantic_label"].startswith("wall_"))
    support_count = sum(1 for plane in planes if plane["semantic_label"] == "support_surface")
    return min(1.0, (0.35 if has_floor else 0.0) + min(wall_count, 4) * 0.12 + min(support_count, 3) * 0.08)


def plane_summary(values: np.ndarray, areas: np.ndarray, *, quantile: float) -> dict[str, Any] | None:
    if values.size == 0:
        return None
    value = float(np.percentile(values, quantile))
    span = max(float(np.percentile(values, 95) - np.percentile(values, 5)), 1e-6)
    near = np.abs(values - value) <= max(span * 0.04, 0.02)
    near_area = float(np.sum(areas[near])) if np.any(near) else 0.0
    return {
        "coordinate": value,
        "quantile": float(quantile),
        "near_face_count": int(np.count_nonzero(near)),
        "near_area": near_area,
    }


def read_obj_triangles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if raw.startswith("v "):
            parts = raw.split()
            if len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif raw.startswith("f "):
            refs = [parse_vertex_ref(token) for token in raw.split()[1:]]
            for idx in range(1, len(refs) - 1):
                faces.append((refs[0], refs[idx], refs[idx + 1]))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def parse_vertex_ref(token: str) -> int:
    return int(token.split("/")[0]) - 1


def triangle_normals(points: np.ndarray) -> np.ndarray:
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-9)


def triangle_areas(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1) * 0.5


if __name__ == "__main__":
    raise SystemExit(main())
