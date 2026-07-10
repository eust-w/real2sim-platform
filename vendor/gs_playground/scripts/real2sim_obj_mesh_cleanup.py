#!/usr/bin/env python3
"""Filter OBJ mesh fragments before Real2Sim texture baking."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ObjFace:
    vertices: tuple[int, int, int]
    texcoords: tuple[int, int, int]


@dataclass(frozen=True)
class ObjMesh:
    vertices: np.ndarray
    colors: np.ndarray
    texcoords: np.ndarray
    faces: tuple[ObjFace, ...]


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter small disconnected OBJ mesh fragments")
    parser.add_argument("--input-obj", type=Path, required=True)
    parser.add_argument("--output-obj", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--keep-largest", type=int, default=1, help="Keep this many largest connected components")
    parser.add_argument("--min-faces", type=int, default=0, help="Also keep components with at least this many faces")
    parser.add_argument("--min-area-ratio", type=float, default=0.0, help="Also keep components with area >= ratio * total area")
    parser.add_argument("--max-face-area-percentile", type=float, default=100.0, help="Drop triangles above this area percentile")
    parser.add_argument("--max-edge-length-percentile", type=float, default=100.0, help="Drop triangles whose longest edge is above this percentile")
    parser.add_argument("--min-centroid-x-percentile", type=float, default=0.0, help="Drop triangles whose centroid x is below this percentile")
    parser.add_argument("--max-centroid-x-percentile", type=float, default=100.0, help="Drop triangles whose centroid x is above this percentile")
    parser.add_argument("--min-centroid-y-percentile", type=float, default=0.0, help="Drop triangles whose centroid y is below this percentile")
    parser.add_argument("--max-centroid-y-percentile", type=float, default=100.0, help="Drop triangles whose centroid y is above this percentile")
    parser.add_argument("--min-centroid-z-percentile", type=float, default=0.0, help="Drop triangles whose centroid z is below this percentile")
    parser.add_argument("--max-centroid-z-percentile", type=float, default=100.0, help="Drop triangles whose centroid z is above this percentile")
    args = parser.parse_args()

    report = cleanup_obj_mesh(
        input_obj=args.input_obj,
        output_obj=args.output_obj,
        keep_largest=max(int(args.keep_largest), 0),
        min_faces=max(int(args.min_faces), 0),
        min_area_ratio=max(float(args.min_area_ratio), 0.0),
        max_face_area_percentile=float(args.max_face_area_percentile),
        max_edge_length_percentile=float(args.max_edge_length_percentile),
        centroid_percentile_bounds={
            "x": (float(args.min_centroid_x_percentile), float(args.max_centroid_x_percentile)),
            "y": (float(args.min_centroid_y_percentile), float(args.max_centroid_y_percentile)),
            "z": (float(args.min_centroid_z_percentile), float(args.max_centroid_z_percentile)),
        },
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cleanup_obj_mesh(
    *,
    input_obj: Path,
    output_obj: Path,
    keep_largest: int = 1,
    min_faces: int = 0,
    min_area_ratio: float = 0.0,
    max_face_area_percentile: float = 100.0,
    max_edge_length_percentile: float = 100.0,
    centroid_percentile_bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    mesh = read_obj_mesh(input_obj)
    components = connected_face_components(mesh)
    face_areas = triangle_areas(mesh)
    face_edges = triangle_max_edges(mesh)
    centroids = triangle_centroids(mesh)
    total_area = float(np.sum(face_areas))
    area_threshold = percentile_threshold(face_areas, max_face_area_percentile)
    edge_threshold = percentile_threshold(face_edges, max_edge_length_percentile)
    centroid_filters = centroid_axis_filters(centroids, centroid_percentile_bounds)
    ranked = sorted(
        (
            {
                "component": idx,
                "face_count": len(face_indices),
                "area": float(np.sum(face_areas[np.asarray(face_indices, dtype=np.int64)])),
                "faces": face_indices,
            }
            for idx, face_indices in enumerate(components)
        ),
        key=lambda item: (item["face_count"], item["area"]),
        reverse=True,
    )
    keep_component_ids: set[int] = set()
    for item in ranked[:keep_largest]:
        keep_component_ids.add(int(item["component"]))
    for item in ranked:
        area_ratio = (float(item["area"]) / total_area) if total_area > 0 else 0.0
        if (min_faces > 0 and int(item["face_count"]) >= min_faces) or (
            min_area_ratio > 0.0 and area_ratio >= min_area_ratio
        ):
            keep_component_ids.add(int(item["component"]))
    kept_faces_before_quality = [
        face_index
        for component_id, face_indices in enumerate(components)
        if component_id in keep_component_ids
        for face_index in face_indices
    ]
    kept_faces = [
        face_index
        for face_index in kept_faces_before_quality
        if (area_threshold is None or face_areas[int(face_index)] <= area_threshold)
        and (edge_threshold is None or face_edges[int(face_index)] <= edge_threshold)
        and centroid_face_allowed(centroids[int(face_index)], centroid_filters)
    ]
    cleaned = remap_mesh(mesh, sorted(kept_faces))
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    write_obj_mesh(cleaned, output_obj)
    kept_area = float(np.sum(face_areas[np.asarray(kept_faces, dtype=np.int64)])) if kept_faces else 0.0
    report = {
        "format": "gs-playground-real2sim-obj-mesh-cleanup-report-v1",
        "input_obj": input_obj.resolve().as_posix(),
        "output_obj": output_obj.resolve().as_posix(),
        "input_vertex_count": int(mesh.vertices.shape[0]),
        "input_texcoord_count": int(mesh.texcoords.shape[0]),
        "input_face_count": int(len(mesh.faces)),
        "component_count": int(len(components)),
        "kept_component_count": int(len(keep_component_ids)),
        "component_kept_face_count": int(len(kept_faces_before_quality)),
        "kept_face_count": int(len(kept_faces)),
        "removed_face_count": int(len(mesh.faces) - len(kept_faces)),
        "quality_removed_face_count": int(len(kept_faces_before_quality) - len(kept_faces)),
        "centroid_filter_removed_face_count": int(
            sum(
                1
                for face_index in kept_faces_before_quality
                if not centroid_face_allowed(centroids[int(face_index)], centroid_filters)
            )
        ),
        "kept_area_ratio": round((kept_area / total_area) if total_area > 0 else 0.0, 6),
        "keep_largest": int(keep_largest),
        "min_faces": int(min_faces),
        "min_area_ratio": float(min_area_ratio),
        "max_face_area_percentile": float(max_face_area_percentile),
        "max_edge_length_percentile": float(max_edge_length_percentile),
        "face_area_threshold": float(area_threshold) if area_threshold is not None else None,
        "edge_length_threshold": float(edge_threshold) if edge_threshold is not None else None,
        "centroid_filters": centroid_filters,
        "largest_components": [
            {
                "component": int(item["component"]),
                "face_count": int(item["face_count"]),
                "area_ratio": round((float(item["area"]) / total_area) if total_area > 0 else 0.0, 6),
            }
            for item in ranked[:10]
        ],
    }
    return report


def connected_face_components(mesh: ObjMesh) -> list[list[int]]:
    vertex_to_faces: dict[int, list[int]] = defaultdict(list)
    for face_index, face in enumerate(mesh.faces):
        for vertex in face.vertices:
            vertex_to_faces[int(vertex)].append(face_index)
    visited = np.zeros(len(mesh.faces), dtype=bool)
    components: list[list[int]] = []
    for start in range(len(mesh.faces)):
        if visited[start]:
            continue
        queue: deque[int] = deque([start])
        visited[start] = True
        component: list[int] = []
        while queue:
            face_index = queue.popleft()
            component.append(face_index)
            for vertex in mesh.faces[face_index].vertices:
                for neighbor in vertex_to_faces[int(vertex)]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
        components.append(component)
    return components


def triangle_areas(mesh: ObjMesh) -> np.ndarray:
    if not mesh.faces:
        return np.zeros((0,), dtype=np.float64)
    indices = np.asarray([face.vertices for face in mesh.faces], dtype=np.int64)
    points = mesh.vertices[indices].astype(np.float64)
    areas = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1) * 0.5
    return areas


def triangle_max_edges(mesh: ObjMesh) -> np.ndarray:
    if not mesh.faces:
        return np.zeros((0,), dtype=np.float64)
    indices = np.asarray([face.vertices for face in mesh.faces], dtype=np.int64)
    points = mesh.vertices[indices].astype(np.float64)
    e01 = np.linalg.norm(points[:, 0] - points[:, 1], axis=1)
    e12 = np.linalg.norm(points[:, 1] - points[:, 2], axis=1)
    e20 = np.linalg.norm(points[:, 2] - points[:, 0], axis=1)
    return np.maximum(np.maximum(e01, e12), e20)


def triangle_centroids(mesh: ObjMesh) -> np.ndarray:
    if not mesh.faces:
        return np.zeros((0, 3), dtype=np.float64)
    indices = np.asarray([face.vertices for face in mesh.faces], dtype=np.int64)
    points = mesh.vertices[indices].astype(np.float64)
    return points.mean(axis=1)


def percentile_threshold(values: np.ndarray, percentile: float) -> float | None:
    if values.size == 0 or percentile >= 100.0:
        return None
    clamped = min(max(float(percentile), 0.0), 100.0)
    return float(np.percentile(values, clamped))


def centroid_axis_filters(
    centroids: np.ndarray,
    percentile_bounds: dict[str, tuple[float, float]] | None,
) -> dict[str, dict[str, float | None]]:
    axes = {"x": 0, "y": 1, "z": 2}
    filters: dict[str, dict[str, float | None]] = {}
    if centroids.size == 0:
        return {
            axis: {"min_percentile": None, "max_percentile": None, "min_value": None, "max_value": None}
            for axis in axes
        }
    percentile_bounds = percentile_bounds or {}
    for axis, index in axes.items():
        lower, upper = percentile_bounds.get(axis, (0.0, 100.0))
        lower = min(max(float(lower), 0.0), 100.0)
        upper = min(max(float(upper), 0.0), 100.0)
        if lower > upper:
            lower, upper = upper, lower
        values = centroids[:, index]
        min_value = float(np.percentile(values, lower)) if lower > 0.0 else None
        max_value = float(np.percentile(values, upper)) if upper < 100.0 else None
        filters[axis] = {
            "min_percentile": lower if min_value is not None else None,
            "max_percentile": upper if max_value is not None else None,
            "min_value": min_value,
            "max_value": max_value,
        }
    return filters


def centroid_face_allowed(centroid: np.ndarray, filters: dict[str, dict[str, float | None]]) -> bool:
    for axis, index in {"x": 0, "y": 1, "z": 2}.items():
        bounds = filters.get(axis, {})
        min_value = bounds.get("min_value")
        max_value = bounds.get("max_value")
        value = float(centroid[index])
        if min_value is not None and value < float(min_value):
            return False
        if max_value is not None and value > float(max_value):
            return False
    return True


def remap_mesh(mesh: ObjMesh, face_indices: list[int]) -> ObjMesh:
    vertex_map: dict[int, int] = {}
    texcoord_map: dict[int, int] = {}
    vertices: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    texcoords: list[np.ndarray] = []
    faces: list[ObjFace] = []
    for face_index in face_indices:
        face = mesh.faces[int(face_index)]
        new_vertices: list[int] = []
        new_texcoords: list[int] = []
        for vertex in face.vertices:
            if vertex not in vertex_map:
                vertex_map[vertex] = len(vertices)
                vertices.append(mesh.vertices[vertex])
                colors.append(mesh.colors[vertex])
            new_vertices.append(vertex_map[vertex])
        for texcoord in face.texcoords:
            if texcoord not in texcoord_map:
                texcoord_map[texcoord] = len(texcoords)
                texcoords.append(mesh.texcoords[texcoord])
            new_texcoords.append(texcoord_map[texcoord])
        faces.append(ObjFace(vertices=tuple(new_vertices), texcoords=tuple(new_texcoords)))  # type: ignore[arg-type]
    return ObjMesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        colors=np.asarray(colors, dtype=np.float32),
        texcoords=np.asarray(texcoords, dtype=np.float32),
        faces=tuple(faces),
    )


def read_obj_mesh(path: Path) -> ObjMesh:
    vertices: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    faces: list[ObjFace] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if raw.startswith("v "):
            parts = raw.split()
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            if len(parts) >= 7:
                colors.append((float(parts[4]), float(parts[5]), float(parts[6])))
            else:
                colors.append((0.5, 0.5, 0.5))
        elif raw.startswith("vt "):
            parts = raw.split()
            if len(parts) >= 3:
                texcoords.append((float(parts[1]), float(parts[2])))
        elif raw.startswith("f "):
            refs = [parse_face_ref(token) for token in raw.split()[1:]]
            if len(refs) < 3:
                continue
            for idx in range(1, len(refs) - 1):
                tri = (refs[0], refs[idx], refs[idx + 1])
                faces.append(
                    ObjFace(
                        vertices=tuple(item[0] for item in tri),
                        texcoords=tuple(item[1] if item[1] is not None else item[0] for item in tri),
                    )
                )
    if not texcoords:
        texcoords = [(0.0, 0.0) for _ in vertices]
    return ObjMesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        colors=np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0),
        texcoords=np.asarray(texcoords, dtype=np.float32),
        faces=tuple(faces),
    )


def parse_face_ref(token: str) -> tuple[int, int | None]:
    parts = token.split("/")
    vertex = int(parts[0]) - 1
    texcoord = int(parts[1]) - 1 if len(parts) >= 2 and parts[1] else None
    return vertex, texcoord


def write_obj_mesh(mesh: ObjMesh, output_obj: Path) -> None:
    material_name = output_obj.stem + "_mat"
    material_file = output_obj.with_suffix(".mtl")
    with material_file.open("w", encoding="utf-8") as dst:
        dst.write("# Generated Real2Sim cleaned OBJ material\n")
        dst.write(f"newmtl {material_name}\n")
        dst.write("Ka 1.000000 1.000000 1.000000\n")
        dst.write("Kd 1.000000 1.000000 1.000000\n")
        dst.write("Ks 0.030000 0.030000 0.030000\n")
        dst.write("Ns 8.000000\n")
        dst.write("d 1.000000\n")
        dst.write("illum 2\n")
    with output_obj.open("w", encoding="utf-8") as dst:
        dst.write("# Real2Sim cleaned scene mesh\n")
        dst.write(f"mtllib {material_file.name}\n")
        dst.write("o real2sim_cleaned_scene\n")
        for vertex, color in zip(mesh.vertices, mesh.colors):
            dst.write(
                f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g} "
                f"{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n"
            )
        for texcoord in mesh.texcoords:
            dst.write(f"vt {texcoord[0]:.7g} {texcoord[1]:.7g}\n")
        dst.write(f"usemtl {material_name}\n")
        for face in mesh.faces:
            dst.write(
                "f "
                f"{face.vertices[0] + 1}/{face.texcoords[0] + 1} "
                f"{face.vertices[1] + 1}/{face.texcoords[1] + 1} "
                f"{face.vertices[2] + 1}/{face.texcoords[2] + 1}\n"
            )


if __name__ == "__main__":
    raise SystemExit(main())
