#!/usr/bin/env python3
"""Bake a photo-derived texture for an OBJ using COLMAP registered cameras."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True)
class CameraModel:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class RegisteredImage:
    image_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    image_name: str


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


@dataclass(frozen=True)
class ProjectionCamera:
    image_name: str
    image_path: Path
    width: int
    height: int
    rotation: np.ndarray
    translation: np.ndarray
    center: np.ndarray
    model: CameraModel


@dataclass(frozen=True)
class CameraVisibilityMap:
    width: int
    height: int
    depth: np.ndarray


def main() -> int:
    parser = argparse.ArgumentParser(description="Bake an OBJ texture from COLMAP registered views")
    parser.add_argument("--input-obj", type=Path, required=True)
    parser.add_argument("--output-obj", type=Path, required=True)
    parser.add_argument("--output-texture", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sfm-report", type=Path, help="SfM report with best_model_text_dir and frames_dir")
    parser.add_argument("--colmap-text", type=Path, help="COLMAP text model directory containing cameras.txt/images.txt")
    parser.add_argument("--images-dir", type=Path, help="Directory containing registered image files")
    parser.add_argument("--atlas-size", type=int, default=4096)
    parser.add_argument("--max-cameras", type=int, default=72)
    parser.add_argument("--max-faces", type=int, default=0, help="Evenly sample faces for baking; 0 means all faces")
    parser.add_argument("--max-output-faces", type=int, default=0, help="Evenly sample output mesh faces before UV unwrapping; 0 means all faces")
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--texture-fill-mode", choices=["face", "warp"], default="face", help="Use flat face colors or affine-warp source image triangles into the UV atlas")
    parser.add_argument("--color-blend-cameras", type=int, default=1, help="Blend flat fallback colors from this many top-scoring projected cameras")
    parser.add_argument("--visibility-depth-tolerance", type=float, default=0.0, help="Reject projected faces behind the per-camera mesh z-buffer; <=0 disables")
    parser.add_argument("--visibility-grid-size", type=int, default=320, help="Longest side of the approximate per-camera visibility z-buffer")
    parser.add_argument(
        "--max-warped-faces",
        type=int,
        default=0,
        help="When using --texture-fill-mode warp, only affine-warp the largest UV triangles; 0 means try every projected face",
    )
    parser.add_argument("--unwrap-uv", action="store_true", help="Use xatlas to create non-overlapping texture UVs before baking")
    parser.add_argument("--require-unwrap-uv", action="store_true", help="Fail if --unwrap-uv is requested but xatlas is unavailable")
    parser.add_argument("--postprocess-median-radius", type=int, default=0, help="Median-filter baked texels inside the UV mask")
    parser.add_argument("--bleed-iterations", type=int, default=0, help="Expand baked texels outside UV islands to reduce texture seams")
    parser.add_argument("--background", default="120,120,120", help="RGB background for unfilled texture pixels")
    args = parser.parse_args()

    report = bake_obj_texture(
        input_obj=args.input_obj,
        output_obj=args.output_obj,
        output_texture=args.output_texture,
        report_path=args.report,
        sfm_report=args.sfm_report,
        colmap_text=args.colmap_text,
        images_dir=args.images_dir,
        atlas_size=max(int(args.atlas_size), 64),
        max_cameras=max(int(args.max_cameras), 1),
        max_faces=max(int(args.max_faces), 0),
        max_output_faces=max(int(args.max_output_faces), 0),
        chunk_size=max(int(args.chunk_size), 512),
        texture_fill_mode=args.texture_fill_mode,
        color_blend_cameras=max(int(args.color_blend_cameras), 1),
        visibility_depth_tolerance=max(float(args.visibility_depth_tolerance), 0.0),
        visibility_grid_size=max(int(args.visibility_grid_size), 32),
        max_warped_faces=max(int(args.max_warped_faces), 0),
        unwrap_uv=bool(args.unwrap_uv),
        require_unwrap_uv=bool(args.require_unwrap_uv),
        postprocess_median_radius=max(int(args.postprocess_median_radius), 0),
        bleed_iterations=max(int(args.bleed_iterations), 0),
        background=parse_rgb(args.background),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def bake_obj_texture(
    *,
    input_obj: Path,
    output_obj: Path,
    output_texture: Path,
    report_path: Path,
    sfm_report: Path | None = None,
    colmap_text: Path | None = None,
    images_dir: Path | None = None,
    atlas_size: int = 4096,
    max_cameras: int = 72,
    max_faces: int = 0,
    max_output_faces: int = 0,
    chunk_size: int = 20000,
    texture_fill_mode: str = "face",
    color_blend_cameras: int = 1,
    visibility_depth_tolerance: float = 0.0,
    visibility_grid_size: int = 320,
    max_warped_faces: int = 0,
    unwrap_uv: bool = False,
    require_unwrap_uv: bool = False,
    postprocess_median_radius: int = 0,
    bleed_iterations: int = 0,
    background: tuple[int, int, int] = (120, 120, 120),
) -> dict[str, Any]:
    resolved_text, resolved_images = resolve_colmap_inputs(
        sfm_report=sfm_report,
        colmap_text=colmap_text,
        images_dir=images_dir,
    )
    mesh = read_obj_mesh(input_obj)
    mesh, mesh_sampling_report = maybe_sample_mesh_faces(mesh, max_output_faces=max_output_faces)
    mesh, unwrap_report = maybe_unwrap_mesh_uvs(
        mesh,
        requested=unwrap_uv,
        require=require_unwrap_uv,
    )
    cameras = load_projection_cameras(resolved_text, resolved_images, max_cameras=max_cameras)
    selected_indices = selected_face_indices(len(mesh.faces), max_faces=max_faces)
    texture_report = bake_texture_image(
        mesh=mesh,
        cameras=cameras,
        face_indices=selected_indices,
        output_texture=output_texture,
        atlas_size=atlas_size,
        chunk_size=chunk_size,
        texture_fill_mode=texture_fill_mode,
        color_blend_cameras=color_blend_cameras,
        visibility_depth_tolerance=visibility_depth_tolerance,
        visibility_grid_size=visibility_grid_size,
        max_warped_faces=max_warped_faces,
        postprocess_median_radius=postprocess_median_radius,
        bleed_iterations=bleed_iterations,
        background=background,
    )
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    output_texture = output_texture.resolve()
    write_textured_obj(
        source_mesh=mesh,
        output_obj=output_obj,
        texture_name=output_texture.name,
    )
    report = {
        "format": "gs-playground-real2sim-colmap-texture-bake-report-v1",
        "input_obj": input_obj.resolve().as_posix(),
        "output_obj": output_obj.resolve().as_posix(),
        "output_texture": output_texture.as_posix(),
        "colmap_text": resolved_text.resolve().as_posix(),
        "images_dir": resolved_images.resolve().as_posix(),
        "vertex_count": int(mesh.vertices.shape[0]),
        "face_count": int(len(mesh.faces)),
        "texcoord_count": int(mesh.texcoords.shape[0]),
        "mesh_sampling": mesh_sampling_report,
        "uv_unwrap": unwrap_report,
        "registered_camera_count": int(len(cameras)),
        "selected_camera_count": int(len(cameras)),
        "selected_face_count": int(len(selected_indices)),
        **texture_report,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def resolve_colmap_inputs(
    *,
    sfm_report: Path | None,
    colmap_text: Path | None,
    images_dir: Path | None,
) -> tuple[Path, Path]:
    if sfm_report is not None:
        data = json.loads(sfm_report.read_text(encoding="utf-8"))
        if colmap_text is None:
            value = data.get("best_model_text_dir")
            if value:
                colmap_text = Path(str(value))
        if images_dir is None:
            value = data.get("frames_dir")
            if value:
                images_dir = Path(str(value))
    if colmap_text is None or not colmap_text.is_dir():
        raise SystemExit(f"COLMAP text model directory not found: {colmap_text}")
    if images_dir is None or not images_dir.is_dir():
        raise SystemExit(f"registered images directory not found: {images_dir}")
    return colmap_text, images_dir


def read_obj_mesh(path: Path) -> ObjMesh:
    vertices: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    faces: list[ObjFace] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if raw.startswith("v "):
            parts = raw.split()
            if len(parts) < 4:
                continue
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
                vi = tuple(item[0] for item in tri)
                vti = tuple(item[1] if item[1] is not None else item[0] for item in tri)
                faces.append(ObjFace(vertices=vi, texcoords=vti))
    if not vertices or not faces:
        raise SystemExit(f"OBJ has no usable vertices/faces: {path}")
    vertex_array = np.asarray(vertices, dtype=np.float32)
    color_array = np.asarray(colors, dtype=np.float32)
    if texcoords:
        texcoord_array = np.asarray(texcoords, dtype=np.float32)
    else:
        texcoord_array = planar_texcoords(vertex_array)
        faces = [
            ObjFace(vertices=face.vertices, texcoords=face.vertices)
            for face in faces
        ]
    return ObjMesh(
        vertices=vertex_array,
        colors=np.clip(color_array, 0.0, 1.0),
        texcoords=np.clip(texcoord_array, 0.0, 1.0),
        faces=tuple(faces),
    )


def maybe_unwrap_mesh_uvs(
    mesh: ObjMesh,
    *,
    requested: bool,
    require: bool,
) -> tuple[ObjMesh, dict[str, Any]]:
    report: dict[str, Any] = {
        "requested": bool(requested),
        "required": bool(require),
        "applied": False,
        "method": "input_obj_uv",
        "input_vertex_count": int(mesh.vertices.shape[0]),
        "input_texcoord_count": int(mesh.texcoords.shape[0]),
        "input_face_count": int(len(mesh.faces)),
    }
    if not requested:
        return mesh, report
    try:
        import xatlas  # type: ignore[import-not-found]
    except Exception as exc:
        report["error"] = f"xatlas unavailable: {exc}"
        if require:
            raise SystemExit(report["error"])
        return mesh, report

    faces = np.asarray([face.vertices for face in mesh.faces], dtype=np.uint32)
    try:
        vmapping, indices, texcoords = xatlas.parametrize(
            np.asarray(mesh.vertices, dtype=np.float32),
            faces,
        )
    except Exception as exc:
        report["error"] = f"xatlas unwrap failed: {exc}"
        if require:
            raise SystemExit(report["error"])
        return mesh, report

    vmapping = np.asarray(vmapping, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    texcoords = np.asarray(texcoords, dtype=np.float32)
    if indices.ndim != 2 or indices.shape[1] != 3 or texcoords.ndim != 2 or texcoords.shape[1] != 2:
        report["error"] = "xatlas returned an unexpected mesh layout"
        if require:
            raise SystemExit(report["error"])
        return mesh, report

    unwrapped = ObjMesh(
        vertices=mesh.vertices[np.clip(vmapping, 0, mesh.vertices.shape[0] - 1)],
        colors=mesh.colors[np.clip(vmapping, 0, mesh.colors.shape[0] - 1)],
        texcoords=np.clip(texcoords, 0.0, 1.0),
        faces=tuple(
            ObjFace(
                vertices=(int(face[0]), int(face[1]), int(face[2])),
                texcoords=(int(face[0]), int(face[1]), int(face[2])),
            )
            for face in indices
        ),
    )
    report.update(
        {
            "applied": True,
            "method": "xatlas",
            "output_vertex_count": int(unwrapped.vertices.shape[0]),
            "output_texcoord_count": int(unwrapped.texcoords.shape[0]),
            "output_face_count": int(len(unwrapped.faces)),
            "duplicated_vertex_count": int(max(0, unwrapped.vertices.shape[0] - mesh.vertices.shape[0])),
        }
    )
    return unwrapped, report


def maybe_sample_mesh_faces(mesh: ObjMesh, *, max_output_faces: int) -> tuple[ObjMesh, dict[str, Any]]:
    report: dict[str, Any] = {
        "requested_max_output_faces": int(max_output_faces),
        "applied": False,
        "method": "all_faces",
        "input_face_count": int(len(mesh.faces)),
        "output_face_count": int(len(mesh.faces)),
    }
    if max_output_faces <= 0 or max_output_faces >= len(mesh.faces):
        return mesh, report
    indices = selected_face_indices(len(mesh.faces), max_faces=max_output_faces)
    sampled_faces = tuple(mesh.faces[int(index)] for index in indices)
    sampled = ObjMesh(
        vertices=mesh.vertices,
        colors=mesh.colors,
        texcoords=mesh.texcoords,
        faces=sampled_faces,
    )
    report.update(
        {
            "applied": True,
            "method": "uniform_face_sampling",
            "output_face_count": int(len(sampled.faces)),
        }
    )
    return sampled, report


def parse_face_ref(token: str) -> tuple[int, int | None]:
    parts = token.split("/")
    vertex = parse_obj_index(parts[0])
    texcoord = parse_obj_index(parts[1]) if len(parts) >= 2 and parts[1] else None
    return vertex, texcoord


def parse_obj_index(value: str) -> int:
    index = int(value)
    if index <= 0:
        raise SystemExit("negative OBJ indices are not supported for texture baking")
    return index - 1


def planar_texcoords(vertices: np.ndarray) -> np.ndarray:
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    axes = np.argsort(extents)[-2:]
    selected = vertices[:, axes]
    low = selected.min(axis=0)
    span = np.maximum(selected.max(axis=0) - low, 1e-9)
    return (selected - low) / span


def read_colmap_cameras(path: Path) -> dict[int, CameraModel]:
    cameras: dict[int, CameraModel] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        cameras[camera_id] = CameraModel(
            camera_id=camera_id,
            model=parts[1].upper(),
            width=int(parts[2]),
            height=int(parts[3]),
            params=tuple(float(value) for value in parts[4:]),
        )
    return cameras


def read_colmap_images(path: Path) -> list[RegisteredImage]:
    images: list[RegisteredImage] = []
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        idx += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        images.append(
            RegisteredImage(
                image_id=int(parts[0]),
                qvec=tuple(float(value) for value in parts[1:5]),  # type: ignore[arg-type]
                tvec=tuple(float(value) for value in parts[5:8]),  # type: ignore[arg-type]
                camera_id=int(parts[8]),
                image_name=" ".join(parts[9:]),
            )
        )
        idx += 1
    return images


def load_projection_cameras(colmap_text: Path, images_dir: Path, *, max_cameras: int) -> list[ProjectionCamera]:
    camera_models = read_colmap_cameras(colmap_text / "cameras.txt")
    registered = read_colmap_images(colmap_text / "images.txt")
    if max_cameras > 0 and len(registered) > max_cameras:
        indices = np.linspace(0, len(registered) - 1, max_cameras).round().astype(int)
        registered = [registered[int(index)] for index in indices]
    cameras: list[ProjectionCamera] = []
    for image in registered:
        model = camera_models.get(image.camera_id)
        image_path = images_dir / image.image_name
        if model is None or not image_path.is_file():
            continue
        rotation = np.asarray(qvec_to_rotmat(image.qvec), dtype=np.float32)
        translation = np.asarray(image.tvec, dtype=np.float32)
        center = -rotation.T @ translation
        cameras.append(
            ProjectionCamera(
                image_name=image.image_name,
                image_path=image_path,
                width=model.width,
                height=model.height,
                rotation=rotation,
                translation=translation,
                center=center,
                model=model,
            )
        )
    if not cameras:
        raise SystemExit("no registered COLMAP image could be loaded for texture baking")
    return cameras


def selected_face_indices(face_count: int, *, max_faces: int) -> np.ndarray:
    if max_faces <= 0 or max_faces >= face_count:
        return np.arange(face_count, dtype=np.int64)
    return np.linspace(0, face_count - 1, max_faces).round().astype(np.int64)


def build_camera_visibility_maps(
    *,
    mesh: ObjMesh,
    cameras: list[ProjectionCamera],
    face_indices: np.ndarray,
    grid_size: int,
) -> list[CameraVisibilityMap]:
    if len(face_indices) == 0:
        return []
    faces = [mesh.faces[int(index)] for index in face_indices]
    vi = np.asarray([face.vertices for face in faces], dtype=np.int64)
    centroids = mesh.vertices[vi].mean(axis=1)
    maps: list[CameraVisibilityMap] = []
    for camera in cameras:
        scale = max(float(camera.width), float(camera.height), 1.0)
        width = max(int(round(camera.width / scale * grid_size)), 8)
        height = max(int(round(camera.height / scale * grid_size)), 8)
        depth = np.full((height, width), np.inf, dtype=np.float32)
        cam_points = centroids @ camera.rotation.T + camera.translation
        projected = project_camera(camera.model, cam_points)
        if projected is not None:
            u, v = projected
            z = cam_points[:, 2]
            finite = np.isfinite(z) & np.isfinite(u) & np.isfinite(v)
            valid = finite & (z > 1e-6) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
            if np.any(valid):
                gx = np.clip((u[valid] / max(camera.width, 1) * width).astype(np.int64), 0, width - 1)
                gy = np.clip((v[valid] / max(camera.height, 1) * height).astype(np.int64), 0, height - 1)
                flat = gy * width + gx
                np.minimum.at(depth.reshape(-1), flat, z[valid].astype(np.float32))
        maps.append(CameraVisibilityMap(width=width, height=height, depth=depth))
    return maps


def camera_visibility_mask(
    *,
    visibility: CameraVisibilityMap | None,
    camera: ProjectionCamera,
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    if visibility is None or tolerance <= 0.0:
        return np.ones(z.shape, dtype=bool)
    gx = np.clip((u / max(camera.width, 1) * visibility.width).astype(np.int64), 0, visibility.width - 1)
    gy = np.clip((v / max(camera.height, 1) * visibility.height).astype(np.int64), 0, visibility.height - 1)
    nearest = visibility.depth[gy, gx].astype(np.float64)
    slack = np.maximum(float(tolerance), np.abs(nearest) * float(tolerance))
    return np.isfinite(nearest) & (z.astype(np.float64) <= nearest + slack)


def warp_allowed_face_indices(
    *,
    mesh: ObjMesh,
    face_indices: np.ndarray,
    atlas_size: int,
    max_warped_faces: int,
) -> set[int] | None:
    if max_warped_faces <= 0 or max_warped_faces >= len(face_indices):
        return None
    areas: list[tuple[float, int]] = []
    for index in face_indices:
        face = mesh.faces[int(index)]
        texcoords = mesh.texcoords[np.clip(face.texcoords, 0, mesh.texcoords.shape[0] - 1)]
        area = polygon_area(uv_polygon_pixels(texcoords, atlas_size))
        if area >= 0.25:
            areas.append((float(area), int(index)))
    if len(areas) <= max_warped_faces:
        return {index for _area, index in areas}
    areas.sort(reverse=True)
    return {index for _area, index in areas[:max_warped_faces]}


def bake_texture_image(
    *,
    mesh: ObjMesh,
    cameras: list[ProjectionCamera],
    face_indices: np.ndarray,
    output_texture: Path,
    atlas_size: int,
    chunk_size: int,
    texture_fill_mode: str,
    color_blend_cameras: int,
    visibility_depth_tolerance: float,
    visibility_grid_size: int,
    max_warped_faces: int,
    postprocess_median_radius: int,
    bleed_iterations: int,
    background: tuple[int, int, int],
) -> dict[str, Any]:
    texture = Image.new("RGB", (atlas_size, atlas_size), background)
    mask = Image.new("L", (atlas_size, atlas_size), 0)
    draw = ImageDraw.Draw(texture)
    mask_draw = ImageDraw.Draw(mask)
    image_cache: OrderedDict[str, np.ndarray] = OrderedDict()
    pil_image_cache: OrderedDict[str, Image.Image] = OrderedDict()
    projected_faces = 0
    fallback_faces = 0
    drawn_faces = 0
    warped_faces = 0
    warp_fallback_faces = 0
    warp_skipped_faces = 0
    warp_allowed_faces = warp_allowed_face_indices(
        mesh=mesh,
        face_indices=face_indices,
        atlas_size=atlas_size,
        max_warped_faces=max_warped_faces,
    )
    visibility_maps = build_camera_visibility_maps(
        mesh=mesh,
        cameras=cameras,
        face_indices=face_indices,
        grid_size=visibility_grid_size,
    ) if visibility_depth_tolerance > 0.0 else None
    for start in range(0, len(face_indices), chunk_size):
        chunk_indices = face_indices[start : start + chunk_size]
        faces = [mesh.faces[int(index)] for index in chunk_indices]
        vi = np.asarray([face.vertices for face in faces], dtype=np.int64)
        vti = np.asarray([face.texcoords for face in faces], dtype=np.int64)
        points = mesh.vertices[vi]
        colors = mesh.colors[vi].mean(axis=1)
        centroids = points.mean(axis=1)
        normals = face_normals(points)
        best_camera, best_uv, best_score = select_best_camera(
            cameras,
            centroids,
            normals,
            visibility_maps=visibility_maps,
            visibility_depth_tolerance=visibility_depth_tolerance,
        )
        sampled_colors = np.clip((colors * 255.0).round(), 0, 255).astype(np.uint8)
        valid = best_camera >= 0
        if np.any(valid):
            projected_faces += int(np.count_nonzero(valid))
            if color_blend_cameras > 1:
                sampled_colors[valid] = blend_projected_face_colors(
                    cameras=cameras,
                    centroids=centroids[valid],
                    normals=normals[valid],
                    image_cache=image_cache,
                    max_cameras=color_blend_cameras,
                    visibility_maps=visibility_maps,
                    visibility_depth_tolerance=visibility_depth_tolerance,
                )
            else:
                for camera_idx in np.unique(best_camera[valid]):
                    face_mask = best_camera == camera_idx
                    camera = cameras[int(camera_idx)]
                    image = cached_image(image_cache, camera.image_path)
                    sampled_colors[face_mask] = sample_image(image, best_uv[face_mask])
        fallback_faces += int(np.count_nonzero(~valid))
        uv = mesh.texcoords[np.clip(vti, 0, mesh.texcoords.shape[0] - 1)]
        for local_idx, tri_uv in enumerate(uv):
            polygon = uv_polygon_pixels(tri_uv, atlas_size)
            if polygon_area(polygon) < 0.25:
                continue
            filled_by_warp = False
            can_warp = (
                texture_fill_mode == "warp"
                and best_camera[local_idx] >= 0
                and (warp_allowed_faces is None or int(chunk_indices[local_idx]) in warp_allowed_faces)
            )
            if can_warp:
                camera = cameras[int(best_camera[local_idx])]
                src_tri = project_triangle_pixels(camera, points[local_idx])
                if src_tri is not None:
                    image = cached_pil_image(pil_image_cache, camera.image_path)
                    filled_by_warp = paste_warped_triangle(
                        texture=texture,
                        image=image,
                        dest_polygon=polygon,
                        source_triangle=src_tri,
                    )
            if filled_by_warp:
                warped_faces += 1
            else:
                if texture_fill_mode == "warp" and best_camera[local_idx] >= 0:
                    if warp_allowed_faces is not None and int(chunk_indices[local_idx]) not in warp_allowed_faces:
                        warp_skipped_faces += 1
                    else:
                        warp_fallback_faces += 1
                color = tuple(int(value) for value in sampled_colors[local_idx])
                draw.polygon(polygon, fill=color)
            mask_draw.polygon(polygon, fill=255)
            drawn_faces += 1
    mask_array = np.asarray(mask, dtype=np.uint8)
    coverage = float(np.count_nonzero(mask_array)) / float(mask_array.size)
    selected_count = max(int(len(face_indices)), 1)
    projected_ratio = float(projected_faces) / float(selected_count)
    drawn_ratio = float(drawn_faces) / float(selected_count)
    postprocess_report = postprocess_texture(
        texture=texture,
        mask=mask,
        median_radius=postprocess_median_radius,
        bleed_iterations=bleed_iterations,
    )
    output_texture.parent.mkdir(parents=True, exist_ok=True)
    texture.save(output_texture)
    return {
        "atlas_size": int(atlas_size),
        "projected_face_count": int(projected_faces),
        "fallback_face_count": int(fallback_faces),
        "drawn_face_count": int(drawn_faces),
        "texture_fill_mode": texture_fill_mode,
        "color_blend_cameras": int(color_blend_cameras),
        "visibility_depth_tolerance": float(visibility_depth_tolerance),
        "visibility_grid_size": int(visibility_grid_size),
        "visibility_camera_count": int(len(visibility_maps)) if visibility_maps is not None else 0,
        "max_warped_faces": int(max_warped_faces),
        "warp_candidate_face_count": int(len(warp_allowed_faces)) if warp_allowed_faces is not None else int(len(face_indices)),
        "warped_face_count": int(warped_faces),
        "warp_fallback_face_count": int(warp_fallback_faces),
        "warp_skipped_face_count": int(warp_skipped_faces),
        "texture_coverage_ratio": round(coverage, 6),
        "atlas_pixel_coverage_ratio": round(coverage, 6),
        "projected_face_ratio": round(projected_ratio, 6),
        "textured_face_ratio": round(drawn_ratio, 6),
        "texture_quality_breakdown": {
            "mesh_face_coverage": round(drawn_ratio, 6),
            "camera_projected_face_coverage": round(projected_ratio, 6),
            "atlas_pixel_coverage": round(coverage, 6),
            "interpretation": "atlas_pixel_coverage measures UV atlas occupancy; mesh_face_coverage measures how much reconstructed mesh received a baked texture.",
        },
        "texture_postprocess": postprocess_report,
    }


def postprocess_texture(
    *,
    texture: Image.Image,
    mask: Image.Image,
    median_radius: int,
    bleed_iterations: int,
) -> dict[str, Any]:
    report = {
        "median_radius": int(max(median_radius, 0)),
        "median_applied": False,
        "bleed_iterations_requested": int(max(bleed_iterations, 0)),
        "bleed_iterations_applied": 0,
        "bleed_filled_pixels": 0,
    }
    if median_radius > 0:
        size = max(3, int(median_radius) * 2 + 1)
        filtered = texture.filter(ImageFilter.MedianFilter(size=size))
        texture.paste(filtered, mask=mask)
        report["median_applied"] = True
        report["median_filter_size"] = int(size)
    if bleed_iterations > 0:
        filled_texture, filled_mask, filled_pixels, applied = bleed_texture_edges(
            texture,
            mask,
            iterations=int(bleed_iterations),
        )
        texture.paste(filled_texture)
        mask.paste(filled_mask)
        report["bleed_iterations_applied"] = int(applied)
        report["bleed_filled_pixels"] = int(filled_pixels)
    return report


def bleed_texture_edges(
    texture: Image.Image,
    mask: Image.Image,
    *,
    iterations: int,
) -> tuple[Image.Image, Image.Image, int, int]:
    colors = np.asarray(texture.convert("RGB"), dtype=np.float32).copy()
    valid = np.asarray(mask, dtype=np.uint8) > 0
    total_filled = 0
    applied = 0
    shifts = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    for _ in range(max(int(iterations), 0)):
        accum = np.zeros_like(colors, dtype=np.float32)
        counts = np.zeros(valid.shape, dtype=np.float32)
        for dy, dx in shifts:
            shifted_valid = shifted_bool(valid, dy=dy, dx=dx)
            shifted_colors = shifted_rgb(colors, dy=dy, dx=dx)
            accum += shifted_colors * shifted_valid[:, :, None]
            counts += shifted_valid.astype(np.float32)
        fill = (~valid) & (counts > 0)
        filled = int(np.count_nonzero(fill))
        if filled <= 0:
            break
        colors[fill] = accum[fill] / counts[fill, None]
        valid[fill] = True
        total_filled += filled
        applied += 1
    out_texture = Image.fromarray(np.clip(colors, 0, 255).astype(np.uint8), "RGB")
    out_mask = Image.fromarray((valid.astype(np.uint8) * 255), "L")
    return out_texture, out_mask, total_filled, applied


def shifted_bool(values: np.ndarray, *, dy: int, dx: int) -> np.ndarray:
    result = np.zeros_like(values, dtype=bool)
    src_y0 = max(0, -dy)
    src_y1 = values.shape[0] - max(0, dy)
    src_x0 = max(0, -dx)
    src_x1 = values.shape[1] - max(0, dx)
    dst_y0 = max(0, dy)
    dst_y1 = values.shape[0] - max(0, -dy)
    dst_x0 = max(0, dx)
    dst_x1 = values.shape[1] - max(0, -dx)
    result[dst_y0:dst_y1, dst_x0:dst_x1] = values[src_y0:src_y1, src_x0:src_x1]
    return result


def shifted_rgb(values: np.ndarray, *, dy: int, dx: int) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    src_y0 = max(0, -dy)
    src_y1 = values.shape[0] - max(0, dy)
    src_x0 = max(0, -dx)
    src_x1 = values.shape[1] - max(0, dx)
    dst_y0 = max(0, dy)
    dst_y1 = values.shape[0] - max(0, -dy)
    dst_x0 = max(0, dx)
    dst_x1 = values.shape[1] - max(0, -dx)
    result[dst_y0:dst_y1, dst_x0:dst_x1] = values[src_y0:src_y1, src_x0:src_x1]
    return result


def face_normals(points: np.ndarray) -> np.ndarray:
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-9)


def select_best_camera(
    cameras: list[ProjectionCamera],
    centroids: np.ndarray,
    normals: np.ndarray,
    *,
    visibility_maps: list[CameraVisibilityMap] | None = None,
    visibility_depth_tolerance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = centroids.shape[0]
    best_camera = np.full(count, -1, dtype=np.int32)
    best_uv = np.zeros((count, 2), dtype=np.float32)
    best_score = np.full(count, -np.inf, dtype=np.float32)
    for camera_idx, camera in enumerate(cameras):
        cam_points = centroids @ camera.rotation.T + camera.translation
        projected = project_camera(camera.model, cam_points)
        if projected is None:
            continue
        u, v = projected
        z = cam_points[:, 2]
        finite = np.isfinite(z) & np.isfinite(u) & np.isfinite(v)
        valid = finite & (z > 1e-6) & (u >= 1) & (u < camera.width - 1) & (v >= 1) & (v < camera.height - 1)
        if visibility_maps is not None and camera_idx < len(visibility_maps):
            visible = camera_visibility_mask(
                visibility=visibility_maps[camera_idx],
                camera=camera,
                u=u,
                v=v,
                z=z,
                tolerance=visibility_depth_tolerance,
            )
            valid &= visible
        if not np.any(valid):
            continue
        valid_idx = np.nonzero(valid)[0]
        rays = camera.center.astype(np.float64)[None, :] - centroids[valid_idx].astype(np.float64)
        ray_norms = np.linalg.norm(rays, axis=1)
        normal_score = np.abs(
            np.sum(normals[valid_idx].astype(np.float64) * rays, axis=1) / np.maximum(ray_norms, 1e-9)
        )
        center_score = np.full(count, -np.inf, dtype=np.float32)
        du = (u[valid_idx] - (camera.width * 0.5)) / max(camera.width * 0.5, 1.0)
        dv = (v[valid_idx] - (camera.height * 0.5)) / max(camera.height * 0.5, 1.0)
        center_score[valid_idx] = (1.0 - np.sqrt(du * du + dv * dv)).astype(np.float32)
        score = np.full(count, -np.inf, dtype=np.float32)
        score[valid_idx] = (0.70 * normal_score + 0.30 * center_score[valid_idx]).astype(np.float32)
        update = score > best_score
        if not np.any(update):
            continue
        best_score[update] = score[update]
        best_camera[update] = camera_idx
        best_uv[update, 0] = u[update]
        best_uv[update, 1] = v[update]
    return best_camera, best_uv, best_score


def blend_projected_face_colors(
    *,
    cameras: list[ProjectionCamera],
    centroids: np.ndarray,
    normals: np.ndarray,
    image_cache: OrderedDict[str, np.ndarray],
    max_cameras: int,
    visibility_maps: list[CameraVisibilityMap] | None = None,
    visibility_depth_tolerance: float = 0.0,
) -> np.ndarray:
    count = centroids.shape[0]
    top_k = max(int(max_cameras), 1)
    top_camera = np.full((count, top_k), -1, dtype=np.int32)
    top_uv = np.zeros((count, top_k, 2), dtype=np.float32)
    top_score = np.full((count, top_k), -np.inf, dtype=np.float32)
    for camera_idx, camera in enumerate(cameras):
        cam_points = centroids @ camera.rotation.T + camera.translation
        projected = project_camera(camera.model, cam_points)
        if projected is None:
            continue
        u, v = projected
        z = cam_points[:, 2]
        finite = np.isfinite(z) & np.isfinite(u) & np.isfinite(v)
        valid = finite & (z > 1e-6) & (u >= 1) & (u < camera.width - 1) & (v >= 1) & (v < camera.height - 1)
        if visibility_maps is not None and camera_idx < len(visibility_maps):
            visible = camera_visibility_mask(
                visibility=visibility_maps[camera_idx],
                camera=camera,
                u=u,
                v=v,
                z=z,
                tolerance=visibility_depth_tolerance,
            )
            valid &= visible
        if not np.any(valid):
            continue
        valid_idx = np.nonzero(valid)[0]
        rays = camera.center.astype(np.float64)[None, :] - centroids[valid_idx].astype(np.float64)
        ray_norms = np.linalg.norm(rays, axis=1)
        normal_score = np.abs(
            np.sum(normals[valid_idx].astype(np.float64) * rays, axis=1) / np.maximum(ray_norms, 1e-9)
        )
        du = (u[valid_idx] - (camera.width * 0.5)) / max(camera.width * 0.5, 1.0)
        dv = (v[valid_idx] - (camera.height * 0.5)) / max(camera.height * 0.5, 1.0)
        center_score = (1.0 - np.sqrt(du * du + dv * dv)).astype(np.float32)
        score = (0.70 * normal_score + 0.30 * center_score).astype(np.float32)
        for local, face_index in enumerate(valid_idx):
            weakest = int(np.argmin(top_score[face_index]))
            if score[local] <= top_score[face_index, weakest]:
                continue
            top_score[face_index, weakest] = score[local]
            top_camera[face_index, weakest] = camera_idx
            top_uv[face_index, weakest, 0] = u[face_index]
            top_uv[face_index, weakest, 1] = v[face_index]

    accum = np.zeros((count, 3), dtype=np.float64)
    weights = np.zeros((count,), dtype=np.float64)
    for slot in range(top_k):
        slot_valid = top_camera[:, slot] >= 0
        if not np.any(slot_valid):
            continue
        for camera_idx in np.unique(top_camera[slot_valid, slot]):
            face_mask = slot_valid & (top_camera[:, slot] == camera_idx)
            camera = cameras[int(camera_idx)]
            image = cached_image(image_cache, camera.image_path)
            colors = sample_image(image, top_uv[face_mask, slot, :]).astype(np.float64)
            score = np.maximum(top_score[face_mask, slot].astype(np.float64), 1e-3)
            accum[face_mask] += colors * score[:, None]
            weights[face_mask] += score
    missing = weights <= 0
    if np.any(missing):
        accum[missing] = 127.0
        weights[missing] = 1.0
    return np.clip(np.rint(accum / weights[:, None]), 0, 255).astype(np.uint8)


def project_camera(model: CameraModel, cam_points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    z = cam_points[:, 2].astype(np.float64)
    x = np.full(z.shape, np.nan, dtype=np.float64)
    y = np.full(z.shape, np.nan, dtype=np.float64)
    valid_z = np.isfinite(z) & (z > 1e-9)
    x[valid_z] = cam_points[valid_z, 0].astype(np.float64) / z[valid_z]
    y[valid_z] = cam_points[valid_z, 1].astype(np.float64) / z[valid_z]
    params = model.params
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if model.model == "SIMPLE_PINHOLE":
            f, cx, cy = params[:3]
            return f * x + cx, f * y + cy
        if model.model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
            return fx * x + cx, fy * y + cy
        if model.model in {"SIMPLE_RADIAL", "RADIAL"}:
            f, cx, cy = params[:3]
            k1 = params[3] if len(params) > 3 else 0.0
            k2 = params[4] if len(params) > 4 else 0.0
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2
            return f * x * radial + cx, f * y * radial + cy
        if model.model == "OPENCV":
            fx, fy, cx, cy = params[:4]
            k1 = params[4] if len(params) > 4 else 0.0
            k2 = params[5] if len(params) > 5 else 0.0
            p1 = params[6] if len(params) > 6 else 0.0
            p2 = params[7] if len(params) > 7 else 0.0
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2
            x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            return fx * x_dist + cx, fy * y_dist + cy
    return None


def cached_image(cache: OrderedDict[str, np.ndarray], path: Path, *, max_entries: int = 6) -> np.ndarray:
    key = path.as_posix()
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    cache[key] = image
    cache.move_to_end(key)
    while len(cache) > max_entries:
        cache.popitem(last=False)
    return image


def cached_pil_image(cache: OrderedDict[str, Image.Image], path: Path, *, max_entries: int = 4) -> Image.Image:
    key = path.as_posix()
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    image = Image.open(path).convert("RGB")
    cache[key] = image
    cache.move_to_end(key)
    while len(cache) > max_entries:
        _old_key, old_image = cache.popitem(last=False)
        old_image.close()
    return image


def sample_image(image: np.ndarray, uv_pixels: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    u = np.clip(np.rint(uv_pixels[:, 0]).astype(np.int64), 0, w - 1)
    v = np.clip(np.rint(uv_pixels[:, 1]).astype(np.int64), 0, h - 1)
    return image[v, u, :]


def project_triangle_pixels(camera: ProjectionCamera, points: np.ndarray) -> list[tuple[float, float]] | None:
    cam_points = points @ camera.rotation.T + camera.translation
    projected = project_camera(camera.model, cam_points)
    if projected is None:
        return None
    u, v = projected
    z = cam_points[:, 2]
    valid = (
        np.isfinite(z)
        & np.isfinite(u)
        & np.isfinite(v)
        & (z > 1e-6)
        & (u >= 0)
        & (u < camera.width)
        & (v >= 0)
        & (v < camera.height)
    )
    if not bool(np.all(valid)):
        return None
    triangle = [(float(u[idx]), float(v[idx])) for idx in range(3)]
    if triangle_area_float(triangle) < 0.5:
        return None
    return triangle


def paste_warped_triangle(
    *,
    texture: Image.Image,
    image: Image.Image,
    dest_polygon: list[tuple[int, int]],
    source_triangle: list[tuple[float, float]],
) -> bool:
    if polygon_area(dest_polygon) < 0.25 or triangle_area_float(source_triangle) < 0.5:
        return False
    min_x = max(min(x for x, _y in dest_polygon), 0)
    max_x = min(max(x for x, _y in dest_polygon), texture.width - 1)
    min_y = max(min(y for _x, y in dest_polygon), 0)
    max_y = min(max(y for _x, y in dest_polygon), texture.height - 1)
    if max_x <= min_x or max_y <= min_y:
        return False
    local_dest = [(x - min_x, y - min_y) for x, y in dest_polygon]
    coeffs = affine_dest_to_source(local_dest, source_triangle)
    if coeffs is None:
        return False
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    patch = image.transform(
        (width, height),
        image_affine_transform_kind(),
        coeffs,
        resample=image_bilinear_resample_kind(),
    )
    tri_mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(tri_mask).polygon(local_dest, fill=255)
    texture.paste(patch, (min_x, min_y), tri_mask)
    return True


def affine_dest_to_source(
    dest_triangle: list[tuple[float, float]],
    source_triangle: list[tuple[float, float]],
) -> tuple[float, float, float, float, float, float] | None:
    matrix = np.asarray(
        [[x, y, 1.0] for x, y in dest_triangle],
        dtype=np.float64,
    )
    src_x = np.asarray([x for x, _y in source_triangle], dtype=np.float64)
    src_y = np.asarray([y for _x, y in source_triangle], dtype=np.float64)
    try:
        ax, bx, cx = np.linalg.solve(matrix, src_x)
        ay, by, cy = np.linalg.solve(matrix, src_y)
    except np.linalg.LinAlgError:
        return None
    coeffs = (float(ax), float(bx), float(cx), float(ay), float(by), float(cy))
    if not all(math.isfinite(value) for value in coeffs):
        return None
    return coeffs


def image_affine_transform_kind() -> Any:
    transform = getattr(Image, "Transform", None)
    if transform is not None and hasattr(transform, "AFFINE"):
        return transform.AFFINE
    return Image.AFFINE


def image_bilinear_resample_kind() -> Any:
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None and hasattr(resampling, "BILINEAR"):
        return resampling.BILINEAR
    return Image.BILINEAR


def uv_polygon_pixels(uv: np.ndarray, atlas_size: int) -> list[tuple[int, int]]:
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)
    x = np.rint(u * (atlas_size - 1)).astype(int)
    y = np.rint((1.0 - v) * (atlas_size - 1)).astype(int)
    return [(int(px), int(py)) for px, py in zip(x, y)]


def polygon_area(points: list[tuple[int, int]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x0, y0) in enumerate(points):
        x1, y1 = points[(idx + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return abs(area) * 0.5


def triangle_area_float(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    (x0, y0), (x1, y1), (x2, y2) = points[:3]
    return abs((x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1)) * 0.5)


def write_textured_obj(*, source_mesh: ObjMesh, output_obj: Path, texture_name: str) -> None:
    material_name = safe_token(output_obj.stem) + "_baked"
    material_file = output_obj.with_suffix(".mtl")
    with material_file.open("w", encoding="utf-8") as dst:
        dst.write("# Generated Real2Sim COLMAP texture bake material\n")
        dst.write(f"newmtl {material_name}\n")
        dst.write("Ka 1.000000 1.000000 1.000000\n")
        dst.write("Kd 1.000000 1.000000 1.000000\n")
        dst.write("Ks 0.030000 0.030000 0.030000\n")
        dst.write("Ns 8.000000\n")
        dst.write("d 1.000000\n")
        dst.write("illum 2\n")
        dst.write(f"map_Kd {texture_name}\n")
    with output_obj.open("w", encoding="utf-8") as dst:
        dst.write("# Real2Sim COLMAP photo-textured scene mesh\n")
        dst.write(f"mtllib {material_file.name}\n")
        dst.write("o real2sim_colmap_textured_scene\n")
        for vertex, color in zip(source_mesh.vertices, source_mesh.colors):
            dst.write(
                f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g} "
                f"{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n"
            )
        for texcoord in source_mesh.texcoords:
            dst.write(f"vt {texcoord[0]:.7g} {texcoord[1]:.7g}\n")
        dst.write(f"usemtl {material_name}\n")
        for face in source_mesh.faces:
            dst.write(
                "f "
                f"{face.vertices[0] + 1}/{face.texcoords[0] + 1} "
                f"{face.vertices[1] + 1}/{face.texcoords[1] + 1} "
                f"{face.vertices[2] + 1}/{face.texcoords[2] + 1}\n"
            )


def qvec_to_rotmat(qvec: tuple[float, float, float, float]) -> list[list[float]]:
    w, x, y, z = qvec
    return [
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ]


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise SystemExit("--background must be R,G,B")
    return tuple(max(0, min(255, part)) for part in parts)  # type: ignore[return-value]


def safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    return token.strip("_") or "scene"


if __name__ == "__main__":
    raise SystemExit(main())
