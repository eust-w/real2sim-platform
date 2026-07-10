#!/usr/bin/env python3
"""SAM 3D Objects adapter for one Real2Sim object.

This follows Meta's public quick-start shape:
    from inference import Inference, load_image, load_single_mask
    output = inference(image, mask)
    output["gs"].save_ply("splat.ply")

The script intentionally fails with a clear message if the gated checkpoints
or SAM 3D environment are not available.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, REPO_ROOT.as_posix())

from real2sim.contracts import Pose, read_json, write_json
from real2sim.image_io import read_gray, write_gray


class PreparedSam3DInputs:
    def __init__(self, image_path: Path, mask_path: Path, mask: np.ndarray, image_size: tuple[int, int], metadata: dict):
        self.image_path = image_path
        self.mask_path = mask_path
        self.mask = mask
        self.image_size = image_size
        self.metadata = metadata


class PlyVertexTable:
    def __init__(
        self,
        *,
        header_lines: list[str],
        body_start: int,
        fmt: str,
        vertex_count: int,
        property_names: list[str],
        property_types: list[str],
        array: np.ndarray,
    ):
        self.header_lines = header_lines
        self.body_start = body_start
        self.fmt = fmt
        self.vertex_count = vertex_count
        self.property_names = property_names
        self.property_types = property_types
        self.array = array


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SAM 3D Objects for one detected object")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, default=None, help="AnySplat cameras JSON used to match image preprocessing")
    parser.add_argument("--match-camera-preprocess", action="store_true", help="Run SAM-3D on the same aspect-preserving center crop used by AnySplat")
    parser.add_argument("--object-centric-preprocess", action="store_true", help="Crop around the instance mask before SAM-3D preprocessing")
    parser.add_argument("--object-preprocess-size", type=int, default=0, help="Override SAM-3D object input size with a square image")
    parser.add_argument("--object-crop-margin", type=float, default=0.12, help="Relative padding around the instance mask for object-centric preprocessing")
    parser.add_argument("--remove-video-overlay", action="store_true", help="Inpaint a central video play overlay before SAM-3D inference")
    parser.add_argument("--canonicalize-gaussian-to-mesh", action="store_true", help="Rotate SAM-3D Gaussian PLY into the mesh canonical frame")
    parser.add_argument("--texture-baking", action="store_true", help="Use SAM-3D's textured GLB postprocess path for the exported preview mesh")
    parser.add_argument("--texture-render-backend", choices=("auto", "inria", "gsplat"), default="gsplat")
    parser.add_argument("--texture-baking-engine", choices=("nvdiffrast", "pytorch3d"), default="pytorch3d")
    parser.add_argument("--sam3d-root", type=Path, default=os.environ.get("REAL2SIM_SAM3D_ROOT"))
    parser.add_argument("--config", type=Path, default=os.environ.get("REAL2SIM_SAM3D_CONFIG"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-fallback-mesh", action="store_true", help="Write a collision proxy if SAM 3D returns no exportable mesh")
    args = parser.parse_args()

    if args.sam3d_root is None:
        raise SystemExit("SAM 3D root is required via --sam3d-root or REAL2SIM_SAM3D_ROOT")
    if args.config is None:
        args.config = args.sam3d_root / "checkpoints" / "hf" / "pipeline.yaml"
    if not args.config.exists():
        raise SystemExit(f"SAM 3D config/checkpoints are not available: {args.config}")

    sys.path.insert(0, (args.sam3d_root / "notebook").as_posix())
    try:
        from inference import Inference, load_image
    except Exception as exc:
        raise SystemExit(f"failed to import SAM 3D inference from {args.sam3d_root}: {exc}") from exc

    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.image
    mask_np = (read_gray(args.mask) > 0.5).astype(np.uint8)
    preprocess_metadata = None
    overlay_metadata = None
    if args.match_camera_preprocess:
        if args.cameras is None:
            raise SystemExit("--match-camera-preprocess requires --cameras")
        target_shape = None
        if args.object_preprocess_size > 0:
            target_shape = (int(args.object_preprocess_size), int(args.object_preprocess_size))
        prepared = prepare_camera_preprocessed_inputs(
            args.image,
            args.mask,
            args.cameras,
            args.output_dir / "camera_preprocess",
            object_centric=args.object_centric_preprocess,
            target_shape=target_shape,
            object_margin_ratio=args.object_crop_margin,
        )
        image_path = prepared.image_path
        mask_np = prepared.mask.astype(np.uint8)
        preprocess_metadata = prepared.metadata
    if args.remove_video_overlay:
        image_path, overlay_metadata = remove_center_video_overlay(image_path, mask_np > 0, args.output_dir / "video_overlay_removal")
    image = load_image(image_path.as_posix())
    inference = Inference(args.config.as_posix(), compile=False)
    with torch.inference_mode():
        output = run_sam3d_inference(
            inference,
            image,
            mask_np,
            seed=args.seed,
            texture_baking=args.texture_baking,
            texture_render_backend=args.texture_render_backend,
            texture_baking_engine=args.texture_baking_engine,
        )

    save_gaussian(output, args.ply)

    wrote_mesh = export_mesh(output, args.mesh, prefer_textured=args.texture_baking)
    if not wrote_mesh and args.allow_fallback_mesh:
        write_bbox_mesh_from_mask(args.mesh, mask_np)
        wrote_mesh = True
    if not wrote_mesh:
        raise SystemExit(
            "SAM 3D output did not include an exportable mesh. "
            "Use --allow-fallback-mesh only for collision-proxy debugging. "
            f"Output keys: {sorted(output.keys())}"
        )
    canonicalization_metadata = None
    if args.canonicalize_gaussian_to_mesh:
        canonicalization_metadata = canonicalize_gaussian_to_mesh_frame(args.ply, args.mesh)

    pose = Pose()
    layout = output.get("layout") or output.get("pose") or output
    if isinstance(layout, dict):
        pose = pose_from_layout(layout)
    write_json(args.pose, pose.to_json())

    metadata = {
        "backend": "sam3d_objects",
        "source_instance": read_json(args.instance),
        "sam3d_config": args.config.as_posix(),
        "texture_baking": bool(args.texture_baking),
        "texture_render_backend": args.texture_render_backend,
        "texture_baking_engine": args.texture_baking_engine,
    }
    if preprocess_metadata is not None:
        metadata["camera_preprocess"] = preprocess_metadata
    if overlay_metadata is not None:
        metadata["video_overlay_removal"] = overlay_metadata
    if canonicalization_metadata is not None:
        metadata["gaussian_canonicalization"] = canonicalization_metadata
    write_json(args.output_dir / "sam3d_metadata.json", metadata)
    return 0


def prepare_camera_preprocessed_inputs(
    image_path: Path,
    mask_path: Path,
    cameras_path: Path,
    output_dir: Path,
    *,
    object_centric: bool = False,
    target_shape: tuple[int, int] | None = None,
    object_margin_ratio: float = 0.12,
) -> PreparedSam3DInputs:
    cameras = read_json(cameras_path)
    target_h, target_w = target_shape or camera_image_shape(cameras)
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    mask = read_gray(mask_path) > 0.5
    if object_centric and mask.any():
        processed_image, processed_mask, crop_meta = object_centric_resize_pad_image_and_mask(
            image,
            mask,
            (target_h, target_w),
            margin_ratio=object_margin_ratio,
            resample=Image.BICUBIC,
        )
    else:
        processed_image, crop_meta = cover_resize_center_crop_image(image, (target_h, target_w), resample=Image.BICUBIC)
        processed_mask = cover_resize_center_crop_mask(mask, (target_h, target_w), crop_meta)
        crop_meta = {"preprocess_mode": "camera_center_crop", **crop_meta}

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_image_path = output_dir / "sam3d_camera_image.png"
    processed_mask_path = output_dir / "sam3d_camera_mask.pgm"
    processed_image.save(processed_image_path)
    write_gray(processed_mask_path, processed_mask.astype(np.float32))

    metadata = {
        "source_image": image_path.as_posix(),
        "source_mask": mask_path.as_posix(),
        "cameras": cameras_path.as_posix(),
        "source_size": [int(image.height), int(image.width)],
        "target_size": [int(target_h), int(target_w)],
        **crop_meta,
        "image": processed_image_path.as_posix(),
        "mask": processed_mask_path.as_posix(),
        "mask_pixels": int(processed_mask.sum()),
    }
    return PreparedSam3DInputs(
        image_path=processed_image_path,
        mask_path=processed_mask_path,
        mask=processed_mask,
        image_size=(target_h, target_w),
        metadata=metadata,
    )


def camera_image_shape(cameras: dict) -> tuple[int, int]:
    shape = cameras.get("image_shape") if isinstance(cameras, dict) else None
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        return (int(shape[0]), int(shape[1]))
    intrinsics = cameras.get("intrinsics", {}) if isinstance(cameras, dict) else {}
    height = int(intrinsics.get("height", 448))
    width = int(intrinsics.get("width", 448))
    return (height, width)


def cover_resize_center_crop_image(image, shape: tuple[int, int], *, resample) -> tuple:
    dst_h, dst_w = shape
    src_w, src_h = image.size
    scale = max(dst_w / float(src_w), dst_h / float(src_h))
    resized_w = max(dst_w, int(src_w * scale))
    resized_h = max(dst_h, int(src_h * scale))
    resized = image.resize((resized_w, resized_h), resample)
    left = max(0, (resized_w - dst_w) // 2)
    top = max(0, (resized_h - dst_h) // 2)
    crop = resized.crop((left, top, left + dst_w, top + dst_h))
    return crop, {
        "resized_size": [int(resized_h), int(resized_w)],
        "crop_xyxy": [int(left), int(top), int(left + dst_w), int(top + dst_h)],
        "scale": float(scale),
    }


def cover_resize_center_crop_mask(mask: np.ndarray, shape: tuple[int, int], crop_meta: dict) -> np.ndarray:
    from PIL import Image

    dst_h, dst_w = shape
    resized_h, resized_w = crop_meta["resized_size"]
    left, top, right, bottom = crop_meta["crop_xyxy"]
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = mask_img.resize((int(resized_w), int(resized_h)), Image.NEAREST)
    cropped = resized.crop((int(left), int(top), int(right), int(bottom)))
    arr = np.asarray(cropped, dtype=np.uint8) > 127
    if arr.shape == (dst_h, dst_w):
        return arr
    out = np.zeros((dst_h, dst_w), dtype=bool)
    out[: arr.shape[0], : arr.shape[1]] = arr
    return out


def object_centric_resize_pad_image_and_mask(image, mask: np.ndarray, shape: tuple[int, int], *, margin_ratio: float, resample) -> tuple:
    from PIL import Image

    dst_h, dst_w = shape
    src_w, src_h = image.size
    x0, y0, x1, y1 = mask_bbox_xyxy(mask)
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    margin = max(1, int(round(max(box_w, box_h) * max(float(margin_ratio), 0.0))))
    crop_x0 = max(0, x0 - margin)
    crop_y0 = max(0, y0 - margin)
    crop_x1 = min(src_w, x1 + margin)
    crop_y1 = min(src_h, y1 + margin)
    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        processed_image, crop_meta = cover_resize_center_crop_image(image, shape, resample=resample)
        processed_mask = cover_resize_center_crop_mask(mask, shape, crop_meta)
        return processed_image, processed_mask, {"preprocess_mode": "camera_center_crop_empty_object", **crop_meta}

    crop = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
    crop_mask = mask[crop_y0:crop_y1, crop_x0:crop_x1]
    crop_w, crop_h = crop.size
    scale = min(dst_w / float(crop_w), dst_h / float(crop_h))
    resized_w = max(1, min(dst_w, int(round(crop_w * scale))))
    resized_h = max(1, min(dst_h, int(round(crop_h * scale))))
    offset_x = (dst_w - resized_w) // 2
    offset_y = (dst_h - resized_h) // 2

    fill = local_background_color(crop, crop_mask)
    canvas = Image.new("RGB", (dst_w, dst_h), fill)
    resized = crop.resize((resized_w, resized_h), resample)
    canvas.paste(resized, (offset_x, offset_y))

    mask_canvas = np.zeros((dst_h, dst_w), dtype=bool)
    resized_mask = Image.fromarray(crop_mask.astype(np.uint8) * 255).resize((resized_w, resized_h), Image.NEAREST)
    mask_arr = np.asarray(resized_mask, dtype=np.uint8) > 127
    mask_canvas[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = mask_arr
    return canvas, mask_canvas, {
        "preprocess_mode": "object_centric",
        "object_bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "source_crop_xyxy": [int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)],
        "resized_size": [int(resized_h), int(resized_w)],
        "pad_xy": [int(offset_x), int(offset_y)],
        "scale": float(scale),
        "object_margin_ratio": float(margin_ratio),
    }


def mask_bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return (0, 0, int(mask.shape[1]), int(mask.shape[0]))
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def local_background_color(image, mask: np.ndarray) -> tuple[int, int, int]:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bg = arr[~np.asarray(mask, dtype=bool)]
    values = bg if bg.size else arr.reshape(-1, 3)
    color = np.median(values.reshape(-1, 3), axis=0)
    return tuple(int(np.clip(round(float(channel)), 0, 255)) for channel in color)


def remove_center_video_overlay(image_path: Path, object_mask: np.ndarray, output_dir: Path) -> tuple[Path, dict]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    mask = np.asarray(object_mask, dtype=bool)
    if mask.shape != arr.shape[:2]:
        mask = resize_bool_mask(mask, arr.shape[:2])

    overlay_mask = detect_center_video_overlay(arr, mask)
    overlay_pixels = int(overlay_mask.sum())
    metadata = {
        "source_image": image_path.as_posix(),
        "detected": overlay_pixels > 0,
        "overlay_pixels": overlay_pixels,
    }
    if overlay_pixels == 0:
        return image_path, metadata

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sam3d_overlay_removed.png"
    try:
        import cv2

        inpaint_mask = (overlay_mask.astype(np.uint8) * 255)
        cleaned = cv2.inpaint(arr, inpaint_mask, 5, cv2.INPAINT_TELEA)
    except Exception:
        cleaned = simple_mask_fill(arr, overlay_mask)
        metadata["fallback"] = "simple_mask_fill"
    Image.fromarray(cleaned.astype(np.uint8)).save(output_path)
    metadata["image"] = output_path.as_posix()
    return output_path, metadata


def detect_center_video_overlay(image: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    cx = width * 0.5
    cy = height * 0.5
    radius = max(8.0, min(height, width) * 0.11)
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    rgb = image.astype(np.float32) / 255.0
    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 1e-6)
    bright_neutral = (max_channel > 0.72) & (saturation < 0.28)
    raw = disk & object_mask & bright_neutral
    if int(raw.sum()) < max(20, int(radius)):
        return np.zeros((height, width), dtype=bool)
    return dilate_bool_mask(raw, radius=3)


def resize_bool_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    height, width = shape
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize((width, height), Image.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 127


def dilate_bool_mask(mask: np.ndarray, *, radius: int) -> np.ndarray:
    try:
        import cv2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
    except Exception:
        padded = np.pad(mask, radius, mode="constant", constant_values=False)
        out = np.zeros_like(mask, dtype=bool)
        for dy in range(radius * 2 + 1):
            for dx in range(radius * 2 + 1):
                if (dx - radius) ** 2 + (dy - radius) ** 2 <= radius ** 2:
                    out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        return out


def simple_mask_fill(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cleaned = image.copy()
    if not mask.any():
        return cleaned
    yy, xx = np.where(mask)
    y0, y1 = max(0, int(yy.min()) - 5), min(image.shape[0], int(yy.max()) + 6)
    x0, x1 = max(0, int(xx.min()) - 5), min(image.shape[1], int(xx.max()) + 6)
    patch = image[y0:y1, x0:x1]
    patch_mask = mask[y0:y1, x0:x1]
    fill_values = patch[~patch_mask]
    fill = np.median(fill_values, axis=0) if fill_values.size else np.median(image.reshape(-1, 3), axis=0)
    cleaned[mask] = fill.astype(np.uint8)
    return cleaned


PLY_NUMPY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def canonicalize_gaussian_to_mesh_frame(gaussian_ply: Path, mesh_obj: Path) -> dict:
    table = load_ply_vertex_table(gaussian_ply)
    mesh_vertices = load_obj_vertices(mesh_obj)
    if table.array.size == 0 or mesh_vertices.size == 0:
        return {
            "enabled": True,
            "status": "skipped_empty_geometry",
            "gaussian": gaussian_ply.as_posix(),
            "mesh": mesh_obj.as_posix(),
        }

    xyz_indices = indices_for(table.property_names, ("x", "y", "z"))
    points = table.array[:, xyz_indices]
    gaussian_extent = extent(points)
    mesh_extent = extent(mesh_vertices)
    rotation, mapping, signs = canonical_axis_rotation(gaussian_extent, mesh_extent)
    changed = not np.allclose(rotation, np.eye(3), atol=1e-6)
    if changed:
        apply_axis_rotation_to_ply_table(table, rotation)
        write_ply_vertex_table(gaussian_ply, table)

    return {
        "enabled": True,
        "status": "ok",
        "changed": bool(changed),
        "gaussian": gaussian_ply.as_posix(),
        "mesh": mesh_obj.as_posix(),
        "gaussian_extent_before": vector_json(gaussian_extent),
        "mesh_extent": vector_json(mesh_extent),
        "axis_mapping": {
            "source": "extent_permutation",
            "gaussian_axes_to_mesh_axes": [int(value) for value in mapping],
            "mesh_axis_signs": [int(value) for value in signs],
        },
        "rotation_matrix": [[round(float(value), 7) for value in row] for row in rotation.tolist()],
    }


def canonical_axis_rotation(gaussian_extent: np.ndarray, mesh_extent: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    gaussian_order = list(np.argsort(-np.asarray(gaussian_extent, dtype=np.float32)))
    mesh_order = list(np.argsort(-np.asarray(mesh_extent, dtype=np.float32)))
    mapping = [0, 1, 2]
    for mesh_axis, gaussian_axis in zip(mesh_order, gaussian_order):
        mapping[int(mesh_axis)] = int(gaussian_axis)
    signs = [1, 1, 1]
    rotation = rotation_from_axis_mapping(mapping, signs)
    if np.linalg.det(rotation) < 0.0:
        signs[int(mesh_order[-1])] = -1
        rotation = rotation_from_axis_mapping(mapping, signs)
    return rotation.astype(np.float32), mapping, signs


def rotation_from_axis_mapping(mapping: list[int], signs: list[int]) -> np.ndarray:
    rotation = np.zeros((3, 3), dtype=np.float32)
    for mesh_axis, gaussian_axis in enumerate(mapping):
        rotation[mesh_axis, gaussian_axis] = float(signs[mesh_axis])
    return rotation


def apply_axis_rotation_to_ply_table(table: PlyVertexTable, rotation: np.ndarray) -> None:
    names = table.property_names
    xyz_indices = indices_for(names, ("x", "y", "z"))
    table.array[:, xyz_indices] = table.array[:, xyz_indices] @ rotation.T
    if all(name in names for name in ("nx", "ny", "nz")):
        normal_indices = indices_for(names, ("nx", "ny", "nz"))
        table.array[:, normal_indices] = table.array[:, normal_indices] @ rotation.T
    if all(name in names for name in ("rot_0", "rot_1", "rot_2", "rot_3")):
        rot_indices = indices_for(names, ("rot_0", "rot_1", "rot_2", "rot_3"))
        transform_quat = quat_wxyz_from_rotation_matrix(rotation)
        for idx in range(table.array.shape[0]):
            table.array[idx, rot_indices] = quat_multiply(transform_quat, normalize_quat(table.array[idx, rot_indices]))


def load_ply_vertex_table(path: Path) -> PlyVertexTable:
    data = path.read_bytes()
    header_marker = b"end_header"
    marker_index = data.find(header_marker)
    if marker_index < 0:
        raise SystemExit(f"PLY missing end_header: {path}")
    body_start = data.find(b"\n", marker_index)
    if body_start < 0:
        raise SystemExit(f"PLY header missing final newline: {path}")
    body_start += 1
    header_lines = data[:body_start].decode("ascii", errors="ignore").splitlines()
    fmt = "ascii"
    vertex_count = 0
    property_names: list[str] = []
    property_types: list[str] = []
    current_element = ""
    for line in header_lines:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0] == "format":
            fmt = parts[1]
        elif len(parts) >= 3 and parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif len(parts) >= 3 and parts[0] == "property" and current_element == "vertex":
            if parts[1] == "list":
                raise SystemExit(f"PLY vertex list properties are not supported for canonicalization: {path}")
            property_types.append(parts[1])
            property_names.append(parts[-1])
    if vertex_count <= 0:
        return PlyVertexTable(
            header_lines=header_lines,
            body_start=body_start,
            fmt=fmt,
            vertex_count=0,
            property_names=property_names,
            property_types=property_types,
            array=np.zeros((0, len(property_names)), dtype=np.float32),
        )
    if not all(name in property_names for name in ("x", "y", "z")):
        raise SystemExit(f"PLY must expose x/y/z vertex properties: {path}")
    body = data[body_start:]
    if fmt == "ascii":
        array = load_ascii_ply_vertex_array(body, vertex_count, len(property_names))
    elif fmt in {"binary_little_endian", "binary_big_endian"}:
        array = load_binary_ply_vertex_array(body, vertex_count, property_types, little_endian=fmt == "binary_little_endian")
    else:
        raise SystemExit(f"unsupported PLY format for canonicalization: {fmt}")
    return PlyVertexTable(
        header_lines=header_lines,
        body_start=body_start,
        fmt=fmt,
        vertex_count=vertex_count,
        property_names=property_names,
        property_types=property_types,
        array=array.astype(np.float32, copy=False),
    )


def load_ascii_ply_vertex_array(body: bytes, vertex_count: int, property_count: int) -> np.ndarray:
    rows = []
    for line in body.decode("ascii", errors="ignore").splitlines():
        if len(rows) >= vertex_count:
            break
        stripped = line.strip()
        if not stripped:
            continue
        values = [float(value) for value in stripped.split()[:property_count]]
        if len(values) == property_count:
            rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def load_binary_ply_vertex_array(body: bytes, vertex_count: int, property_types: list[str], *, little_endian: bool) -> np.ndarray:
    byte_order = "<" if little_endian else ">"
    try:
        dtype = np.dtype([(f"f{idx}", byte_order + PLY_NUMPY_DTYPES[kind]) for idx, kind in enumerate(property_types)])
    except KeyError as exc:
        raise SystemExit(f"unsupported PLY scalar type for canonicalization: {exc}") from exc
    structured = np.frombuffer(body, dtype=dtype, count=vertex_count)
    columns = [structured[f"f{idx}"].astype(np.float32) for idx in range(len(property_types))]
    return np.stack(columns, axis=1) if columns else np.zeros((vertex_count, 0), dtype=np.float32)


def write_ply_vertex_table(path: Path, table: PlyVertexTable) -> None:
    header = "\n".join(table.header_lines) + "\n"
    if table.fmt == "ascii":
        lines = [" ".join(f"{float(value):.9g}" for value in row) for row in table.array]
        path.write_text(header + "\n".join(lines) + "\n", encoding="ascii")
        return
    if table.fmt in {"binary_little_endian", "binary_big_endian"}:
        byte_order = "<" if table.fmt == "binary_little_endian" else ">"
        try:
            dtype = np.dtype([(f"f{idx}", byte_order + PLY_NUMPY_DTYPES[kind]) for idx, kind in enumerate(table.property_types)])
        except KeyError as exc:
            raise SystemExit(f"unsupported PLY scalar type for canonicalization: {exc}") from exc
        structured = np.empty(table.array.shape[0], dtype=dtype)
        for idx in range(table.array.shape[1]):
            structured[f"f{idx}"] = table.array[:, idx].astype(dtype[idx])
        path.write_bytes(header.encode("ascii") + structured.tobytes())
        return
    raise SystemExit(f"unsupported PLY format for canonicalization: {table.fmt}")


def load_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return np.asarray(vertices, dtype=np.float32)


def indices_for(names: list[str], required: tuple[str, ...]) -> list[int]:
    return [names.index(name) for name in required]


def extent(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(points.max(axis=0) - points.min(axis=0), dtype=np.float32)


def normalize_quat(raw: np.ndarray) -> np.ndarray:
    quat = np.asarray(raw, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = normalize_quat(a)
    bw, bx, by, bz = normalize_quat(b)
    return normalize_quat(
        np.asarray(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=np.float32,
        )
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
    return normalize_quat(quat.astype(np.float32))


def vector_json(values: np.ndarray) -> list[float]:
    return [round(float(value), 7) for value in np.asarray(values, dtype=np.float32).reshape(-1).tolist()]


def pose_from_layout(layout: dict) -> Pose:
    position = layout.get("position", layout.get("translation", [0.0, 0.0, 0.0]))
    scale = layout.get("scale", [1.0, 1.0, 1.0])
    quat = layout.get("quat_wxyz", layout.get("quaternion_wxyz", layout.get("rotation", [1.0, 0.0, 0.0, 0.0])))
    if "quaternion_xyzw" in layout and "quat_wxyz" not in layout and "quaternion_wxyz" not in layout and "rotation" not in layout:
        qx, qy, qz, qw = vector_from_value(layout["quaternion_xyzw"], 4, "quaternion_xyzw")
        quat = (qw, qx, qy, qz)
    return Pose(
        position=vector_from_value(position, 3, "position"),
        quat_wxyz=vector_from_value(quat, 4, "quat_wxyz"),
        scale=vector_from_value(scale, 3, "scale", broadcast_scalar=True),
    )


def run_sam3d_inference(
    inference,
    image: np.ndarray,
    mask_np: np.ndarray,
    *,
    seed: int,
    texture_baking: bool,
    texture_render_backend: str = "gsplat",
    texture_baking_engine: str = "pytorch3d",
) -> dict:
    if not texture_baking:
        return inference(image, mask_np, seed=seed)

    patch_texture_baking_render_backend(texture_render_backend)
    pipeline = getattr(inference, "_pipeline", None)
    merge_mask_to_rgba = getattr(inference, "merge_mask_to_rgba", None)
    if pipeline is None or not hasattr(pipeline, "run") or merge_mask_to_rgba is None:
        raise SystemExit("--texture-baking requires SAM-3D notebook Inference._pipeline.run support")
    if hasattr(pipeline, "rendering_engine"):
        pipeline.rendering_engine = texture_baking_engine

    rgba = merge_mask_to_rgba(image, mask_np)
    options = {
        "stage1_only": False,
        "with_mesh_postprocess": False,
        "with_texture_baking": True,
        "with_layout_postprocess": False,
        "use_vertex_color": False,
        "stage1_inference_steps": None,
        "pointmap": None,
    }
    return pipeline.run(rgba, None, seed, **supported_kwargs(pipeline.run, options))


def patch_texture_baking_render_backend(backend: str) -> bool:
    if backend in {"", "auto", "inria"}:
        return False
    if backend != "gsplat":
        raise SystemExit(f"unsupported SAM-3D texture render backend: {backend}")

    render_utils = importlib.import_module("sam3d_objects.model.backbone.tdfy_dit.utils.render_utils")
    postprocessing_utils = importlib.import_module("sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils")

    def render_multiview_with_backend(sample, resolution=512, nviews=30):
        radius = 2
        fov = 40
        cameras = [render_utils.sphere_hammersley_sequence(index, nviews) for index in range(nviews)]
        yaws = [camera[0] for camera in cameras]
        pitches = [camera[1] for camera in cameras]
        extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitches, radius, fov)
        result = render_utils.render_frames(
            sample,
            extrinsics,
            intrinsics,
            {"resolution": resolution, "bg_color": (0, 0, 0), "backend": "gsplat"},
        )
        return result["color"], extrinsics, intrinsics

    postprocessing_utils.render_multiview = render_multiview_with_backend
    return True


def supported_kwargs(callable_obj, options: dict[str, object]) -> dict[str, object]:
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return options
    return {key: value for key, value in options.items() if key in signature.parameters}


def save_gaussian(output: dict, path: Path) -> None:
    for key in ("gs", "gaussian", "gaussians"):
        if key not in output:
            continue
        value = first_value(output[key])
        if hasattr(value, "save_ply"):
            path.parent.mkdir(parents=True, exist_ok=True)
            value.save_ply(path.as_posix())
            return
    raise SystemExit(f"SAM 3D output does not expose a Gaussian save_ply method; keys={sorted(output.keys())}")


def export_mesh(output: dict, path: Path, *, prefer_textured: bool = False) -> bool:
    keys = ("glb", "mesh", "trimesh", "obj") if prefer_textured else ("mesh", "trimesh", "glb", "obj")
    for key in keys:
        if key not in output:
            continue
        value = first_value(output[key])
        if hasattr(value, "export"):
            path.parent.mkdir(parents=True, exist_ok=True)
            value.export(path.as_posix())
            return True
        if isinstance(value, (str, os.PathLike)):
            source = Path(value)
            if source.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.read_bytes())
                return True
    return False


def first_value(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else value
    return value


def vector_from_value(value, size: int, field_name: str, *, broadcast_scalar: bool = False) -> tuple[float, ...]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float64).squeeze()
    if arr.ndim == 0:
        if broadcast_scalar:
            return tuple(round(float(arr), 7) for _ in range(size))
        if size == 1:
            return (round(float(arr), 7),)
    flat = arr.reshape(-1)
    if flat.size != size:
        raise SystemExit(f"{field_name} must have {size} values, got shape {arr.shape}")
    return tuple(round(float(x), 7) for x in flat)


def write_bbox_mesh_from_mask(path: Path, mask) -> None:
    import numpy as np

    ys, xs = np.where(mask)
    if ys.size == 0:
        extent = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    else:
        extent = np.array([max((xs.max() - xs.min()) / 512.0, 0.05), 0.08, max((ys.max() - ys.min()) / 512.0, 0.05)], dtype=np.float32)
    half = extent / 2.0
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
        f.write("o sam3d_collision_fallback\n")
        for v in vertices:
            f.write(f"v {v[0]:.7g} {v[1]:.7g} {v[2]:.7g}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


if __name__ == "__main__":
    raise SystemExit(main())
