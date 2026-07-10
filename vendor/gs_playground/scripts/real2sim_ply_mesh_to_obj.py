#!/usr/bin/env python3
"""Convert a colored PLY mesh to a Motrix-friendly OBJ.

The converter keeps vertex colors for tools that understand them and also emits
an OBJ material palette so viewers that ignore OBJ vertex colors still show the
scene with explicit diffuse materials.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


SCALAR_TYPES: dict[str, tuple[str, int]] = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}

COLOR_NAMES = (
    ("red", "green", "blue"),
    ("diffuse_red", "diffuse_green", "diffuse_blue"),
    ("r", "g", "b"),
)


@dataclass(frozen=True)
class PlyHeader:
    ply_format: str
    vertex_count: int
    face_count: int
    vertex_properties: tuple[tuple[str, str], ...]
    face_count_type: str
    face_index_type: str
    data_offset: int


@dataclass(frozen=True)
class ObjVertex:
    x: float
    y: float
    z: float
    r: float
    g: float
    b: float


@dataclass(frozen=True)
class ObjTriangle:
    a: int
    b: int
    c: int


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a colored PLY mesh to a Motrix-compatible OBJ")
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-obj", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--no-material-palette", action="store_true", help="Write vertex-color-only OBJ without a sidecar MTL")
    parser.add_argument("--material-palette-levels", type=int, default=18, help="RGB quantization levels for generated diffuse materials")
    args = parser.parse_args()

    report = convert_ply_mesh_to_obj(
        args.input_ply,
        args.output_obj,
        material_palette=not args.no_material_palette,
        material_palette_levels=max(int(args.material_palette_levels), 2),
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def convert_ply_mesh_to_obj(
    input_ply: Path,
    output_obj: Path,
    *,
    material_palette: bool = True,
    material_palette_levels: int = 18,
) -> dict[str, Any]:
    header = read_ply_header(input_ply)
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    with input_ply.open("rb") as src:
        src.seek(header.data_offset)
        if header.ply_format == "ascii":
            vertices, triangles = read_ascii_mesh(src, header)
        elif header.ply_format == "binary_little_endian":
            vertices, triangles = read_binary_mesh(src, header, endian="<")
        elif header.ply_format == "binary_big_endian":
            vertices, triangles = read_binary_mesh(src, header, endian=">")
        else:
            raise SystemExit(f"unsupported PLY format: {header.ply_format}")
    material_count = write_obj_mesh(
        input_ply=input_ply,
        output_obj=output_obj,
        vertices=vertices,
        triangles=triangles,
        material_palette=material_palette,
        material_palette_levels=max(int(material_palette_levels), 2),
    )
    return {
        "format": "gs-playground-real2sim-ply-mesh-to-obj-report-v1",
        "input_ply": input_ply.resolve().as_posix(),
        "output_obj": output_obj.resolve().as_posix(),
        "vertex_count": int(header.vertex_count),
        "face_count": int(len(triangles)),
        "texcoord_count": int(len(vertices)),
        "material_palette": bool(material_palette),
        "material_palette_levels": int(max(int(material_palette_levels), 2)),
        "material_count": int(material_count),
        "material_library": output_obj.with_suffix(".mtl").resolve().as_posix() if material_palette else None,
    }


def read_ply_header(path: Path) -> PlyHeader:
    ply_format = "ascii"
    vertex_count = 0
    face_count = 0
    vertex_properties: list[tuple[str, str]] = []
    face_count_type = "uchar"
    face_index_type = "int"
    active_element: str | None = None
    with path.open("rb") as handle:
        first = handle.readline().decode("ascii", errors="ignore").strip()
        if first != "ply":
            raise SystemExit(f"not a PLY file: {path}")
        while True:
            raw = handle.readline()
            if not raw:
                raise SystemExit(f"PLY header missing end_header: {path}")
            line = raw.decode("ascii", errors="ignore").strip()
            if line == "end_header":
                data_offset = handle.tell()
                break
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format" and len(parts) >= 2:
                ply_format = parts[1]
            elif parts[0] == "element" and len(parts) >= 3:
                active_element = parts[1]
                if active_element == "vertex":
                    vertex_count = int(parts[2])
                elif active_element == "face":
                    face_count = int(parts[2])
            elif parts[0] == "property" and active_element == "vertex" and len(parts) == 3:
                vertex_properties.append((parts[2], parts[1]))
            elif parts[0] == "property" and active_element == "face" and len(parts) >= 5 and parts[1] == "list":
                face_count_type = parts[2]
                face_index_type = parts[3]
    if vertex_count <= 0:
        raise SystemExit(f"PLY has no vertices: {path}")
    return PlyHeader(
        ply_format=ply_format,
        vertex_count=vertex_count,
        face_count=face_count,
        vertex_properties=tuple(vertex_properties),
        face_count_type=face_count_type,
        face_index_type=face_index_type,
        data_offset=data_offset,
    )


def read_ascii_mesh(src: BinaryIO, header: PlyHeader) -> tuple[list[ObjVertex], list[ObjTriangle]]:
    property_names = [name for name, _kind in header.vertex_properties]
    index = {name: idx for idx, name in enumerate(property_names)}
    color_indices = find_color_indices(index)
    vertices: list[ObjVertex] = []
    for _ in range(header.vertex_count):
        parts = src.readline().decode("utf-8", errors="ignore").split()
        if len(parts) < len(property_names):
            raise SystemExit("truncated ASCII PLY vertex table")
        vertices.append(make_obj_vertex(parts, index, color_indices))
    triangles: list[ObjTriangle] = []
    for _ in range(header.face_count):
        parts = src.readline().decode("utf-8", errors="ignore").split()
        if not parts:
            continue
        count = int(parts[0])
        indices = [int(value) + 1 for value in parts[1 : 1 + count]]
        triangles.extend(triangulate(indices))
    return vertices, triangles


def read_binary_mesh(src: BinaryIO, header: PlyHeader, *, endian: str) -> tuple[list[ObjVertex], list[ObjTriangle]]:
    property_names = [name for name, _kind in header.vertex_properties]
    index = {name: idx for idx, name in enumerate(property_names)}
    color_indices = find_color_indices(index)
    vertex_struct = struct.Struct(endian + "".join(scalar_format(kind) for _name, kind in header.vertex_properties))
    vertices: list[ObjVertex] = []
    for _ in range(header.vertex_count):
        data = src.read(vertex_struct.size)
        if len(data) != vertex_struct.size:
            raise SystemExit("truncated binary PLY vertex table")
        values = vertex_struct.unpack(data)
        vertices.append(make_obj_vertex(values, index, color_indices))
    count_struct = struct.Struct(endian + scalar_format(header.face_count_type))
    index_struct = struct.Struct(endian + scalar_format(header.face_index_type))
    triangles: list[ObjTriangle] = []
    for _ in range(header.face_count):
        raw_count = src.read(count_struct.size)
        if len(raw_count) != count_struct.size:
            raise SystemExit("truncated binary PLY face table")
        count = int(count_struct.unpack(raw_count)[0])
        raw_indices = src.read(index_struct.size * count)
        if len(raw_indices) != index_struct.size * count:
            raise SystemExit("truncated binary PLY face indices")
        indices = [
            int(index_struct.unpack_from(raw_indices, offset)[0]) + 1
            for offset in range(0, len(raw_indices), index_struct.size)
        ]
        triangles.extend(triangulate(indices))
    return vertices, triangles


def scalar_format(kind: str) -> str:
    if kind not in SCALAR_TYPES:
        raise SystemExit(f"unsupported PLY scalar type: {kind}")
    return SCALAR_TYPES[kind][0]


def find_color_indices(index: dict[str, int]) -> tuple[int, int, int] | None:
    for names in COLOR_NAMES:
        if all(name in index for name in names):
            return tuple(index[name] for name in names)
    return None


def make_obj_vertex(values: Any, index: dict[str, int], color_indices: tuple[int, int, int] | None) -> ObjVertex:
    x = float(values[index["x"]])
    y = float(values[index["y"]])
    z = float(values[index["z"]])
    if color_indices is None:
        r, g, b = 0.72, 0.72, 0.72
    else:
        channels = [float(values[idx]) for idx in color_indices]
        if max(channels) > 1.0:
            channels = [value / 255.0 for value in channels]
        r, g, b = [min(max(value, 0.0), 1.0) for value in channels]
    return ObjVertex(x=x, y=y, z=z, r=r, g=g, b=b)


def triangulate(indices: list[int]) -> list[ObjTriangle]:
    if len(indices) < 3:
        return []
    triangles: list[ObjTriangle] = []
    for idx in range(1, len(indices) - 1):
        triangles.append(ObjTriangle(indices[0], indices[idx], indices[idx + 1]))
    return triangles


def write_obj_mesh(
    *,
    input_ply: Path,
    output_obj: Path,
    vertices: list[ObjVertex],
    triangles: list[ObjTriangle],
    material_palette: bool,
    material_palette_levels: int,
) -> int:
    material_path = output_obj.with_suffix(".mtl")
    uv_axes = choose_uv_axes(vertices)
    uv_bounds = axis_bounds(vertices, uv_axes)
    materials: dict[str, tuple[float, float, float]] = {}
    with output_obj.open("w", encoding="utf-8") as dst:
        dst.write(f"# Converted from {input_ply.name} for Real2Sim Motrix scene mesh\n")
        if material_palette:
            dst.write(f"mtllib {material_path.name}\n")
        dst.write("o real2sim_colmap_dense_scene\n")
        for vertex in vertices:
            dst.write(
                f"v {vertex.x:.9g} {vertex.y:.9g} {vertex.z:.9g} "
                f"{vertex.r:.6f} {vertex.g:.6f} {vertex.b:.6f}\n"
            )
        for vertex in vertices:
            u, v = vertex_uv(vertex, uv_axes, uv_bounds)
            dst.write(f"vt {u:.7g} {v:.7g}\n")
        current_material = None
        for triangle in triangles:
            if material_palette:
                material_name, color = triangle_material(vertices, triangle, output_obj.stem, material_palette_levels)
                materials.setdefault(material_name, color)
                if material_name != current_material:
                    dst.write(f"usemtl {material_name}\n")
                    current_material = material_name
            dst.write(f"f {triangle.a}/{triangle.a} {triangle.b}/{triangle.b} {triangle.c}/{triangle.c}\n")
    if material_palette:
        write_material_library(material_path, materials)
    elif material_path.exists():
        material_path.unlink()
    return len(materials)


def choose_uv_axes(vertices: list[ObjVertex]) -> tuple[str, str]:
    if not vertices:
        return ("x", "y")
    extents = {}
    for axis in ("x", "y", "z"):
        values = [getattr(vertex, axis) for vertex in vertices]
        extents[axis] = max(values) - min(values)
    ranked = sorted(extents, key=lambda axis: extents[axis], reverse=True)
    return ranked[0], ranked[1]


def axis_bounds(vertices: list[ObjVertex], axes: tuple[str, str]) -> tuple[tuple[float, float], tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for axis in axes:
        values = [getattr(vertex, axis) for vertex in vertices]
        low = min(values) if values else 0.0
        high = max(values) if values else 1.0
        if math.isclose(high, low):
            high = low + 1.0
        bounds.append((low, high))
    return bounds[0], bounds[1]


def vertex_uv(vertex: ObjVertex, axes: tuple[str, str], bounds: tuple[tuple[float, float], tuple[float, float]]) -> tuple[float, float]:
    first, second = axes
    (u_low, u_high), (v_low, v_high) = bounds
    u = (getattr(vertex, first) - u_low) / max(u_high - u_low, 1e-9)
    v = (getattr(vertex, second) - v_low) / max(v_high - v_low, 1e-9)
    return min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0)


def triangle_material(
    vertices: list[ObjVertex],
    triangle: ObjTriangle,
    stem: str,
    levels: int,
) -> tuple[str, tuple[float, float, float]]:
    tri_vertices = [vertices[index - 1] for index in (triangle.a, triangle.b, triangle.c)]
    color = tuple(sum(getattr(vertex, channel) for vertex in tri_vertices) / 3.0 for channel in ("r", "g", "b"))
    quantized = tuple(round(channel * (levels - 1)) / (levels - 1) for channel in color)
    suffix = "_".join(f"{int(round(channel * 255)):02x}" for channel in quantized)
    return f"{safe_token(stem)}_mat_{suffix}", quantized


def safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    return token.strip("_") or "scene"


def write_material_library(path: Path, materials: dict[str, tuple[float, float, float]]) -> None:
    with path.open("w", encoding="utf-8") as dst:
        dst.write("# Generated Real2Sim dense scene material palette\n")
        for name, (r, g, b) in sorted(materials.items()):
            dst.write(f"newmtl {name}\n")
            dst.write(f"Ka {r:.6f} {g:.6f} {b:.6f}\n")
            dst.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
            dst.write("Ks 0.030000 0.030000 0.030000\n")
            dst.write("Ns 8.000000\n")
            dst.write("d 1.000000\n")
            dst.write("illum 2\n\n")


if __name__ == "__main__":
    raise SystemExit(main())
