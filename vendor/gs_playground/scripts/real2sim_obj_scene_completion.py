#!/usr/bin/env python3
"""Add coarse room/tabletop completion planes to a reconstructed OBJ scene.

The COLMAP/Delaunay mesh path often produces visually useful fragments but
leaves large holes in walls, tabletops, or floors. This utility keeps the
original OBJ intact and appends a conservative, quantile-bounded planar shell
so MotrixSim has a continuous scene scaffold for inspection and collision.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Bounds:
    low: tuple[float, float, float]
    high: tuple[float, float, float]


@dataclass(frozen=True)
class Plane:
    name: str
    axis: int
    value: float
    u_axis: int
    v_axis: int
    u_min: float
    u_max: float
    v_min: float
    v_max: float
    material: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Append quantile-bounded scene completion planes to an OBJ")
    parser.add_argument("--input-obj", type=Path, required=True)
    parser.add_argument("--output-obj", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--low-quantile", type=float, default=0.05)
    parser.add_argument("--high-quantile", type=float, default=0.95)
    parser.add_argument("--margin-ratio", type=float, default=0.03)
    parser.add_argument("--up-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--grid", type=int, default=24, help="Subdivisions per plane axis")
    parser.add_argument(
        "--planes",
        default="floor,ceiling,left,right,back,front",
        help="Comma-separated planes to append: floor,ceiling,left,right,back,front",
    )
    args = parser.parse_args(argv)

    vertices, texcoord_count = read_obj_counts(args.input_obj)
    if len(vertices) < 3:
        raise SystemExit(f"OBJ has too few vertices for scene completion: {args.input_obj}")
    low_q = clamp(args.low_quantile, 0.0, 0.49)
    high_q = clamp(args.high_quantile, 0.51, 1.0)
    if low_q >= high_q:
        raise SystemExit("--low-quantile must be below --high-quantile")
    bounds = quantile_bounds(vertices, low_q, high_q, margin_ratio=max(args.margin_ratio, 0.0))
    up_axis = "xyz".index(args.up_axis)
    plane_names = tuple(name.strip().lower() for name in args.planes.split(",") if name.strip())
    planes = build_planes(bounds, up_axis=up_axis, names=plane_names)
    if not planes:
        raise SystemExit("No completion planes selected")

    args.output_obj.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = args.output_obj.with_name(f"{args.output_obj.stem}_completion.mtl")
    append_completion(
        args.input_obj,
        args.output_obj,
        mtl_path,
        vertices_start=len(vertices),
        texcoords_start=texcoord_count,
        planes=planes,
        grid=max(args.grid, 1),
    )
    report = {
        "format": "gs-playground-real2sim-obj-scene-completion-report-v1",
        "input_obj": args.input_obj.as_posix(),
        "output_obj": args.output_obj.as_posix(),
        "material_library": mtl_path.name,
        "input_vertex_count": len(vertices),
        "input_texcoord_count": texcoord_count,
        "bounds_low": list(bounds.low),
        "bounds_high": list(bounds.high),
        "low_quantile": low_q,
        "high_quantile": high_q,
        "margin_ratio": max(args.margin_ratio, 0.0),
        "up_axis": args.up_axis,
        "grid": max(args.grid, 1),
        "plane_count": len(planes),
        "planes": [plane.name for plane in planes],
        "added_vertex_count": len(planes) * (max(args.grid, 1) + 1) ** 2,
        "added_texcoord_count": len(planes) * (max(args.grid, 1) + 1) ** 2,
        "added_face_count": len(planes) * max(args.grid, 1) * max(args.grid, 1) * 2,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def read_obj_counts(path: Path) -> tuple[list[tuple[float, float, float]], int]:
    vertices: list[tuple[float, float, float]] = []
    texcoord_count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("vt "):
                texcoord_count += 1
                continue
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                continue
    return vertices, texcoord_count


def quantile_bounds(
    vertices: Iterable[tuple[float, float, float]],
    low_quantile: float,
    high_quantile: float,
    *,
    margin_ratio: float,
) -> Bounds:
    columns = list(zip(*vertices))
    lows: list[float] = []
    highs: list[float] = []
    for values in columns:
        ordered = sorted(values)
        low = percentile(ordered, low_quantile)
        high = percentile(ordered, high_quantile)
        span = max(high - low, 1e-6)
        lows.append(low - span * margin_ratio)
        highs.append(high + span * margin_ratio)
    return Bounds(tuple(lows), tuple(highs))


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def build_planes(bounds: Bounds, *, up_axis: int, names: tuple[str, ...]) -> list[Plane]:
    axes = {0, 1, 2}
    horizontal = [axis for axis in axes if axis != up_axis]
    a0, a1 = horizontal
    low, high = bounds.low, bounds.high
    specs = {
        "floor": Plane("floor", up_axis, low[up_axis], a0, a1, low[a0], high[a0], low[a1], high[a1], "real2sim_completion_floor"),
        "ceiling": Plane("ceiling", up_axis, high[up_axis], a0, a1, low[a0], high[a0], low[a1], high[a1], "real2sim_completion_ceiling"),
        "left": Plane("left", a0, low[a0], a1, up_axis, low[a1], high[a1], low[up_axis], high[up_axis], "real2sim_completion_wall"),
        "right": Plane("right", a0, high[a0], a1, up_axis, low[a1], high[a1], low[up_axis], high[up_axis], "real2sim_completion_wall"),
        "back": Plane("back", a1, low[a1], a0, up_axis, low[a0], high[a0], low[up_axis], high[up_axis], "real2sim_completion_wall"),
        "front": Plane("front", a1, high[a1], a0, up_axis, low[a0], high[a0], low[up_axis], high[up_axis], "real2sim_completion_wall"),
    }
    return [specs[name] for name in names if name in specs]


def append_completion(
    input_obj: Path,
    output_obj: Path,
    mtl_path: Path,
    *,
    vertices_start: int,
    texcoords_start: int,
    planes: list[Plane],
    grid: int,
) -> None:
    source = input_obj.read_text(encoding="utf-8", errors="ignore")
    output = [source]
    if not source.endswith("\n"):
        output.append("\n")
    output.append(f"mtllib {mtl_path.name}\n")
    output.append("o real2sim_scene_completion\n")
    vertex_index = vertices_start
    for plane in planes:
        output.append(f"g real2sim_completion_{plane.name}\n")
        output.append(f"usemtl {plane.material}\n")
        vertex_start = vertex_index + 1
        texcoord_start = texcoords_start + 1
        for iv in range(grid + 1):
            v_fraction = iv / grid
            v = lerp(plane.v_min, plane.v_max, v_fraction)
            for iu in range(grid + 1):
                u_fraction = iu / grid
                u = lerp(plane.u_min, plane.u_max, u_fraction)
                coords = [0.0, 0.0, 0.0]
                coords[plane.axis] = plane.value
                coords[plane.u_axis] = u
                coords[plane.v_axis] = v
                output.append(f"v {coords[0]:.8f} {coords[1]:.8f} {coords[2]:.8f}\n")
                output.append(f"vt {u_fraction:.8f} {v_fraction:.8f}\n")
        for iv in range(grid):
            for iu in range(grid):
                a = vertex_start + iv * (grid + 1) + iu
                b = a + 1
                c = a + (grid + 1)
                d = c + 1
                ta = texcoord_start + iv * (grid + 1) + iu
                tb = ta + 1
                tc = ta + (grid + 1)
                td = tc + 1
                output.append(f"f {a}/{ta} {b}/{tb} {d}/{td}\n")
                output.append(f"f {a}/{ta} {d}/{td} {c}/{tc}\n")
        vertex_index += (grid + 1) * (grid + 1)
        texcoords_start += (grid + 1) * (grid + 1)
    output_obj.write_text("".join(output), encoding="utf-8")
    write_completion_mtl(mtl_path)


def write_completion_mtl(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Generated Real2Sim scene completion materials",
                "newmtl real2sim_completion_floor",
                "Kd 0.420000 0.410000 0.390000",
                "Ka 0.120000 0.120000 0.120000",
                "Ks 0.020000 0.020000 0.020000",
                "Ns 8.000000",
                "",
                "newmtl real2sim_completion_wall",
                "Kd 0.680000 0.650000 0.590000",
                "Ka 0.150000 0.145000 0.135000",
                "Ks 0.015000 0.015000 0.015000",
                "Ns 6.000000",
                "",
                "newmtl real2sim_completion_ceiling",
                "Kd 0.720000 0.700000 0.660000",
                "Ka 0.160000 0.155000 0.145000",
                "Ks 0.010000 0.010000 0.010000",
                "Ns 4.000000",
                "",
            ]
        ),
        encoding="utf-8",
    )


def lerp(a: float, b: float, t: float) -> float:
    return a * (1.0 - t) + b * t


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


if __name__ == "__main__":
    raise SystemExit(main())
