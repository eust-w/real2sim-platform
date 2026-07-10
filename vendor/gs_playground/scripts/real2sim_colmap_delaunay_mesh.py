#!/usr/bin/env python3
"""Run COLMAP Delaunay meshing from an existing dense workspace."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


class ColmapCommandError(RuntimeError):
    def __init__(self, command: list[str], returncode: int) -> None:
        super().__init__(f"COLMAP command failed with exit code {returncode}: {' '.join(command[:3])}")
        self.command = command
        self.returncode = returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run COLMAP Delaunay meshing from an existing dense workspace")
    parser.add_argument("--dense-workspace", type=Path, required=True, help="COLMAP dense workspace containing images/stereo/sparse")
    parser.add_argument("--output-ply", type=Path, required=True, help="Output Delaunay mesh PLY")
    parser.add_argument("--report", type=Path, required=True, help="Output Delaunay report JSON")
    parser.add_argument("--colmap-bin", default="colmap")
    parser.add_argument("--fused-ply", type=Path, help="Existing fused point cloud; defaults to dense-workspace/fused.ply")
    parser.add_argument("--ensure-vis", action="store_true", help="Run stereo_fusion into the workspace when fused.ply.vis is missing")
    parser.add_argument("--geom-consistency", action="store_true", default=True)
    parser.add_argument("--no-geom-consistency", action="store_false", dest="geom_consistency")
    parser.add_argument("--stereo-fusion-min-num-pixels", type=int, default=5)
    parser.add_argument("--stereo-fusion-max-reproj-error", type=float, default=2.0)
    parser.add_argument("--stereo-fusion-max-depth-error", type=float, default=0.01)
    parser.add_argument("--max-proj-dist", type=float, default=12.0)
    parser.add_argument("--max-depth-dist", type=float, default=0.035)
    parser.add_argument("--visibility-sigma", type=float, default=3.0)
    parser.add_argument("--quality-regularization", type=float, default=1.0)
    parser.add_argument("--max-side-length-factor", type=float, default=20.0)
    parser.add_argument("--max-side-length-percentile", type=float, default=95.0)
    parser.add_argument("--min-mesh-faces", type=int, default=100000)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    report = run_delaunay_meshing(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.require_pass and not report.get("passed", False):
        return 1
    return 0


def run_delaunay_meshing(args: argparse.Namespace) -> dict[str, Any]:
    colmap = resolve_colmap(args.colmap_bin)
    dense_workspace = args.dense_workspace.resolve()
    output_ply = args.output_ply.resolve()
    fused_ply = (args.fused_ply or dense_workspace / "fused.ply").resolve()
    workspace_fused = dense_workspace / "fused.ply"
    workspace_vis = dense_workspace / "fused.ply.vis"
    commands: list[dict[str, Any]] = []
    warnings: list[str] = []

    if not dense_workspace.is_dir():
        warnings.append(f"dense workspace is missing: {dense_workspace}")
    else:
        output_ply.parent.mkdir(parents=True, exist_ok=True)
        try:
            prepare_workspace_fused(
                colmap=colmap,
                dense_workspace=dense_workspace,
                fused_ply=fused_ply,
                workspace_fused=workspace_fused,
                workspace_vis=workspace_vis,
                args=args,
                commands=commands,
            )
            run_logged(build_delaunay_command(colmap, dense_workspace, output_ply, args), commands)
        except ColmapCommandError as exc:
            warnings.append(str(exc))

    mesh_counts = ply_counts(output_ply) if output_ply.is_file() else {}
    mesh_face_count = int(mesh_counts.get("face_count") or 0)
    passed = mesh_face_count >= int(args.min_mesh_faces)
    if mesh_face_count and not passed:
        warnings.append(f"Delaunay mesh face count {mesh_face_count} < {int(args.min_mesh_faces)}")
    if not mesh_face_count and not any("COLMAP command failed" in warning for warning in warnings):
        warnings.append("COLMAP Delaunay meshing produced no mesh")

    return {
        "format": "gs-playground-real2sim-colmap-delaunay-report-v1",
        "dense_workspace": dense_workspace.as_posix(),
        "fused_ply": workspace_fused.as_posix() if workspace_fused.is_file() else None,
        "fused_visibility": workspace_vis.as_posix() if workspace_vis.is_file() else None,
        "output_ply": output_ply.as_posix() if output_ply.is_file() else None,
        "mesh_vertex_count": int(mesh_counts.get("vertex_count") or 0),
        "mesh_face_count": mesh_face_count,
        "min_mesh_faces": int(args.min_mesh_faces),
        "ensure_vis": bool(args.ensure_vis),
        "geom_consistency": bool(args.geom_consistency),
        "delaunay_options": {
            "max_proj_dist": float(args.max_proj_dist),
            "max_depth_dist": float(args.max_depth_dist),
            "visibility_sigma": float(args.visibility_sigma),
            "quality_regularization": float(args.quality_regularization),
            "max_side_length_factor": float(args.max_side_length_factor),
            "max_side_length_percentile": float(args.max_side_length_percentile),
        },
        "passed": bool(passed),
        "warnings": warnings,
        "commands": commands,
    }


def prepare_workspace_fused(
    *,
    colmap: str,
    dense_workspace: Path,
    fused_ply: Path,
    workspace_fused: Path,
    workspace_vis: Path,
    args: argparse.Namespace,
    commands: list[dict[str, Any]],
) -> None:
    if workspace_fused.resolve() != fused_ply.resolve() and fused_ply.is_file():
        workspace_fused.unlink(missing_ok=True)
        try:
            workspace_fused.symlink_to(fused_ply)
        except OSError:
            shutil.copy2(fused_ply, workspace_fused)
    if workspace_fused.is_file() and workspace_vis.is_file():
        return
    if not args.ensure_vis:
        raise ColmapCommandError(["missing", workspace_vis.as_posix()], 2)
    workspace_fused.unlink(missing_ok=True)
    run_logged(build_stereo_fusion_command(colmap, dense_workspace, workspace_fused, args), commands)


def build_stereo_fusion_command(
    colmap: str,
    dense_workspace: Path,
    workspace_fused: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        colmap,
        "stereo_fusion",
        "--workspace_path",
        dense_workspace.as_posix(),
        "--workspace_format",
        "COLMAP",
        "--input_type",
        "geometric" if args.geom_consistency else "photometric",
        "--output_type",
        "PLY",
        "--output_path",
        workspace_fused.as_posix(),
    ]
    append_option_if_supported(colmap, "stereo_fusion", command, ["--StereoFusion.min_num_pixels"], str(max(int(args.stereo_fusion_min_num_pixels), 1)))
    append_option_if_supported(colmap, "stereo_fusion", command, ["--StereoFusion.max_reproj_error"], str(max(float(args.stereo_fusion_max_reproj_error), 0.1)))
    append_option_if_supported(colmap, "stereo_fusion", command, ["--StereoFusion.max_depth_error"], str(max(float(args.stereo_fusion_max_depth_error), 0.0001)))
    return command


def build_delaunay_command(colmap: str, dense_workspace: Path, output_ply: Path, args: argparse.Namespace) -> list[str]:
    command = [
        colmap,
        "delaunay_mesher",
        "--input_path",
        dense_workspace.as_posix(),
        "--input_type",
        "dense",
        "--output_path",
        output_ply.as_posix(),
    ]
    append_option_if_supported(colmap, "delaunay_mesher", command, ["--DelaunayMeshing.max_proj_dist"], str(max(float(args.max_proj_dist), 0.1)))
    append_option_if_supported(colmap, "delaunay_mesher", command, ["--DelaunayMeshing.max_depth_dist"], str(max(float(args.max_depth_dist), 0.0001)))
    append_option_if_supported(colmap, "delaunay_mesher", command, ["--DelaunayMeshing.visibility_sigma"], str(max(float(args.visibility_sigma), 0.1)))
    append_option_if_supported(colmap, "delaunay_mesher", command, ["--DelaunayMeshing.quality_regularization"], str(max(float(args.quality_regularization), 0.0)))
    append_option_if_supported(colmap, "delaunay_mesher", command, ["--DelaunayMeshing.max_side_length_factor"], str(max(float(args.max_side_length_factor), 1.0)))
    append_option_if_supported(colmap, "delaunay_mesher", command, ["--DelaunayMeshing.max_side_length_percentile"], str(min(max(float(args.max_side_length_percentile), 1.0), 100.0)))
    return command


def run_logged(command: list[str], commands: list[dict[str, Any]], *, required: bool = True) -> None:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    commands.append(
        {
            "argv": list(command),
            "returncode": completed.returncode,
            "output_tail": "\n".join(completed.stdout.splitlines()[-100:]),
        }
    )
    if completed.returncode != 0 and required:
        sys.stderr.write(completed.stdout)
        raise ColmapCommandError(command, completed.returncode)


def resolve_colmap(colmap_bin: str) -> str:
    path = shutil.which(colmap_bin)
    if path is not None:
        return path
    candidate = Path(colmap_bin)
    if candidate.is_file():
        return candidate.resolve().as_posix()
    raise SystemExit(f"COLMAP binary not found: {colmap_bin}")


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


def colmap_option(colmap: str, command: str, names: list[str]) -> str | None:
    completed = subprocess.run([colmap, command, "-h"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    help_text = completed.stdout
    for name in names:
        if name in help_text:
            return name
    return None


def ply_counts(path: Path) -> dict[str, int]:
    counts = {"vertex_count": 0, "face_count": 0}
    if not path.is_file():
        return counts
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                counts["vertex_count"] = int(line.split()[-1])
            elif line.startswith("element face "):
                counts["face_count"] = int(line.split()[-1])
            elif line == "end_header":
                break
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
