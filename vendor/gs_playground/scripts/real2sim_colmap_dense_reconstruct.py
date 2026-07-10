#!/usr/bin/env python3
"""Run COLMAP dense MVS from a passing Real2Sim SfM report."""

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
    parser = argparse.ArgumentParser(description="Run COLMAP dense MVS from a Real2Sim SfM report")
    parser.add_argument("--sfm-report", type=Path, required=True, help="SfM report produced by real2sim_colmap_capture_report.py")
    parser.add_argument("--output-dir", type=Path, required=True, help="Dense artifact output directory")
    parser.add_argument("--report", type=Path, required=True, help="Output dense reconstruction report JSON")
    parser.add_argument("--colmap-bin", default="colmap")
    parser.add_argument("--max-image-size", type=int, default=960)
    parser.add_argument("--min-fused-points", type=int, default=100000)
    parser.add_argument("--geom-consistency", action="store_true", default=True)
    parser.add_argument("--no-geom-consistency", action="store_false", dest="geom_consistency")
    parser.add_argument("--gpu-index", default="")
    parser.add_argument("--patch-match-window-radius", type=int, default=5)
    parser.add_argument("--patch-match-window-step", type=int, default=1)
    parser.add_argument("--patch-match-num-samples", type=int, default=15)
    parser.add_argument("--stereo-fusion-min-num-pixels", type=int, default=5)
    parser.add_argument("--stereo-fusion-max-reproj-error", type=float, default=2.0)
    parser.add_argument("--stereo-fusion-max-depth-error", type=float, default=0.01)
    parser.add_argument("--run-poisson", action="store_true")
    parser.add_argument("--poisson-depth", type=int, default=10)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    report = run_dense_reconstruction(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.require_pass and not report.get("passed", False):
        return 1
    return 0


def run_dense_reconstruction(args: argparse.Namespace) -> dict[str, Any]:
    sfm = json.loads(args.sfm_report.read_text(encoding="utf-8"))
    colmap = resolve_colmap(args.colmap_bin)
    output_dir = args.output_dir.resolve()
    dense_workspace = output_dir / "workspace"
    fused_ply = output_dir / "fused.ply"
    mesh_ply = output_dir / "mesh_poisson.ply"
    commands: list[dict[str, Any]] = []
    warnings: list[str] = []

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = Path(str(sfm.get("frames_dir", ""))).resolve()
    model_dir = Path(str(sfm.get("model_dir", ""))).resolve()
    if not sfm.get("passed", False):
        warnings.append("source SfM report did not pass capture gates")
    if not frames_dir.is_dir():
        warnings.append(f"SfM frames_dir is missing: {frames_dir}")
    if not model_dir.is_dir():
        warnings.append(f"SfM model_dir is missing: {model_dir}")

    if not warnings:
        try:
            run_logged(build_image_undistorter_command(colmap, frames_dir, model_dir, dense_workspace, args), commands)
            run_logged(build_patch_match_command(colmap, dense_workspace, args), commands)
            run_logged(build_stereo_fusion_command(colmap, dense_workspace, fused_ply, args), commands)
            if args.run_poisson and fused_ply.is_file():
                run_logged(build_poisson_command(colmap, fused_ply, mesh_ply, args), commands, required=False)
        except ColmapCommandError as exc:
            warnings.append(str(exc))

    fused_counts = ply_counts(fused_ply) if fused_ply.is_file() else {}
    mesh_counts = ply_counts(mesh_ply) if mesh_ply.is_file() else {}
    fused_point_count = int(fused_counts.get("vertex_count") or 0)
    passed = fused_point_count >= int(args.min_fused_points)
    if fused_point_count and not passed:
        warnings.append(f"fused point count {fused_point_count} < {int(args.min_fused_points)}")
    if not fused_point_count and not any("COLMAP command failed" in warning for warning in warnings):
        warnings.append("COLMAP dense reconstruction produced no fused point cloud")

    return {
        "format": "gs-playground-real2sim-colmap-dense-report-v1",
        "sfm_report": args.sfm_report.resolve().as_posix(),
        "frames_dir": frames_dir.as_posix(),
        "model_dir": model_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "dense_workspace": dense_workspace.as_posix(),
        "fused_ply": fused_ply.as_posix() if fused_ply.is_file() else None,
        "mesh_ply": mesh_ply.as_posix() if mesh_ply.is_file() else None,
        "max_image_size": int(args.max_image_size),
        "geom_consistency": bool(args.geom_consistency),
        "gpu_index": str(args.gpu_index),
        "fused_point_count": fused_point_count,
        "mesh_vertex_count": int(mesh_counts.get("vertex_count") or 0),
        "mesh_face_count": int(mesh_counts.get("face_count") or 0),
        "min_fused_points": int(args.min_fused_points),
        "passed": bool(passed),
        "warnings": warnings,
        "commands": commands,
    }


def build_image_undistorter_command(
    colmap: str,
    frames_dir: Path,
    model_dir: Path,
    dense_workspace: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        colmap,
        "image_undistorter",
        "--image_path",
        frames_dir.as_posix(),
        "--input_path",
        model_dir.as_posix(),
        "--output_path",
        dense_workspace.as_posix(),
        "--output_type",
        "COLMAP",
    ]
    if int(args.max_image_size) > 0:
        append_option_if_supported(
            colmap,
            "image_undistorter",
            command,
            ["--max_image_size", "--ImageUndistorter.max_image_size"],
            str(int(args.max_image_size)),
        )
    return command


def build_patch_match_command(colmap: str, dense_workspace: Path, args: argparse.Namespace) -> list[str]:
    command = [
        colmap,
        "patch_match_stereo",
        "--workspace_path",
        dense_workspace.as_posix(),
        "--workspace_format",
        "COLMAP",
    ]
    append_option_if_supported(
        colmap,
        "patch_match_stereo",
        command,
        ["--PatchMatchStereo.geom_consistency"],
        "true" if args.geom_consistency else "false",
    )
    if str(args.gpu_index).strip():
        append_option_if_supported(
            colmap,
            "patch_match_stereo",
            command,
            ["--PatchMatchStereo.gpu_index"],
            str(args.gpu_index).strip(),
        )
    append_option_if_supported(
        colmap,
        "patch_match_stereo",
        command,
        ["--PatchMatchStereo.window_radius"],
        str(max(int(args.patch_match_window_radius), 1)),
    )
    append_option_if_supported(
        colmap,
        "patch_match_stereo",
        command,
        ["--PatchMatchStereo.window_step"],
        str(max(int(args.patch_match_window_step), 1)),
    )
    append_option_if_supported(
        colmap,
        "patch_match_stereo",
        command,
        ["--PatchMatchStereo.num_samples"],
        str(max(int(args.patch_match_num_samples), 1)),
    )
    return command


def build_stereo_fusion_command(
    colmap: str,
    dense_workspace: Path,
    fused_ply: Path,
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
        "--output_path",
        fused_ply.as_posix(),
    ]
    append_option_if_supported(
        colmap,
        "stereo_fusion",
        command,
        ["--StereoFusion.min_num_pixels"],
        str(max(int(args.stereo_fusion_min_num_pixels), 1)),
    )
    append_option_if_supported(
        colmap,
        "stereo_fusion",
        command,
        ["--StereoFusion.max_reproj_error"],
        str(max(float(args.stereo_fusion_max_reproj_error), 0.1)),
    )
    append_option_if_supported(
        colmap,
        "stereo_fusion",
        command,
        ["--StereoFusion.max_depth_error"],
        str(max(float(args.stereo_fusion_max_depth_error), 0.0001)),
    )
    return command


def build_poisson_command(colmap: str, fused_ply: Path, mesh_ply: Path, args: argparse.Namespace) -> list[str]:
    command = [
        colmap,
        "poisson_mesher",
        "--input_path",
        fused_ply.as_posix(),
        "--output_path",
        mesh_ply.as_posix(),
    ]
    append_option_if_supported(
        colmap,
        "poisson_mesher",
        command,
        ["--PoissonMeshing.depth"],
        str(max(int(args.poisson_depth), 5)),
    )
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


def colmap_option(colmap: str, command: str, names: list[str]) -> str | None:
    completed = subprocess.run([colmap, command, "-h"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    help_text = completed.stdout
    for name in names:
        if name in help_text:
            return name
    return None


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
