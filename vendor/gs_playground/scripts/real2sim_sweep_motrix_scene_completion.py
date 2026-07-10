#!/usr/bin/env python3
"""Sweep MotrixSim scene-completion parameters for a Real2Sim package.

The sweep keeps the reconstructed 3DGS and textured scene mesh fixed, varies
the photo-completion backdrop margin, packages each candidate, runs the MuJoCo
offscreen verifier, and ranks candidates by the verifier's render-quality
score. It is intended for same-capture candidate comparison, not as an absolute
commercial-quality metric.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep Real2Sim Motrix scene-completion backdrop margins")
    parser.add_argument("--source", type=Path, required=True, help="Original capture directory or source path")
    parser.add_argument("--scene-ply", type=Path, required=True)
    parser.add_argument("--scene-mesh", type=Path, required=True)
    parser.add_argument("--cameras-json", type=Path, required=True)
    parser.add_argument("--preview-image", type=Path, required=True)
    parser.add_argument("--mesh-cleanup-report", type=Path, required=True)
    parser.add_argument("--texture-bake-report", type=Path, required=True)
    parser.add_argument("--completion-report", type=Path, required=True)
    parser.add_argument("--completion-texture", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name-prefix", default="real2sim_scene_completion_sweep")
    parser.add_argument("--margins", default="0,0.15,0.25,0.35,0.55", help="Comma-separated backdrop margins to test")
    parser.add_argument("--scene-faces", type=int, default=30_000, help="Default scene mesh LOD face budget")
    parser.add_argument("--scene-face-options", default="", help="Comma-separated scene mesh LOD face budgets to sweep")
    parser.add_argument("--min-scene-vertices", type=int, default=100_000)
    parser.add_argument("--robot", default="go2", choices=["go1", "go2", "g1"])
    parser.add_argument("--prompt", default="photogrammetry textured scene with swept photo completion backdrop")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--archive", action="store_true", help="Write a .tar.gz for each candidate package")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse package and verifier output directories when present")
    args = parser.parse_args()

    margins = parse_margins(args.margins)
    face_options = parse_int_options(args.scene_face_options) if args.scene_face_options else [int(args.scene_faces)]
    args.output_root.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for margin in margins:
        for scene_faces in face_options:
            candidates.append(run_candidate(args, margin, scene_faces))

    ranked = rank_candidates(candidates)
    report = {
        "format": "gs-playground-real2sim-scene-completion-sweep-v1",
        "target_quality": "polycam_realitycapture_like_space",
        "margins": margins,
        "scene_face_options": face_options,
        "best": ranked[0] if ranked else None,
        "candidates": ranked,
        "notes": [
            "Scores are comparable only within the same capture, cameras, render size, and verifier version.",
            "A high score does not prove Polycam/RealityCapture-level fidelity; inspect contact sheets before accepting.",
        ],
    }
    report_path = args.output_root / "sweep_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def parse_margins(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0:
            raise SystemExit(f"margin must be non-negative: {value}")
        values.append(value)
    if not values:
        raise SystemExit("at least one margin is required")
    return values


def parse_int_options(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise SystemExit(f"scene face option must be positive: {value}")
        values.append(value)
    if not values:
        raise SystemExit("at least one scene face option is required")
    return values


def run_candidate(args: argparse.Namespace, margin: float, scene_faces: int) -> dict[str, Any]:
    token = candidate_token(margin, scene_faces)
    package_dir = args.output_root / f"{args.name_prefix}_{token}_package"
    verify_dir = args.output_root / f"{args.name_prefix}_{token}_verify"
    report_path = verify_dir / "report.json"

    if not (args.skip_existing and report_path.is_file()):
        if package_dir.exists():
            shutil.rmtree(package_dir)
        if verify_dir.exists():
            shutil.rmtree(verify_dir)
        verify_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update(
            {
                "REAL2SIM_SCENE_COMPLETION_REPORT": args.completion_report.as_posix(),
                "REAL2SIM_SCENE_COMPLETION_TEXTURE": args.completion_texture.as_posix(),
                "REAL2SIM_SCENE_COMPLETION_BACKDROP_MARGIN": str(margin),
                "REAL2SIM_MOTRIX_SCENE_VISUAL": "photogrammetry",
                "REAL2SIM_MOTRIX_PREVIEW_MESH_FACES": str(max(int(scene_faces), 1)),
            }
        )
        package_cmd = [
            args.python,
            "-m",
            "real2sim",
            "package-existing",
            "--source",
            args.source.as_posix(),
            "--scene-ply",
            args.scene_ply.as_posix(),
            "--scene-mesh",
            args.scene_mesh.as_posix(),
            "--cameras-json",
            args.cameras_json.as_posix(),
            "--preview-image",
            args.preview_image.as_posix(),
            "--mesh-cleanup-report",
            args.mesh_cleanup_report.as_posix(),
            "--texture-bake-report",
            args.texture_bake_report.as_posix(),
            "--output",
            package_dir.as_posix(),
            "--prompt",
            args.prompt,
            "--robot",
            args.robot,
        ]
        run(package_cmd, env=env, cwd=REPO_ROOT)

        verify_cmd = [
            args.python,
            (REPO_ROOT / "scripts" / "real2sim_verify_motrix_package.py").as_posix(),
            "--package",
            package_dir.as_posix(),
            "--min-scene-vertices",
            str(int(args.min_scene_vertices)),
            "--mujoco-offscreen-render-dir",
            verify_dir.as_posix(),
            "--visual-report-dir",
            verify_dir.as_posix(),
        ]
        with report_path.open("w", encoding="utf-8") as handle:
            run(verify_cmd, env={"MUJOCO_GL": "egl", **env}, cwd=REPO_ROOT, stdout=handle)

        if args.archive:
            archive_path = package_dir.with_suffix(".tar.gz")
            run(["tar", "-C", package_dir.parent.as_posix(), "-czf", archive_path.as_posix(), package_dir.name], cwd=REPO_ROOT)

    verification = json.loads(report_path.read_text(encoding="utf-8"))
    return candidate_summary(margin, scene_faces, package_dir, verify_dir, verification)


def run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    stdout: Any | None = None,
) -> None:
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    subprocess.run(cmd, cwd=cwd, env=process_env, stdout=stdout, check=True)


def candidate_summary(
    margin: float,
    scene_faces: int,
    package_dir: Path,
    verify_dir: Path,
    verification: dict[str, Any],
) -> dict[str, Any]:
    renders = verification.get("mujoco_offscreen_render", {}).get("renders", [])
    scores = [
        float(render.get("quality", {}).get("score", 0.0))
        for render in renders
        if isinstance(render, dict)
    ]
    overview = next((render for render in renders if render.get("camera") == "overview"), {}) if isinstance(renders, list) else {}
    closeup = next((render for render in renders if render.get("camera") == "closeup"), {}) if isinstance(renders, list) else {}
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "margin": float(margin),
        "scene_faces": int(scene_faces),
        "score": float(avg_score),
        "package": package_dir.as_posix(),
        "verify_dir": verify_dir.as_posix(),
        "contact_sheet": (verify_dir / "visual_acceptance_contact_sheet.png").as_posix(),
        "overview": render_quality_summary(overview),
        "closeup": render_quality_summary(closeup),
    }


def render_quality_summary(render: dict[str, Any]) -> dict[str, Any]:
    quality = render.get("quality") if isinstance(render, dict) else {}
    return {
        "path": render.get("path") if isinstance(render, dict) else None,
        "score": float(quality.get("score", 0.0)) if isinstance(quality, dict) else 0.0,
        "black_ratio": float(quality.get("black_ratio", 0.0)) if isinstance(quality, dict) else 0.0,
        "high_saturation_ratio": float(quality.get("high_saturation_ratio", 0.0)) if isinstance(quality, dict) else 0.0,
    }


def rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (
            float(item.get("score", 0.0)),
            int(item.get("scene_faces", 0)),
            -float(item.get("margin", 0.0)),
        ),
        reverse=True,
    )


def margin_token(margin: float) -> str:
    return f"margin_{margin:.3f}".replace(".", "p").replace("-", "m")


def candidate_token(margin: float, scene_faces: int) -> str:
    return f"{margin_token(margin)}_faces_{int(scene_faces)}"


if __name__ == "__main__":
    raise SystemExit(main())
