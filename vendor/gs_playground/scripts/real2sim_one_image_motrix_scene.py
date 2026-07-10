#!/usr/bin/env python3
"""One-image Real2Sim scene reconstruction workflow for MotrixSim.

This script is intentionally orchestration-only: it does not store API keys and
does not replace the project CLI. It chains the existing image-edit multiview
adapter, scene generation CLI, package verification, preview projection, and
archive creation into one reproducible command.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PIL_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from real2sim.camera import load_anysplat_source_camera


DEFAULT_PROMPT = (
    "Generate a reconstruction-grade, geometry-consistent 8-view image-to-image contact sheet for Real2Sim and MotrixSim import. "
    "Use the input photo as exact source truth. Preserve all visible object identities, relative spacing, material colors, reflections, "
    "shadows, the supporting surface, static background, camera scale, and lighting. Output exactly one 4 columns x 2 rows contact sheet, "
    "8 equal portrait panels, thin white gutters, no labels, no captions, no extra text, no watermarks. Views in order: original front, "
    "front-left 45 degrees, left 90 degrees, rear-left 135 degrees, rear/back 180 degrees, rear-right 225 degrees, right 270 degrees, "
    "top oblique 35 degrees. Do not add people, hands, robots, extra objects, UI, a different room, or a product render background. "
    "Prefer sharp contours and reconstruction-useful texture over artistic beautification."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and verify a MotrixSim-importable scene package from one image")
    parser.add_argument("--image", type=Path, required=True, help="Source RGB image")
    parser.add_argument("--output", type=Path, required=True, help="Output Real2Sim scene package directory")
    parser.add_argument("--work-dir", type=Path, required=True, help="Intermediate workflow directory")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Image-to-image multiview prompt")
    parser.add_argument("--scene-prompt", default="", help="Scene reconstruction prompt; defaults to --prompt")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/real2sim.external.example.json")
    parser.add_argument("--backend", default="builtin", choices=["builtin", "external", "hybrid", "auto", "anysplat-direct"])
    parser.add_argument("--api-key-env", default="REAL2SIM_IMAGE_EDIT_API_KEY")
    parser.add_argument("--api-url", default=os.environ.get("REAL2SIM_IMAGE_EDIT_API_URL", "https://chatapi.applyma.com/v1/images/edits"))
    parser.add_argument("--model", default=os.environ.get("REAL2SIM_IMAGE_EDIT_MODEL", "gpt-image-2"))
    parser.add_argument("--size", default=os.environ.get("REAL2SIM_IMAGE_EDIT_SIZE", "1536x1024"))
    parser.add_argument("--quality", default=os.environ.get("REAL2SIM_IMAGE_EDIT_QUALITY", "auto"))
    parser.add_argument(
        "--candidates",
        type=int,
        default=int(os.environ.get("REAL2SIM_IMAGE_EDIT_CANDIDATES", "1")),
        help="Generate and score multiple image-edit multiview candidates, then reconstruct the best one",
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=float(os.environ.get("REAL2SIM_MIN_MULTIVIEW_QUALITY", "0.42")),
        help="Fail before reconstruction when synthetic multiview quality is below this score",
    )
    parser.add_argument(
        "--allow-low-quality",
        action="store_true",
        help="Continue even when the best synthetic multiview candidate is below --min-quality-score",
    )
    parser.add_argument("--frames-dir", type=Path, default=None, help="Reuse existing multiview frames instead of calling the image API")
    parser.add_argument("--no-image-edit", action="store_true", help="Require --frames-dir and skip the image-edit API call")
    parser.add_argument("--no-prune", action="store_true", help="Pass --no-prune to scene generation")
    parser.add_argument("--anysplat-python", default=os.environ.get("REAL2SIM_ANY_SPLAT_PYTHON", ""), help="Python executable for AnySplat direct mode")
    parser.add_argument("--anysplat-root", type=Path, default=os.environ.get("REAL2SIM_ANYSPLAT_ROOT"), help="AnySplat repository root for direct mode")
    parser.add_argument("--anysplat-max-views", type=int, default=int(os.environ.get("REAL2SIM_ANYSPLAT_MAX_VIEWS", "8")))
    parser.add_argument("--anysplat-mesh-grid", type=int, default=int(os.environ.get("REAL2SIM_ANYSPLAT_MESH_GRID", "48")))
    parser.add_argument("--source-gs-preview", type=Path, default=None, help="Optional source-view 3DGS preview PNG to include in the Motrix package")
    parser.add_argument(
        "--min-source-preview-psnr",
        type=float,
        default=float(os.environ.get("REAL2SIM_MIN_SOURCE_PREVIEW_PSNR", "18.0")),
        help="Fail AnySplat direct mode when source-view render PSNR is below this threshold",
    )
    parser.add_argument(
        "--max-source-preview-mae",
        type=float,
        default=float(os.environ.get("REAL2SIM_MAX_SOURCE_PREVIEW_MAE", "0.10")),
        help="Fail AnySplat direct mode when source-view render MAE is above this threshold",
    )
    parser.add_argument(
        "--min-scene-vertices",
        type=int,
        default=int(os.environ.get("REAL2SIM_MIN_SCENE_VERTICES", "100000")),
        help="Fail final package verification when scene.ply has fewer vertices",
    )
    parser.add_argument(
        "--skip-final-package-verify",
        action="store_true",
        help="Skip final Motrix package archive verification",
    )
    parser.add_argument("--archive", type=Path, default=None, help="Optional .tar.gz path; defaults to <output>.tar.gz")
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"source image not found: {args.image}")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = args.frames_dir
    multiview_dir = args.work_dir / "synthetic_multiview"
    quality_report = args.work_dir / "synthetic_multiview_quality.json"
    if args.no_image_edit:
        if frames_dir is None:
            raise SystemExit("--no-image-edit requires --frames-dir")
    elif frames_dir is None:
        frames_dir = run_best_image_edit_candidate(args, multiview_dir, quality_report)

    if frames_dir is None or not frames_dir.is_dir():
        raise SystemExit(f"frames directory not found: {frames_dir}")
    if not quality_report.exists():
        write_multiview_quality_report(args.image, frames_dir, quality_report)

    anysplat_metadata = None
    if args.backend == "anysplat-direct":
        anysplat_metadata = run_anysplat_direct_generation(args, frames_dir)
    else:
        run_scene_generation(args, frames_dir)
    report = verify_motrix_package(args.output)
    preview = write_projection_preview(args.output / "3dgs" / "scene.ply", args.output / "pointcloud_projection_preview.png")
    if args.backend == "anysplat-direct" and (args.output / "motrix_import.json").is_file():
        import_files = {
            "manifest": args.output / "motrix_import.json",
            "readme": args.output / "MOTRIX_IMPORT.md",
        }
        copy_quality_report(args.output, quality_report)
        if anysplat_metadata is not None:
            attach_anysplat_metadata_to_package(args.output, anysplat_metadata)
    else:
        import_files = write_motrix_import_files(args.output, report, preview, quality_report)
    archive = create_archive(args.output, args.archive or args.output.with_suffix(".tar.gz"))
    final_verification = None
    if not args.skip_final_package_verify and args.backend == "anysplat-direct":
        final_verification = run_final_package_verification(args, archive)
        archive = create_archive(args.output, archive)

    print(
        json.dumps(
            {
                "package": report,
                "preview": preview.as_posix(),
                "motrix_import": import_files["manifest"].as_posix(),
                "motrix_readme": import_files["readme"].as_posix(),
                "archive": archive.as_posix(),
                "anysplat_metadata": (args.output / "artifacts" / "anysplat_metadata.json").as_posix()
                if (args.output / "artifacts" / "anysplat_metadata.json").is_file()
                else None,
                "final_package_verification": final_verification.as_posix() if final_verification is not None else None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def run_final_package_verification(args: argparse.Namespace, archive: Path) -> Path:
    report_path = args.output / "motrix_package_verification.json"
    command = [
        sys.executable,
        "scripts/real2sim_verify_motrix_package.py",
        "--package",
        archive.as_posix(),
        "--min-scene-vertices",
        str(args.min_scene_vertices),
        "--min-source-preview-psnr",
        str(args.min_source_preview_psnr),
        "--max-source-preview-mae",
        str(args.max_source_preview_mae),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, file=sys.stderr)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise SystemExit(completed.returncode)
    report_path.write_text(completed.stdout, encoding="utf-8")
    return report_path


def run_best_image_edit_candidate(args: argparse.Namespace, multiview_dir: Path, quality_report: Path) -> Path:
    if args.candidates <= 0:
        raise SystemExit("--candidates must be positive")
    candidate_reports: list[dict[str, Any]] = []
    best_report: dict[str, Any] | None = None
    best_frames: Path | None = None
    for index in range(args.candidates):
        suffix = f"candidate_{index + 1:02d}" if args.candidates > 1 else "candidate_01"
        candidate_dir = multiview_dir / suffix
        frames_dir = candidate_dir / "frames"
        run_image_edit(args, candidate_dir, frames_dir)
        report = score_multiview_frames(args.image, frames_dir)
        report["candidate"] = index + 1
        report["frames_dir"] = frames_dir.as_posix()
        report["contact_sheet"] = (candidate_dir / "contact_sheet.png").as_posix()
        candidate_reports.append(report)
        if best_report is None or float(report["score"]) > float(best_report["score"]):
            best_report = report
            best_frames = frames_dir

    if best_report is None or best_frames is None:
        raise SystemExit("image-edit API produced no synthetic multiview candidates")
    quality = {
        "selected_candidate": best_report["candidate"],
        "selected_frames_dir": best_frames.as_posix(),
        "min_quality_score": args.min_quality_score,
        "passed": float(best_report["score"]) >= args.min_quality_score,
        "candidates": candidate_reports,
    }
    quality_report.parent.mkdir(parents=True, exist_ok=True)
    quality_report.write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not quality["passed"] and not args.allow_low_quality:
        raise SystemExit(
            "synthetic multiview quality gate failed: "
            f"best score {float(best_report['score']):.3f} < {args.min_quality_score:.3f}; "
            "inspect synthetic_multiview_quality.json or rerun with --allow-low-quality"
        )
    return best_frames


def run_image_edit(args: argparse.Namespace, multiview_dir: Path, frames_dir: Path) -> None:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing image-edit API key: set {args.api_key_env}")
    command = [
        sys.executable,
        "scripts/real2sim_external/synthetic_multiview_expand.py",
        "--image",
        args.image.as_posix(),
        "--output-dir",
        multiview_dir.as_posix(),
        "--frames-dir",
        frames_dir.as_posix(),
        "--mode",
        "scene",
        "--api-url",
        args.api_url,
        "--model",
        args.model,
        "--size",
        args.size,
        "--quality",
        args.quality,
        "--prompt",
        args.prompt,
    ]
    env = os.environ.copy()
    env["REAL2SIM_IMAGE_EDIT_API_KEY"] = api_key
    run(command, env=env)


def write_multiview_quality_report(source_image: Path, frames_dir: Path, output: Path) -> None:
    report = score_multiview_frames(source_image, frames_dir)
    quality = {
        "selected_candidate": None,
        "selected_frames_dir": frames_dir.as_posix(),
        "min_quality_score": None,
        "passed": True,
        "candidates": [report],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def score_multiview_frames(source_image: Path, frames_dir: Path) -> dict[str, Any]:
    frame_paths = sorted(
        path
        for path in frames_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    frames = [Image.open(path).convert("RGB") for path in frame_paths]
    source = Image.open(source_image).convert("RGB")

    frame_count_score = min(len(frames), 8) / 8.0
    resolution_score = 0.0
    exposure_score = 0.0
    sharpness_score = 0.0
    if frames:
        resolution_score = float(np.mean([min((img.width * img.height) / (360 * 360), 1.0) for img in frames]))
        exposure_score = float(np.mean([exposure_quality(img) for img in frames]))
        sharpness_score = float(np.mean([sharpness_quality(img) for img in frames]))
    source_similarity = first_view_similarity(source, frames[0]) if frames else 0.0
    diversity_score = view_diversity(frames)

    score = (
        0.20 * frame_count_score
        + 0.12 * resolution_score
        + 0.18 * source_similarity
        + 0.22 * diversity_score
        + 0.18 * sharpness_score
        + 0.10 * exposure_score
    )
    return {
        "score": round(float(score), 4),
        "frame_count": len(frames),
        "frame_count_score": round(float(frame_count_score), 4),
        "resolution_score": round(float(resolution_score), 4),
        "source_similarity": round(float(source_similarity), 4),
        "view_diversity": round(float(diversity_score), 4),
        "sharpness_score": round(float(sharpness_score), 4),
        "exposure_score": round(float(exposure_score), 4),
        "frames": [path.as_posix() for path in frame_paths],
        "warnings": multiview_quality_warnings(
            frame_count=len(frames),
            source_similarity=source_similarity,
            diversity_score=diversity_score,
            sharpness_score=sharpness_score,
            exposure_score=exposure_score,
        ),
    }


def image_array(image: Image.Image, size: tuple[int, int] = (160, 120)) -> np.ndarray:
    return np.asarray(image.resize(size, PIL_BILINEAR), dtype=np.float32) / 255.0


def first_view_similarity(source: Image.Image, first_view: Image.Image) -> float:
    src = image_array(source)
    view = image_array(first_view)
    mean_abs = float(np.mean(np.abs(src - view)))
    color_delta = float(np.mean(np.abs(src.mean(axis=(0, 1)) - view.mean(axis=(0, 1)))))
    return max(0.0, min(1.0, 1.0 - (1.7 * mean_abs + 0.8 * color_delta)))


def view_diversity(frames: list[Image.Image]) -> float:
    if len(frames) < 2:
        return 0.0
    arrays = [image_array(frame, size=(96, 72)) for frame in frames]
    diffs = [float(np.mean(np.abs(arrays[index] - arrays[index - 1]))) for index in range(1, len(arrays))]
    mean_diff = float(np.mean(diffs))
    if mean_diff < 0.035:
        return mean_diff / 0.035 * 0.45
    if mean_diff > 0.42:
        return max(0.0, 1.0 - (mean_diff - 0.42) / 0.30)
    return min(1.0, 0.45 + (mean_diff - 0.035) / (0.42 - 0.035) * 0.55)


def sharpness_quality(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L").resize((192, 144), PIL_BILINEAR), dtype=np.float32) / 255.0
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    energy = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))
    return max(0.0, min(1.0, energy / 0.11))


def exposure_quality(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L").resize((96, 72), PIL_BILINEAR), dtype=np.float32) / 255.0
    clipped = float(np.mean((gray < 0.025) | (gray > 0.975)))
    contrast = float(np.std(gray))
    return max(0.0, min(1.0, (1.0 - clipped * 2.0) * min(contrast / 0.22, 1.0)))


def multiview_quality_warnings(
    *,
    frame_count: int,
    source_similarity: float,
    diversity_score: float,
    sharpness_score: float,
    exposure_score: float,
) -> list[str]:
    warnings: list[str] = []
    if frame_count < 8:
        warnings.append("expected 8 synthetic views")
    if source_similarity < 0.35:
        warnings.append("first synthetic view may not preserve the source image")
    if diversity_score < 0.35:
        warnings.append("synthetic views may be too similar for scene reconstruction")
    if sharpness_score < 0.35:
        warnings.append("synthetic views look blurry or low-detail")
    if exposure_score < 0.35:
        warnings.append("synthetic views have weak contrast or clipped exposure")
    return warnings


def run_scene_generation(args: argparse.Namespace, frames_dir: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "real2sim",
        "generate-scene",
        "--input",
        frames_dir.as_posix(),
        "--output",
        args.output.as_posix(),
        "--work-dir",
        (args.work_dir / "scene_work").as_posix(),
        "--backend",
        args.backend,
        "--config",
        args.config.as_posix(),
        "--prompt",
        args.scene_prompt or args.prompt,
    ]
    if args.no_prune:
        command.append("--no-prune")
    run(command)


def run_anysplat_direct_generation(args: argparse.Namespace, frames_dir: Path) -> Path | None:
    scene_dir = args.work_dir / "anysplat_scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_ply = scene_dir / "scene.ply"
    scene_mesh = scene_dir / "mesh.obj"
    cameras_json = scene_dir / "cameras.json"
    depth_map = scene_dir / "depth.pgm"
    source_preview = args.source_gs_preview or (scene_dir / "source_gs_preview.png")
    anysplat_python = args.anysplat_python or sys.executable

    command = [
        anysplat_python,
        "scripts/real2sim_external/anysplat_scene.py",
        "--image",
        args.image.as_posix(),
        "--frames-dir",
        frames_dir.as_posix(),
        "--output-dir",
        scene_dir.as_posix(),
        "--ply",
        scene_ply.as_posix(),
        "--mesh",
        scene_mesh.as_posix(),
        "--cameras",
        cameras_json.as_posix(),
        "--depth",
        depth_map.as_posix(),
        "--source-preview",
        source_preview.as_posix(),
        "--max-views",
        str(args.anysplat_max_views),
        "--mesh-grid",
        str(max(8, int(getattr(args, "anysplat_mesh_grid", 48)))),
    ]
    if args.anysplat_root is not None:
        command.extend(["--anysplat-root", args.anysplat_root.as_posix()])
    run(command)
    metadata = scene_dir / "anysplat_metadata.json"
    if metadata.exists():
        enforce_anysplat_preview_quality(
            metadata,
            min_psnr=args.min_source_preview_psnr,
            max_mae=args.max_source_preview_mae,
            allow_low_quality=args.allow_low_quality,
        )

    package_command = [
        sys.executable,
        "-m",
        "real2sim",
        "package-existing",
        "--source",
        args.image.as_posix(),
        "--scene-ply",
        scene_ply.as_posix(),
        "--scene-mesh",
        scene_mesh.as_posix(),
        "--cameras-json",
        cameras_json.as_posix(),
        "--depth-map",
        depth_map.as_posix(),
        "--background-image",
        args.image.as_posix(),
        "--output",
        args.output.as_posix(),
        "--prompt",
        args.scene_prompt or args.prompt,
    ]
    package_command.extend(["--preview-image", source_preview.as_posix()])
    run(package_command)
    return metadata if metadata.exists() else None


def copy_quality_report(package_dir: Path, quality_report: Path) -> None:
    if quality_report.exists():
        quality_dest = package_dir / "synthetic_multiview_quality.json"
        quality_dest.write_text(quality_report.read_text(encoding="utf-8"), encoding="utf-8")


def enforce_anysplat_preview_quality(
    metadata_path: Path,
    *,
    min_psnr: float,
    max_mae: float,
    allow_low_quality: bool,
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = metadata.get("source_preview_metrics")
    if not isinstance(metrics, dict):
        raise SystemExit(f"AnySplat metadata missing source_preview_metrics: {metadata_path}")
    psnr = float(metrics.get("psnr", 0.0))
    mae = float(metrics.get("mae", 1.0))
    passed = psnr >= min_psnr and mae <= max_mae
    gate = {
        "min_source_preview_psnr": min_psnr,
        "max_source_preview_mae": max_mae,
        "passed": passed,
    }
    metadata["source_preview_quality_gate"] = gate
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not passed and not allow_low_quality:
        raise SystemExit(
            "AnySplat source-view quality gate failed: "
            f"psnr={psnr:.3f} < {min_psnr:.3f} or mae={mae:.4f} > {max_mae:.4f}; "
            "rerun with --allow-low-quality to keep the package"
        )
    return metadata


def attach_anysplat_metadata_to_package(package_dir: Path, metadata_path: Path) -> Path:
    artifact_dir = package_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dest = artifact_dir / "anysplat_metadata.json"
    dest.write_text(metadata_path.read_text(encoding="utf-8"), encoding="utf-8")

    import_path = package_dir / "motrix_import.json"
    if import_path.is_file():
        import_manifest = json.loads(import_path.read_text(encoding="utf-8"))
        import_manifest["reconstruction_metadata"] = dest.relative_to(package_dir).as_posix()
        metadata = json.loads(dest.read_text(encoding="utf-8"))
        metrics = metadata.get("source_preview_metrics")
        gate = metadata.get("source_preview_quality_gate")
        if isinstance(metrics, dict):
            import_manifest.setdefault("verification", {})["source_preview_metrics"] = metrics
        if isinstance(gate, dict):
            import_manifest.setdefault("verification", {})["source_preview_quality_gate"] = gate
        import_path.write_text(json.dumps(import_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dest


def verify_motrix_package(package_dir: Path) -> dict[str, Any]:
    scene_xml = package_dir / "mjcf" / "scene.xml"
    scene_ply = package_dir / "3dgs" / "scene.ply"
    config_json = package_dir / "configs" / "go2_real2sim.json"
    manifest_json = package_dir / "package_manifest.json"
    cameras_json = package_dir / "artifacts" / "cameras_json.json"
    for path in (scene_xml, scene_ply, config_json, manifest_json, cameras_json):
        if not path.is_file() or path.stat().st_size <= 0:
            raise SystemExit(f"package verification failed; missing or empty file: {path}")

    ET.parse(scene_xml)
    config = json.loads(config_json.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    cameras = json.loads(cameras_json.read_text(encoding="utf-8"))
    load_anysplat_source_camera(cameras_json)

    scene_gaussians = config.get("scene_gaussians")
    if not isinstance(scene_gaussians, dict) or "scene" not in scene_gaussians:
        raise SystemExit("package verification failed; config missing scene_gaussians.scene")
    gaussian_path = resolve_package_config_path(config_json, scene_gaussians["scene"])
    if not gaussian_path.is_file():
        raise SystemExit(f"package verification failed; scene Gaussian PLY does not exist: {gaussian_path}")

    manifest_scene = manifest.get("scene", {})
    manifest_ply = package_dir / str(manifest_scene.get("gaussian_ply", ""))
    if not manifest_ply.is_file():
        raise SystemExit(f"package verification failed; manifest scene gaussian_ply does not exist: {manifest_ply}")

    return {
        "root": package_dir.as_posix(),
        "scene_xml": scene_xml.as_posix(),
        "config_json": config_json.as_posix(),
        "scene_ply": scene_ply.as_posix(),
        "scene_ply_vertices": count_ply_vertices(scene_ply),
        "capture_kind": manifest.get("metadata", {}).get("capture_kind"),
        "scene_fusion": manifest.get("metadata", {}).get("builtin_scene_fusion"),
        "scene_view_count": manifest.get("metadata", {}).get("builtin_scene_view_count"),
        "camera_view_count": len(cameras.get("views", [])) if isinstance(cameras.get("views"), list) else 0,
    }


def resolve_package_config_path(config_json: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (config_json.parent / path).resolve()


def count_ply_vertices(path: Path) -> int:
    with path.open("r", encoding="ascii", errors="ignore") as handle:
        for line in handle:
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line.strip() == "end_header":
                break
    raise SystemExit(f"PLY vertex count not found: {path}")


def write_projection_preview(ply: Path, output: Path) -> Path:
    points, colors = load_ply_points_for_preview(ply, max_points=90000)
    width, height = 900, 520
    canvas = Image.new("RGB", (width * 2 + 20, height), (245, 245, 245))
    draw_projection(canvas, points, colors, (0, 2), 0, width, height, "front projection: x/z")
    draw_projection(canvas, points, colors, (0, 1), width + 20, width, height, "top projection: x/y")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def write_motrix_import_files(
    package_dir: Path,
    report: dict[str, Any],
    preview: Path,
    quality_report: Path | None = None,
) -> dict[str, Path]:
    """Write explicit MotrixSim import entrypoints into the generated package."""
    manifest_path = package_dir / "motrix_import.json"
    readme_path = package_dir / "MOTRIX_IMPORT.md"
    package_name = package_dir.name
    recommended_camera = "source_3dgs_preview"
    viewer_args = [
        "--package",
        f"/path/to/{package_name}",
        "--camera",
        "closeup",
        "--gs-width",
        "900",
        "--gs-height",
        "700",
        "--static-gs-preview",
        "--with-gs",
        "--show-source-gs",
        "--source-gs-only",
    ]
    import_manifest = {
        "format": "gs-playground-real2sim-motrix-import-v1",
        "import_type": "mjcf_scene_with_3dgs",
        "scene_xml": "mjcf/scene.xml",
        "config": "configs/go2_real2sim.json",
        "scene_gaussians": {"scene": "3dgs/scene.ply"},
        "camera_artifacts": {"cameras_json": "artifacts/cameras_json.json"},
        "preview": preview.relative_to(package_dir).as_posix(),
        "recommended_camera": recommended_camera,
        "viewer_args": viewer_args,
        "wuying_archive_launch": (
            "scripts/real2sim_wuying_launch_package.sh "
            f"--archive /path/to/{package_name}.tar.gz "
            f"--dest /tmp/{package_name}_import --with-gs"
        ),
        "verification": {
            "scene_ply_vertices": report.get("scene_ply_vertices"),
            "scene_fusion": report.get("scene_fusion"),
            "scene_view_count": report.get("scene_view_count"),
            "camera_view_count": report.get("camera_view_count"),
            "recommended_preview": recommended_camera,
        },
    }
    if quality_report is not None and quality_report.exists():
        quality_dest = package_dir / "synthetic_multiview_quality.json"
        quality_dest.write_text(quality_report.read_text(encoding="utf-8"), encoding="utf-8")
        import_manifest["synthetic_multiview_quality"] = quality_dest.relative_to(package_dir).as_posix()
    manifest_path.write_text(json.dumps(import_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    readme_path.write_text(
        "\n".join(
            [
                "# MotrixSim Import",
                "",
                "This package is a scene-level Real2Sim reconstruction from one source image plus synthetic multiview completion.",
                "",
                "Required entrypoints:",
                "",
                "- MJCF scene: `mjcf/scene.xml`",
                "- Scene Gaussian asset: `3dgs/scene.ply`",
                "- Viewer config: `configs/go2_real2sim.json`",
                "- Source cameras: `artifacts/cameras_json.json`",
                "- Projection preview: `pointcloud_projection_preview.png`",
                "",
                "Wuying launch command:",
                "",
                "```bash",
                import_manifest["wuying_archive_launch"],
                "```",
                "",
                "Direct viewer arguments:",
                "",
                "```bash",
                " ".join(import_manifest["viewer_args"]),
                "```",
                "",
                "Verification:",
                "",
                f"- scene vertices: {report.get('scene_ply_vertices')}",
                f"- scene fusion: {report.get('scene_fusion')}",
                f"- scene views: {report.get('scene_view_count')}",
                f"- camera views: {report.get('camera_view_count')}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"manifest": manifest_path, "readme": readme_path}


def load_ply_points_for_preview(path: Path, *, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    ply_format = read_ply_format(path)
    if ply_format != "ascii":
        return load_binary_ply_points_for_preview(path, max_points=max_points)

    with path.open("r", encoding="ascii", errors="ignore") as handle:
        vertex_count = 0
        for line in handle:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        stride = max(1, vertex_count // max_points)
        points = []
        colors = []
        for index, line in enumerate(handle):
            if index >= vertex_count:
                break
            if index % stride:
                continue
            values = line.split()
            if len(values) < 9:
                continue
            points.append(tuple(map(float, values[:3])))
            rgb = np.asarray(list(map(float, values[6:9])), dtype=np.float32)
            rgb = 1.0 / (1.0 + np.exp(-rgb))
            colors.append((rgb * 255.0).clip(0.0, 255.0).astype(np.uint8))
    if not points:
        raise SystemExit(f"no previewable points found in {path}")
    return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.uint8)


def read_ply_format(path: Path) -> str:
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("format "):
                parts = line.split()
                return parts[1] if len(parts) > 1 else "ascii"
            if line == "end_header":
                break
    return "ascii"


def load_binary_ply_points_for_preview(path: Path, *, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        from plyfile import PlyData
    except Exception as exc:
        return load_binary_ply_points_without_plyfile(path, max_points=max_points, import_error=exc)

    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {"x", "y", "z"}
    if not required.issubset(names):
        raise SystemExit(f"binary PLY missing required xyz fields: {path}")

    vertex_count = len(vertex)
    stride = max(1, vertex_count // max_points)
    sampled = vertex[::stride]
    points = np.column_stack([sampled["x"], sampled["y"], sampled["z"]]).astype(np.float32, copy=False)
    if {"red", "green", "blue"}.issubset(names):
        colors = np.column_stack([sampled["red"], sampled["green"], sampled["blue"]]).astype(np.uint8, copy=False)
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        rgb = np.column_stack([sampled["f_dc_0"], sampled["f_dc_1"], sampled["f_dc_2"]]).astype(np.float32, copy=False)
        colors = (1.0 / (1.0 + np.exp(-rgb)) * 255.0).clip(0.0, 255.0).astype(np.uint8)
    else:
        colors = np.full((len(points), 3), 220, dtype=np.uint8)
    if len(points) == 0:
        raise SystemExit(f"no previewable points found in {path}")
    return points, colors


def load_binary_ply_points_without_plyfile(
    path: Path,
    *,
    max_points: int,
    import_error: Exception | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    header_end = data.find(b"end_header\n")
    newline_len = 1
    if header_end < 0:
        header_end = data.find(b"end_header\r\n")
        newline_len = 2
    if header_end < 0:
        raise SystemExit(f"binary PLY missing end_header: {path}") from import_error

    header_text = data[: header_end + len("end_header")].decode("ascii", errors="ignore")
    header_lines = [line.strip() for line in header_text.splitlines()]
    ply_format = "binary_little_endian"
    vertex_count = 0
    vertex_properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        if line.startswith("format "):
            parts = line.split()
            if len(parts) > 1:
                ply_format = parts[1]
        elif line.startswith("element "):
            parts = line.split()
            in_vertex = len(parts) >= 3 and parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif in_vertex and line.startswith("property "):
            parts = line.split()
            if len(parts) == 3:
                vertex_properties.append((parts[1], parts[2]))
            elif len(parts) > 3 and parts[1] == "list":
                raise SystemExit(f"binary PLY vertex list properties are unsupported for preview: {path}") from import_error

    if vertex_count <= 0 or not vertex_properties:
        raise SystemExit(f"binary PLY has no previewable vertex table: {path}") from import_error
    endian = "<" if ply_format == "binary_little_endian" else ">"
    dtype = np.dtype([(name, numpy_dtype_for_ply_type(kind, endian)) for kind, name in vertex_properties])
    offset = header_end + len("end_header") + newline_len
    vertex = np.frombuffer(data, dtype=dtype, count=vertex_count, offset=offset)
    names = set(vertex.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise SystemExit(f"binary PLY missing required xyz fields: {path}") from import_error

    stride = max(1, vertex_count // max_points)
    sampled = vertex[::stride]
    points = np.column_stack([sampled["x"], sampled["y"], sampled["z"]]).astype(np.float32, copy=False)
    if {"red", "green", "blue"}.issubset(names):
        colors = np.column_stack([sampled["red"], sampled["green"], sampled["blue"]]).astype(np.uint8, copy=False)
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        rgb = np.column_stack([sampled["f_dc_0"], sampled["f_dc_1"], sampled["f_dc_2"]]).astype(np.float32, copy=False)
        colors = (1.0 / (1.0 + np.exp(-rgb)) * 255.0).clip(0.0, 255.0).astype(np.uint8)
    else:
        colors = np.full((len(points), 3), 220, dtype=np.uint8)
    return points, colors


def numpy_dtype_for_ply_type(kind: str, endian: str) -> str:
    mapping = {
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
    if kind not in mapping:
        raise SystemExit(f"unsupported binary PLY property type: {kind}")
    code = mapping[kind]
    return code if code.endswith("1") else endian + code


def draw_projection(
    canvas: Image.Image,
    points: np.ndarray,
    colors: np.ndarray,
    axes: tuple[int, int],
    offset: int,
    width: int,
    height: int,
    title: str,
) -> None:
    sub = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(sub)
    xy = points[:, list(axes)]
    low = xy.min(axis=0)
    high = xy.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    uv = (xy - low) / span
    px = (uv[:, 0] * (width - 60) + 30).astype(int)
    py = ((1.0 - uv[:, 1]) * (height - 60) + 30).astype(int)
    order = np.argsort(points[:, 1] if axes != (0, 1) else points[:, 2])
    for index in order:
        x = int(px[index])
        y = int(py[index])
        if 0 <= x < width and 0 <= y < height:
            sub.putpixel((x, y), tuple(int(v) for v in colors[index]))
    draw.rectangle((0, 0, width - 1, height - 1), outline=(190, 190, 190))
    draw.text((14, 12), title, fill=(30, 30, 30))
    canvas.paste(sub, (offset, 0))


def create_archive(package_dir: Path, archive: Path) -> Path:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(package_dir, arcname=package_dir.name)
    return archive


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
