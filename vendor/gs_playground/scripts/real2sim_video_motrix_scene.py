#!/usr/bin/env python3
"""Video Real2Sim scene reconstruction workflow for MotrixSim.

This script upgrades short video captures from the built-in pseudo-depth path
to the AnySplat-direct path. It extracts video frames, selects sharp and
temporally spread keyframes, reconstructs a scene-level 3DGS, packages it for
MotrixSim, and writes verification/visual acceptance artifacts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageOps

PIL_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

VIDEO_PROMPT = "office aisle scene with desks, cabinets, floor, plants and fixtures"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}
DEFAULT_VIDEO_ANYSPLAT_MAX_VIEWS = 12
DEFAULT_VIDEO_MIN_SCENE_VERTICES = 600_000
MOTRIX_HIGH_FRAME_FPS = 3.0
MOTRIX_HIGH_MESH_GRID = 160
MOTRIX_HIGH_MIN_SCENE_VERTICES = 900_000
MOTRIX_HIGH_SCENE_SURFELS = 30_000
MOTRIX_HIGH_POSE_INIT_STEPS = 320
MOTRIX_HIGH_POSE_INIT_MAX_FRAMES = 10
MOTRIX_HIGH_POSE_INIT_IMAGE_MAX_SIZE = 448
MOTRIX_HIGH_POSE_INIT_TARGET_GAUSSIANS = 60_000
MOTRIX_HIGH_POSE_INIT_MAX_POINTS = 90_000
MOTRIX_HIGH_POSE_INIT_DENSE_POINTS_PER_FRAME = 2_600
MOTRIX_HIGH_POSE_INIT_ADAPTIVE_EVERY = 25
MOTRIX_HIGH_POSE_INIT_ADAPTIVE_STOP = 125
MOTRIX_HIGH_POSE_INIT_ADAPTIVE_POINTS = 2_500
MOTRIX_HIGH_POSE_INIT_ADAPTIVE_ERROR_PERCENTILE = 88.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and verify a MotrixSim scene package from a video")
    parser.add_argument("--video", type=Path, default=None, help="Source video")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Use pre-extracted video frames or a still-image capture directory")
    parser.add_argument("--images-dir", type=Path, default=None, help="Alias for --frames-dir when the input is a still-image capture set")
    parser.add_argument("--output", type=Path, required=True, help="Output Real2Sim scene package directory")
    parser.add_argument("--work-dir", type=Path, required=True, help="Intermediate workflow directory")
    parser.add_argument("--prompt", default=VIDEO_PROMPT)
    parser.add_argument("--scene-prompt", default="", help="Scene reconstruction prompt; defaults to --prompt")
    parser.add_argument(
        "--quality-preset",
        default=os.environ.get("REAL2SIM_VIDEO_QUALITY_PRESET", "balanced"),
        choices=["balanced", "motrix-high", "capture-high"],
        help="Balanced keeps default package size; capture-high also runs COLMAP/SfM capture diagnostics",
    )
    parser.add_argument("--frame-fps", type=float, default=float(os.environ.get("REAL2SIM_VIDEO_FRAME_FPS", "2.0")))
    parser.add_argument("--anysplat-python", default=os.environ.get("REAL2SIM_ANY_SPLAT_PYTHON", ""))
    parser.add_argument("--anysplat-root", type=Path, default=os.environ.get("REAL2SIM_ANYSPLAT_ROOT"))
    parser.add_argument(
        "--anysplat-max-views",
        type=int,
        default=int(os.environ.get("REAL2SIM_VIDEO_ANYSPLAT_MAX_VIEWS", str(DEFAULT_VIDEO_ANYSPLAT_MAX_VIEWS))),
    )
    parser.add_argument("--anysplat-mesh-grid", type=int, default=int(os.environ.get("REAL2SIM_VIDEO_ANYSPLAT_MESH_GRID", "96")))
    parser.add_argument(
        "--anysplat-keyframe-strategy",
        choices=["quality-diversity", "sfm-registered"],
        default=os.environ.get("REAL2SIM_ANYSPLAT_KEYFRAME_STRATEGY"),
        help="How to select frames passed to AnySplat; capture-high defaults to SfM-registered pose bins",
    )
    parser.add_argument(
        "--anysplat-keyframe-orientation-weight",
        type=float,
        default=float(os.environ["REAL2SIM_ANYSPLAT_KEYFRAME_ORIENTATION_WEIGHT"])
        if os.environ.get("REAL2SIM_ANYSPLAT_KEYFRAME_ORIENTATION_WEIGHT")
        else None,
        help="SfM-registered keyframe score weight for camera up-vector consistency",
    )
    parser.add_argument("--min-video-quality", type=float, default=float(os.environ.get("REAL2SIM_MIN_VIDEO_FRAME_QUALITY", "0.38")))
    parser.add_argument("--allow-low-quality", action="store_true")
    parser.add_argument("--min-source-preview-psnr", type=float, default=float(os.environ.get("REAL2SIM_MIN_SOURCE_PREVIEW_PSNR", "18.0")))
    parser.add_argument("--max-source-preview-mae", type=float, default=float(os.environ.get("REAL2SIM_MAX_SOURCE_PREVIEW_MAE", "0.10")))
    parser.add_argument(
        "--min-scene-vertices",
        type=int,
        default=int(os.environ.get("REAL2SIM_VIDEO_MIN_SCENE_VERTICES", str(DEFAULT_VIDEO_MIN_SCENE_VERTICES))),
    )
    parser.add_argument("--skip-final-package-verify", action="store_true")
    parser.add_argument("--sfm-report", action="store_true", default=_env_bool("REAL2SIM_VIDEO_SFM_REPORT"))
    parser.add_argument("--colmap-bin", default=os.environ.get("REAL2SIM_COLMAP_BIN", "colmap"))
    parser.add_argument("--sfm-matcher", choices=["exhaustive", "sequential"], default=os.environ.get("REAL2SIM_COLMAP_MATCHER", "sequential"))
    parser.add_argument("--sfm-sequential-overlap", type=int, default=int(os.environ.get("REAL2SIM_COLMAP_SEQUENTIAL_OVERLAP", "10")))
    parser.add_argument("--sfm-camera-model", default=os.environ.get("REAL2SIM_COLMAP_CAMERA_MODEL"))
    parser.add_argument("--sfm-max-image-size", type=int, default=_env_int("REAL2SIM_COLMAP_MAX_IMAGE_SIZE"))
    parser.add_argument("--sfm-sift-max-num-features", type=int, default=int(os.environ.get("REAL2SIM_COLMAP_SIFT_MAX_FEATURES", "8192")))
    parser.add_argument("--sfm-sift-peak-threshold", type=float, default=float(os.environ["REAL2SIM_COLMAP_SIFT_PEAK_THRESHOLD"]) if os.environ.get("REAL2SIM_COLMAP_SIFT_PEAK_THRESHOLD") else None)
    parser.add_argument("--sfm-feature-threads", type=int, default=_env_int("REAL2SIM_COLMAP_FEATURE_THREADS"))
    parser.add_argument("--sfm-matcher-threads", type=int, default=_env_int("REAL2SIM_COLMAP_MATCHER_THREADS"))
    parser.add_argument("--sfm-guided-matching", action="store_true", default=_env_bool("REAL2SIM_COLMAP_GUIDED_MATCHING"))
    parser.add_argument("--sfm-frames", choices=["extracted", "selected"], default=os.environ.get("REAL2SIM_SFM_FRAMES", "extracted"))
    parser.add_argument("--min-sfm-registered-ratio", type=float, default=float(os.environ.get("REAL2SIM_MIN_SFM_REGISTERED_RATIO", "0.70")))
    parser.add_argument("--min-sfm-points", type=int, default=int(os.environ.get("REAL2SIM_MIN_SFM_POINTS", "500")))
    parser.add_argument("--require-sfm", action="store_true", default=_env_bool("REAL2SIM_REQUIRE_SFM"))
    parser.add_argument("--colmap-dense", action="store_true", default=_env_bool("REAL2SIM_COLMAP_DENSE"))
    parser.add_argument("--require-colmap-dense", action="store_true", default=_env_bool("REAL2SIM_REQUIRE_COLMAP_DENSE"))
    parser.add_argument("--colmap-dense-bin", default=os.environ.get("REAL2SIM_COLMAP_DENSE_BIN"))
    parser.add_argument("--colmap-dense-max-image-size", type=int, default=int(os.environ.get("REAL2SIM_COLMAP_DENSE_MAX_IMAGE_SIZE", "960")))
    parser.add_argument("--colmap-dense-min-fused-points", type=int, default=int(os.environ.get("REAL2SIM_COLMAP_DENSE_MIN_FUSED_POINTS", "100000")))
    parser.add_argument("--colmap-dense-gpu-index", default=os.environ.get("REAL2SIM_COLMAP_DENSE_GPU_INDEX", ""))
    parser.add_argument("--colmap-dense-run-poisson", action="store_true", default=_env_bool("REAL2SIM_COLMAP_DENSE_POISSON"))
    parser.add_argument("--colmap-dense-poisson-depth", type=int, default=int(os.environ.get("REAL2SIM_COLMAP_DENSE_POISSON_DEPTH", "10")))
    parser.add_argument("--colmap-dense-cleanup-keep-largest", type=int, default=int(os.environ.get("REAL2SIM_COLMAP_DENSE_CLEANUP_KEEP_LARGEST", "1")))
    parser.add_argument("--colmap-dense-cleanup-min-faces", type=int, default=int(os.environ.get("REAL2SIM_COLMAP_DENSE_CLEANUP_MIN_FACES", "0")))
    parser.add_argument("--colmap-dense-cleanup-min-area-ratio", type=float, default=float(os.environ.get("REAL2SIM_COLMAP_DENSE_CLEANUP_MIN_AREA_RATIO", "0")))
    parser.add_argument("--colmap-dense-cleanup-max-face-area-percentile", type=float, default=float(os.environ.get("REAL2SIM_COLMAP_DENSE_CLEANUP_MAX_FACE_AREA_PERCENTILE", "100")))
    parser.add_argument("--colmap-dense-cleanup-max-edge-length-percentile", type=float, default=float(os.environ.get("REAL2SIM_COLMAP_DENSE_CLEANUP_MAX_EDGE_LENGTH_PERCENTILE", "100")))
    parser.add_argument("--colmap-dense-cleanup-max-centroid-z-percentile", type=float, default=float(os.environ.get("REAL2SIM_COLMAP_DENSE_CLEANUP_MAX_CENTROID_Z_PERCENTILE", "100")))
    parser.add_argument(
        "--colmap-dense-replace-scene-mesh",
        action="store_true",
        default=_env_bool("REAL2SIM_COLMAP_DENSE_REPLACE_SCENE_MESH"),
        help="Repackage the Motrix scene with the COLMAP dense mesh as the scene mesh",
    )
    parser.add_argument("--colmap-texture-bake", action="store_true", default=_env_bool("REAL2SIM_COLMAP_TEXTURE_BAKE"))
    parser.add_argument("--require-colmap-texture-bake", action="store_true", default=_env_bool("REAL2SIM_REQUIRE_COLMAP_TEXTURE_BAKE"))
    parser.add_argument("--colmap-texture-bake-unwrap-uv", action="store_true", default=_env_bool("REAL2SIM_COLMAP_TEXTURE_BAKE_UNWRAP_UV"))
    parser.add_argument("--require-colmap-texture-bake-unwrap-uv", action="store_true", default=_env_bool("REAL2SIM_REQUIRE_COLMAP_TEXTURE_BAKE_UNWRAP_UV"))
    parser.add_argument("--colmap-texture-bake-atlas-size", type=int, default=int(os.environ.get("REAL2SIM_COLMAP_TEXTURE_BAKE_ATLAS_SIZE", "4096")))
    parser.add_argument("--colmap-texture-bake-max-cameras", type=int, default=int(os.environ.get("REAL2SIM_COLMAP_TEXTURE_BAKE_MAX_CAMERAS", "72")))
    parser.add_argument("--colmap-texture-bake-max-faces", type=int, default=int(os.environ.get("REAL2SIM_COLMAP_TEXTURE_BAKE_MAX_FACES", "0")))
    parser.add_argument("--colmap-texture-bake-max-output-faces", type=int, default=int(os.environ.get("REAL2SIM_COLMAP_TEXTURE_BAKE_MAX_OUTPUT_FACES", "0")))
    parser.add_argument("--colmap-texture-bake-fill-mode", choices=["face", "warp"], default=os.environ.get("REAL2SIM_COLMAP_TEXTURE_BAKE_FILL_MODE", "face"))
    parser.add_argument("--colmap-texture-bake-color-blend-cameras", type=int, default=int(os.environ.get("REAL2SIM_COLMAP_TEXTURE_BAKE_COLOR_BLEND_CAMERAS", "1")))
    parser.add_argument("--colmap-texture-bake-visibility-depth-tolerance", type=float, default=float(os.environ.get("REAL2SIM_COLMAP_TEXTURE_BAKE_VISIBILITY_DEPTH_TOLERANCE", "0")))
    parser.add_argument("--colmap-texture-bake-visibility-grid-size", type=int, default=int(os.environ.get("REAL2SIM_COLMAP_TEXTURE_BAKE_VISIBILITY_GRID_SIZE", "320")))
    parser.add_argument("--colmap-texture-bake-max-warped-faces", type=int, default=int(os.environ.get("REAL2SIM_COLMAP_TEXTURE_BAKE_MAX_WARPED_FACES", "0")))
    parser.add_argument("--colmap-texture-bake-postprocess-median-radius", type=int, default=int(os.environ.get("REAL2SIM_COLMAP_TEXTURE_BAKE_POSTPROCESS_MEDIAN_RADIUS", "0")))
    parser.add_argument("--colmap-texture-bake-bleed-iterations", type=int, default=int(os.environ.get("REAL2SIM_COLMAP_TEXTURE_BAKE_BLEED_ITERATIONS", "0")))
    parser.add_argument("--motrix-preview-mesh-faces", type=int, default=int(os.environ.get("REAL2SIM_MOTRIX_PREVIEW_MESH_FACES", "0")))
    parser.add_argument("--motrix-preview-mode", choices=["proxy", "lod"], default=os.environ.get("REAL2SIM_MOTRIX_PREVIEW_MODE"))
    parser.add_argument("--capture-acceptance", action="store_true", default=_env_bool("REAL2SIM_CAPTURE_ACCEPTANCE"))
    parser.add_argument("--require-capture-acceptance", action="store_true", default=_env_bool("REAL2SIM_REQUIRE_CAPTURE_ACCEPTANCE"))
    parser.add_argument(
        "--min-capture-camera-path-length",
        type=float,
        default=float(os.environ.get("REAL2SIM_MIN_CAPTURE_CAMERA_PATH_LENGTH", "1.0")),
    )
    parser.add_argument(
        "--min-capture-camera-planar-coverage-ratio",
        type=float,
        default=float(os.environ.get("REAL2SIM_MIN_CAPTURE_CAMERA_PLANAR_COVERAGE_RATIO", "0.12")),
    )
    parser.add_argument(
        "--max-capture-camera-step-ratio",
        type=float,
        default=float(os.environ.get("REAL2SIM_MAX_CAPTURE_CAMERA_STEP_RATIO", "0.55")),
    )
    parser.add_argument(
        "--skip-pose-init-dataset",
        action="store_true",
        help="Do not copy registered COLMAP frames/poses into artifacts/pose_init_dataset",
    )
    parser.add_argument("--pose-init-train", action="store_true", default=_env_bool("REAL2SIM_POSE_INIT_TRAIN"))
    parser.add_argument(
        "--pose-init-train-python",
        default=os.environ.get("REAL2SIM_POSE_INIT_TRAIN_PYTHON", os.environ.get("REAL2SIM_ANY_SPLAT_PYTHON", "")),
        help="Python interpreter with torch and gsplat for pose-init training",
    )
    parser.add_argument("--pose-init-train-steps", type=int, default=int(os.environ.get("REAL2SIM_POSE_INIT_TRAIN_STEPS", "200")))
    parser.add_argument("--pose-init-train-max-frames", type=int, default=int(os.environ.get("REAL2SIM_POSE_INIT_TRAIN_MAX_FRAMES", "16")))
    parser.add_argument("--pose-init-train-image-max-size", type=int, default=int(os.environ.get("REAL2SIM_POSE_INIT_TRAIN_IMAGE_MAX_SIZE", "448")))
    parser.add_argument("--pose-init-train-target-gaussians", type=int, default=int(os.environ.get("REAL2SIM_POSE_INIT_TRAIN_TARGET_GAUSSIANS", "50000")))
    parser.add_argument("--pose-init-train-max-points", type=int, default=int(os.environ.get("REAL2SIM_POSE_INIT_TRAIN_MAX_POINTS", "80000")))
    parser.add_argument(
        "--pose-init-train-dense-image-points-per-frame",
        type=int,
        default=int(os.environ.get("REAL2SIM_POSE_INIT_TRAIN_DENSE_IMAGE_POINTS_PER_FRAME", "1600")),
    )
    parser.add_argument("--pose-init-train-dense-depth-neighbors", type=int, default=int(os.environ.get("REAL2SIM_POSE_INIT_TRAIN_DENSE_DEPTH_NEIGHBORS", "4")))
    parser.add_argument("--pose-init-train-dense-depth-max-distance", type=float, default=float(os.environ.get("REAL2SIM_POSE_INIT_TRAIN_DENSE_DEPTH_MAX_DISTANCE", "48.0")))
    parser.add_argument("--pose-init-train-adaptive-densify-every", type=int, default=int(os.environ.get("REAL2SIM_POSE_INIT_TRAIN_ADAPTIVE_DENSIFY_EVERY", "0")))
    parser.add_argument("--pose-init-train-adaptive-densify-start", type=int, default=int(os.environ.get("REAL2SIM_POSE_INIT_TRAIN_ADAPTIVE_DENSIFY_START", "50")))
    parser.add_argument("--pose-init-train-adaptive-densify-stop", type=int, default=int(os.environ.get("REAL2SIM_POSE_INIT_TRAIN_ADAPTIVE_DENSIFY_STOP", "0")))
    parser.add_argument("--pose-init-train-adaptive-densify-points", type=int, default=int(os.environ.get("REAL2SIM_POSE_INIT_TRAIN_ADAPTIVE_DENSIFY_POINTS", "2500")))
    parser.add_argument("--pose-init-train-adaptive-densify-error-percentile", type=float, default=float(os.environ.get("REAL2SIM_POSE_INIT_TRAIN_ADAPTIVE_DENSIFY_ERROR_PERCENTILE", "82.0")))
    parser.add_argument("--pose-init-train-preview-frames", type=int, default=int(os.environ.get("REAL2SIM_POSE_INIT_TRAIN_PREVIEW_FRAMES", "4")))
    parser.add_argument(
        "--pose-init-train-replace-scene",
        action="store_true",
        default=_env_bool("REAL2SIM_POSE_INIT_TRAIN_REPLACE_SCENE"),
        help="Replace 3dgs/scene.ply with the trained pose-init scene after training",
    )
    parser.add_argument("--archive", type=Path, default=None, help="Optional .tar.gz path; defaults to <output>.tar.gz")
    parser.add_argument("--visual-report-dir", type=Path, default=None)
    parser.add_argument("--motrix-viewer-render", type=Path, default=None)
    args = parser.parse_args()
    if args.images_dir is not None:
        if args.frames_dir is not None and args.frames_dir != args.images_dir:
            raise SystemExit("--images-dir and --frames-dir point to different directories")
        args.frames_dir = args.images_dir
    apply_quality_preset(args)

    result = run_video_workflow(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def run_video_workflow(args: argparse.Namespace) -> dict[str, Any]:
    if args.frames_dir is None and args.video is None:
        raise SystemExit("either --video or --frames-dir/--images-dir is required")
    if args.frames_dir is None and not args.video.is_file():
        raise SystemExit(f"source video not found: {args.video}")
    if args.anysplat_max_views <= 0:
        raise SystemExit("--anysplat-max-views must be positive")
    if args.frame_fps <= 0:
        raise SystemExit("--frame-fps must be positive")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.frames_dir is not None:
        extracted_frames = normalize_existing_frames(
            collect_existing_frames(args.frames_dir),
            args.work_dir / "input_frames",
        )
    else:
        extracted_frames = extract_video_frames(args.video, args.work_dir / "video_frames", fps=args.frame_fps)
    selected_frames = args.work_dir / "selected_frames"
    video_quality = args.work_dir / "video_frame_quality.json"
    selected = select_video_keyframes(
        extracted_frames,
        selected_frames,
        max_views=args.anysplat_max_views,
        quality_report=video_quality,
        min_quality_score=args.min_video_quality,
    )
    annotate_video_quality_report(video_quality, args)
    quality = json.loads(video_quality.read_text(encoding="utf-8"))
    if not quality["passed"] and not args.allow_low_quality:
        raise SystemExit(
            "video frame quality gate failed: "
            f"score {float(quality['score']):.3f} < {args.min_video_quality:.3f}; "
            "inspect video_frame_quality.json or rerun with --allow-low-quality"
        )
    sfm_report = run_sfm_capture_report(args, sfm_frames_dir(args, extracted_frames, selected_frames)) if args.sfm_report else None
    acceptance_report = run_capture_acceptance_report(args, sfm_report) if args.capture_acceptance and sfm_report is not None else None
    selected = maybe_reselect_anysplat_keyframes_from_sfm(
        args,
        extracted_frames,
        selected_frames,
        video_quality,
        sfm_report,
        selected,
    )

    helpers = load_one_image_helpers()
    helpers.run = run_child_command
    source_image = selected[0]
    helper_args = SimpleNamespace(
        image=source_image,
        output=args.output,
        work_dir=args.work_dir,
        prompt=args.prompt,
        scene_prompt=args.scene_prompt or args.prompt,
        anysplat_python=args.anysplat_python,
        anysplat_root=args.anysplat_root,
        anysplat_max_views=args.anysplat_max_views,
        anysplat_mesh_grid=args.anysplat_mesh_grid,
        source_gs_preview=None,
        min_source_preview_psnr=args.min_source_preview_psnr,
        max_source_preview_mae=args.max_source_preview_mae,
        allow_low_quality=args.allow_low_quality,
        min_scene_vertices=args.min_scene_vertices,
    )
    anysplat_metadata = helpers.run_anysplat_direct_generation(helper_args, selected_frames)
    package_report = helpers.verify_motrix_package(args.output)
    projection_preview = helpers.write_projection_preview(args.output / "3dgs" / "scene.ply", args.output / "pointcloud_projection_preview.png")
    if anysplat_metadata is not None:
        helpers.attach_anysplat_metadata_to_package(args.output, anysplat_metadata)
    attach_video_quality_to_package(args.output, video_quality)
    if sfm_report is not None:
        attach_sfm_report_to_package(args.output, sfm_report)
    if acceptance_report is not None:
        attach_capture_acceptance_to_package(args.output, acceptance_report)
    colmap_dense_report = (
        run_colmap_dense_reconstruction_to_package(args, args.output, sfm_report, helpers)
        if should_run_colmap_dense(args, sfm_report)
        else None
    )
    if colmap_dense_report is not None and bool(args.colmap_dense_replace_scene_mesh):
        if anysplat_metadata is not None:
            helpers.attach_anysplat_metadata_to_package(args.output, anysplat_metadata)
        attach_video_quality_to_package(args.output, video_quality)
        if sfm_report is not None:
            attach_sfm_report_to_package(args.output, sfm_report)
        if acceptance_report is not None:
            attach_capture_acceptance_to_package(args.output, acceptance_report)
        attach_colmap_dense_to_package(args.output, colmap_dense_report)
    pose_init_dataset = (
        export_pose_init_dataset_to_package(sfm_report, args.output)
        if should_export_pose_init_dataset(args, sfm_report)
        else None
    )
    pose_init_training = run_pose_init_training_to_package(args, args.output, pose_init_dataset)

    archive = helpers.create_archive(args.output, args.archive or args.output.with_suffix(".tar.gz"))
    final_verification = None
    if not args.skip_final_package_verify:
        final_verification = helpers.run_final_package_verification(helper_args, archive)
        archive = helpers.create_archive(args.output, archive)

    visual_report = None
    visual_dir = args.visual_report_dir or (args.work_dir / "visual_acceptance")
    visual_report = write_visual_report(
        args.output,
        visual_dir,
        min_scene_vertices=args.min_scene_vertices,
        min_source_preview_psnr=args.min_source_preview_psnr,
        max_source_preview_mae=args.max_source_preview_mae,
        motrix_viewer_render=args.motrix_viewer_render,
    )

    return {
        "input": args.video.as_posix() if args.video is not None else args.frames_dir.as_posix(),
        "input_kind": "video" if args.frames_dir is None else "image_directory",
        "video": args.video.as_posix() if args.video is not None else None,
        "frames_dir": args.frames_dir.as_posix() if args.frames_dir is not None else None,
        "quality_preset": args.quality_preset,
        "selected_frames": [path.as_posix() for path in selected],
        "video_frame_quality": video_quality.as_posix(),
        "sfm_capture_report": sfm_report.as_posix() if sfm_report is not None else None,
        "capture_acceptance_report": acceptance_report.as_posix() if acceptance_report is not None else None,
        "package": package_report,
        "projection_preview": projection_preview.as_posix(),
        "archive": archive.as_posix(),
        "final_package_verification": final_verification.as_posix() if final_verification is not None else None,
        "visual_acceptance_report": visual_report,
        "colmap_dense_reconstruction": colmap_dense_report.as_posix() if colmap_dense_report is not None else None,
        "pose_init_dataset": pose_init_dataset.as_posix() if pose_init_dataset is not None else None,
        "pose_init_training": pose_init_training.as_posix() if pose_init_training is not None else None,
    }


def apply_quality_preset(args: argparse.Namespace) -> None:
    apply_quality_preset_arg_defaults(args)
    if args.require_colmap_dense:
        args.colmap_dense = True
    if args.colmap_dense:
        args.sfm_report = True
        if args.sfm_camera_model is None:
            args.sfm_camera_model = "SIMPLE_RADIAL"
        if args.sfm_max_image_size is None or int(args.sfm_max_image_size) <= 0:
            args.sfm_max_image_size = 1600
        if args.sfm_feature_threads is None or int(args.sfm_feature_threads) <= 0:
            args.sfm_feature_threads = min(8, max(2, (os.cpu_count() or 2) // 2))
        if args.sfm_matcher_threads is None or int(args.sfm_matcher_threads) <= 0:
            args.sfm_matcher_threads = min(8, max(2, (os.cpu_count() or 2) // 2))
        args.sfm_sequential_overlap = max(int(args.sfm_sequential_overlap), 20)
        args.sfm_sift_max_num_features = max(int(args.sfm_sift_max_num_features), 16_384)
        if args.sfm_sift_peak_threshold is None:
            args.sfm_sift_peak_threshold = 0.003
        args.colmap_dense_max_image_size = max(int(args.colmap_dense_max_image_size), 1)
        args.colmap_dense_min_fused_points = max(int(args.colmap_dense_min_fused_points), 1)
    if args.quality_preset not in {"motrix-high", "capture-high"}:
        return
    args.frame_fps = max(float(args.frame_fps), MOTRIX_HIGH_FRAME_FPS)
    args.anysplat_max_views = max(int(args.anysplat_max_views), DEFAULT_VIDEO_ANYSPLAT_MAX_VIEWS)
    args.anysplat_mesh_grid = max(int(args.anysplat_mesh_grid), MOTRIX_HIGH_MESH_GRID)
    args.min_scene_vertices = max(int(args.min_scene_vertices), MOTRIX_HIGH_MIN_SCENE_VERTICES)
    os.environ.setdefault("REAL2SIM_MOTRIX_PREVIEW_MODE", "lod")
    os.environ.setdefault("REAL2SIM_MOTRIX_SCENE_VISUAL", "hybrid")
    os.environ.setdefault("REAL2SIM_MOTRIX_SCENE_SURFELS", str(MOTRIX_HIGH_SCENE_SURFELS))
    args.pose_init_train_steps = max(int(args.pose_init_train_steps), MOTRIX_HIGH_POSE_INIT_STEPS)
    args.pose_init_train_max_frames = max(int(args.pose_init_train_max_frames), MOTRIX_HIGH_POSE_INIT_MAX_FRAMES)
    args.pose_init_train_image_max_size = max(int(args.pose_init_train_image_max_size), MOTRIX_HIGH_POSE_INIT_IMAGE_MAX_SIZE)
    args.pose_init_train_target_gaussians = max(int(args.pose_init_train_target_gaussians), MOTRIX_HIGH_POSE_INIT_TARGET_GAUSSIANS)
    args.pose_init_train_max_points = max(int(args.pose_init_train_max_points), MOTRIX_HIGH_POSE_INIT_MAX_POINTS)
    args.pose_init_train_dense_image_points_per_frame = max(
        int(args.pose_init_train_dense_image_points_per_frame),
        MOTRIX_HIGH_POSE_INIT_DENSE_POINTS_PER_FRAME,
    )
    args.pose_init_train_adaptive_densify_every = max(
        int(args.pose_init_train_adaptive_densify_every),
        MOTRIX_HIGH_POSE_INIT_ADAPTIVE_EVERY,
    )
    if int(args.pose_init_train_adaptive_densify_stop) <= 0:
        args.pose_init_train_adaptive_densify_stop = MOTRIX_HIGH_POSE_INIT_ADAPTIVE_STOP
    args.pose_init_train_adaptive_densify_points = max(
        int(args.pose_init_train_adaptive_densify_points),
        MOTRIX_HIGH_POSE_INIT_ADAPTIVE_POINTS,
    )
    args.pose_init_train_adaptive_densify_error_percentile = max(
        float(args.pose_init_train_adaptive_densify_error_percentile),
        MOTRIX_HIGH_POSE_INIT_ADAPTIVE_ERROR_PERCENTILE,
    )
    if args.quality_preset == "capture-high":
        args.sfm_report = True
        args.capture_acceptance = True
        if args.anysplat_keyframe_strategy is None:
            args.anysplat_keyframe_strategy = "sfm-registered"
        if args.anysplat_keyframe_orientation_weight is None:
            args.anysplat_keyframe_orientation_weight = 0.30
        if args.sfm_camera_model is None:
            args.sfm_camera_model = "SIMPLE_RADIAL"
        if args.sfm_max_image_size is None or int(args.sfm_max_image_size) <= 0:
            args.sfm_max_image_size = 1600
        if args.sfm_feature_threads is None or int(args.sfm_feature_threads) <= 0:
            args.sfm_feature_threads = min(8, max(2, (os.cpu_count() or 2) // 2))
        if args.sfm_matcher_threads is None or int(args.sfm_matcher_threads) <= 0:
            args.sfm_matcher_threads = min(8, max(2, (os.cpu_count() or 2) // 2))
        args.sfm_sequential_overlap = max(int(args.sfm_sequential_overlap), 20)
        args.sfm_sift_max_num_features = max(int(args.sfm_sift_max_num_features), 16_384)
        if args.sfm_sift_peak_threshold is None:
            args.sfm_sift_peak_threshold = 0.003
    if args.anysplat_keyframe_strategy is None:
        args.anysplat_keyframe_strategy = "quality-diversity"
    if args.anysplat_keyframe_orientation_weight is None:
        args.anysplat_keyframe_orientation_weight = 0.0


def apply_quality_preset_arg_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "require_colmap_dense": False,
        "colmap_dense": False,
        "sfm_camera_model": None,
        "sfm_max_image_size": None,
        "sfm_feature_threads": None,
        "sfm_matcher_threads": None,
        "sfm_sequential_overlap": 10,
        "sfm_sift_max_num_features": 8192,
        "sfm_sift_peak_threshold": None,
        "colmap_dense_max_image_size": 960,
        "colmap_dense_min_fused_points": 100000,
        "pose_init_train_steps": 200,
        "pose_init_train_max_frames": 16,
        "pose_init_train_image_max_size": 448,
        "pose_init_train_target_gaussians": 50000,
        "pose_init_train_max_points": 80000,
        "pose_init_train_dense_image_points_per_frame": 1600,
        "pose_init_train_adaptive_densify_every": 0,
        "pose_init_train_adaptive_densify_stop": 0,
        "pose_init_train_adaptive_densify_points": 2500,
        "pose_init_train_adaptive_densify_error_percentile": 82.0,
        "anysplat_keyframe_strategy": None,
        "anysplat_keyframe_orientation_weight": None,
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)


def annotate_video_quality_report(path: Path, args: argparse.Namespace) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    report["quality_preset"] = args.quality_preset
    report["frame_fps"] = float(args.frame_fps)
    report["anysplat_max_views"] = int(args.anysplat_max_views)
    report["anysplat_mesh_grid"] = int(args.anysplat_mesh_grid)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_sfm_capture_report(args: argparse.Namespace, frames_dir: Path) -> Path:
    report = args.work_dir / "sfm_capture_report.json"
    command = [
        sys.executable,
        (REPO_ROOT / "scripts" / "real2sim_colmap_capture_report.py").as_posix(),
        "--frames-dir",
        frames_dir.as_posix(),
        "--work-dir",
        (args.work_dir / "colmap_sfm").as_posix(),
        "--output",
        report.as_posix(),
        "--colmap-bin",
        str(args.colmap_bin),
        "--matcher",
        str(args.sfm_matcher),
        "--sequential-overlap",
        str(max(int(args.sfm_sequential_overlap), 1)),
        "--sift-max-num-features",
        str(max(int(args.sfm_sift_max_num_features), 1)),
        "--min-registered-ratio",
        str(args.min_sfm_registered_ratio),
        "--min-sparse-points",
        str(args.min_sfm_points),
    ]
    if args.sfm_camera_model is not None:
        command.extend(["--camera-model", str(args.sfm_camera_model)])
    if args.sfm_max_image_size is not None and int(args.sfm_max_image_size) > 0:
        command.extend(["--max-image-size", str(int(args.sfm_max_image_size))])
    if args.sfm_feature_threads is not None and int(args.sfm_feature_threads) > 0:
        command.extend(["--feature-threads", str(int(args.sfm_feature_threads))])
    if args.sfm_matcher_threads is not None and int(args.sfm_matcher_threads) > 0:
        command.extend(["--matcher-threads", str(int(args.sfm_matcher_threads))])
    if args.sfm_sift_peak_threshold is not None:
        command.extend(["--sift-peak-threshold", str(float(args.sfm_sift_peak_threshold))])
    if args.sfm_guided_matching:
        command.append("--guided-matching")
    if args.require_sfm:
        command.append("--require-pass")
    run_child_command(command)
    return report


def sfm_frames_dir(args: argparse.Namespace, extracted_frames: list[Path], selected_frames: Path) -> Path:
    if args.sfm_frames == "selected":
        return selected_frames
    if extracted_frames:
        return extracted_frames[0].parent
    return selected_frames


def run_capture_acceptance_report(args: argparse.Namespace, sfm_report: Path) -> Path:
    report = args.work_dir / "capture_acceptance_report.json"
    command = [
        sys.executable,
        (REPO_ROOT / "scripts" / "real2sim_capture_acceptance.py").as_posix(),
        "--sfm-report",
        sfm_report.as_posix(),
        "--output-json",
        report.as_posix(),
        "--output-md",
        (args.work_dir / "capture_acceptance_report.md").as_posix(),
        "--min-camera-path-length",
        str(float(args.min_capture_camera_path_length)),
        "--min-camera-planar-coverage-ratio",
        str(float(args.min_capture_camera_planar_coverage_ratio)),
        "--max-camera-step-ratio",
        str(float(args.max_capture_camera_step_ratio)),
    ]
    run_child_command_allowing_exit(command, allowed={0, 2})
    acceptance = json.loads(report.read_text(encoding="utf-8"))
    if args.require_capture_acceptance and not acceptance.get("accepted", False):
        raise SystemExit("capture acceptance gate failed; inspect capture_acceptance_report.json")
    return report


def maybe_reselect_anysplat_keyframes_from_sfm(
    args: argparse.Namespace,
    frames: list[Path],
    output_dir: Path,
    quality_report: Path,
    sfm_report: Path | None,
    fallback_selected: list[Path],
) -> list[Path]:
    if args.anysplat_keyframe_strategy != "sfm-registered" or sfm_report is None:
        return fallback_selected
    if not sfm_report.is_file():
        return fallback_selected
    try:
        report = json.loads(sfm_report.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback_selected
    if not report.get("passed"):
        return fallback_selected
    selected = select_sfm_registered_keyframes(
        frames,
        output_dir,
        sfm_report=sfm_report,
        sfm_report_data=report,
        max_views=max(int(args.anysplat_max_views), 1),
        orientation_weight=max(0.0, min(float(args.anysplat_keyframe_orientation_weight), 1.0)),
        quality_report=quality_report,
        min_quality_score=float(args.min_video_quality),
    )
    return selected or fallback_selected


def attach_capture_acceptance_to_package(package_dir: Path, acceptance_report: Path) -> Path:
    artifact_dir = package_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dest = artifact_dir / "capture_acceptance_report.json"
    dest.write_text(acceptance_report.read_text(encoding="utf-8"), encoding="utf-8")
    source_md = acceptance_report.with_suffix(".md")
    if source_md.is_file():
        (artifact_dir / "capture_acceptance_report.md").write_text(source_md.read_text(encoding="utf-8"), encoding="utf-8")

    import_path = package_dir / "motrix_import.json"
    if import_path.is_file():
        manifest = json.loads(import_path.read_text(encoding="utf-8"))
        report = json.loads(dest.read_text(encoding="utf-8"))
        manifest["capture_acceptance_report"] = dest.relative_to(package_dir).as_posix()
        metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
        manifest.setdefault("verification", {})["capture_acceptance"] = {
            "accepted": report.get("accepted"),
            "decision": report.get("decision"),
            "blockers": report.get("blockers", []),
            "camera_path_length": metrics.get("camera_path_length"),
            "camera_planar_coverage_ratio": metrics.get("camera_planar_coverage_ratio"),
            "camera_max_step_ratio": metrics.get("camera_max_step_ratio"),
        }
        import_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dest


def attach_sfm_report_to_package(package_dir: Path, sfm_report: Path) -> Path:
    artifact_dir = package_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dest = artifact_dir / "sfm_capture_report.json"
    report = json.loads(sfm_report.read_text(encoding="utf-8"))
    coverage_dest = copy_sfm_coverage_preview(report, artifact_dir, package_dir)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    import_path = package_dir / "motrix_import.json"
    if import_path.is_file():
        manifest = json.loads(import_path.read_text(encoding="utf-8"))
        manifest["sfm_capture_report"] = dest.relative_to(package_dir).as_posix()
        if coverage_dest is not None:
            manifest["sfm_capture_coverage_preview"] = coverage_dest.relative_to(package_dir).as_posix()
        manifest.setdefault("verification", {})["sfm_capture"] = {
            "passed": report.get("passed"),
            "frame_count": report.get("frame_count"),
            "registered_image_count": report.get("registered_image_count"),
            "registered_ratio": report.get("registered_ratio"),
            "sparse_point_count": report.get("sparse_point_count"),
            "camera_pose_count": (report.get("camera_path") or {}).get("pose_count"),
            "camera_path_length": (report.get("camera_path") or {}).get("path_length"),
            "coverage_preview": coverage_dest.relative_to(package_dir).as_posix() if coverage_dest is not None else None,
            "warnings": report.get("warnings", []),
        }
        import_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dest


def should_run_colmap_dense(args: argparse.Namespace, sfm_report: Path | None) -> bool:
    if not getattr(args, "colmap_dense", False):
        return False
    if sfm_report is None:
        if getattr(args, "require_colmap_dense", False):
            raise SystemExit("--colmap-dense requires --sfm-report")
        return False
    try:
        report = json.loads(sfm_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        if getattr(args, "require_colmap_dense", False):
            raise SystemExit("COLMAP dense reconstruction requires a readable SfM report")
        return False
    if not report.get("passed", False):
        if getattr(args, "require_colmap_dense", False):
            raise SystemExit("COLMAP dense reconstruction requires a passing SfM report")
        return False
    return True


def run_colmap_dense_reconstruction_to_package(
    args: argparse.Namespace,
    package_dir: Path,
    sfm_report: Path | None,
    helpers: Any,
) -> Path:
    if sfm_report is None:
        raise SystemExit("--colmap-dense requires a passing SfM report")
    output_dir = (
        args.work_dir / "colmap_dense"
        if bool(args.colmap_dense_replace_scene_mesh)
        else package_dir / "artifacts" / "colmap_dense"
    )
    report_path = output_dir / "colmap_dense_report.json"
    command = [
        sys.executable,
        (REPO_ROOT / "scripts" / "real2sim_colmap_dense_reconstruct.py").as_posix(),
        "--sfm-report",
        sfm_report.as_posix(),
        "--output-dir",
        output_dir.as_posix(),
        "--report",
        report_path.as_posix(),
        "--colmap-bin",
        str(args.colmap_dense_bin or args.colmap_bin),
        "--max-image-size",
        str(max(int(args.colmap_dense_max_image_size), 1)),
        "--min-fused-points",
        str(max(int(args.colmap_dense_min_fused_points), 1)),
        "--poisson-depth",
        str(max(int(args.colmap_dense_poisson_depth), 5)),
    ]
    if str(args.colmap_dense_gpu_index).strip():
        command.extend(["--gpu-index", str(args.colmap_dense_gpu_index).strip()])
    if bool(args.colmap_dense_run_poisson):
        command.append("--run-poisson")
    if bool(args.require_colmap_dense):
        command.append("--require-pass")
    run_child_command(command)
    attach_colmap_dense_preview(report_path, package_dir, helpers)
    if bool(args.colmap_dense_replace_scene_mesh):
        replace_package_scene_mesh_with_colmap_dense(args, package_dir, report_path)
        report_path = materialize_colmap_dense_package_artifacts(package_dir, report_path)
    attach_colmap_dense_to_package(package_dir, report_path)
    return report_path


def attach_colmap_dense_preview(report_path: Path, package_dir: Path, helpers: Any) -> None:
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fused_value = report.get("fused_ply")
    if not fused_value:
        return
    fused = Path(str(fused_value))
    if not fused.is_file():
        return
    preview = report_path.parent / "fused_projection_preview.png"
    try:
        helpers.write_projection_preview(fused, preview)
        report["projection_preview"] = preview.as_posix()
    except (Exception, SystemExit) as exc:
        report.setdefault("warnings", []).append(f"failed to write dense projection preview: {exc}")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def replace_package_scene_mesh_with_colmap_dense(
    args: argparse.Namespace,
    package_dir: Path,
    report_path: Path,
) -> None:
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mesh_value = report.get("mesh_ply") or report.get("fused_ply")
    if not mesh_value:
        report.setdefault("warnings", []).append("dense scene mesh replacement skipped: no dense mesh/fused PLY")
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return
    mesh_ply = Path(str(mesh_value))
    if not mesh_ply.is_file():
        report.setdefault("warnings", []).append(f"dense scene mesh replacement skipped: missing {mesh_ply}")
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return

    staging = args.work_dir / "colmap_dense_repackage"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((package_dir / "motrix_import.json").read_text(encoding="utf-8"))
    package_manifest = json.loads((package_dir / "package_manifest.json").read_text(encoding="utf-8"))

    scene_ply = copy_package_relative(package_dir, manifest["scene_gaussians"]["scene"], staging / "scene.ply")
    camera_artifacts = {
        key: copy_package_relative(package_dir, value, staging / Path(value).name)
        for key, value in (manifest.get("camera_artifacts") or {}).items()
        if value
    }
    preview = copy_package_relative(package_dir, manifest.get("preview"), staging / "source_gs_preview.png")
    source = Path(str(package_manifest.get("source") or ""))
    if not source.is_file():
        source = camera_artifacts.get("background_image") or preview or scene_ply

    dense_obj = staging / "scene_colmap_dense.obj"
    dense_obj_report = staging / "scene_colmap_dense_obj_report.json"
    run_child_command(
        [
            sys.executable,
            (REPO_ROOT / "scripts" / "real2sim_ply_mesh_to_obj.py").as_posix(),
            "--input-ply",
            mesh_ply.as_posix(),
            "--output-obj",
            dense_obj.as_posix(),
            "--report",
            dense_obj_report.as_posix(),
        ]
    )
    dense_obj = maybe_cleanup_colmap_dense_obj(args, dense_obj, staging, report)
    scene_structure_report = maybe_write_scene_structure_report(dense_obj, staging, report)
    scene_mesh_for_package = maybe_texture_bake_colmap_dense_obj(args, dense_obj, staging, report)

    package_command = [
        sys.executable,
        "-m",
        "real2sim",
        "package-existing",
        "--source",
        source.as_posix(),
        "--scene-ply",
        scene_ply.as_posix(),
        "--scene-mesh",
        scene_mesh_for_package.as_posix(),
        "--output",
        package_dir.as_posix(),
        "--prompt",
        args.scene_prompt or args.prompt,
    ]
    optional_args = {
        "--cameras-json": camera_artifacts.get("cameras_json"),
        "--depth-map": camera_artifacts.get("depth_map"),
        "--background-image": camera_artifacts.get("background_image"),
        "--preview-image": preview,
    }
    for flag, path in optional_args.items():
        if path is not None and path.is_file():
            package_command.extend([flag, path.as_posix()])
    if report.get("mesh_cleanup_report"):
        package_command.extend(["--mesh-cleanup-report", str(report["mesh_cleanup_report"])])
    if report.get("texture_bake_report"):
        package_command.extend(["--texture-bake-report", str(report["texture_bake_report"])])
    if scene_structure_report is not None:
        package_command.extend(["--scene-structure-report", scene_structure_report.as_posix()])
    run_child_command(package_command, env=motrix_package_env(args))

    obj_report = json.loads(dense_obj_report.read_text(encoding="utf-8"))
    report["scene_mesh_replacement"] = {
        "enabled": True,
        "mesh_source": mesh_ply.as_posix(),
        "obj_report": obj_report,
        "mesh_cleanup_report": report.get("mesh_cleanup_report"),
        "scene_structure_report": report.get("scene_structure_report"),
        "texture_bake_report": report.get("texture_bake_report"),
        "package_scene_mesh": "meshes/scene.obj",
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def maybe_write_scene_structure_report(
    dense_obj: Path,
    staging: Path,
    report: dict[str, Any],
) -> Path | None:
    structure_report = staging / "scene_structure_report.json"
    command = [
        sys.executable,
        (REPO_ROOT / "scripts" / "real2sim_obj_structure_report.py").as_posix(),
        "--input-obj",
        dense_obj.as_posix(),
        "--report",
        structure_report.as_posix(),
    ]
    try:
        run_child_command(command)
    except (Exception, SystemExit) as exc:
        report.setdefault("warnings", []).append(f"scene structure report failed: {exc}")
        return None
    if not structure_report.is_file():
        report.setdefault("warnings", []).append("scene structure report did not produce an output file")
        return None
    data = json.loads(structure_report.read_text(encoding="utf-8"))
    report["scene_structure_report"] = structure_report.as_posix()
    report["scene_structure_summary"] = {
        "format": data.get("format"),
        "structure_score": data.get("structure_score"),
        "support_area_ratio": (data.get("support_planes") or {}).get("support_area_ratio")
        if isinstance(data.get("support_planes"), dict)
        else None,
    }
    return structure_report


def maybe_cleanup_colmap_dense_obj(
    args: argparse.Namespace,
    dense_obj: Path,
    staging: Path,
    report: dict[str, Any],
) -> Path:
    max_face_percentile = float(getattr(args, "colmap_dense_cleanup_max_face_area_percentile", 100.0) or 100.0)
    max_edge_percentile = float(getattr(args, "colmap_dense_cleanup_max_edge_length_percentile", 100.0) or 100.0)
    max_centroid_z_percentile = float(getattr(args, "colmap_dense_cleanup_max_centroid_z_percentile", 100.0) or 100.0)
    min_faces = int(getattr(args, "colmap_dense_cleanup_min_faces", 0) or 0)
    min_area_ratio = float(getattr(args, "colmap_dense_cleanup_min_area_ratio", 0.0) or 0.0)
    keep_largest = int(getattr(args, "colmap_dense_cleanup_keep_largest", 1) or 0)
    cleanup_requested = (
        max_face_percentile < 100.0
        or max_edge_percentile < 100.0
        or max_centroid_z_percentile < 100.0
        or min_faces > 0
        or min_area_ratio > 0.0
        or keep_largest != 1
    )
    if not cleanup_requested:
        return dense_obj
    cleaned_obj = staging / "scene_colmap_dense_cleaned.obj"
    cleanup_report = staging / "scene_colmap_dense_cleanup_report.json"
    command = [
        sys.executable,
        (REPO_ROOT / "scripts" / "real2sim_obj_mesh_cleanup.py").as_posix(),
        "--input-obj",
        dense_obj.as_posix(),
        "--output-obj",
        cleaned_obj.as_posix(),
        "--report",
        cleanup_report.as_posix(),
        "--keep-largest",
        str(max(keep_largest, 0)),
        "--min-faces",
        str(max(min_faces, 0)),
        "--min-area-ratio",
        str(max(min_area_ratio, 0.0)),
        "--max-face-area-percentile",
        str(max_face_percentile),
        "--max-edge-length-percentile",
        str(max_edge_percentile),
        "--max-centroid-z-percentile",
        str(max_centroid_z_percentile),
    ]
    try:
        run_child_command(command)
    except (Exception, SystemExit) as exc:
        message = f"COLMAP dense OBJ cleanup failed: {exc}"
        report.setdefault("warnings", []).append(message)
        return dense_obj
    if not cleaned_obj.is_file() or not cleanup_report.is_file():
        report.setdefault("warnings", []).append("COLMAP dense OBJ cleanup produced incomplete outputs")
        return dense_obj
    cleanup_data = json.loads(cleanup_report.read_text(encoding="utf-8"))
    report["mesh_cleanup_report"] = cleanup_report.as_posix()
    report["mesh_cleanup_summary"] = {
        "format": cleanup_data.get("format"),
        "input_face_count": cleanup_data.get("input_face_count"),
        "kept_face_count": cleanup_data.get("kept_face_count"),
        "removed_face_count": cleanup_data.get("removed_face_count"),
        "centroid_filter_removed_face_count": cleanup_data.get("centroid_filter_removed_face_count"),
        "kept_area_ratio": cleanup_data.get("kept_area_ratio"),
    }
    return cleaned_obj


def motrix_package_env(args: argparse.Namespace) -> dict[str, str] | None:
    env = os.environ.copy()
    changed = False
    preview_faces = int(getattr(args, "motrix_preview_mesh_faces", 0) or 0)
    if preview_faces > 0:
        env["REAL2SIM_MOTRIX_PREVIEW_MESH_FACES"] = str(preview_faces)
        changed = True
    preview_mode = getattr(args, "motrix_preview_mode", None)
    if preview_mode:
        env["REAL2SIM_MOTRIX_PREVIEW_MODE"] = str(preview_mode)
        changed = True
    return env if changed else None


def maybe_texture_bake_colmap_dense_obj(
    args: argparse.Namespace,
    dense_obj: Path,
    staging: Path,
    report: dict[str, Any],
) -> Path:
    if not bool(getattr(args, "colmap_texture_bake", False)):
        return dense_obj
    sfm_report = Path(str(report.get("sfm_report") or ""))
    if not sfm_report.is_file():
        message = "COLMAP texture bake skipped: dense report has no readable sfm_report"
        report.setdefault("warnings", []).append(message)
        if bool(getattr(args, "require_colmap_texture_bake", False)):
            raise SystemExit(message)
        return dense_obj
    textured_obj = staging / "scene_colmap_dense_textured.obj"
    textured_png = staging / "scene_colmap_dense_texture.png"
    texture_report = staging / "scene_colmap_dense_texture_bake_report.json"
    command = [
        sys.executable,
        (REPO_ROOT / "scripts" / "real2sim_colmap_texture_bake_obj.py").as_posix(),
        "--input-obj",
        dense_obj.as_posix(),
        "--output-obj",
        textured_obj.as_posix(),
        "--output-texture",
        textured_png.as_posix(),
        "--report",
        texture_report.as_posix(),
        "--sfm-report",
        sfm_report.as_posix(),
        "--atlas-size",
        str(max(int(args.colmap_texture_bake_atlas_size), 64)),
        "--max-cameras",
        str(max(int(args.colmap_texture_bake_max_cameras), 1)),
        "--max-faces",
        str(max(int(args.colmap_texture_bake_max_faces), 0)),
        "--max-output-faces",
        str(max(int(args.colmap_texture_bake_max_output_faces), 0)),
        "--texture-fill-mode",
        args.colmap_texture_bake_fill_mode,
        "--color-blend-cameras",
        str(max(int(args.colmap_texture_bake_color_blend_cameras), 1)),
        "--visibility-depth-tolerance",
        str(max(float(args.colmap_texture_bake_visibility_depth_tolerance), 0.0)),
        "--visibility-grid-size",
        str(max(int(args.colmap_texture_bake_visibility_grid_size), 32)),
        "--max-warped-faces",
        str(max(int(args.colmap_texture_bake_max_warped_faces), 0)),
        "--postprocess-median-radius",
        str(max(int(args.colmap_texture_bake_postprocess_median_radius), 0)),
        "--bleed-iterations",
        str(max(int(args.colmap_texture_bake_bleed_iterations), 0)),
    ]
    if bool(getattr(args, "colmap_texture_bake_unwrap_uv", False)):
        command.append("--unwrap-uv")
    if bool(getattr(args, "require_colmap_texture_bake_unwrap_uv", False)):
        command.append("--require-unwrap-uv")
    try:
        run_child_command(command)
    except (Exception, SystemExit) as exc:
        message = f"COLMAP texture bake failed: {exc}"
        report.setdefault("warnings", []).append(message)
        if bool(getattr(args, "require_colmap_texture_bake", False)):
            raise
        return dense_obj
    if not textured_obj.is_file() or not textured_png.is_file() or not texture_report.is_file():
        message = "COLMAP texture bake produced incomplete OBJ/texture/report outputs"
        report.setdefault("warnings", []).append(message)
        if bool(getattr(args, "require_colmap_texture_bake", False)):
            raise SystemExit(message)
        return dense_obj
    texture_data = json.loads(texture_report.read_text(encoding="utf-8"))
    report["texture_bake_report"] = texture_report.as_posix()
    report["texture_baked_obj"] = textured_obj.as_posix()
    report["texture_baked_texture"] = textured_png.as_posix()
    report["texture_bake_summary"] = {
        "format": texture_data.get("format"),
        "projected_face_count": texture_data.get("projected_face_count"),
        "drawn_face_count": texture_data.get("drawn_face_count"),
        "texture_coverage_ratio": texture_data.get("texture_coverage_ratio"),
        "texture_fill_mode": texture_data.get("texture_fill_mode"),
        "warped_face_count": texture_data.get("warped_face_count"),
        "warp_fallback_face_count": texture_data.get("warp_fallback_face_count"),
        "texture_postprocess": texture_data.get("texture_postprocess"),
        "atlas_size": texture_data.get("atlas_size"),
        "selected_camera_count": texture_data.get("selected_camera_count"),
        "selected_face_count": texture_data.get("selected_face_count"),
    }
    return textured_obj


def copy_package_relative(package_dir: Path, value: Any, dest: Path) -> Path | None:
    if not value:
        return None
    source = package_dir / str(value)
    if not source.is_file():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest


def materialize_colmap_dense_package_artifacts(package_dir: Path, report_path: Path) -> Path:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dest_dir = package_dir / "artifacts" / "colmap_dense"
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, str] = {}
    for key, name in (
        ("fused_ply", "fused.ply"),
        ("mesh_ply", "mesh_poisson.ply"),
        ("projection_preview", "fused_projection_preview.png"),
        ("texture_bake_report", "texture_bake_report.json"),
        ("texture_baked_texture", "scene_texture_bake.png"),
    ):
        value = report.get(key)
        if not value:
            continue
        source = Path(str(value))
        if not source.is_file():
            continue
        dest = dest_dir / name
        shutil.copy2(source, dest)
        copied[key] = dest.as_posix()
    report.update(copied)
    report["output_dir"] = dest_dir.as_posix()
    report["package_artifacts_are_summary_only"] = True
    packaged_report = dest_dir / "colmap_dense_report.json"
    packaged_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return packaged_report


def attach_colmap_dense_to_package(package_dir: Path, report_path: Path) -> None:
    import_path = package_dir / "motrix_import.json"
    if not import_path.is_file() or not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(import_path.read_text(encoding="utf-8"))
    relative_root = report_path.parent.relative_to(package_dir).as_posix()
    manifest["colmap_dense_reconstruction"] = {
        "root": relative_root,
        "report": report_path.relative_to(package_dir).as_posix(),
        "fused_ply": relative_package_file(package_dir, report.get("fused_ply")),
        "mesh_ply": relative_package_file(package_dir, report.get("mesh_ply")),
        "projection_preview": relative_package_file(package_dir, report.get("projection_preview")),
        "texture_bake_report": relative_package_file(package_dir, report.get("texture_bake_report")),
        "texture_baked_texture": relative_package_file(package_dir, report.get("texture_baked_texture")),
    }
    manifest.setdefault("verification", {})["colmap_dense"] = {
        "format": report.get("format"),
        "passed": report.get("passed"),
        "fused_point_count": report.get("fused_point_count"),
        "mesh_vertex_count": report.get("mesh_vertex_count"),
        "mesh_face_count": report.get("mesh_face_count"),
        "max_image_size": report.get("max_image_size"),
        "min_fused_points": report.get("min_fused_points"),
        "scene_mesh_replacement": report.get("scene_mesh_replacement"),
        "texture_bake_summary": report.get("texture_bake_summary"),
        "warnings": report.get("warnings", []),
        "projection_preview": relative_package_file(package_dir, report.get("projection_preview")),
    }
    import_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    append_colmap_dense_readme(package_dir, manifest["colmap_dense_reconstruction"])


def relative_package_file(package_dir: Path, value: Any) -> str | None:
    if value is None:
        return None
    path = Path(str(value))
    if not path.exists():
        return None
    try:
        return path.resolve().relative_to(package_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def append_colmap_dense_readme(package_dir: Path, dense_manifest: dict[str, Any]) -> None:
    readme = package_dir / "MOTRIX_IMPORT.md"
    if not readme.is_file():
        return
    lines = [
        "",
        "COLMAP dense MVS artifacts:",
        "",
        f"- Report: `{dense_manifest.get('report')}`",
        f"- Fused point cloud: `{dense_manifest.get('fused_ply')}`",
        f"- Poisson mesh: `{dense_manifest.get('mesh_ply')}`",
        f"- Projection preview: `{dense_manifest.get('projection_preview')}`",
        "",
    ]
    with readme.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def should_export_pose_init_dataset(args: argparse.Namespace, sfm_report: Path | None) -> bool:
    if sfm_report is None or bool(getattr(args, "skip_pose_init_dataset", False)):
        return False
    try:
        report = json.loads(sfm_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(report.get("passed"))


def export_pose_init_dataset_to_package(sfm_report: Path, package_dir: Path) -> Path:
    output_dir = package_dir / "artifacts" / "pose_init_dataset"
    command = [
        sys.executable,
        (REPO_ROOT / "scripts" / "real2sim_export_pose_init_dataset.py").as_posix(),
        "--sfm-report",
        sfm_report.as_posix(),
        "--output-dir",
        output_dir.as_posix(),
    ]
    run_child_command(command)
    attach_pose_init_dataset_to_manifest(package_dir, output_dir)
    return output_dir


def run_pose_init_training_to_package(args: argparse.Namespace, package_dir: Path, pose_init_dir: Path | None) -> Path | None:
    if not getattr(args, "pose_init_train", False):
        return None
    if pose_init_dir is None:
        raise SystemExit("--pose-init-train requires a passing SfM report and exported pose-init dataset")
    python_bin = str(args.pose_init_train_python or sys.executable)
    output_dir = package_dir / "artifacts" / "pose_init_trained"
    output_ply = output_dir / "pose_init_trained_scene.ply"
    preview = output_dir / "pose_init_training_preview.png"
    multi_preview = output_dir / "pose_init_training_contact_sheet.png"
    metrics = output_dir / "pose_init_training_metrics.json"
    command = [
        python_bin,
        (REPO_ROOT / "scripts" / "real2sim_pose_init_gsplat_train.py").as_posix(),
        "--pose-init-dir",
        pose_init_dir.as_posix(),
        "--output-dir",
        output_dir.as_posix(),
        "--output-ply",
        output_ply.as_posix(),
        "--preview-output",
        preview.as_posix(),
        "--multi-preview-output",
        multi_preview.as_posix(),
        "--metrics-output",
        metrics.as_posix(),
        "--max-steps",
        str(max(int(args.pose_init_train_steps), 1)),
        "--max-frames",
        str(max(int(args.pose_init_train_max_frames), 1)),
        "--image-max-size",
        str(max(int(args.pose_init_train_image_max_size), 1)),
        "--target-gaussians",
        str(max(int(args.pose_init_train_target_gaussians), 1)),
        "--max-points",
        str(max(int(args.pose_init_train_max_points), 1)),
        "--dense-image-points-per-frame",
        str(max(int(args.pose_init_train_dense_image_points_per_frame), 1)),
        "--dense-depth-neighbors",
        str(max(int(args.pose_init_train_dense_depth_neighbors), 1)),
        "--dense-depth-max-distance",
        str(max(float(args.pose_init_train_dense_depth_max_distance), 1.0)),
        "--adaptive-densify-every",
        str(max(int(args.pose_init_train_adaptive_densify_every), 0)),
        "--adaptive-densify-start",
        str(max(int(args.pose_init_train_adaptive_densify_start), 1)),
        "--adaptive-densify-stop",
        str(max(int(args.pose_init_train_adaptive_densify_stop), 0)),
        "--adaptive-densify-points",
        str(max(int(args.pose_init_train_adaptive_densify_points), 1)),
        "--adaptive-densify-error-percentile",
        str(max(min(float(args.pose_init_train_adaptive_densify_error_percentile), 99.0), 1.0)),
        "--preview-frames",
        str(max(int(args.pose_init_train_preview_frames), 1)),
    ]
    run_child_command(command)
    if args.pose_init_train_replace_scene:
        scene_ply = package_dir / "3dgs" / "scene.ply"
        backup = output_dir / "scene_before_pose_init_train.ply"
        if scene_ply.is_file():
            shutil.copy2(scene_ply, backup)
        shutil.copy2(output_ply, scene_ply)
    attach_pose_init_training_to_manifest(
        package_dir,
        output_dir,
        replace_scene=bool(args.pose_init_train_replace_scene),
    )
    return metrics


def attach_pose_init_dataset_to_manifest(package_dir: Path, pose_init_dir: Path) -> None:
    import_path = package_dir / "motrix_import.json"
    summary_path = pose_init_dir / "pose_init_dataset.json"
    if not import_path.is_file() or not summary_path.is_file():
        return
    manifest = json.loads(import_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    relative_root = pose_init_dir.relative_to(package_dir).as_posix()
    manifest["pose_init_dataset"] = relative_root
    manifest.setdefault("verification", {})["pose_init_dataset"] = {
        "format": summary.get("format"),
        "root": relative_root,
        "transforms": f"{relative_root}/{summary.get('transforms', 'transforms.json')}",
        "images": f"{relative_root}/{summary.get('images', 'images')}",
        "image_count": summary.get("image_count"),
        "sparse_points_ply": f"{relative_root}/{summary.get('sparse_points_ply')}"
        if summary.get("sparse_points_ply")
        else None,
        "sparse_point_count": summary.get("sparse_point_count"),
        "registered_ratio": summary.get("registered_ratio"),
        "ready_for_training": summary.get("ready_for_training"),
    }
    import_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def attach_pose_init_training_to_manifest(package_dir: Path, training_dir: Path, *, replace_scene: bool) -> None:
    import_path = package_dir / "motrix_import.json"
    metrics_path = training_dir / "pose_init_training_metrics.json"
    output_ply = training_dir / "pose_init_trained_scene.ply"
    preview = training_dir / "pose_init_training_preview.png"
    multi_preview = training_dir / "pose_init_training_contact_sheet.png"
    if not import_path.is_file() or not metrics_path.is_file():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(import_path.read_text(encoding="utf-8"))
    relative_root = training_dir.relative_to(package_dir).as_posix()
    manifest["pose_init_training"] = {
        "root": relative_root,
        "scene_ply": output_ply.relative_to(package_dir).as_posix() if output_ply.is_file() else None,
        "preview": preview.relative_to(package_dir).as_posix() if preview.is_file() else None,
        "multi_preview": multi_preview.relative_to(package_dir).as_posix() if multi_preview.is_file() else None,
        "metrics": metrics_path.relative_to(package_dir).as_posix(),
        "replaced_package_scene": replace_scene,
    }
    manifest.setdefault("verification", {})["pose_init_training"] = {
        "format": metrics.get("format"),
        "trained_gaussian_count": metrics.get("trained_gaussian_count"),
        "frame_count": metrics.get("frame_count"),
        "max_steps": metrics.get("max_steps"),
        "final_loss": metrics.get("final_loss"),
        "final_preview_psnr": metrics.get("final_preview_psnr"),
        "final_preview_mae": metrics.get("final_preview_mae"),
        "mean_step_psnr_last_10": metrics.get("mean_step_psnr_last_10"),
        "mean_step_mae_last_10": metrics.get("mean_step_mae_last_10"),
        "multi_preview_mean_psnr": metrics.get("multi_preview_mean_psnr"),
        "multi_preview_mean_mae": metrics.get("multi_preview_mean_mae"),
        "adaptive_densify": {
            "enabled": (metrics.get("adaptive_densify") or {}).get("enabled"),
            "event_count": (metrics.get("adaptive_densify") or {}).get("event_count"),
            "added_point_count": (metrics.get("adaptive_densify") or {}).get("added_point_count"),
        },
    }
    import_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_sfm_coverage_preview(report: dict[str, Any], artifact_dir: Path, package_dir: Path) -> Path | None:
    coverage_value = report.get("coverage_preview")
    if not coverage_value:
        return None
    source = Path(str(coverage_value))
    if not source.is_file():
        return None
    dest = artifact_dir / "sfm_capture_coverage.png"
    shutil.copy2(source, dest)
    report["coverage_preview"] = dest.relative_to(package_dir).as_posix()
    return dest


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def extract_video_frames(video: Path, frames_dir: Path, *, fps: float) -> list[Path]:
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("video workflow requires ffmpeg")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video.as_posix(),
        "-vf",
        f"fps={fps:g}",
        (frames_dir / "frame_%06d.png").as_posix(),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"ffmpeg frame extraction failed with exit code {completed.returncode}")
    frames = sorted(path for path in frames_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not frames:
        raise SystemExit(f"ffmpeg produced no frames in {frames_dir}")
    return frames


def collect_existing_frames(frames_dir: Path) -> list[Path]:
    if not frames_dir.is_dir():
        raise SystemExit(f"frames directory not found: {frames_dir}")
    frames = sorted(path for path in frames_dir.iterdir() if is_video_frame_image(path))
    if not frames:
        raise SystemExit(f"frames directory contains no supported images: {frames_dir}")
    return frames


def normalize_existing_frames(frames: list[Path], output_dir: Path) -> list[Path]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized: list[Path] = []
    for index, src in enumerate(frames, start=1):
        dst = output_dir / f"frame_{index:06d}.png"
        write_normalized_frame(src, dst)
        normalized.append(dst)
    return normalized


def is_video_frame_image(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    name = path.name.lower()
    return "contact_sheet" not in name and "preview" not in name


def select_video_keyframes(
    frames: list[Path],
    output_dir: Path,
    *,
    max_views: int,
    quality_report: Path,
    min_quality_score: float,
) -> list[Path]:
    if not frames:
        raise SystemExit("no video frames available for keyframe selection")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_metrics = {index: frame_quality(frames[index]) for index in range(len(frames))}
    selected_indices = select_diverse_keyframe_indices(frames, frame_metrics, max_views=max_views)
    selected_paths: list[Path] = []
    for output_index, source_index in enumerate(selected_indices, start=1):
        src = frames[source_index]
        dst = output_dir / f"frame_{output_index:06d}.png"
        write_normalized_frame(src, dst)
        selected_paths.append(dst)
    contact_sheet = write_keyframe_contact_sheet(selected_paths, output_dir / "selected_keyframes_contact_sheet.png")

    selected_metrics = [frame_metrics[index] for index in selected_indices]
    diversity = selected_frame_diversity(selected_paths)
    mean_sharpness = float(np.mean([metric["sharpness"] for metric in selected_metrics]))
    mean_exposure = float(np.mean([metric["exposure"] for metric in selected_metrics]))
    frame_count_score = min(1.0, len(selected_paths) / max(float(max_views), 1.0))
    diversity_score = max(0.0, min(1.0, diversity / 0.10))
    score = 0.26 * frame_count_score + 0.28 * mean_sharpness + 0.22 * mean_exposure + 0.24 * diversity_score
    warnings = video_quality_warnings(
        frame_count=len(selected_paths),
        requested_views=max_views,
        mean_sharpness=mean_sharpness,
        mean_exposure=mean_exposure,
        diversity=diversity,
    )
    report = {
        "passed": float(score) >= min_quality_score,
        "score": round(float(score), 4),
        "min_quality_score": min_quality_score,
        "source_frame_count": len(frames),
        "selected_frame_count": len(selected_paths),
        "requested_views": max_views,
        "frame_fusion": "video_keyframes",
        "selection_strategy": "coverage_anchored_quality_diversity_v1",
        "mean_sharpness": round(mean_sharpness, 4),
        "mean_exposure": round(mean_exposure, 4),
        "mean_temporal_diversity": round(float(diversity), 4),
        "selected_source_indices": selected_indices,
        "selected_frames": [path.as_posix() for path in selected_paths],
        "contact_sheet": contact_sheet.as_posix(),
        "warnings": warnings,
    }
    quality_report.parent.mkdir(parents=True, exist_ok=True)
    quality_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return selected_paths


def select_sfm_registered_keyframes(
    frames: list[Path],
    output_dir: Path,
    *,
    sfm_report: Path,
    sfm_report_data: dict[str, Any],
    max_views: int,
    orientation_weight: float,
    quality_report: Path,
    min_quality_score: float,
) -> list[Path]:
    frame_by_name = {path.name: path for path in frames}
    frame_index_by_name = {path.name: index for index, path in enumerate(frames)}
    ordered_names = sfm_registered_pose_names(sfm_report_data, frame_by_name)
    if not ordered_names:
        return []
    target_count = min(max_views, len(ordered_names))
    if target_count <= 0:
        return []

    frame_metrics = {
        frame_index_by_name[name]: frame_quality(frame_by_name[name])
        for name in ordered_names
        if name in frame_index_by_name
    }
    pose_alignment = sfm_camera_up_alignment(sfm_report_data)
    selected_names = select_pose_binned_registered_names(
        ordered_names,
        frame_index_by_name=frame_index_by_name,
        frame_metrics=frame_metrics,
        pose_alignment=pose_alignment,
        orientation_weight=orientation_weight,
        target_count=target_count,
    )
    if not selected_names:
        return []

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_paths: list[Path] = []
    selected_indices: list[int] = []
    for output_index, name in enumerate(selected_names, start=1):
        src = frame_by_name[name]
        dst = output_dir / f"frame_{output_index:06d}.png"
        write_normalized_frame(src, dst)
        selected_paths.append(dst)
        selected_indices.append(frame_index_by_name[name])
    contact_sheet = write_keyframe_contact_sheet(selected_paths, output_dir / "selected_keyframes_contact_sheet.png")

    selected_metrics = [frame_metrics[index] for index in selected_indices if index in frame_metrics]
    diversity = selected_frame_diversity(selected_paths)
    mean_sharpness = float(np.mean([metric["sharpness"] for metric in selected_metrics])) if selected_metrics else 0.0
    mean_exposure = float(np.mean([metric["exposure"] for metric in selected_metrics])) if selected_metrics else 0.0
    frame_count_score = min(1.0, len(selected_paths) / max(float(max_views), 1.0))
    diversity_score = max(0.0, min(1.0, diversity / 0.10))
    registered_ratio = float(sfm_report_data.get("registered_ratio") or 0.0)
    selected_alignment = [pose_alignment.get(name, 0.0) for name in selected_names]
    score = 0.22 * frame_count_score + 0.20 * mean_sharpness + 0.18 * mean_exposure + 0.20 * diversity_score + 0.20 * min(registered_ratio, 1.0)
    warnings = video_quality_warnings(
        frame_count=len(selected_paths),
        requested_views=max_views,
        mean_sharpness=mean_sharpness,
        mean_exposure=mean_exposure,
        diversity=diversity,
    )
    report = {
        "passed": float(score) >= min_quality_score,
        "score": round(float(score), 4),
        "min_quality_score": min_quality_score,
        "source_frame_count": len(frames),
        "registered_frame_count": len(ordered_names),
        "selected_frame_count": len(selected_paths),
        "requested_views": max_views,
        "frame_fusion": "image_set_sfm_registered_keyframes",
        "selection_strategy": "sfm_registered_pose_bins_v1",
        "source_sfm_report": sfm_report.as_posix(),
        "registered_ratio": round(registered_ratio, 4),
        "orientation_weight": round(float(orientation_weight), 4),
        "mean_pose_up_alignment": round(float(np.mean(selected_alignment)), 4) if selected_alignment else None,
        "min_pose_up_alignment": round(float(np.min(selected_alignment)), 4) if selected_alignment else None,
        "mean_sharpness": round(mean_sharpness, 4),
        "mean_exposure": round(mean_exposure, 4),
        "mean_temporal_diversity": round(float(diversity), 4),
        "selected_source_indices": selected_indices,
        "selected_registered_images": selected_names,
        "selected_frames": [path.as_posix() for path in selected_paths],
        "contact_sheet": contact_sheet.as_posix(),
        "warnings": warnings,
    }
    quality_report.parent.mkdir(parents=True, exist_ok=True)
    quality_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return selected_paths


def sfm_registered_pose_names(sfm_report_data: dict[str, Any], frame_by_name: dict[str, Path]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for pose in sfm_report_data.get("camera_poses", []):
        if not isinstance(pose, dict):
            continue
        name = str(pose.get("image_name") or "")
        if name in frame_by_name and name not in seen:
            names.append(name)
            seen.add(name)
    if not names:
        for name in sfm_report_data.get("registered_images", []):
            name = str(name)
            if name in frame_by_name and name not in seen:
                names.append(name)
                seen.add(name)
    return sorted(names, key=frame_name_sort_key)


def select_pose_binned_registered_names(
    ordered_names: list[str],
    *,
    frame_index_by_name: dict[str, int],
    frame_metrics: dict[int, dict[str, float]],
    pose_alignment: dict[str, float],
    orientation_weight: float,
    target_count: int,
) -> list[str]:
    if target_count >= len(ordered_names):
        return list(ordered_names)
    selected: list[str] = []
    edges = np.linspace(0, len(ordered_names), target_count + 1, dtype=int)
    for bin_index in range(target_count):
        start = int(edges[bin_index])
        stop = int(edges[bin_index + 1])
        if stop <= start:
            stop = min(start + 1, len(ordered_names))
        candidates = ordered_names[start:stop]
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda name: registered_frame_score(
                name,
                frame_index_by_name,
                frame_metrics,
                pose_alignment=pose_alignment,
                orientation_weight=orientation_weight,
            ),
        )
        if best not in selected:
            selected.append(best)
    if len(selected) < target_count:
        selected_set = set(selected)
        for name in sorted(
            ordered_names,
            key=lambda item: registered_frame_score(
                item,
                frame_index_by_name,
                frame_metrics,
                pose_alignment=pose_alignment,
                orientation_weight=orientation_weight,
            ),
            reverse=True,
        ):
            if name not in selected_set:
                selected.append(name)
                selected_set.add(name)
            if len(selected) >= target_count:
                break
    return sorted(selected, key=frame_name_sort_key)


def registered_frame_score(
    name: str,
    frame_index_by_name: dict[str, int],
    frame_metrics: dict[int, dict[str, float]],
    *,
    pose_alignment: dict[str, float] | None = None,
    orientation_weight: float = 0.0,
) -> float:
    index = frame_index_by_name.get(name)
    if index is None:
        return 0.0
    metrics = frame_metrics.get(index, {})
    base = 0.62 * float(metrics.get("sharpness", 0.0)) + 0.38 * float(metrics.get("exposure", 0.0))
    alignment = float((pose_alignment or {}).get(name, 0.0))
    weight = max(0.0, min(float(orientation_weight), 1.0))
    return (1.0 - weight) * base + weight * alignment


def sfm_camera_up_alignment(sfm_report_data: dict[str, Any]) -> dict[str, float]:
    up_by_name: dict[str, np.ndarray] = {}
    for pose in sfm_report_data.get("camera_poses", []):
        if not isinstance(pose, dict):
            continue
        name = str(pose.get("image_name") or "")
        qvec = pose.get("qvec_wxyz")
        if not name or not isinstance(qvec, list) or len(qvec) != 4:
            continue
        rotation = qvec_wxyz_to_rotation([float(value) for value in qvec])
        up = rotation.T @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
        norm = float(np.linalg.norm(up))
        if norm > 1e-9:
            up_by_name[name] = up / norm
    if not up_by_name:
        return {}
    dominant = np.mean(np.stack(list(up_by_name.values()), axis=0), axis=0)
    dominant_norm = float(np.linalg.norm(dominant))
    if dominant_norm <= 1e-9:
        return {name: 0.0 for name in up_by_name}
    dominant = dominant / dominant_norm
    return {
        name: max(0.0, float(np.dot(up, dominant)))
        for name, up in up_by_name.items()
    }


def qvec_wxyz_to_rotation(qvec: list[float]) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * z * w, 2.0 * x * z + 2.0 * y * w],
            [2.0 * x * y + 2.0 * z * w, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * x * w],
            [2.0 * x * z - 2.0 * y * w, 2.0 * y * z + 2.0 * x * w, 1.0 - 2.0 * x * x - 2.0 * y * y],
        ],
        dtype=np.float64,
    )


def frame_name_sort_key(name: str) -> tuple[int, str]:
    stem = Path(name).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return (int(digits), name)
    return (0, name)


def select_diverse_keyframe_indices(
    frames: list[Path],
    frame_metrics: dict[int, dict[str, float]],
    *,
    max_views: int,
) -> list[int]:
    target_count = min(max_views, len(frames))
    if target_count <= 0:
        return []
    if target_count == len(frames):
        return list(range(len(frames)))

    features = [frame_feature(path) for path in frames]
    base_scores = {
        index: 0.62 * frame_metrics[index]["sharpness"] + 0.38 * frame_metrics[index]["exposure"]
        for index in range(len(frames))
    }
    selected: list[int] = []

    anchor_width = max(1, int(np.ceil(len(frames) / max(float(target_count), 1.0))))
    selected.append(best_frame_in_range(0, anchor_width, base_scores))
    if target_count > 1:
        tail_start = max(0, len(frames) - anchor_width)
        selected.append(best_frame_in_range(tail_start, len(frames), base_scores))

    selected = sorted(dict.fromkeys(selected))
    target_spacing = max(1.0, (len(frames) - 1) / max(float(target_count - 1), 1.0))
    while len(selected) < target_count:
        selected_set = set(selected)
        best_index = None
        best_score = -1.0
        for index in range(len(frames)):
            if index in selected_set:
                continue
            visual_novelty = min(
                frame_feature_distance(features[index], features[selected_index])
                for selected_index in selected
            )
            visual_score = max(0.0, min(1.0, visual_novelty / 0.10))
            temporal_novelty = min(abs(index - selected_index) for selected_index in selected) / target_spacing
            temporal_score = max(0.0, min(1.0, temporal_novelty))
            score = 0.48 * base_scores[index] + 0.34 * visual_score + 0.18 * temporal_score
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
        selected.sort()
    return selected


def best_frame_in_range(start: int, stop: int, base_scores: dict[int, float]) -> int:
    return max(range(start, stop), key=lambda index: base_scores[index])


def frame_feature(path: Path) -> np.ndarray:
    return np.asarray(
        open_rgb_image(path).resize((96, 54), PIL_BILINEAR),
        dtype=np.float32,
    ) / 255.0


def frame_feature_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def write_keyframe_contact_sheet(frames: list[Path], output: Path) -> Path:
    if not frames:
        raise SystemExit("cannot write keyframe contact sheet without frames")
    columns = min(4, len(frames))
    rows = int(np.ceil(len(frames) / columns))
    tile = (320, 180)
    header_h = 26
    canvas = Image.new("RGB", (columns * tile[0], rows * (tile[1] + header_h)), "white")
    for index, path in enumerate(frames):
        image = open_rgb_image(path)
        image.thumbnail(tile)
        col = index % columns
        row = index // columns
        x = col * tile[0] + (tile[0] - image.width) // 2
        y = row * (tile[1] + header_h) + header_h + (tile[1] - image.height) // 2
        canvas.paste(image, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def frame_quality(path: Path) -> dict[str, float]:
    image = open_rgb_image(path)
    gray = np.asarray(image.convert("L").resize((192, 108), PIL_BILINEAR), dtype=np.float32) / 255.0
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    sharpness = max(0.0, min(1.0, float(np.mean(np.abs(gx)) + np.mean(np.abs(gy))) / 0.10))
    clipped = float(np.mean((gray < 0.025) | (gray > 0.975)))
    contrast = float(np.std(gray))
    exposure = max(0.0, min(1.0, (1.0 - clipped * 2.0) * min(contrast / 0.22, 1.0)))
    return {"sharpness": sharpness, "exposure": exposure}


def selected_frame_diversity(frames: list[Path]) -> float:
    if len(frames) < 2:
        return 0.0
    arrays = [
        np.asarray(open_rgb_image(path).resize((128, 72), PIL_BILINEAR), dtype=np.float32) / 255.0
        for path in frames
    ]
    diffs = [float(np.mean(np.abs(arrays[index] - arrays[index - 1]))) for index in range(1, len(arrays))]
    return float(np.mean(diffs))


def open_rgb_image(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def write_normalized_frame(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    open_rgb_image(src).save(dst)


def video_quality_warnings(
    *,
    frame_count: int,
    requested_views: int,
    mean_sharpness: float,
    mean_exposure: float,
    diversity: float,
) -> list[str]:
    warnings: list[str] = []
    if frame_count < min(4, requested_views):
        warnings.append("too few selected video keyframes for scene reconstruction")
    if mean_sharpness < 0.32:
        warnings.append("selected video frames are blurry or low-detail")
    if mean_exposure < 0.32:
        warnings.append("selected video frames have weak contrast or clipped exposure")
    if diversity < 0.025 and frame_count > 1:
        warnings.append("video camera motion may be too small for scene reconstruction")
    return warnings


def attach_video_quality_to_package(package_dir: Path, quality_report: Path) -> Path:
    dest = package_dir / "video_frame_quality.json"
    dest.write_text(quality_report.read_text(encoding="utf-8"), encoding="utf-8")
    import_path = package_dir / "motrix_import.json"
    if import_path.is_file():
        manifest = json.loads(import_path.read_text(encoding="utf-8"))
        manifest["video_frame_quality"] = dest.relative_to(package_dir).as_posix()
        manifest["recommended_camera"] = "source_3dgs_preview"
        manifest["viewer_args"] = source_preview_viewer_args(manifest.get("viewer_args"))
        quality = json.loads(dest.read_text(encoding="utf-8"))
        manifest.setdefault("verification", {})["video_frame_quality"] = {
            "score": quality.get("score"),
            "selected_frame_count": quality.get("selected_frame_count"),
            "source_frame_count": quality.get("source_frame_count"),
            "mean_temporal_diversity": quality.get("mean_temporal_diversity"),
        }
        import_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dest


def source_preview_viewer_args(value: Any) -> list[str]:
    args = [str(item) for item in value] if isinstance(value, list) else []
    if not args:
        args = ["--package", "/path/to/package", "--camera", "overview"]
    args = set_arg_value(args, "--camera", "closeup")
    for item in ("--static-gs-preview", "--show-source-gs", "--source-gs-only"):
        if item not in args:
            args.append(item)
    return args


def set_arg_value(args: list[str], flag: str, value: str) -> list[str]:
    updated = list(args)
    if flag in updated:
        index = updated.index(flag)
        if index + 1 < len(updated):
            updated[index + 1] = value
            return updated
    updated.extend([flag, value])
    return updated


def write_visual_report(
    package_dir: Path,
    output_dir: Path,
    *,
    min_scene_vertices: int,
    min_source_preview_psnr: float,
    max_source_preview_mae: float,
    motrix_viewer_render: Path | None,
) -> dict[str, Any]:
    verifier = load_verifier()
    report = verifier.verify_package(
        package_dir,
        min_scene_vertices=min_scene_vertices,
        min_source_preview_psnr=min_source_preview_psnr,
        max_source_preview_mae=max_source_preview_mae,
    )
    return verifier.write_visual_acceptance_report(
        package_dir,
        report,
        output_dir,
        motrix_viewer_render=motrix_viewer_render,
    )


def load_one_image_helpers():
    return load_module("real2sim_one_image_motrix_scene_helpers", REPO_ROOT / "scripts" / "real2sim_one_image_motrix_scene.py")


def load_verifier():
    return load_module("real2sim_verify_motrix_package_helpers", REPO_ROOT / "scripts" / "real2sim_verify_motrix_package.py")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_child_command(command: list[str], *, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, stdout=sys.stderr, stderr=sys.stderr)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def run_child_command_allowing_exit(command: list[str], *, allowed: set[int], env: dict[str, str] | None = None) -> None:
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, stdout=sys.stderr, stderr=sys.stderr)
    if result.returncode not in allowed:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
