"""Command line interface for GS-Playground Real2Sim."""

from __future__ import annotations

import argparse
from pathlib import Path

from .contracts import write_json
from .paper_strict import check_paper_strict, format_report, validate_paper_strict
from .pipeline import Real2SimPipeline
from .pruning import PruneOptions


REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m real2sim", description="GS-Playground Real2Sim pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full image/video-to-sim-ready-scene pipeline")
    run_parser.add_argument("--input", type=Path, required=True, help="Input RGB image or video")
    run_parser.add_argument("--output", type=Path, required=True, help="Output package directory")
    run_parser.add_argument("--work-dir", type=Path, default=None, help="Intermediate artifact directory")
    run_parser.add_argument("--prompt", default="all task-relevant objects", help="Open-vocabulary detection prompt")
    run_parser.add_argument("--config", type=Path, default=None, help="Real2Sim stage command config JSON")
    run_parser.add_argument("--robot", default="go2", choices=["go1", "go2", "g1"], help="Demo robot config to emit")
    run_parser.add_argument("--backend", default="auto", choices=["auto", "external", "builtin", "hybrid"], help="Real2Sim backend")
    run_parser.add_argument("--dynamic-objects", action="store_true", help="Emit free joints for reconstructed objects")
    run_parser.add_argument("--resume", action="store_true", help="Reuse existing work-dir artifacts")
    run_parser.add_argument("--dry-run", action="store_true", help="Print/validate configured commands without requiring outputs")
    run_parser.add_argument("--paper-strict", action="store_true", help="Require exact paper-equivalent external stages and runtime prerequisites")
    run_parser.add_argument("--synthetic-multiview", action="store_true", help="Expand a single input image into synthetic multiview frames before scene reconstruction")
    run_parser.add_argument(
        "--synthetic-multiview-mode",
        default="object",
        choices=["object", "scene"],
        help="Prompt style for synthetic multiview expansion: object turntable or scene completion",
    )
    run_parser.add_argument(
        "--pipeline-mode",
        default="full",
        choices=["full", "scene-only", "object-only"],
        help="Pipeline scope: full (scene+objects), scene-only, or object-only",
    )
    run_parser.add_argument("--skip-scene-reconstruction", action="store_true", help="Skip scene reconstruction in an existing work-dir run")
    run_parser.add_argument(
        "--skip-object-reconstruction",
        action="store_true",
        help="Skip object reconstruction and keep only the reconstructed scene",
    )
    run_parser.add_argument("--no-prune", action="store_true", help="Skip 3DGS pruning")
    run_parser.add_argument("--keep-ratio", type=float, default=None, help="Pruning keep ratio")
    run_parser.set_defaults(func=_run)

    scene_parser = subparsers.add_parser("generate-scene", help="Generate a scene-only Real2Sim package")
    scene_parser.add_argument("--input", type=Path, required=True, help="Input RGB image, image directory, or video")
    scene_parser.add_argument("--output", type=Path, required=True, help="Output scene package directory")
    scene_parser.add_argument("--work-dir", type=Path, default=None, help="Intermediate artifact directory")
    scene_parser.add_argument("--prompt", default="static navigable scene", help="Scene reconstruction prompt/context")
    scene_parser.add_argument("--config", type=Path, default=None, help="Real2Sim stage command config JSON")
    scene_parser.add_argument("--robot", default="go2", choices=["go1", "go2", "g1"], help="Demo robot config to emit")
    scene_parser.add_argument("--backend", default="auto", choices=["auto", "external", "builtin", "hybrid"], help="Scene generation backend")
    scene_parser.add_argument("--dynamic-objects", action="store_true", help=argparse.SUPPRESS)
    scene_parser.add_argument("--resume", action="store_true", help="Reuse existing work-dir artifacts")
    scene_parser.add_argument("--dry-run", action="store_true", help="Print/validate configured commands without requiring outputs")
    scene_parser.add_argument("--paper-strict", action="store_true", help="Require exact paper-equivalent external scene stages")
    scene_parser.add_argument("--synthetic-multiview", action="store_true", help="Expand a single input image into synthetic multiview frames before scene reconstruction")
    scene_parser.add_argument(
        "--scene-completion",
        action="store_true",
        help="Alias for scene-style synthetic multiview completion when a single image has insufficient scene evidence",
    )
    scene_parser.add_argument("--no-prune", action="store_true", help="Skip 3DGS pruning")
    scene_parser.add_argument("--keep-ratio", type=float, default=None, help="Pruning keep ratio")
    scene_parser.set_defaults(func=_generate_scene)

    package_parser = subparsers.add_parser("package-existing", help="Package existing 3DGS/mesh assets as a GS-Playground scene")
    package_parser.add_argument("--source", type=Path, required=True, help="Original capture or source path")
    package_parser.add_argument("--scene-ply", type=Path, required=True, help="Scene/background 3DGS PLY")
    package_parser.add_argument("--scene-mesh", type=Path, default=None, help="Optional scene mesh OBJ")
    package_parser.add_argument("--objects-json", type=Path, default=None, help="Optional object asset list JSON")
    package_parser.add_argument("--cameras-json", type=Path, default=None, help="Optional AnySplat camera intrinsics/extrinsics JSON")
    package_parser.add_argument("--depth-map", type=Path, default=None, help="Optional monocular depth artifact used for alignment")
    package_parser.add_argument("--background-image", type=Path, default=None, help="Optional inpainted background image artifact")
    package_parser.add_argument("--preview-image", type=Path, default=None, help="Optional source-view 3DGS preview image artifact")
    package_parser.add_argument("--mesh-cleanup-report", type=Path, default=None, help="Optional scene mesh cleanup report JSON")
    package_parser.add_argument("--texture-bake-report", type=Path, default=None, help="Optional COLMAP texture bake report JSON")
    package_parser.add_argument("--scene-structure-report", type=Path, default=None, help="Optional scene structure report JSON")
    package_parser.add_argument("--output", type=Path, required=True, help="Output package directory")
    package_parser.add_argument("--prompt", default="Navigate in the reconstructed Real2Sim scene")
    package_parser.add_argument("--robot", default="go2", choices=["go1", "go2", "g1"])
    package_parser.add_argument("--dynamic-objects", action="store_true")
    package_parser.set_defaults(func=_package_existing)

    check_parser = subparsers.add_parser("check", help="Check configured external model commands and local packages")
    check_parser.add_argument("--config", type=Path, default=None)
    check_parser.add_argument("--robot", default="go2", choices=["go1", "go2", "g1"])
    check_parser.add_argument("--backend", default="auto", choices=["auto", "external", "builtin", "hybrid"])
    check_parser.add_argument("--dynamic-objects", action="store_true")
    check_parser.add_argument("--paper-strict", action="store_true", help="Validate exact paper-equivalent runtime prerequisites")
    check_parser.set_defaults(func=_check)

    example_parser = subparsers.add_parser("write-example-config", help="Write a config template for external model commands")
    example_parser.add_argument("--output", type=Path, required=True)
    example_parser.set_defaults(func=_write_example_config)

    args = parser.parse_args(argv)
    return int(args.func(args))


def _pipeline(args) -> Real2SimPipeline:
    return Real2SimPipeline.from_config_file(
        repo_root=REPO_ROOT,
        config_file=getattr(args, "config", None),
        robot=args.robot,
        dynamic_objects=args.dynamic_objects,
        dry_run=getattr(args, "dry_run", False),
        backend="external" if getattr(args, "paper_strict", False) else getattr(args, "backend", "auto"),
    )


def _run(args) -> int:
    mode = str(getattr(args, "pipeline_mode", "full"))
    skip_scene = bool(getattr(args, "skip_scene_reconstruction", False))
    skip_objects = bool(getattr(args, "skip_object_reconstruction", False))

    if mode != "full" and (skip_scene or skip_objects):
        raise SystemExit("cannot combine --pipeline-mode with --skip-scene-reconstruction/--skip-object-reconstruction")

    enable_scene_reconstruction = True
    enable_object_reconstruction = True
    if mode == "scene-only":
        enable_scene_reconstruction = True
        enable_object_reconstruction = False
    elif mode == "object-only":
        enable_scene_reconstruction = False
        enable_object_reconstruction = True
    else:
        enable_scene_reconstruction = not skip_scene
        enable_object_reconstruction = not skip_objects

    if not enable_scene_reconstruction and not enable_object_reconstruction:
        raise SystemExit("both scene and object reconstruction are disabled")

    resume = args.resume
    if mode == "object-only":
        resume = True

    pipeline = _pipeline(args)
    paper_strict = bool(getattr(args, "paper_strict", False))
    if paper_strict and (args.no_prune or args.keep_ratio is not None):
        raise SystemExit("--paper-strict cannot be combined with --no-prune or --keep-ratio; use pruning.mode=speedy_splat")
    if paper_strict:
        report = check_paper_strict(pipeline.config, REPO_ROOT)
        for line in format_report(report):
            print(line)
        if args.dry_run:
            return 0 if report["ok"] else 1
        validate_paper_strict(pipeline.config, REPO_ROOT)
    if args.dry_run:
        status = pipeline.check()
        required = {"grounding_dino", "sam2", "instance_refinement", "lama", "anysplat", "sam3d", "pose_alignment"}
        if args.backend in {"auto", "builtin", "hybrid"} and status["builtin_ready"]:
            print("builtin: ready (numpy)")
            if args.backend == "builtin":
                return 0
        for stage, info in status["stages"].items():
            state = "configured" if info["configured"] else "missing"
            print(f"{stage}: {state} ({info['env']})")
        external_ready = all(status["stages"][stage]["configured"] for stage in required)
        return 0 if external_ready or (args.backend in {"auto", "hybrid"} and status["builtin_ready"]) else 1

    prune = None
    if args.no_prune or args.keep_ratio is not None:
        keep_ratio = args.keep_ratio if args.keep_ratio is not None else 0.3
        prune = PruneOptions(enabled=not args.no_prune, keep_ratio=keep_ratio)
    result = pipeline.run(
        input_path=args.input,
        output_dir=args.output,
        work_dir=args.work_dir,
        prompt=args.prompt,
        resume=resume,
        prune_options=prune,
        enable_scene_reconstruction=enable_scene_reconstruction,
        enable_object_reconstruction=enable_object_reconstruction,
        synthetic_multiview=args.synthetic_multiview,
        synthetic_multiview_mode=getattr(args, "synthetic_multiview_mode", "object"),
    )
    print(f"Real2Sim package: {result.package.root}")
    print(f"Scene XML: {result.package.scene_xml}")
    print(f"Demo config: {result.package.config_json}")
    print(f"Manifest: {result.package.manifest_json}")
    return 0


def _generate_scene(args) -> int:
    pipeline = _pipeline(args)
    paper_strict = bool(getattr(args, "paper_strict", False))
    if paper_strict and (args.no_prune or args.keep_ratio is not None):
        raise SystemExit("--paper-strict cannot be combined with --no-prune or --keep-ratio; use pruning.mode=speedy_splat")
    if paper_strict:
        report = check_paper_strict(pipeline.config, REPO_ROOT)
        for line in format_report(report):
            print(line)
        if args.dry_run:
            return 0 if report["ok"] else 1
        validate_paper_strict(pipeline.config, REPO_ROOT)
    if args.dry_run:
        status = pipeline.check()
        if args.backend in {"auto", "builtin", "hybrid"} and status["builtin_ready"]:
            print("builtin: ready (numpy)")
            return 0 if args.backend == "builtin" else 0
        required = {"grounding_dino", "sam2", "instance_refinement", "lama", "anysplat"}
        for stage, info in status["stages"].items():
            state = "configured" if info["configured"] else "missing"
            print(f"{stage}: {state} ({info['env']})")
        return 0 if all(status["stages"][stage]["configured"] for stage in required) else 1

    prune = None
    if args.no_prune or args.keep_ratio is not None:
        keep_ratio = args.keep_ratio if args.keep_ratio is not None else 0.3
        prune = PruneOptions(enabled=not args.no_prune, keep_ratio=keep_ratio)
    result = pipeline.run(
        input_path=args.input,
        output_dir=args.output,
        work_dir=args.work_dir,
        prompt=args.prompt,
        resume=args.resume,
        prune_options=prune,
        enable_scene_reconstruction=True,
        enable_object_reconstruction=False,
        synthetic_multiview=args.synthetic_multiview or getattr(args, "scene_completion", False),
        synthetic_multiview_mode="scene",
    )
    print(f"Real2Sim scene package: {result.package.root}")
    print(f"Scene PLY: {result.manifest.scene.gaussian_ply}")
    print(f"Scene XML: {result.package.scene_xml}")
    print(f"Demo config: {result.package.config_json}")
    print(f"Manifest: {result.package.manifest_json}")
    return 0


def _package_existing(args) -> int:
    result = _pipeline(args).package_existing(
        source=args.source,
        scene_ply=args.scene_ply,
        scene_mesh=args.scene_mesh,
        objects_json=args.objects_json,
        output_dir=args.output,
        prompt=args.prompt,
        cameras_json=args.cameras_json,
        depth_map=args.depth_map,
        background_image=args.background_image,
        preview_image=args.preview_image,
        mesh_cleanup_report=args.mesh_cleanup_report,
        texture_bake_report=args.texture_bake_report,
        scene_structure_report=args.scene_structure_report,
    )
    print(f"Real2Sim package: {result.root}")
    print(f"Scene XML: {result.scene_xml}")
    print(f"Demo config: {result.config_json}")
    print(f"Manifest: {result.manifest_json}")
    return 0


def _check(args) -> int:
    status = _pipeline(args).check()
    paper_strict = bool(getattr(args, "paper_strict", False))
    if paper_strict:
        status["paper_strict"] = check_paper_strict(_pipeline(args).config, REPO_ROOT)
    write_json(Path("/tmp/gsplayground_real2sim_check.json"), status)
    for stage, info in status["stages"].items():
        state = "configured" if info["configured"] else "missing"
        print(f"{stage}: {state} ({info['env']})")
    for module, ok in status["python_packages"].items():
        prefix = "controller_python." if paper_strict else ""
        print(f"{prefix}{module}: {'ok' if ok else 'missing'}")
    if paper_strict:
        for module in status["paper_strict"].get("python_modules", []):
            stage = module.get("stage", "unknown")
            name = module.get("name", "unknown")
            state = "ok" if module.get("ok") else "missing"
            print(f"paper_strict.{stage}.{name}: {state}")
        for line in format_report(status["paper_strict"]):
            print(line)
    print("Wrote /tmp/gsplayground_real2sim_check.json")
    return 0 if not paper_strict or status["paper_strict"]["ok"] else 1


def _write_example_config(args) -> int:
    write_json(args.output, example_config())
    print(f"Wrote {args.output}")
    return 0


def example_config() -> dict:
    return {
        "stages": {
            "synthetic_multiview": {
                "command": [
                    "{python}",
                    "/path/to/synthetic_multiview_expand.py",
                    "--image",
                    "{image}",
                    "--output-dir",
                    "{synthetic_multiview_dir}",
                    "--frames-dir",
                    "{synthetic_frames_dir}",
                    "--prompt",
                    "{synthetic_multiview_prompt}",
                    "--size",
                    "1536x1024",
                ]
            },
            "grounding_dino": {
                "command": [
                    "{python}",
                    "/path/to/grounding_dino_detect.py",
                    "--image",
                    "{image}",
                    "--prompt",
                    "{prompt}",
                    "--output",
                    "{detections_json}",
                ]
            },
            "sam2": {
                "command": [
                    "{python}",
                    "/path/to/sam2_segment.py",
                    "--image",
                    "{image}",
                    "--detections",
                    "{detections_json}",
                    "--masks-dir",
                    "{masks_dir}",
                    "--output",
                    "{instances_json}",
                ]
            },
            "instance_refinement": {
                "command": [
                    "{python}",
                    "/path/to/instance_refine.py",
                    "--instances",
                    "{instances_json}",
                    "--output",
                    "{instances_json}",
                ]
            },
            "lama": {
                "command": [
                    "{python}",
                    "/path/to/lama_inpaint.py",
                    "--image",
                    "{image}",
                    "--instances",
                    "{instances_json}",
                    "--output",
                    "{background_image}",
                    "--prompt",
                    "{prompt}",
                    "--work-dir",
                    "{work_dir}",
                    "--redetect-command-env",
                    "REAL2SIM_GROUNDING_DINO_COMMAND",
                    "--segment-command-env",
                    "REAL2SIM_SAM2_COMMAND",
                    "--require-redetect-loop",
                ]
            },
            "depth_estimation": {
                "command": [
                    "{python}",
                    "/path/to/depth_anything_depth.py",
                    "--image",
                    "{background_image}",
                    "--output",
                    "{depth_map}",
                    "--model",
                    "depth-anything/Depth-Anything-V2-Small-hf",
                ]
            },
            "anysplat": {
                "command": [
                    "{python}",
                    "/path/to/anysplat_scene.py",
                    "--image",
                    "{background_image}",
                    "--frames-dir",
                    "{frames_dir}",
                    "--output-dir",
                    "{scene_reconstruction_dir}",
                    "--ply",
                    "{scene_ply}",
                    "--mesh",
                    "{scene_mesh}",
                    "--cameras",
                    "{cameras_json}",
                    "--depth",
                    "{depth_map}",
                    "--max-views",
                    "8",
                ]
            },
            "sam3d": {
                "command": [
                    "{python}",
                    "/path/to/sam3d_object.py",
                    "--image",
                    "{image}",
                    "--mask",
                    "{mask}",
                    "--instance",
                    "{instance_json}",
                    "--output-dir",
                    "{object_dir}",
                    "--ply",
                    "{object_ply}",
                    "--mesh",
                    "{object_mesh}",
                    "--pose",
                    "{object_pose}",
                    "--cameras",
                    "{cameras_json}",
                    "--match-camera-preprocess",
                    "--object-centric-preprocess",
                    "--object-preprocess-size",
                    "896",
                    "--object-crop-margin",
                    "0.12",
                ]
            },
            "pose_alignment": {
                "command": [
                    "{python}",
                    "/path/to/refine_pose_alignment.py",
                    "--assets",
                    "{aligned_assets_json}",
                    "--instances",
                    "{instances_json}",
                    "--depth",
                    "{depth_map}",
                    "--cameras",
                    "{cameras_json}",
                    "--work-dir",
                    "{work_dir}",
                    "--output",
                    "{aligned_assets_json}",
                ]
            },
        },
        "builtin": {
            "max_scene_points": 180000,
            "max_object_points": 50000,
            "max_objects": 10,
            "min_component_area_ratio": 0.003,
            "stage_min_component_area_ratio": 0.0005,
            "focal_scale": 0.9,
            "external_depth_weight": 0.65,
            "color_enhance_strength": 0.35,
            "scene_point_scale": 0.018,
            "object_point_scale": 0.014,
            "opacity_logit": 4.5,
            "mask_close_passes": 2,
            "mask_dilate_passes": 1,
            "max_mask_overlap": 0.6,
        },
        "pruning": {
            "enabled": True,
            "keep_ratio": 0.85,
            "mode": "speedy_splat",
            "external_command": [
                "{python}",
                "scripts/real2sim_external/speedy_splat_prune.py",
                "--input",
                "{input}",
                "--output",
                "{output}",
                "--image",
                "{source_image}",
                "--cameras",
                "{cameras_json}",
                "--keep-ratio",
                "{keep_ratio}",
            ],
        },
    }
