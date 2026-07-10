"""Automated Real2Sim pipeline orchestration.

This follows the GS-Playground paper structure:

1. GroundingDINO instance detection.
2. SAM/SAM2 instance segmentation.
3. Visual instance de-duplication/refinement.
4. LaMa background inpainting.
5. SAM-3D object reconstruction.
6. AnySplat scene reconstruction.
7. Speedy-Splat-compatible 3DGS pruning.
8. GS-Playground MJCF/config package creation.

The heavy model stages are intentionally external command adapters. The public
preview repository does not ship those model checkpoints or code, so a run only
succeeds when real tools are configured and produce the declared artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .commands import StageCommand, command_preferring_env, resolve_command, run_command
from .builtin import run_builtin_real2sim
from .contracts import (
    Pose,
    Real2SimAsset,
    Real2SimManifest,
    load_instances,
    read_json,
    require_file,
    validate_detections,
    write_json,
)
from .errors import MissingStageCommand, PipelineError, StageArtifactError
from .image_io import read_rgb, write_gray, write_rgb
from .packaging import PackageResult, PackageWriter
from .pruning import PruneOptions, prune_ply


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


STAGES = {
    "synthetic_multiview": StageCommand(
        "synthetic_multiview",
        "REAL2SIM_SYNTHETIC_MULTIVIEW_COMMAND",
        "optional single-image to multi-view expansion; must write image frames under {synthetic_frames_dir}",
        required=False,
    ),
    "grounding_dino": StageCommand(
        "grounding_dino",
        "REAL2SIM_GROUNDING_DINO_COMMAND",
        "open-vocabulary detection; must write {detections_json}",
    ),
    "sam2": StageCommand(
        "sam2",
        "REAL2SIM_SAM2_COMMAND",
        "instance segmentation/tracking; must write {instances_json} and masks under {masks_dir}",
    ),
    "instance_refinement": StageCommand(
        "instance_refinement",
        "REAL2SIM_INSTANCE_REFINEMENT_COMMAND",
        "visual mask de-duplication and confidence ranking; may update {instances_json}",
    ),
    "lama": StageCommand(
        "lama",
        "REAL2SIM_LAMA_COMMAND",
        "sequential background inpainting; must write {background_image}",
    ),
    "depth_estimation": StageCommand(
        "depth_estimation",
        "REAL2SIM_DEPTH_COMMAND",
        "optional monocular depth estimation; should write {depth_map}",
        required=False,
    ),
    "anysplat": StageCommand(
        "anysplat",
        "REAL2SIM_ANYSPLAT_COMMAND",
        "scene-level 3DGS reconstruction; must write {scene_ply}",
    ),
    "sam3d": StageCommand(
        "sam3d",
        "REAL2SIM_SAM3D_COMMAND",
        "object-level 3DGS/mesh/pose reconstruction; must write per-object gaussian.ply, mesh.obj, pose.json",
    ),
    "pose_alignment": StageCommand(
        "pose_alignment",
        "REAL2SIM_POSE_ALIGNMENT_COMMAND",
        "optional pose/scale refinement; may update {aligned_assets_json}",
        required=False,
    ),
}


@dataclass(frozen=True)
class CaptureInfo:
    source: Path
    image: Path
    frames_dir: Path | None
    kind: str


@dataclass(frozen=True)
class PipelineRunResult:
    work_dir: Path
    package: PackageResult
    manifest: Real2SimManifest


class Real2SimPipeline:
    def __init__(
        self,
        *,
        repo_root: Path,
        config: Mapping[str, Any] | None = None,
        robot: str = "go2",
        dynamic_objects: bool = False,
        dry_run: bool = False,
        backend: str = "auto",
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.config = dict(config or {})
        self.robot = robot
        self.dynamic_objects = dynamic_objects
        self.dry_run = dry_run
        self.backend = backend

    @classmethod
    def from_config_file(
        cls,
        *,
        repo_root: Path,
        config_file: Path | None,
        robot: str,
        dynamic_objects: bool,
        dry_run: bool = False,
        backend: str = "auto",
    ) -> "Real2SimPipeline":
        config = read_json(config_file) if config_file else {}
        if not isinstance(config, Mapping):
            raise PipelineError(f"Real2Sim config must be a JSON object: {config_file}")
        return cls(
            repo_root=repo_root,
            config=config,
            robot=robot,
            dynamic_objects=dynamic_objects,
            dry_run=dry_run,
            backend=backend,
        )

    def run(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        prompt: str,
        work_dir: Path | None = None,
        resume: bool = False,
        prune_options: PruneOptions | None = None,
        synthetic_multiview: bool = False,
        synthetic_multiview_mode: str = "object",
        enable_scene_reconstruction: bool = True,
        enable_object_reconstruction: bool = True,
    ) -> PipelineRunResult:
        work = (work_dir or (output_dir / "_work")).resolve()
        output = output_dir.resolve()
        if not enable_scene_reconstruction and not work.exists():
            raise PipelineError(
                "object-only mode requires an existing work-dir containing reconstructed scene artifacts; "
                "provide a real output/work directory via --work-dir (or set --pipeline-mode scene-only first)."
            )
        if work.exists() and not resume:
            shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)

        capture = self.prepare_capture(input_path.resolve(), work)
        context = self._base_context(capture, work, output, prompt, synthetic_multiview_mode=synthetic_multiview_mode)
        capture = self._run_synthetic_multiview_expansion(capture, context, resume=resume, enabled=synthetic_multiview)

        if self.backend == "hybrid" and not enable_scene_reconstruction:
            raise PipelineError("hybrid backend cannot run object-only mode; use backend=external with cached scene artifacts")
        if self.backend == "hybrid":
            return self._run_hybrid_backend(
                capture=capture,
                context=context,
                prompt=prompt,
                work=work,
                output=output,
                resume=resume,
                prune_options=prune_options,
                enable_object_reconstruction=enable_object_reconstruction,
            )

        if self._should_use_builtin_backend():
            if not enable_scene_reconstruction:
                raise PipelineError("built-in backend cannot run object-only mode; use an existing external scene package or backend=external")
            manifest = run_builtin_real2sim(
                capture,
                context,
                prompt,
                self.config,
                include_objects=enable_object_reconstruction,
            )
            manifest = self._prune_assets(manifest, work, prune_options or self._prune_options())
            package = PackageWriter(
                robot=self.robot,
                dynamic_objects=self.dynamic_objects,
                include_scene_mesh=manifest.scene.mesh_obj is not None,
            ).write(manifest, output)
            return PipelineRunResult(work_dir=work, package=package, manifest=manifest)

        detections_json = self._run_detection(context, resume=resume)
        context["detections_json"] = detections_json
        instances_json = self._run_segmentation(context, resume=resume)
        context["instances_json"] = instances_json
        instances_json = self._run_instance_refinement(context, resume=resume)
        context["instances_json"] = instances_json
        background_image = self._run_inpainting(context, resume=resume)
        context["background_image"] = background_image
        if enable_scene_reconstruction:
            scene_ply, scene_mesh, camera_artifacts = self._run_scene_reconstruction(context, resume=resume)
            context["scene_ply"] = scene_ply
            if scene_mesh:
                context["scene_mesh"] = scene_mesh
        else:
            scene_ply = Path(context["scene_ply"])
            scene_mesh = Path(context["scene_mesh"])
            scene_ply = require_file(scene_ply, "precomputed scene 3DGS PLY")
            if scene_mesh.exists():
                scene_mesh = require_file(scene_mesh, "precomputed scene mesh OBJ")
            else:
                scene_mesh = None
            camera_artifacts = {
                "source_image": Path(context["image"]),
            }
            if scene_mesh is not None:
                camera_artifacts["scene_mesh"] = scene_mesh
            if Path(context["cameras_json"]).exists():
                camera_artifacts["cameras_json"] = Path(context["cameras_json"])
            if Path(context["depth_map"]).exists():
                camera_artifacts["depth_map"] = Path(context["depth_map"])

        instances = load_instances(instances_json) if enable_object_reconstruction else []
        object_assets = self._run_object_reconstruction(context, instances, resume=resume) if enable_object_reconstruction else []
        scene_asset = Real2SimAsset(name="scene", label="background_scene", gaussian_ply=scene_ply, mesh_obj=scene_mesh)
        manifest = Real2SimManifest(
            source=capture.source,
            prompt=prompt,
            scene=scene_asset,
            objects=tuple(object_assets),
            camera_artifacts=camera_artifacts,
            metadata={
                "pipeline": "GS-Playground Real2Sim MVP",
                "capture_kind": capture.kind,
                "stage_order": list(STAGES.keys()),
            },
        )

        if enable_object_reconstruction:
            manifest = self._run_pose_alignment(context, manifest, resume=resume)
        manifest = self._prune_assets(manifest, work, prune_options or self._prune_options())
        package = PackageWriter(
            robot=self.robot,
            dynamic_objects=self.dynamic_objects,
            include_scene_mesh=manifest.scene.mesh_obj is not None,
        ).write(manifest, output)
        return PipelineRunResult(work_dir=work, package=package, manifest=manifest)

    def package_existing(
        self,
        *,
        source: Path,
        scene_ply: Path,
        output_dir: Path,
        prompt: str,
        scene_mesh: Path | None = None,
        objects_json: Path | None = None,
        cameras_json: Path | None = None,
        depth_map: Path | None = None,
        background_image: Path | None = None,
        preview_image: Path | None = None,
        mesh_cleanup_report: Path | None = None,
        texture_bake_report: Path | None = None,
        scene_structure_report: Path | None = None,
    ) -> PackageResult:
        scene = Real2SimAsset(
            name="scene",
            label="background_scene",
            gaussian_ply=require_file(scene_ply.resolve(), "scene 3DGS PLY"),
            mesh_obj=require_file(scene_mesh.resolve(), "scene mesh OBJ") if scene_mesh else None,
        )
        objects: list[Real2SimAsset] = []
        if objects_json is not None:
            raw = read_json(objects_json)
            entries = raw.get("objects", raw) if isinstance(raw, Mapping) else raw
            if not isinstance(entries, list):
                raise StageArtifactError("objects JSON must be a list or object with objects list")
            for entry in entries:
                objects.append(self._asset_from_json_entry(entry, objects_json.parent))

        camera_artifacts = {}
        if cameras_json is not None:
            camera_artifacts["cameras_json"] = require_file(cameras_json.resolve(), "camera JSON")
        if depth_map is not None:
            camera_artifacts["depth_map"] = require_file(depth_map.resolve(), "depth map")
        if background_image is not None:
            camera_artifacts["background_image"] = require_file(background_image.resolve(), "background image")
        if preview_image is not None:
            camera_artifacts["source_gs_preview"] = require_file(preview_image.resolve(), "source-view 3DGS preview image")

        manifest = Real2SimManifest(
            source=source.resolve(),
            prompt=prompt,
            scene=scene,
            objects=tuple(objects),
            camera_artifacts=camera_artifacts,
            metadata={"pipeline": "manual-existing-assets"},
        )
        result = PackageWriter(
            robot=self.robot,
            dynamic_objects=self.dynamic_objects,
            include_scene_mesh=manifest.scene.mesh_obj is not None,
        ).write(manifest, output_dir)
        copy_optional_package_report(
            mesh_cleanup_report,
            result.root / "artifacts" / "mesh_cleanup" / "cleanup_report.json",
            "mesh cleanup report",
        )
        copy_optional_package_report(
            texture_bake_report,
            result.root / "artifacts" / "colmap_texture_bake" / "texture_bake_report.json",
            "texture bake report",
        )
        copy_optional_package_report(
            scene_structure_report,
            result.root / "artifacts" / "scene_structure" / "scene_structure_report.json",
            "scene structure report",
        )
        self._attach_scene_structure_report_to_import(result.root)
        return result

    def _attach_scene_structure_report_to_import(self, root: Path) -> None:
        report = root / "artifacts" / "scene_structure" / "scene_structure_report.json"
        import_path = root / "motrix_import.json"
        if not report.is_file() or not import_path.is_file():
            return
        scene_xml = root / "mjcf" / "scene.xml"
        report_data = read_json(report)
        structure_layers_path = root / "artifacts" / "scene_structure" / "static_structure.json"
        structure_layers = scene_structure_layers(report_data, proxy_count=0)
        source_proxy_count = self._attach_scene_structure_collision_proxies(scene_xml, report_data)
        completion_proxy_count = self._attach_scene_completion_collision_proxies(scene_xml, structure_layers)
        proxy_count = source_proxy_count + completion_proxy_count
        structure_layers.setdefault("quality", {})["source_collision_proxy_count"] = source_proxy_count
        structure_layers["quality"]["completion_collision_proxy_count"] = completion_proxy_count
        structure_layers["quality"]["total_collision_proxy_count"] = proxy_count
        write_json(structure_layers_path, structure_layers)
        commercial_readiness_path = root / "artifacts" / "quality" / "commercial_readiness.json"
        room_completeness_path = root / "artifacts" / "quality" / "room_completeness.json"
        metric_scale_path = root / "artifacts" / "quality" / "metric_scale.json"
        navigation_acceptance_path = root / "artifacts" / "quality" / "navigation_acceptance.json"
        material_readiness_path = root / "artifacts" / "quality" / "material_readiness.json"
        task_affordance_path = root / "artifacts" / "quality" / "task_affordance.json"
        commercial_acceptance_path = root / "artifacts" / "quality" / "commercial_acceptance.json"
        movable_assets_path = root / "artifacts" / "assets" / "movable_assets.json"
        movable_assets = movable_asset_layers(root, structure_layers=structure_layers)
        movable_assets_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(movable_assets_path, movable_assets)
        data = read_json(import_path)
        movable_tasks_path = root / "artifacts" / "assets" / "movable_extraction_tasks.json"
        movable_tasks = movable_extraction_tasks(root, data, movable_assets)
        write_json(movable_tasks_path, movable_tasks)
        metric_scale = metric_scale_report(root, structure_layers, report_data)
        navigation_acceptance = navigation_acceptance_report(root, structure_layers)
        material_readiness = material_readiness_report(root)
        task_affordance = task_affordance_report(root, movable_assets)
        commercial_readiness = commercial_readiness_report(
            root,
            structure_layers,
            report_data,
            proxy_count=proxy_count,
            metric_scale=metric_scale,
            navigation_acceptance=navigation_acceptance,
            material_readiness=material_readiness,
            task_affordance=task_affordance,
        )
        room_completeness = room_completeness_report(root, structure_layers, report_data)
        commercial_readiness_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(commercial_readiness_path, commercial_readiness)
        write_json(room_completeness_path, room_completeness)
        write_json(metric_scale_path, metric_scale)
        write_json(navigation_acceptance_path, navigation_acceptance)
        write_json(material_readiness_path, material_readiness)
        write_json(task_affordance_path, task_affordance)
        data["scene_structure_report"] = report.relative_to(root).as_posix()
        data["scene_structure_layers"] = structure_layers_path.relative_to(root).as_posix()
        data["movable_asset_layers"] = movable_assets_path.relative_to(root).as_posix()
        data["movable_extraction_tasks"] = movable_tasks_path.relative_to(root).as_posix()
        data["commercial_readiness_report"] = commercial_readiness_path.relative_to(root).as_posix()
        data["room_completeness_report"] = room_completeness_path.relative_to(root).as_posix()
        data["metric_scale_report"] = metric_scale_path.relative_to(root).as_posix()
        data["navigation_acceptance_report"] = navigation_acceptance_path.relative_to(root).as_posix()
        data["material_readiness_report"] = material_readiness_path.relative_to(root).as_posix()
        data["task_affordance_report"] = task_affordance_path.relative_to(root).as_posix()
        data["commercial_acceptance_report"] = commercial_acceptance_path.relative_to(root).as_posix()
        data.setdefault("verification", {})["scene_structure"] = report_data
        data["verification"]["scene_structure_collision_proxies"] = proxy_count
        data["verification"]["scene_structure_layers"] = {
            "static_collision_proxy_count": len(structure_layers["static_collision_proxies"]),
            "completion_proxy_count": len(structure_layers.get("completion_proxies", [])),
            "navigation_surface_count": len(structure_layers["navigation_surfaces"]),
            "support_surface_count": len(structure_layers["support_surfaces"]),
            "wall_count": len(structure_layers["walls"]),
        }
        data["verification"]["movable_asset_layers"] = {
            "movable_asset_count": len(movable_assets["movable_assets"]),
            "dynamic_asset_count": sum(1 for asset in movable_assets["movable_assets"] if asset.get("dynamic")),
            "candidate_slot_count": len(movable_assets["candidate_slots"]),
            "extraction_task_count": len(movable_tasks["tasks"]),
            "production_asset_count": movable_assets["quality"].get("production_asset_count"),
            "provisional_asset_count": movable_assets["quality"].get("provisional_asset_count"),
            "incomplete_asset_count": movable_assets["quality"].get("incomplete_asset_count"),
            "commercial_asset_grade": movable_assets["quality"].get("commercial_asset_grade"),
        }
        data["verification"]["commercial_readiness"] = commercial_readiness["summary"]
        data["verification"]["room_completeness"] = room_completeness["summary"]
        data["verification"]["metric_scale"] = metric_scale["summary"]
        data["verification"]["navigation_acceptance"] = navigation_acceptance["summary"]
        data["verification"]["material_readiness"] = material_readiness["summary"]
        data["verification"]["task_affordance"] = task_affordance["summary"]
        commercial_acceptance = commercial_acceptance_manifest(
            root,
            commercial_readiness=commercial_readiness,
            room_completeness=room_completeness,
            metric_scale=metric_scale,
            navigation_acceptance=navigation_acceptance,
            material_readiness=material_readiness,
            task_affordance=task_affordance,
            movable_assets=movable_assets,
            structure_layers=structure_layers,
        )
        data["verification"]["commercial_acceptance"] = commercial_acceptance["summary"]
        write_json(import_path, data)
        commercial_acceptance = commercial_acceptance_manifest(
            root,
            commercial_readiness=commercial_readiness,
            room_completeness=room_completeness,
            metric_scale=metric_scale,
            navigation_acceptance=navigation_acceptance,
            material_readiness=material_readiness,
            task_affordance=task_affordance,
            movable_assets=movable_assets,
            structure_layers=structure_layers,
        )
        write_json(commercial_acceptance_path, commercial_acceptance)

    def _attach_scene_structure_collision_proxies(self, scene_xml: Path, report: Mapping[str, Any]) -> int:
        if not scene_xml.is_file():
            return 0
        semantic_planes = report.get("semantic_planes")
        if not isinstance(semantic_planes, list):
            return 0
        tree = ET.parse(scene_xml)
        world = tree.getroot().find("worldbody")
        if world is None:
            return 0
        scene_body = world.find("./body[@name='scene']")
        body = scene_body if scene_body is not None else world
        added = 0
        for index, plane in enumerate(semantic_planes[:12]):
            if not isinstance(plane, Mapping):
                continue
            proxy = plane.get("collision_proxy_hint")
            if not isinstance(proxy, Mapping) or proxy.get("type") != "box":
                continue
            center = numeric_triplet(proxy.get("center"))
            size = numeric_triplet(proxy.get("size"))
            if center is None or size is None:
                continue
            label = str(plane.get("semantic_label") or proxy.get("label") or "structure")
            if label not in {"floor", "ceiling", "support_surface", "wall_visible"} and not label.startswith("wall_"):
                continue
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"scene_structure_collision_{slug_token(label)}_{index:03d}",
                    "type": "box",
                    "pos": vec(center),
                    "size": vec(size),
                    "rgba": "0.1 0.7 1 0.08",
                    "group": "3",
                    "contype": "1",
                    "conaffinity": "1",
                },
            )
            added += 1
        if added:
            indent_xml(tree.getroot())
            tree.write(scene_xml, encoding="utf-8", xml_declaration=True)
        return added

    def _attach_scene_completion_collision_proxies(self, scene_xml: Path, structure_layers: Mapping[str, Any]) -> int:
        if not scene_xml.is_file():
            return 0
        proxies = structure_layers.get("completion_proxies")
        if not isinstance(proxies, list) or not proxies:
            return 0
        tree = ET.parse(scene_xml)
        world = tree.getroot().find("worldbody")
        if world is None:
            return 0
        scene_body = world.find("./body[@name='scene']")
        body = scene_body if scene_body is not None else world
        added = 0
        for proxy in proxies[:8]:
            if not isinstance(proxy, Mapping) or proxy.get("type") != "box":
                continue
            center = numeric_triplet(proxy.get("center"))
            size = numeric_triplet(proxy.get("size"))
            if center is None or size is None:
                continue
            name = str(proxy.get("mjcf_geom_name") or f"scene_completion_collision_{added:03d}")
            ET.SubElement(
                body,
                "geom",
                {
                    "name": name,
                    "type": "box",
                    "pos": vec(center),
                    "size": vec(size),
                    "rgba": "0.2 0.9 0.35 0.10",
                    "group": "3",
                    "contype": "1",
                    "conaffinity": "1",
                },
            )
            added += 1
        if added:
            indent_xml(tree.getroot())
            tree.write(scene_xml, encoding="utf-8", xml_declaration=True)
        return added

    def check(self) -> dict[str, Any]:
        status: dict[str, Any] = {"backend": self.backend, "stages": {}, "python_packages": {}}
        for name, stage in STAGES.items():
            try:
                command = resolve_command(stage, self.config)
                status["stages"][name] = {"configured": command is not None, "env": stage.env_var}
            except MissingStageCommand as exc:
                status["stages"][name] = {"configured": False, "env": stage.env_var, "error": str(exc)}

        for module in ("numpy", "torch", "gaussian_renderer", "gsplat", "plyfile"):
            try:
                __import__(module)
                status["python_packages"][module] = True
            except Exception:
                status["python_packages"][module] = False
        status["builtin_ready"] = status["python_packages"]["numpy"]
        return status

    def prepare_capture(self, input_path: Path, work: Path) -> CaptureInfo:
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        input_dir = work / "00_input"
        input_dir.mkdir(parents=True, exist_ok=True)

        if input_path.is_dir():
            source_images = sorted(path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
            if not source_images:
                raise StageArtifactError(f"image directory contains no supported images: {input_path}")
            frames_dir = input_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            frames: list[Path] = []
            for index, source_image in enumerate(source_images, start=1):
                frame = frames_dir / f"frame_{index:06d}{source_image.suffix.lower()}"
                shutil.copy2(source_image, frame)
                frames.append(frame)
            return CaptureInfo(source=input_path, image=frames[0], frames_dir=frames_dir, kind="multi_image")

        suffix = input_path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            image = input_dir / f"input{suffix}"
            shutil.copy2(input_path, image)
            return CaptureInfo(source=input_path, image=image, frames_dir=None, kind="single_image")

        if suffix in VIDEO_SUFFIXES:
            frames_dir = input_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise PipelineError("video input requires ffmpeg to extract frames")
            cmd = [ffmpeg, "-y", "-i", input_path.as_posix(), "-vf", "fps=2", (frames_dir / "frame_%06d.png").as_posix()]
            completed = subprocess.run(cmd, check=False)
            if completed.returncode != 0:
                raise PipelineError(f"ffmpeg frame extraction failed with exit code {completed.returncode}")
            frames = sorted(frames_dir.glob("frame_*.png"))
            if not frames:
                raise StageArtifactError(f"ffmpeg produced no frames in {frames_dir}")
            return CaptureInfo(source=input_path, image=frames[0], frames_dir=frames_dir, kind="video")

        raise PipelineError(f"unsupported input type: {input_path}")

    def _base_context(
        self,
        capture: CaptureInfo,
        work: Path,
        output: Path,
        prompt: str,
        *,
        synthetic_multiview_mode: str = "object",
    ) -> dict[str, Any]:
        mode = self._synthetic_multiview_mode(synthetic_multiview_mode)
        return {
            "source": capture.source,
            "input": capture.source,
            "python": sys.executable,
            "image": capture.image,
            "frames_dir": capture.frames_dir or "",
            "synthetic_multiview_dir": work / "00_input" / "synthetic_multiview",
            "synthetic_frames_dir": work / "00_input" / "synthetic_multiview" / "frames",
            "synthetic_contact_sheet": work / "00_input" / "synthetic_multiview" / "contact_sheet.png",
            "synthetic_multiview_mode": mode,
            "synthetic_multiview_prompt": self._synthetic_multiview_prompt(prompt, mode=mode),
            "work_dir": work,
            "output_dir": output,
            "prompt": prompt,
            "detections_json": work / "01_detection" / "detections.json",
            "instances_json": work / "02_segmentation" / "instances.json",
            "masks_dir": work / "02_segmentation" / "masks",
            "background_image": work / "03_inpainting" / "background.png",
            "scene_reconstruction_dir": work / "04_reconstruction" / "scene",
            "scene_ply": work / "04_reconstruction" / "scene" / "point_cloud.ply",
            "scene_mesh": work / "04_reconstruction" / "scene" / "mesh.obj",
            "depth_map": work / "04_reconstruction" / "scene" / "depth.pgm",
            "cameras_json": work / "04_reconstruction" / "scene" / "cameras.json",
            "objects_dir": work / "04_reconstruction" / "objects",
            "aligned_assets_json": work / "05_alignment" / "assets.json",
        }

    def _should_use_builtin_backend(self) -> bool:
        if self.backend == "builtin":
            return True
        if self.backend == "hybrid":
            return False
        if self.backend == "external":
            return False
        if self.backend != "auto":
            raise PipelineError(f"unknown Real2Sim backend: {self.backend}")
        return not self._external_stages_configured()

    def _external_stages_configured(self) -> bool:
        required = ("grounding_dino", "sam2", "instance_refinement", "lama", "anysplat", "sam3d")
        for name in required:
            try:
                resolve_command(STAGES[name], self.config)
            except MissingStageCommand:
                return False
        return True

    def _run_hybrid_backend(
        self,
        *,
        capture: CaptureInfo,
        context: dict[str, Any],
        prompt: str,
        work: Path,
        output: Path,
        resume: bool,
        prune_options: PruneOptions | None,
        enable_object_reconstruction: bool = True,
    ) -> PipelineRunResult:
        try:
            self._run_detection(context, resume=resume)
            self._run_segmentation(context, resume=resume)
            self._run_instance_refinement(context, resume=resume)
            self._run_inpainting(context, resume=resume)
            self._run_depth_estimation(context, resume=resume)
        except MissingStageCommand:
            pass

        hybrid_config = dict(self.config)
        builtin_config = dict(hybrid_config.get("builtin", {})) if isinstance(hybrid_config.get("builtin", {}), Mapping) else {}
        builtin_config["prefer_stage_artifacts"] = True
        hybrid_config["builtin"] = builtin_config
        manifest = run_builtin_real2sim(
            capture,
            context,
            prompt,
            hybrid_config,
            include_objects=enable_object_reconstruction,
        )
        manifest = self._prune_assets(manifest, work, prune_options or self._prune_options())
        package = PackageWriter(
            robot=self.robot,
            dynamic_objects=self.dynamic_objects,
            include_scene_mesh=manifest.scene.mesh_obj is not None,
        ).write(manifest, output)
        return PipelineRunResult(work_dir=work, package=package, manifest=manifest)

    def _run_detection(self, context: dict[str, Any], *, resume: bool) -> Path:
        output = Path(context["detections_json"])
        if resume and output.exists():
            validate_detections(output)
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        command = resolve_command(STAGES["grounding_dino"], self.config)
        run_command(STAGES["grounding_dino"], command, context, cwd=self.repo_root, dry_run=self.dry_run)
        if not self.dry_run:
            validate_detections(output)
        return output

    def _run_segmentation(self, context: dict[str, Any], *, resume: bool) -> Path:
        output = Path(context["instances_json"])
        if resume and output.exists():
            load_instances(output)
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        Path(context["masks_dir"]).mkdir(parents=True, exist_ok=True)
        command = resolve_command(STAGES["sam2"], self.config)
        run_command(STAGES["sam2"], command, context, cwd=self.repo_root, dry_run=self.dry_run)
        if not self.dry_run:
            load_instances(output)
        return output

    def _run_instance_refinement(self, context: dict[str, Any], *, resume: bool) -> Path:
        output = Path(context["instances_json"])
        command = resolve_command(STAGES["instance_refinement"], self.config)
        if resume and output.exists() and instances_are_refined(output):
            load_instances(output)
            return output
        run_command(STAGES["instance_refinement"], command, context, cwd=self.repo_root, dry_run=self.dry_run)
        if not self.dry_run:
            load_instances(output)
        return output

    def _run_inpainting(self, context: dict[str, Any], *, resume: bool) -> Path:
        output = Path(context["background_image"])
        if resume and output.exists():
            return require_file(output, "inpainted background image")
        output.parent.mkdir(parents=True, exist_ok=True)
        command = resolve_command(STAGES["lama"], self.config)
        run_command(STAGES["lama"], command, context, cwd=self.repo_root, dry_run=self.dry_run)
        if not self.dry_run:
            require_file(output, "inpainted background image")
        return output

    def _run_depth_estimation(self, context: dict[str, Any], *, resume: bool) -> Path | None:
        command = resolve_command(STAGES["depth_estimation"], self.config)
        if command is None:
            return None
        output = Path(context["depth_map"])
        if resume and output.exists():
            return require_file(output, "monocular depth map")
        output.parent.mkdir(parents=True, exist_ok=True)
        run_command(STAGES["depth_estimation"], command, context, cwd=self.repo_root, dry_run=self.dry_run)
        if not self.dry_run:
            require_file(output, "monocular depth map")
        return output

    def _run_scene_reconstruction(self, context: dict[str, Any], *, resume: bool) -> tuple[Path, Path | None, dict[str, Path]]:
        scene_dir = Path(context["scene_reconstruction_dir"])
        scene_dir.mkdir(parents=True, exist_ok=True)
        ply = Path(context["scene_ply"])
        if not (resume and ply.exists()):
            command = resolve_command(STAGES["anysplat"], self.config)
            run_command(STAGES["anysplat"], command, context, cwd=self.repo_root, dry_run=self.dry_run)
        if not self.dry_run:
            require_file(ply, "scene 3DGS PLY")

        mesh = Path(context["scene_mesh"])
        camera_artifacts = {}
        source_image = Path(context["image"])
        if source_image.exists():
            camera_artifacts["source_image"] = source_image
        cameras_json = Path(context["cameras_json"])
        depth_map = Path(context["depth_map"])
        if mesh.exists():
            camera_artifacts["scene_mesh"] = mesh
        if cameras_json.exists():
            camera_artifacts["cameras_json"] = cameras_json
        if depth_map.exists():
            camera_artifacts["depth_map"] = depth_map
        return ply, mesh if mesh.exists() else None, camera_artifacts

    def _run_object_reconstruction(
        self,
        context: dict[str, Any],
        instances: list[Any],
        *,
        resume: bool,
    ) -> list[Real2SimAsset]:
        if not instances:
            return []

        command = resolve_command(STAGES["sam3d"], self.config)
        objects: list[Real2SimAsset] = []
        objects_dir = Path(context["objects_dir"])
        objects_dir.mkdir(parents=True, exist_ok=True)
        for instance in instances:
            object_dir = objects_dir / _safe_name(instance.instance_id)
            object_dir.mkdir(parents=True, exist_ok=True)
            instance_json = object_dir / "instance.json"
            write_json(
                instance_json,
                {
                    "id": instance.instance_id,
                    "label": instance.label,
                    "score": instance.score,
                    "bbox_xyxy": list(instance.bbox_xyxy) if instance.bbox_xyxy else None,
                    "mask_path": instance.mask_path.as_posix() if instance.mask_path else None,
                },
            )
            object_context = {
                **context,
                "object_dir": object_dir,
                "instance_json": instance_json,
                "instance_id": instance.instance_id,
                "instance_label": instance.label,
                "mask": instance.mask_path or "",
                "object_ply": object_dir / "gaussian.ply",
                "object_mesh": object_dir / "mesh.obj",
                "object_pose": object_dir / "pose.json",
            }
            if not (resume and Path(object_context["object_ply"]).exists()):
                run_command(STAGES["sam3d"], command, object_context, cwd=self.repo_root, dry_run=self.dry_run)
            if not self.dry_run:
                ply = require_file(Path(object_context["object_ply"]), f"object 3DGS PLY for {instance.instance_id}")
                mesh = require_file(Path(object_context["object_mesh"]), f"object mesh OBJ for {instance.instance_id}")
                pose = Pose.from_json(Path(object_context["object_pose"]))
                objects.append(
                    Real2SimAsset(
                        name=_safe_name(instance.instance_id),
                        label=instance.label,
                        gaussian_ply=ply,
                        mesh_obj=mesh,
                        pose=pose,
                        dynamic=True,
                        source_instance_id=instance.instance_id,
                    )
                )
        return objects

    def _run_pose_alignment(self, context: dict[str, Any], manifest: Real2SimManifest, *, resume: bool) -> Real2SimManifest:
        try:
            command = resolve_command(STAGES["pose_alignment"], self.config)
        except MissingStageCommand:
            command = None
        if command is None:
            return manifest
        aligned_json = Path(context["aligned_assets_json"])
        if not (resume and aligned_json.exists()):
            aligned_json.parent.mkdir(parents=True, exist_ok=True)
            write_json(aligned_json, manifest.to_json(context["work_dir"]))
            run_command(STAGES["pose_alignment"], command, context, cwd=self.repo_root, dry_run=self.dry_run)
        if self.dry_run or not aligned_json.exists():
            return manifest
        return _manifest_from_alignment_json(aligned_json, manifest)

    def _run_synthetic_multiview_expansion(
        self,
        capture: CaptureInfo,
        context: dict[str, Any],
        *,
        resume: bool,
        enabled: bool,
    ) -> CaptureInfo:
        if not enabled:
            return capture
        if capture.kind != "single_image":
            return capture
        command = resolve_command(STAGES["synthetic_multiview"], self.config)
        if command is None:
            raise MissingStageCommand(
                "synthetic_multiview was requested but no command was configured. "
                "Set stages.synthetic_multiview.command or export REAL2SIM_SYNTHETIC_MULTIVIEW_COMMAND."
            )

        frames_dir = Path(context["synthetic_frames_dir"])
        if not _supported_images(frames_dir):
            frames_dir.mkdir(parents=True, exist_ok=True)
            run_command(STAGES["synthetic_multiview"], command, context, cwd=self.repo_root, dry_run=self.dry_run)
        if not self.dry_run:
            frames = _supported_images(frames_dir)
            if not frames:
                raise StageArtifactError(f"synthetic multiview produced no supported images in {frames_dir}")
        context["frames_dir"] = frames_dir
        return CaptureInfo(source=capture.source, image=capture.image, frames_dir=frames_dir, kind="synthetic_multiview")

    def _synthetic_multiview_mode(self, requested: str) -> str:
        cfg = self.config.get("synthetic_multiview", {})
        if isinstance(cfg, Mapping) and isinstance(cfg.get("mode"), str) and cfg["mode"].strip():
            requested = str(cfg["mode"])
        mode = requested.strip().lower()
        if mode not in {"object", "scene"}:
            raise PipelineError(f"synthetic_multiview mode must be 'object' or 'scene', got: {requested}")
        return mode

    def _synthetic_multiview_prompt(self, prompt: str, *, mode: str = "object") -> str:
        cfg = self.config.get("synthetic_multiview", {})
        if isinstance(cfg, Mapping) and isinstance(cfg.get("prompt"), str) and cfg["prompt"].strip():
            return str(cfg["prompt"])
        if mode == "scene":
            return (
                "Create a photorealistic 4 columns x 2 rows multiview capture contact sheet for scene-level real2sim reconstruction. "
                f"Scene context: {prompt}. Treat the input as the canonical source view and preserve the exact room or outdoor layout, "
                "floor plane, walls, doors, windows, large furniture, static fixtures, lighting direction, material colors, and visible object identities. "
                "Infer only physically plausible nearby camera views from the same static environment; do not invent a different room, replace furniture, "
                "add people, add text, add UI, or move structural elements. Produce exactly 8 equal portrait panels with thin white gutters and no labels, "
                "numbers, captions, icons, or watermarks. Views in reading order: source/front view, front-left 45 degrees, left 90 degrees, rear-left "
                "135 degrees, rear/back, rear-right 225 degrees, right 270 degrees, and top-oblique 30 to 45 degrees. Keep camera height, focal length, "
                "scale, horizon, navigable ground continuity, occlusion relationships, and shadows consistent across all panels. When unseen regions are "
                "not directly observable, complete them conservatively as simple continuations of the visible surfaces so AnySplat receives coherent "
                "scene geometry evidence rather than object-only turntable views."
            )
        return (
            "Create a photorealistic 4 columns x 2 rows multiview capture contact sheet for real2sim reconstruction. "
            f"Subject: {prompt}. Preserve the exact object identity, geometry, material, color, labels, markings, and size relationships. "
            "Produce exactly 8 equal portrait panels with thin white gutters and no text, titles, numbers, captions, UI icons, or watermarks. "
            "Views in reading order: front main, front-left 45 degrees, left 90 degrees, rear-left 135 degrees, rear/back, rear-right 225 degrees, "
            "right 270 degrees, and top-oblique 30 to 45 degrees. Keep the same stationary object or objects on the same tabletop, consistent focal length, "
            "stable soft lighting, sharp focus, full silhouettes with 10 to 15 percent margin, no hands, no people blocking the object, no play overlays, "
            "and avoid blown-out reflections."
        )

    def _prune_assets(self, manifest: Real2SimManifest, work: Path, options: PruneOptions) -> Real2SimManifest:
        pruned_dir = work / "06_pruned"
        prune_context = self._prune_command_context(manifest)
        scene_ply = prune_ply(
            manifest.scene.gaussian_ply,
            pruned_dir / "scene.ply",
            options,
            self.repo_root,
            extra_context=prune_context,
        )
        scene = Real2SimAsset(
            name=manifest.scene.name,
            label=manifest.scene.label,
            gaussian_ply=scene_ply,
            mesh_obj=manifest.scene.mesh_obj,
            pose=manifest.scene.pose,
            dynamic=manifest.scene.dynamic,
            source_instance_id=manifest.scene.source_instance_id,
        )
        objects = []
        for asset in manifest.objects:
            obj_ply = prune_ply(
                asset.gaussian_ply,
                pruned_dir / "objects" / f"{asset.name}.ply",
                options,
                self.repo_root,
                extra_context={**prune_context, "asset_name": asset.name, "asset_label": asset.label},
            )
            objects.append(
                Real2SimAsset(
                    name=asset.name,
                    label=asset.label,
                    gaussian_ply=obj_ply,
                    mesh_obj=asset.mesh_obj,
                    pose=asset.pose,
                    dynamic=asset.dynamic,
                    source_instance_id=asset.source_instance_id,
                )
            )
        return Real2SimManifest(
            source=manifest.source,
            prompt=manifest.prompt,
            scene=scene,
            objects=tuple(objects),
            camera_artifacts=manifest.camera_artifacts,
            metadata={**manifest.metadata, "pruning": options.__dict__},
        )

    def _prune_command_context(self, manifest: Real2SimManifest) -> dict[str, Path | str]:
        cameras = manifest.camera_artifacts.get("cameras_json")
        source = manifest.source
        source_image = manifest.camera_artifacts.get("source_image", source)
        return {
            "source": source,
            "source_image": source_image,
            "image": source_image,
            "cameras_json": cameras or "",
            "cameras": cameras or "",
        }

    def _prune_options(self) -> PruneOptions:
        cfg = self.config.get("pruning", {})
        if not isinstance(cfg, Mapping):
            cfg = {}
        return PruneOptions(
            enabled=bool(cfg.get("enabled", True)),
            keep_ratio=float(cfg.get("keep_ratio", 0.3)),
            keep_count=int(cfg["keep_count"]) if cfg.get("keep_count") is not None else None,
            mode=str(cfg.get("mode", "opacity_area")),
            external_command=command_preferring_env(cfg.get("external_command"), os.environ.get("REAL2SIM_SPEEDY_SPLAT_COMMAND")),
        )

    def _asset_from_json_entry(self, entry: Mapping[str, Any], base_dir: Path) -> Real2SimAsset:
        if not isinstance(entry, Mapping):
            raise StageArtifactError(f"object asset entry must be a JSON object: {entry}")
        name = str(entry.get("name", entry.get("id", entry.get("label", "object"))))
        label = str(entry.get("label", name))
        ply = _resolve_path(base_dir, entry.get("gaussian_ply", entry.get("ply")))
        mesh_raw = entry.get("mesh_obj", entry.get("mesh"))
        pose_raw = entry.get("pose")
        if isinstance(pose_raw, Mapping):
            pose_file = base_dir / f".{_safe_name(name)}_pose.json"
            write_json(pose_file, pose_raw)
            pose = Pose.from_json(pose_file)
            pose_file.unlink(missing_ok=True)
        elif pose_raw:
            pose = Pose.from_json(_resolve_path(base_dir, pose_raw))
        else:
            pose = Pose()
        return Real2SimAsset(
            name=_safe_name(name),
            label=label,
            gaussian_ply=require_file(ply, f"object gaussian PLY for {name}"),
            mesh_obj=require_file(_resolve_path(base_dir, mesh_raw), f"object mesh OBJ for {name}") if mesh_raw else None,
            pose=pose,
            dynamic=bool(entry.get("dynamic", True)),
            source_instance_id=str(entry.get("source_instance_id", name)),
        )


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = read_json(path)
    if not isinstance(data, dict):
        raise PipelineError(f"config must be a JSON object: {path}")
    return data


def copy_optional_package_report(source: Path | None, destination: Path, label: str) -> None:
    if source is None:
        return
    src = require_file(source.resolve(), label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, destination)


def scene_structure_layers(report: Mapping[str, Any], *, proxy_count: int = 0) -> dict[str, Any]:
    semantic_planes = report.get("semantic_planes")
    planes = semantic_planes if isinstance(semantic_planes, list) else []
    static_proxies: list[dict[str, Any]] = []
    walls: list[dict[str, Any]] = []
    support_surfaces: list[dict[str, Any]] = []
    navigation_surfaces: list[dict[str, Any]] = []
    has_floor = False
    for index, plane in enumerate(planes[:32]):
        if not isinstance(plane, Mapping):
            continue
        label = str(plane.get("semantic_label") or "unknown")
        has_floor = has_floor or label == "floor"
        proxy = plane.get("collision_proxy_hint")
        normalized_proxy = normalized_collision_proxy(proxy, label=label, index=index)
        if normalized_proxy is not None and is_static_structure_label(label):
            static_proxies.append(normalized_proxy)
        surface = {
            "id": f"{slug_token(label)}_{index:03d}",
            "semantic_label": label,
            "axis": plane.get("axis"),
            "coordinate": plane.get("coordinate"),
            "area_ratio": plane.get("area_ratio"),
            "extent": plane.get("extent"),
            "collision_proxy_id": normalized_proxy["id"] if normalized_proxy else None,
        }
        if label.startswith("wall_") or label == "wall_visible":
            walls.append(surface)
        if label == "support_surface":
            support_surfaces.append(surface)
            navigation_surfaces.append({**surface, "navigation_role": "placeable_support"})
        elif label == "floor":
            navigation_surfaces.append({**surface, "navigation_role": "walkable_floor"})
    completion_proxies = room_completion_proxies(report, has_floor=has_floor, walls=walls, support_surfaces=support_surfaces)
    for completion_proxy in completion_proxies:
        static_proxies.append(completion_proxy)
        if completion_proxy.get("semantic_label") == "floor_completion":
            navigation_surfaces.append(
                {
                    "id": completion_proxy["id"],
                    "semantic_label": "floor_completion",
                    "axis": completion_proxy.get("axis"),
                    "coordinate": completion_proxy.get("coordinate"),
                    "area_ratio": None,
                    "extent": completion_proxy.get("extent"),
                    "collision_proxy_id": completion_proxy["id"],
                    "navigation_role": "walkable_floor_inferred",
                    "inferred_completion": True,
                }
            )
    readiness = report.get("commercial_readiness") if isinstance(report.get("commercial_readiness"), Mapping) else {}
    return {
        "format": "gs-playground-real2sim-static-structure-v1",
        "source_report_format": report.get("format"),
        "static_collision_proxies": static_proxies,
        "completion_proxies": completion_proxies,
        "walls": walls,
        "support_surfaces": support_surfaces,
        "navigation_surfaces": navigation_surfaces,
        "quality": {
            "structure_score": report.get("structure_score"),
            "semantic_structure_score": readiness.get("semantic_structure_score"),
            "source_collision_proxy_count": proxy_count,
            "completion_proxy_count": len(completion_proxies),
            "has_floor": readiness.get("has_floor"),
            "has_floor_completion": any(proxy.get("semantic_label") == "floor_completion" for proxy in completion_proxies),
            "has_wall": readiness.get("has_wall"),
            "has_support_surface": readiness.get("has_support_surface"),
        },
        "commercial_gap_notes": [
            "Static structure layers are now consumable by simulation code; inferred completion proxies are marked and should not be confused with directly observed mesh.",
            "Navigation surfaces are proxy hints; production navmesh generation still needs topology cleanup and traversability validation.",
        ],
    }


def room_completion_proxies(
    report: Mapping[str, Any],
    *,
    has_floor: bool,
    walls: list[dict[str, Any]],
    support_surfaces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if has_floor:
        return []
    bounds = report.get("bounds")
    if not isinstance(bounds, Mapping):
        return []
    low = numeric_triplet(bounds.get("p05") or bounds.get("min"))
    high = numeric_triplet(bounds.get("p95") or bounds.get("max"))
    if low is None or high is None:
        return []
    up_axis_name = str(report.get("up_axis") or "z")
    up_axis = "xyz".find(up_axis_name)
    if up_axis < 0:
        up_axis = 2
    spans = [max(high[idx] - low[idx], 0.02) for idx in range(3)]
    if not walls and not support_surfaces:
        return []
    center = [(low[idx] + high[idx]) * 0.5 for idx in range(3)]
    center[up_axis] = low[up_axis]
    size = [max(spans[idx] * 0.5, 0.05) for idx in range(3)]
    size[up_axis] = max(spans[up_axis] * 0.01, 0.02)
    extent = {name: [round(low[idx], 6), round(high[idx], 6)] for idx, name in enumerate("xyz")}
    return [
        {
            "id": "floor_completion_000",
            "type": "box",
            "semantic_label": "floor_completion",
            "center": [round(value, 6) for value in center],
            "size": [round(value, 6) for value in size],
            "axis": up_axis_name if up_axis_name in {"x", "y", "z"} else "z",
            "coordinate": round(float(center[up_axis]), 6),
            "extent": extent,
            "mjcf_geom_name": "scene_completion_collision_floor_000",
            "inferred_completion": True,
            "inference_source": {
                "reason": "observed walls/support surfaces but no explicit floor plane",
                "bounds_source": "scene_structure_report.bounds.p05/p95",
            },
            "physics_material": default_static_physics_material("floor"),
        }
    ]


def normalized_collision_proxy(proxy: Any, *, label: str, index: int) -> dict[str, Any] | None:
    if not isinstance(proxy, Mapping) or proxy.get("type") != "box":
        return None
    center = numeric_triplet(proxy.get("center"))
    size = numeric_triplet(proxy.get("size"))
    if center is None or size is None:
        return None
    return {
        "id": f"{slug_token(label)}_{index:03d}",
        "type": "box",
        "semantic_label": label,
        "center": [round(value, 6) for value in center],
        "size": [round(value, 6) for value in size],
        "mjcf_geom_name": f"scene_structure_collision_{slug_token(label)}_{index:03d}",
        "physics_material": default_static_physics_material(label),
    }


def is_static_structure_label(label: str) -> bool:
    return label in {"floor", "ceiling", "support_surface", "wall_visible"} or label.startswith("wall_")


def default_static_physics_material(label: str) -> dict[str, Any]:
    if label == "support_surface":
        return {"movable": False, "friction": [0.85, 0.02, 0.001], "restitution": 0.02, "contact_role": "support"}
    if label == "floor":
        return {"movable": False, "friction": [0.9, 0.02, 0.001], "restitution": 0.01, "contact_role": "walkable"}
    if label == "ceiling":
        return {"movable": False, "friction": [0.7, 0.02, 0.001], "restitution": 0.01, "contact_role": "overhead_boundary"}
    return {"movable": False, "friction": [0.75, 0.02, 0.001], "restitution": 0.01, "contact_role": "static_boundary"}


def commercial_readiness_report(
    root: Path,
    structure_layers: Mapping[str, Any],
    scene_structure_report: Mapping[str, Any],
    *,
    proxy_count: int,
    metric_scale: Mapping[str, Any] | None = None,
    navigation_acceptance: Mapping[str, Any] | None = None,
    material_readiness: Mapping[str, Any] | None = None,
    task_affordance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = read_json(root / "package_manifest.json") if (root / "package_manifest.json").is_file() else {}
    texture_report = read_optional_json(root / "artifacts" / "colmap_texture_bake" / "texture_bake_report.json")
    mesh_cleanup = read_optional_json(root / "artifacts" / "mesh_cleanup" / "cleanup_report.json")
    movable_report = read_optional_json(root / "artifacts" / "assets" / "movable_assets.json")
    objects = manifest.get("objects") if isinstance(manifest, Mapping) else []
    object_count = len(objects) if isinstance(objects, list) else 0
    scene = manifest.get("scene") if isinstance(manifest, Mapping) else {}
    scene_mesh_present = isinstance(scene, Mapping) and bool(scene.get("mesh_obj"))
    static_proxies = structure_layers.get("static_collision_proxies") if isinstance(structure_layers, Mapping) else []
    navigation_surfaces = structure_layers.get("navigation_surfaces") if isinstance(structure_layers, Mapping) else []
    walls = structure_layers.get("walls") if isinstance(structure_layers, Mapping) else []
    support_surfaces = structure_layers.get("support_surfaces") if isinstance(structure_layers, Mapping) else []
    texture_metrics = texture_quality_metrics(texture_report)
    candidate_slots = movable_report.get("candidate_slots") if isinstance(movable_report, Mapping) else []
    candidate_slot_count = len(candidate_slots) if isinstance(candidate_slots, list) else 0
    movable_quality = movable_report.get("quality") if isinstance(movable_report, Mapping) and isinstance(movable_report.get("quality"), Mapping) else {}
    movable_assets = movable_report.get("movable_assets") if isinstance(movable_report, Mapping) and isinstance(movable_report.get("movable_assets"), list) else []
    movable_physics = movable_physics_material_quality(movable_assets)
    commercial_movable_assets = bool(movable_quality.get("commercial_asset_grade"))
    production_asset_count = int(movable_quality.get("production_asset_count") or 0)
    provisional_asset_count = int(movable_quality.get("provisional_asset_count") or 0)
    incomplete_asset_count = int(movable_quality.get("incomplete_asset_count") or 0)
    static_material_hint_count = sum(
        1 for proxy in static_proxies if isinstance(proxy, Mapping) and valid_static_physics_material(proxy.get("physics_material"))
    )
    static_materials_ready = bool(static_proxies) and static_material_hint_count == len(static_proxies)
    room_report = room_completeness_report(root, structure_layers, scene_structure_report)
    room_summary = room_report.get("summary") if isinstance(room_report.get("summary"), Mapping) else {}
    metric_scale_summary = metric_scale.get("summary") if isinstance(metric_scale, Mapping) and isinstance(metric_scale.get("summary"), Mapping) else {}
    navigation_summary = (
        navigation_acceptance.get("summary")
        if isinstance(navigation_acceptance, Mapping) and isinstance(navigation_acceptance.get("summary"), Mapping)
        else {}
    )
    material_summary = (
        material_readiness.get("summary")
        if isinstance(material_readiness, Mapping) and isinstance(material_readiness.get("summary"), Mapping)
        else {}
    )
    task_affordance_summary = (
        task_affordance.get("summary")
        if isinstance(task_affordance, Mapping) and isinstance(task_affordance.get("summary"), Mapping)
        else {}
    )
    gates = [
        readiness_gate(
            "dense_scene_geometry",
            bool(scene_mesh_present and int(scene_structure_report.get("face_count") or 0) >= 100_000),
            0.15,
            "scene mesh with at least 100k analyzed faces",
            {"face_count": scene_structure_report.get("face_count"), "scene_mesh_present": scene_mesh_present},
        ),
        readiness_gate(
            "texture_bake",
            bool(texture_metrics["passed"]),
            0.12,
            "projected texture covers most reconstructed mesh faces, with atlas pixel coverage reported separately",
            texture_metrics,
        ),
        readiness_gate(
            "material_readiness",
            bool(material_summary.get("passed")),
            0.05,
            "package contains texture-backed MJCF/OBJ material assets with all referenced texture files present",
            {
                "score": material_summary.get("score"),
                "texture_backed_material_count": material_summary.get("texture_backed_material_count"),
                "missing_texture_file_count": material_summary.get("missing_texture_file_count"),
                "scene_mesh_has_texture": material_summary.get("scene_mesh_has_texture"),
                "blockers": material_summary.get("blockers"),
            },
        ),
        readiness_gate(
            "static_structure_layers",
            bool(static_proxies and (walls or support_surfaces)),
            0.16,
            "walls/support surfaces are exported as static structure layers",
            {"static_collision_proxy_count": len(static_proxies), "wall_count": len(walls), "support_surface_count": len(support_surfaces)},
        ),
        readiness_gate(
            "physics_collision",
            proxy_count > 0,
            0.14,
            "semantic collision proxies are injected into MJCF",
            {"scene_structure_collision_proxies": proxy_count},
        ),
        readiness_gate(
            "navigation_surface_hints",
            bool(navigation_surfaces),
            0.07,
            "walkable or placeable navigation surfaces are available",
            {"navigation_surface_count": len(navigation_surfaces)},
        ),
        readiness_gate(
            "movable_asset_instances",
            commercial_movable_assets,
            0.18,
            "movable objects are separated from the static scene as production-grade reconstructed assets",
            {
                "object_count": object_count,
                "candidate_slot_count": candidate_slot_count,
                "production_asset_count": production_asset_count,
                "provisional_asset_count": provisional_asset_count,
                "incomplete_asset_count": incomplete_asset_count,
                "commercial_asset_grade": commercial_movable_assets,
            },
        ),
        readiness_gate(
            "physical_material_hints",
            static_materials_ready and bool(movable_physics["passed"]),
            0.08,
            "static structure and movable assets include mass, friction, restitution, and contact-role physical material hints where applicable",
            {
                "static_material_hint_count": static_material_hint_count,
                "static_proxy_count": len(static_proxies),
                "movable_material_hint_count": movable_physics["ready_count"],
                "movable_asset_count": movable_physics["asset_count"],
                "movable_material_incomplete_count": movable_physics["incomplete_count"],
                "movable_material_blockers": movable_physics["blockers"],
            },
        ),
        readiness_gate(
            "task_affordance_targets",
            bool(task_affordance_summary.get("passed")),
            0.04,
            "production movable assets expose finite task poses and actionable contact roles for robot manipulation probes",
            {
                "score": task_affordance_summary.get("score"),
                "target_count": task_affordance_summary.get("target_count"),
                "ready_target_count": task_affordance_summary.get("ready_target_count"),
                "blockers": task_affordance_summary.get("blockers"),
            },
        ),
        readiness_gate(
            "room_completeness",
            bool(room_summary.get("passed")),
            0.07,
            "capture or explicit completion contains enough structure for floor plus wall-level room context",
            {
                "score": room_summary.get("score"),
                "has_observed_floor": room_summary.get("has_observed_floor"),
                "has_floor_completion": room_summary.get("has_floor_completion"),
                "wall_count": len(walls),
                "missing_required_elements": room_summary.get("missing_required_elements"),
            },
        ),
        readiness_gate(
            "metric_scale_plausibility",
            bool(metric_scale_summary.get("passed")),
            0.05,
            "scene bounds and structure proxies are finite and plausible for metric robot simulation",
            {
                "score": metric_scale_summary.get("score"),
                "bounds_extent": metric_scale_summary.get("bounds_extent"),
                "horizontal_extent_range": metric_scale_summary.get("horizontal_extent_range"),
                "vertical_extent": metric_scale_summary.get("vertical_extent"),
                "blockers": metric_scale_summary.get("blockers"),
            },
        ),
        readiness_gate(
            "navigation_acceptance",
            bool(navigation_summary.get("passed")),
            0.03,
            "robot navigation has a walkable surface with enough metric footprint for route planning",
            {
                "score": navigation_summary.get("score"),
                "walkable_surface_count": navigation_summary.get("walkable_surface_count"),
                "largest_walkable_area": navigation_summary.get("largest_walkable_area"),
                "blockers": navigation_summary.get("blockers"),
            },
        ),
    ]
    total_weight = sum(float(gate["weight"]) for gate in gates)
    raw_score = sum(float(gate["weight"]) for gate in gates if gate["passed"])
    score = round(raw_score / max(total_weight, 1e-9), 6)
    blockers = [gate["id"] for gate in gates if not gate["passed"]]
    return {
        "format": "gs-playground-real2sim-commercial-readiness-v1",
        "summary": {
            "score": score,
            "target": 0.85,
            "commercial_grade": score >= 0.85 and not blockers,
            "blocker_count": len(blockers),
            "blockers": blockers,
        },
        "gates": gates,
        "evidence": {
            "object_count": object_count,
            "movable_candidate_slot_count": candidate_slot_count,
            "movable_production_asset_count": production_asset_count,
            "movable_provisional_asset_count": provisional_asset_count,
            "movable_incomplete_asset_count": incomplete_asset_count,
            "movable_commercial_asset_grade": commercial_movable_assets,
            "movable_physics_material_ready_count": movable_physics["ready_count"],
            "movable_physics_material_incomplete_count": movable_physics["incomplete_count"],
            "task_affordance_score": task_affordance_summary.get("score"),
            "scene_structure_score": scene_structure_report.get("structure_score"),
            "semantic_structure_score": scene_structure_report.get("commercial_readiness", {}).get("semantic_structure_score")
            if isinstance(scene_structure_report.get("commercial_readiness"), Mapping)
            else None,
            "mesh_cleanup_kept_area_ratio": mesh_cleanup.get("kept_area_ratio") if isinstance(mesh_cleanup, Mapping) else None,
            "texture_coverage_ratio": texture_metrics.get("texture_coverage_ratio"),
            "textured_face_ratio": texture_metrics.get("textured_face_ratio"),
            "projected_face_ratio": texture_metrics.get("projected_face_ratio"),
            "atlas_pixel_coverage_ratio": texture_metrics.get("atlas_pixel_coverage_ratio"),
            "room_completeness_score": room_summary.get("score"),
            "metric_scale_score": metric_scale_summary.get("score"),
            "navigation_acceptance_score": navigation_summary.get("score"),
            "material_readiness_score": material_summary.get("score"),
        },
        "next_actions": commercial_next_actions(blockers),
    }


def commercial_acceptance_manifest(
    root: Path,
    *,
    commercial_readiness: Mapping[str, Any],
    room_completeness: Mapping[str, Any],
    metric_scale: Mapping[str, Any],
    navigation_acceptance: Mapping[str, Any],
    material_readiness: Mapping[str, Any],
    task_affordance: Mapping[str, Any],
    movable_assets: Mapping[str, Any],
    structure_layers: Mapping[str, Any],
) -> dict[str, Any]:
    reports = {
        "commercial_readiness": "artifacts/quality/commercial_readiness.json",
        "room_completeness": "artifacts/quality/room_completeness.json",
        "metric_scale": "artifacts/quality/metric_scale.json",
        "navigation_acceptance": "artifacts/quality/navigation_acceptance.json",
        "material_readiness": "artifacts/quality/material_readiness.json",
        "task_affordance": "artifacts/quality/task_affordance.json",
        "movable_assets": "artifacts/assets/movable_assets.json",
        "static_structure_layers": "artifacts/scene_structure/static_structure.json",
    }
    entrypoints = {
        "motrix_import": "motrix_import.json",
        "package_manifest": "package_manifest.json",
        "scene_xml": "mjcf/scene.xml",
        "viewer_config": "configs/go2_real2sim.json",
        "scene_gaussians": "3dgs/scene.ply",
    }
    entrypoint_audits = {key: package_file_audit(root, path) for key, path in entrypoints.items()}
    report_audits = {key: package_file_audit(root, path) for key, path in reports.items()}
    manifest = read_json(root / "package_manifest.json") if (root / "package_manifest.json").is_file() else {}
    metadata = manifest.get("metadata") if isinstance(manifest, Mapping) and isinstance(manifest.get("metadata"), Mapping) else {}
    checks = [
        acceptance_check(
            "package_entrypoints",
            {},
            explicit_passed=all(item["present"] and item["size"] > 0 for item in entrypoint_audits.values()),
            evidence=entrypoint_audits,
        ),
        acceptance_check(
            "quality_report_files",
            {},
            explicit_passed=all(item["present"] and item["size"] > 0 for item in report_audits.values()),
            evidence=report_audits,
        ),
        acceptance_check("commercial_readiness", commercial_readiness, summary_key="commercial_grade"),
        acceptance_check("room_completeness", room_completeness),
        acceptance_check("metric_scale", metric_scale),
        acceptance_check("navigation_acceptance", navigation_acceptance),
        acceptance_check("material_readiness", material_readiness),
        acceptance_check("task_affordance", task_affordance),
        acceptance_check(
            "movable_assets",
            movable_assets,
            summary_key="commercial_asset_grade",
            quality_summary=True,
        ),
        acceptance_check(
            "static_structure_layers",
            structure_layers,
            explicit_passed=bool(
                structure_layers.get("static_collision_proxies")
                and structure_layers.get("navigation_surfaces")
                and structure_layers.get("quality", {}).get("total_collision_proxy_count")
            )
            if isinstance(structure_layers, Mapping)
            else False,
        ),
    ]
    blockers = [check["id"] for check in checks if not check["passed"]]
    score = sum(1 for check in checks if check["passed"]) / float(max(len(checks), 1))
    return {
        "format": "gs-playground-real2sim-commercial-acceptance-v1",
        "package_root": root.as_posix(),
        "summary": {
            "score": round(float(score), 6),
            "target": 1.0,
            "accepted": not blockers,
            "check_count": len(checks),
            "passed_check_count": len(checks) - len(blockers),
            "blocker_count": len(blockers),
            "blockers": blockers,
        },
        "package": {
            "format": metadata.get("package_format"),
            "robot": metadata.get("robot"),
            "motrix_scene_visual_mode": metadata.get("motrix_scene_visual_mode"),
            "motrix_visual_maxgeom": metadata.get("motrix_visual_maxgeom"),
        },
        "entrypoints": entrypoints,
        "reports": reports,
        "artifact_integrity": {
            "entrypoints": entrypoint_audits,
            "reports": report_audits,
        },
        "checks": checks,
        "next_actions": commercial_next_actions(
            commercial_readiness.get("summary", {}).get("blockers", [])
            if isinstance(commercial_readiness.get("summary"), Mapping)
            else []
        ),
        "notes": [
            "This manifest is the package-level commercial delivery ledger.",
            "It summarizes generated quality artifacts and points consumers to the detailed reports.",
        ],
    }


def package_file_audit(root: Path, package_path: str) -> dict[str, Any]:
    path = root / package_path
    present = path.is_file()
    size = path.stat().st_size if present else 0
    return {
        "path": package_path,
        "present": present,
        "size": size,
        "sha256": sha256_file(path) if present and size > 0 else None,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acceptance_check(
    check_id: str,
    report: Mapping[str, Any],
    *,
    summary_key: str = "passed",
    quality_summary: bool = False,
    explicit_passed: bool | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if explicit_passed is not None:
        return {"id": check_id, "passed": bool(explicit_passed), "summary": {}, "evidence": dict(evidence or {})}
    summary = report.get("quality") if quality_summary else report.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    return {
        "id": check_id,
        "passed": bool(summary.get(summary_key)),
        "summary": dict(summary),
        "evidence": dict(evidence or {}),
    }


def material_readiness_report(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "package_manifest.json") if (root / "package_manifest.json").is_file() else {}
    scene = manifest.get("scene") if isinstance(manifest, Mapping) else {}
    objects = manifest.get("objects") if isinstance(manifest.get("objects"), list) else []
    scene_mesh = resolve_package_relative_path(root, scene.get("mesh_obj")) if isinstance(scene, Mapping) else None
    object_meshes = [
        path
        for entry in objects
        if isinstance(entry, Mapping)
        for path in [resolve_package_relative_path(root, entry.get("mesh_obj"))]
        if path is not None
    ]
    mjcf_materials = mjcf_material_readiness(root / "mjcf" / "scene.xml", root)
    mesh_materials = obj_material_readiness([path for path in [scene_mesh, *object_meshes] if path is not None], root)
    all_texture_files = sorted({*mjcf_materials["texture_files"], *mesh_materials["texture_files"]})
    missing_texture_files = sorted({*mjcf_materials["missing_texture_files"], *mesh_materials["missing_texture_files"]})
    scene_mesh_has_texture = bool(scene_mesh is not None and mesh_materials["mesh_texture_counts"].get(scene_mesh.as_posix(), 0) > 0)
    texture_backed_material_count = int(mjcf_materials["texture_backed_material_count"]) + int(mesh_materials["texture_backed_material_count"])
    blockers: list[str] = []
    if scene_mesh is not None and not scene_mesh_has_texture:
        blockers.append("scene_mesh_missing_texture_material")
    if texture_backed_material_count <= 0:
        blockers.append("missing_texture_backed_material")
    if missing_texture_files:
        blockers.append("missing_texture_files")
    if scene_mesh is None and not object_meshes:
        blockers.append("missing_mesh_assets")
    score = 1.0
    if "missing_mesh_assets" in blockers:
        score = 0.0
    elif "missing_texture_backed_material" in blockers:
        score = 0.25
    elif "scene_mesh_missing_texture_material" in blockers:
        score = 0.55
    elif "missing_texture_files" in blockers:
        score = 0.7
    return {
        "format": "gs-playground-real2sim-material-readiness-v1",
        "summary": {
            "score": round(score, 6),
            "passed": not blockers,
            "scene_mesh_present": scene_mesh is not None,
            "scene_mesh_has_texture": scene_mesh_has_texture,
            "object_mesh_count": len(object_meshes),
            "texture_backed_material_count": texture_backed_material_count,
            "texture_file_count": len(all_texture_files),
            "missing_texture_file_count": len(missing_texture_files),
            "blockers": blockers,
        },
        "inputs": {
            "package_manifest": "package_manifest.json",
            "scene_xml": "mjcf/scene.xml",
            "scene_mesh": scene.get("mesh_obj") if isinstance(scene, Mapping) else None,
            "object_meshes": [path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix() for path in object_meshes],
        },
        "evidence": {
            "mjcf": mjcf_materials["evidence"],
            "meshes": mesh_materials["evidence"],
            "texture_files": [path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix() for path in all_texture_files],
            "missing_texture_files": [path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix() for path in missing_texture_files],
        },
        "diagnostics": material_readiness_diagnostics(blockers),
    }


def resolve_package_relative_path(root: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path if path.is_file() else None
    resolved = (root / path).resolve()
    return resolved if resolved.is_file() else None


def mjcf_material_readiness(scene_xml: Path, root: Path) -> dict[str, Any]:
    if not scene_xml.is_file():
        return {"texture_files": [], "missing_texture_files": [], "texture_backed_material_count": 0, "evidence": {"present": False}}
    try:
        tree = ET.parse(scene_xml)
    except ET.ParseError:
        return {"texture_files": [], "missing_texture_files": [], "texture_backed_material_count": 0, "evidence": {"present": True, "parse_error": True}}
    texture_by_name: dict[str, Path] = {}
    for texture in tree.findall(".//texture"):
        name = texture.get("name")
        file_value = texture.get("file")
        if not name or not file_value:
            continue
        path = (scene_xml.parent / "../meshes" / file_value).resolve()
        if (root / file_value).is_file():
            path = (root / file_value).resolve()
        texture_by_name[name] = path
    backed = 0
    for material in tree.findall(".//material"):
        if material.get("texture") in texture_by_name:
            backed += 1
    texture_files = [path for path in texture_by_name.values() if path.is_file()]
    missing = [path for path in texture_by_name.values() if not path.is_file()]
    return {
        "texture_files": texture_files,
        "missing_texture_files": missing,
        "texture_backed_material_count": backed,
        "evidence": {
            "present": True,
            "texture_count": len(texture_by_name),
            "material_with_texture_count": backed,
            "missing_texture_file_count": len(missing),
        },
    }


def obj_material_readiness(meshes: list[Path], root: Path) -> dict[str, Any]:
    texture_files: list[Path] = []
    missing: list[Path] = []
    backed = 0
    mesh_texture_counts: dict[str, int] = {}
    mesh_evidence: list[dict[str, Any]] = []
    for mesh in meshes:
        mesh_textures, mesh_missing, material_count = obj_texture_materials(mesh)
        texture_files.extend(mesh_textures)
        missing.extend(mesh_missing)
        backed += material_count
        mesh_texture_counts[mesh.as_posix()] = material_count
        mesh_evidence.append(
            {
                "mesh": mesh.relative_to(root).as_posix() if mesh.is_relative_to(root) else mesh.as_posix(),
                "texture_backed_material_count": material_count,
                "texture_file_count": len(mesh_textures),
                "missing_texture_file_count": len(mesh_missing),
            }
        )
    return {
        "texture_files": texture_files,
        "missing_texture_files": missing,
        "texture_backed_material_count": backed,
        "mesh_texture_counts": mesh_texture_counts,
        "evidence": mesh_evidence,
    }


def obj_texture_materials(mesh: Path) -> tuple[list[Path], list[Path], int]:
    texture_files: list[Path] = []
    missing: list[Path] = []
    material_count = 0
    if not mesh.is_file():
        return texture_files, missing, material_count
    mtl_names: list[str] = []
    for line in mesh.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("mtllib "):
            mtl_names.extend(line.split()[1:])
    for name in mtl_names:
        mtl = (mesh.parent / name).resolve()
        if not mtl.is_file():
            missing.append(mtl)
            continue
        for texture_name in mtl_diffuse_texture_names(mtl):
            texture_path = (mtl.parent / texture_name).resolve()
            material_count += 1
            if texture_path.is_file():
                texture_files.append(texture_path)
            else:
                missing.append(texture_path)
    return texture_files, missing, material_count


def mtl_diffuse_texture_names(mtl: Path) -> list[str]:
    names: list[str] = []
    for line in mtl.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if parts and parts[0] == "map_Kd" and len(parts) >= 2:
            names.append(parts[-1])
    return names


def metric_scale_report(root: Path, structure_layers: Mapping[str, Any], scene_structure_report: Mapping[str, Any]) -> dict[str, Any]:
    bounds = scene_structure_report.get("bounds") if isinstance(scene_structure_report.get("bounds"), Mapping) else {}
    low = numeric_triplet(bounds.get("p05") or bounds.get("min")) if isinstance(bounds, Mapping) else None
    high = numeric_triplet(bounds.get("p95") or bounds.get("max")) if isinstance(bounds, Mapping) else None
    extents = tuple(max(float(high[idx] - low[idx]), 0.0) for idx in range(3)) if low is not None and high is not None else None
    up_axis_name = str(scene_structure_report.get("up_axis") or "z")
    up_axis = "xyz".find(up_axis_name)
    if up_axis < 0:
        up_axis = 2
    horizontal = [extents[idx] for idx in range(3) if idx != up_axis] if extents is not None else []
    horizontal_max = max(horizontal) if horizontal else 0.0
    horizontal_min = min(horizontal) if horizontal else 0.0
    vertical = extents[up_axis] if extents is not None else 0.0
    proxies = structure_layers.get("static_collision_proxies") if isinstance(structure_layers, Mapping) else []
    proxies = proxies if isinstance(proxies, list) else []
    proxy_sizes: list[tuple[float, float, float]] = []
    for proxy in proxies:
        if not isinstance(proxy, Mapping):
            continue
        size = numeric_triplet(proxy.get("size"))
        if size is not None:
            proxy_sizes.append(size)
    proxy_max_extent = max((max(size) for size in proxy_sizes), default=0.0)
    proxy_min_extent = min((min(size) for size in proxy_sizes), default=0.0)
    blockers: list[str] = []
    if extents is None:
        blockers.append("missing_metric_bounds")
    elif not all(0.0 < value < 50.0 for value in extents):
        blockers.append("invalid_or_extreme_bounds")
    if extents is not None and not (0.5 <= horizontal_max <= 20.0 and 0.2 <= horizontal_min <= 20.0):
        blockers.append("implausible_horizontal_extent")
    if extents is not None and not (0.4 <= vertical <= 6.0):
        blockers.append("implausible_vertical_extent")
    if not proxy_sizes:
        blockers.append("missing_structure_proxy_sizes")
    elif not (0.005 <= proxy_min_extent and proxy_max_extent <= 20.0):
        blockers.append("implausible_structure_proxy_size")
    score = 1.0
    if "missing_metric_bounds" in blockers or "invalid_or_extreme_bounds" in blockers:
        score -= 0.45
    if "implausible_horizontal_extent" in blockers:
        score -= 0.22
    if "implausible_vertical_extent" in blockers:
        score -= 0.18
    if "missing_structure_proxy_sizes" in blockers or "implausible_structure_proxy_size" in blockers:
        score -= 0.15
    score = round(max(score, 0.0), 6)
    return {
        "format": "gs-playground-real2sim-metric-scale-v1",
        "package_root": root.as_posix(),
        "summary": {
            "score": score,
            "target": 0.8,
            "passed": score >= 0.8 and not blockers,
            "blocker_count": len(blockers),
            "blockers": blockers,
            "bounds_extent": [round(value, 6) for value in extents] if extents is not None else None,
            "horizontal_extent_range": [round(horizontal_min, 6), round(horizontal_max, 6)] if horizontal else None,
            "vertical_extent": round(float(vertical), 6) if extents is not None else None,
            "structure_proxy_count": len(proxy_sizes),
            "structure_proxy_extent_range": [round(float(proxy_min_extent), 6), round(float(proxy_max_extent), 6)] if proxy_sizes else None,
        },
        "inputs": {
            "scene_structure_report": "artifacts/scene_structure/scene_structure_report.json",
            "static_structure_layers": "artifacts/scene_structure/static_structure.json",
            "bounds": scene_structure_report.get("bounds"),
            "up_axis": up_axis_name if up_axis_name in {"x", "y", "z"} else "z",
        },
        "diagnostics": metric_scale_diagnostics(blockers),
    }


def navigation_acceptance_report(root: Path, structure_layers: Mapping[str, Any]) -> dict[str, Any]:
    surfaces = structure_layers.get("navigation_surfaces") if isinstance(structure_layers, Mapping) else []
    surfaces = surfaces if isinstance(surfaces, list) else []
    walkable: list[dict[str, Any]] = []
    for surface in surfaces:
        if not isinstance(surface, Mapping):
            continue
        role = str(surface.get("navigation_role") or "")
        if not role.startswith("walkable_floor"):
            continue
        extent = surface.get("extent")
        size = navigation_surface_size(extent)
        area = size[0] * size[1] if size is not None else 0.0
        walkable.append(
            {
                "id": surface.get("id"),
                "semantic_label": surface.get("semantic_label"),
                "navigation_role": role,
                "inferred_completion": bool(surface.get("inferred_completion")),
                "size": [round(float(value), 6) for value in size] if size is not None else None,
                "area": round(float(area), 6),
            }
        )
    largest_area = max((float(item["area"]) for item in walkable), default=0.0)
    blockers: list[str] = []
    if not walkable:
        blockers.append("missing_walkable_floor_surface")
    if walkable and largest_area < 0.5:
        blockers.append("walkable_surface_too_small")
    if walkable and all(item.get("inferred_completion") for item in walkable):
        blockers.append("walkable_surface_inferred_only")
    score = 1.0
    if "missing_walkable_floor_surface" in blockers:
        score -= 0.55
    if "walkable_surface_too_small" in blockers:
        score -= 0.25
    if "walkable_surface_inferred_only" in blockers:
        score -= 0.2
    score = round(max(score, 0.0), 6)
    return {
        "format": "gs-playground-real2sim-navigation-acceptance-v1",
        "package_root": root.as_posix(),
        "summary": {
            "score": score,
            "target": 0.8,
            "passed": score >= 0.8 and not blockers,
            "blocker_count": len(blockers),
            "blockers": blockers,
            "navigation_surface_count": len(surfaces),
            "walkable_surface_count": len(walkable),
            "largest_walkable_area": round(float(largest_area), 6),
        },
        "inputs": {
            "static_structure_layers": "artifacts/scene_structure/static_structure.json",
        },
        "walkable_surfaces": walkable,
        "diagnostics": navigation_acceptance_diagnostics(blockers),
    }


def navigation_surface_size(extent: Any) -> tuple[float, float] | None:
    if not isinstance(extent, Mapping):
        return None
    spans: list[float] = []
    for axis in "xyz":
        values = extent.get(axis)
        if isinstance(values, list) and len(values) == 2:
            try:
                spans.append(max(float(values[1]) - float(values[0]), 0.0))
            except (TypeError, ValueError):
                return None
    if len(spans) != 3:
        return None
    horizontal = sorted(spans, reverse=True)[:2]
    return float(horizontal[0]), float(horizontal[1])


def texture_quality_metrics(texture_report: Any) -> dict[str, Any]:
    if not isinstance(texture_report, Mapping):
        return {
            "passed": False,
            "texture_coverage_ratio": None,
            "atlas_pixel_coverage_ratio": None,
            "textured_face_ratio": None,
            "projected_face_ratio": None,
            "reason": "texture bake report is missing",
        }
    face_count = int(texture_report.get("face_count") or texture_report.get("selected_face_count") or 0)
    selected_count = int(texture_report.get("selected_face_count") or face_count or 0)
    drawn_faces = int(texture_report.get("drawn_face_count") or 0)
    projected_faces = int(texture_report.get("projected_face_count") or 0)
    denominator = max(selected_count or face_count, 1)
    textured_face_ratio = texture_report.get("textured_face_ratio")
    if not isinstance(textured_face_ratio, (int, float)):
        textured_face_ratio = drawn_faces / float(denominator)
    projected_face_ratio = texture_report.get("projected_face_ratio")
    if not isinstance(projected_face_ratio, (int, float)):
        projected_face_ratio = projected_faces / float(denominator)
    atlas_pixel_coverage = texture_report.get("atlas_pixel_coverage_ratio")
    if not isinstance(atlas_pixel_coverage, (int, float)):
        atlas_pixel_coverage = texture_report.get("texture_coverage_ratio")
    texture_coverage = texture_report.get("texture_coverage_ratio")
    passed = bool(float(textured_face_ratio) >= 0.95 and float(projected_face_ratio) >= 0.90)
    if not passed and isinstance(atlas_pixel_coverage, (int, float)):
        passed = float(atlas_pixel_coverage) >= 0.45 and float(textured_face_ratio) >= 0.80
    return {
        "passed": passed,
        "texture_coverage_ratio": texture_coverage,
        "atlas_pixel_coverage_ratio": round(float(atlas_pixel_coverage), 6) if isinstance(atlas_pixel_coverage, (int, float)) else None,
        "textured_face_ratio": round(float(textured_face_ratio), 6),
        "projected_face_ratio": round(float(projected_face_ratio), 6),
        "drawn_face_count": drawn_faces,
        "projected_face_count": projected_faces,
        "selected_face_count": selected_count,
        "face_count": face_count,
        "acceptance_basis": "mesh_face_coverage",
        "note": "Atlas pixel coverage is retained as a UV packing diagnostic; the commercial gate uses mesh face coverage to avoid penalizing valid UV island padding.",
    }


def movable_physics_material_quality(assets: Any) -> dict[str, Any]:
    entries = assets if isinstance(assets, list) else []
    ready_count = 0
    blockers: list[str] = []
    details: list[dict[str, Any]] = []
    for index, asset in enumerate(entries):
        if not isinstance(asset, Mapping):
            blockers.append("invalid_movable_asset_entry")
            details.append({"index": index, "ready": False, "blockers": ["invalid_movable_asset_entry"]})
            continue
        material = asset.get("physics_material")
        item_blockers = movable_physics_material_blockers(material)
        if item_blockers:
            blockers.extend(item_blockers)
        else:
            ready_count += 1
        details.append(
            {
                "id": asset.get("id") or asset.get("name") or f"asset_{index:03d}",
                "ready": not item_blockers,
                "blockers": item_blockers,
            }
        )
    unique_blockers = sorted(set(blockers))
    return {
        "asset_count": len(entries),
        "ready_count": ready_count,
        "incomplete_count": len(entries) - ready_count,
        "passed": not entries or ready_count == len(entries),
        "blockers": unique_blockers,
        "details": details,
    }


def task_affordance_report(root: Path, movable_assets: Mapping[str, Any]) -> dict[str, Any]:
    assets = movable_assets.get("movable_assets") if isinstance(movable_assets.get("movable_assets"), list) else []
    targets: list[dict[str, Any]] = []
    aggregate_blockers: list[str] = []
    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            blockers = ["invalid_movable_asset_entry"]
            aggregate_blockers.extend(blockers)
            targets.append({"id": f"asset_{index:03d}", "ready": False, "blockers": blockers})
            continue
        pose = asset.get("pose") if isinstance(asset.get("pose"), Mapping) else {}
        material = asset.get("physics_material") if isinstance(asset.get("physics_material"), Mapping) else {}
        reconstruction = asset.get("reconstruction_quality") if isinstance(asset.get("reconstruction_quality"), Mapping) else {}
        blockers = task_affordance_blockers(asset, pose, material, reconstruction)
        aggregate_blockers.extend(blockers)
        targets.append(
            {
                "id": asset.get("id") or asset.get("name") or f"asset_{index:03d}",
                "label": asset.get("label"),
                "dynamic": bool(asset.get("dynamic")),
                "contact_role": material.get("contact_role"),
                "pose": {
                    "position": pose.get("position"),
                    "quat_wxyz": pose.get("quat_wxyz"),
                    "scale": pose.get("scale"),
                },
                "ready": not blockers,
                "blockers": blockers,
            }
        )
    ready_count = sum(1 for target in targets if target.get("ready"))
    blockers = sorted(set(aggregate_blockers))
    score = ready_count / float(max(len(targets), 1)) if targets else 0.0
    return {
        "format": "gs-playground-real2sim-task-affordance-v1",
        "package_root": root.as_posix(),
        "summary": {
            "score": round(float(score), 6),
            "target": 1.0,
            "passed": bool(targets and ready_count == len(targets) and not blockers),
            "target_count": len(targets),
            "ready_target_count": ready_count,
            "blocker_count": len(blockers),
            "blockers": blockers,
        },
        "inputs": {
            "movable_asset_layers": "artifacts/assets/movable_assets.json",
        },
        "targets": targets,
        "diagnostics": task_affordance_diagnostics(blockers),
    }


def task_affordance_blockers(
    asset: Mapping[str, Any],
    pose: Mapping[str, Any],
    material: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not bool(asset.get("dynamic")):
        blockers.append("target_not_dynamic")
    if not reconstruction.get("commercial_asset_grade"):
        blockers.append("target_not_production_asset")
    if material.get("contact_role") not in {"graspable", "pushable", "movable_asset"}:
        blockers.append("missing_actionable_contact_role")
    if not finite_pose_triplet(pose.get("position"), max_abs=10.0):
        blockers.append("invalid_target_position")
    if not finite_pose_triplet(pose.get("scale"), max_abs=20.0, positive=True):
        blockers.append("invalid_target_scale")
    quat = pose.get("quat_wxyz")
    if not isinstance(quat, (list, tuple)) or len(quat) != 4:
        blockers.append("invalid_target_orientation")
    else:
        try:
            if not all(np.isfinite(float(value)) for value in quat):
                blockers.append("invalid_target_orientation")
        except (TypeError, ValueError):
            blockers.append("invalid_target_orientation")
    return blockers


def finite_pose_triplet(value: Any, *, max_abs: float, positive: bool = False) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    if not all(np.isfinite(item) and abs(item) <= max_abs for item in values):
        return False
    if positive and not all(item > 0.0 for item in values):
        return False
    return True


def movable_physics_material_blockers(material: Any) -> list[str]:
    if not isinstance(material, Mapping):
        return ["missing_movable_physics_material"]
    blockers: list[str] = []
    if material.get("movable") is not True:
        blockers.append("movable_material_not_marked_movable")
    mass = material.get("mass")
    if not isinstance(mass, (int, float)) or not (0.02 <= float(mass) <= 5.0):
        blockers.append("invalid_or_implausible_movable_mass")
    friction = material.get("friction")
    if not valid_friction_triplet(friction):
        blockers.append("invalid_movable_friction")
    role = str(material.get("contact_role") or "")
    if role not in {"graspable", "pushable", "movable_asset"}:
        blockers.append("missing_movable_contact_role")
    restitution = material.get("restitution")
    if not isinstance(restitution, (int, float)) or not (0.0 <= float(restitution) <= 1.0):
        blockers.append("invalid_movable_restitution")
    return blockers


def valid_static_physics_material(material: Any) -> bool:
    if not isinstance(material, Mapping):
        return False
    if material.get("movable") is not False:
        return False
    if not valid_friction_triplet(material.get("friction")):
        return False
    return bool(str(material.get("contact_role") or ""))


def valid_friction_triplet(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    try:
        slide, torsion, rolling = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return False
    return slide >= 0.2 and torsion >= 0.0 and rolling >= 0.0


def room_completeness_report(root: Path, structure_layers: Mapping[str, Any], scene_structure_report: Mapping[str, Any]) -> dict[str, Any]:
    walls = structure_layers.get("walls") if isinstance(structure_layers, Mapping) else []
    support_surfaces = structure_layers.get("support_surfaces") if isinstance(structure_layers, Mapping) else []
    navigation_surfaces = structure_layers.get("navigation_surfaces") if isinstance(structure_layers, Mapping) else []
    completion_proxies = structure_layers.get("completion_proxies") if isinstance(structure_layers, Mapping) else []
    walls = walls if isinstance(walls, list) else []
    support_surfaces = support_surfaces if isinstance(support_surfaces, list) else []
    navigation_surfaces = navigation_surfaces if isinstance(navigation_surfaces, list) else []
    completion_proxies = completion_proxies if isinstance(completion_proxies, list) else []
    readiness = scene_structure_report.get("commercial_readiness")
    readiness = readiness if isinstance(readiness, Mapping) else {}
    has_observed_floor = bool(readiness.get("has_floor")) or any(
        isinstance(surface, Mapping) and surface.get("navigation_role") == "walkable_floor" for surface in navigation_surfaces
    )
    has_floor_completion = any(
        isinstance(proxy, Mapping) and proxy.get("semantic_label") == "floor_completion" for proxy in completion_proxies
    )
    wall_count = len(walls)
    support_count = len(support_surfaces)
    navigation_count = len(navigation_surfaces)
    score = (
        (0.45 if has_observed_floor else 0.34 if has_floor_completion else 0.0)
        + min(wall_count, 2) * 0.18
        + (0.12 if support_count else 0.0)
        + (0.07 if navigation_count else 0.0)
    )
    score = round(min(score, 1.0), 6)
    missing: list[str] = []
    if not (has_observed_floor or has_floor_completion):
        missing.append("floor_or_floor_completion")
    if wall_count < 2:
        missing.append("at_least_two_wall_planes")
    if navigation_count < 1:
        missing.append("walkable_or_placeable_navigation_surface")
    passed = score >= 0.7 and not missing
    return {
        "format": "gs-playground-real2sim-room-completeness-v1",
        "package_root": root.as_posix(),
        "summary": {
            "score": score,
            "target": 0.7,
            "passed": passed,
            "has_observed_floor": has_observed_floor,
            "has_floor_completion": has_floor_completion,
            "wall_count": wall_count,
            "support_surface_count": support_count,
            "navigation_surface_count": navigation_count,
            "completion_proxy_count": len(completion_proxies),
            "missing_required_elements": missing,
        },
        "inputs": {
            "scene_structure_report": "artifacts/scene_structure/scene_structure_report.json",
            "static_structure_layers": "artifacts/scene_structure/static_structure.json",
            "bounds": scene_structure_report.get("bounds"),
            "semantic_planes_count": len(scene_structure_report.get("semantic_planes", []))
            if isinstance(scene_structure_report.get("semantic_planes"), list)
            else 0,
        },
        "outputs": {
            "completion_proxies": completion_proxies,
            "navigation_surfaces": navigation_surfaces,
        },
        "diagnostics": room_completeness_diagnostics(
            has_observed_floor=has_observed_floor,
            has_floor_completion=has_floor_completion,
            wall_count=wall_count,
            support_count=support_count,
            navigation_count=navigation_count,
        ),
    }


def movable_asset_layers(root: Path, *, structure_layers: Mapping[str, Any] | None = None) -> dict[str, Any]:
    manifest = read_json(root / "package_manifest.json") if (root / "package_manifest.json").is_file() else {}
    objects = manifest.get("objects") if isinstance(manifest, Mapping) else []
    assets: list[dict[str, Any]] = []
    if isinstance(objects, list):
        for index, asset in enumerate(objects):
            if not isinstance(asset, Mapping):
                continue
            name = str(asset.get("name") or f"object_{index:03d}")
            pose = asset.get("pose") if isinstance(asset.get("pose"), Mapping) else {}
            assets.append(
                {
                    "id": slug_token(name),
                    "name": name,
                    "label": asset.get("label"),
                    "dynamic": bool(asset.get("dynamic", True)),
                    "source_instance_id": asset.get("source_instance_id"),
                    "gaussian_ply": asset.get("gaussian_ply"),
                    "mesh_obj": asset.get("mesh_obj"),
                    "pose": {
                        "position": pose.get("position", [0.0, 0.0, 0.0]),
                        "quat_wxyz": pose.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
                        "scale": pose.get("scale", [1.0, 1.0, 1.0]),
                    },
                    "physics_material": default_movable_physics_material(str(asset.get("label") or name)),
                    "reconstruction_quality": movable_asset_reconstruction_quality(asset),
                }
            )
    candidate_slots = movable_candidate_slots(structure_layers or {}, enabled=not assets)
    production_assets = [asset for asset in assets if asset["reconstruction_quality"]["commercial_asset_grade"]]
    provisional_assets = [asset for asset in assets if asset["reconstruction_quality"]["reconstruction_source"] == "provisional_extraction_task"]
    incomplete_assets = [asset for asset in assets if asset["reconstruction_quality"]["asset_completeness_blockers"]]
    physics_material_quality = movable_physics_material_quality(assets)
    return {
        "format": "gs-playground-real2sim-movable-assets-v1",
        "movable_assets": assets,
        "candidate_slots": candidate_slots,
        "quality": {
            "movable_asset_count": len(assets),
            "dynamic_asset_count": sum(1 for asset in assets if asset.get("dynamic")),
            "has_separate_movable_assets": bool(assets),
            "candidate_slot_count": len(candidate_slots),
            "production_asset_count": len(production_assets),
            "provisional_asset_count": len(provisional_assets),
            "incomplete_asset_count": len(incomplete_assets),
            "physics_material_ready_count": physics_material_quality["ready_count"],
            "physics_material_incomplete_count": physics_material_quality["incomplete_count"],
            "commercial_asset_grade": bool(assets) and len(production_assets) == len(assets),
        },
        "commercial_gap_notes": [
            "Commercial-grade Real2Sim requires movable assets to be segmented and reconstructed separately from the static scene.",
            "Candidate slots identify support/navigation surfaces where object segmentation and reconstruction should run next.",
            "An empty movable asset list means the package is scene-only even if movable objects are visually embedded in the mesh.",
        ]
        if not assets
        else [
            "Movable assets exist, but at least one asset is provisional and should be replaced by SAM/SAM3D or multi-view object reconstruction."
        ]
        if provisional_assets
        else [
            "Movable assets exist, but at least one asset is missing gaussian, mesh, or pose artifacts required for commercial import."
        ]
        if incomplete_assets
        else [],
    }


def movable_asset_reconstruction_quality(asset: Mapping[str, Any]) -> dict[str, Any]:
    source_id = str(asset.get("source_instance_id") or "")
    name = str(asset.get("name") or "")
    explicit_provisional = bool(asset.get("provisional"))
    inferred_provisional = explicit_provisional or source_id.startswith("candidate_slot_") or "_bottle" in name and source_id.startswith("candidate_")
    source = "provisional_extraction_task" if inferred_provisional else "external_or_sam3d_reconstruction"
    has_gaussian = bool(asset.get("gaussian_ply"))
    has_mesh = bool(asset.get("mesh_obj"))
    has_pose = isinstance(asset.get("pose"), Mapping)
    completeness_blockers: list[str] = []
    if not has_gaussian:
        completeness_blockers.append("missing_gaussian_asset")
    if not has_mesh:
        completeness_blockers.append("missing_mesh_asset")
    if not has_pose:
        completeness_blockers.append("missing_pose")
    return {
        "reconstruction_source": source,
        "commercial_asset_grade": not inferred_provisional and not completeness_blockers,
        "has_gaussian_asset": has_gaussian,
        "has_mesh_asset": has_mesh,
        "has_pose": has_pose,
        "asset_completeness_blockers": completeness_blockers,
        "source_instance_id": source_id or None,
        "required_upgrade": "replace_with_sam3d_or_multiview_object_reconstruction"
        if inferred_provisional
        else "complete_gaussian_mesh_pose_assets"
        if completeness_blockers
        else None,
    }


def movable_candidate_slots(structure_layers: Mapping[str, Any], *, enabled: bool) -> list[dict[str, Any]]:
    if not enabled:
        return []
    navigation = structure_layers.get("navigation_surfaces") if isinstance(structure_layers, Mapping) else []
    if not isinstance(navigation, list):
        return []
    slots: list[dict[str, Any]] = []
    for index, surface in enumerate(navigation[:8]):
        if not isinstance(surface, Mapping):
            continue
        role = str(surface.get("navigation_role") or "")
        if role not in {"placeable_support", "walkable_floor"}:
            continue
        extent = surface.get("extent") if isinstance(surface.get("extent"), Mapping) else {}
        slots.append(
            {
                "id": f"candidate_slot_{index:03d}",
                "source_surface_id": surface.get("id"),
                "source_surface_label": surface.get("semantic_label"),
                "navigation_role": role,
                "extent": extent,
                "candidate_type": "movable_objects_on_support" if role == "placeable_support" else "movable_objects_on_floor",
                "recommended_prompts": ["bottle", "cup", "book", "box", "monitor", "tool"]
                if role == "placeable_support"
                else ["chair", "box", "bag", "movable obstacle"],
                "required_next_stage": "segment_and_reconstruct_objects",
            }
        )
    return slots


def movable_extraction_tasks(root: Path, motrix_import: Mapping[str, Any], movable_assets: Mapping[str, Any]) -> dict[str, Any]:
    slots = movable_assets.get("candidate_slots") if isinstance(movable_assets, Mapping) else []
    if not isinstance(slots, list) or not slots:
        return {
            "format": "gs-playground-real2sim-movable-extraction-tasks-v1",
            "tasks": [],
            "quality": {"task_count": 0, "candidate_slot_count": 0},
        }
    preview_value = motrix_import.get("preview")
    preview_path = root / str(preview_value) if preview_value else None
    task_dir = root / "artifacts" / "assets" / "movable_candidates"
    task_dir.mkdir(parents=True, exist_ok=True)
    try:
        rgb = read_rgb_with_pillow_fallback(preview_path) if preview_path is not None else None
    except Exception as exc:
        return {
            "format": "gs-playground-real2sim-movable-extraction-tasks-v1",
            "tasks": [],
            "quality": {"task_count": 0, "candidate_slot_count": len(slots)},
            "warnings": [f"failed to read preview image for movable extraction tasks: {exc}"],
        }
    if rgb is None:
        return {
            "format": "gs-playground-real2sim-movable-extraction-tasks-v1",
            "tasks": [],
            "quality": {"task_count": 0, "candidate_slot_count": len(slots)},
            "warnings": ["motrix import does not include a preview image"],
        }
    detections = image_movable_candidate_boxes(rgb, max_candidates=max(1, min(6, len(slots) * 3)))
    tasks: list[dict[str, Any]] = []
    for slot_index, slot in enumerate(slots):
        if not isinstance(slot, Mapping):
            continue
        prompts = slot.get("recommended_prompts") if isinstance(slot.get("recommended_prompts"), list) else ["movable object"]
        for det_index, detection in enumerate(detections[:3]):
            bbox = detection["bbox_xyxy"]
            x0, y0, x1, y1 = bbox
            crop = rgb[y0:y1, x0:x1, :]
            mask = np.zeros(rgb.shape[:2], dtype=np.float32)
            mask[y0:y1, x0:x1] = 1.0
            task_id = f"{slot.get('id', f'slot_{slot_index:03d}')}_{det_index:03d}"
            crop_path = task_dir / f"{task_id}_crop.ppm"
            mask_path = task_dir / f"{task_id}_mask.pgm"
            write_rgb(crop_path, crop)
            write_gray(mask_path, mask)
            tasks.append(
                {
                    "id": task_id,
                    "source_slot_id": slot.get("id"),
                    "source_surface_id": slot.get("source_surface_id"),
                    "prompt_candidates": prompts,
                    "bbox_xyxy": bbox,
                    "score": detection["score"],
                    "crop": crop_path.relative_to(root).as_posix(),
                    "mask": mask_path.relative_to(root).as_posix(),
                    "required_next_stage": "sam_segment_then_sam3d_reconstruct",
                    "expected_output": "objects_json entry with gaussian_ply, mesh_obj, pose, dynamic=true",
                }
            )
    return {
        "format": "gs-playground-real2sim-movable-extraction-tasks-v1",
        "preview": str(preview_value) if preview_value else None,
        "tasks": tasks,
        "quality": {
            "task_count": len(tasks),
            "candidate_slot_count": len(slots),
            "image_candidate_count": len(detections),
        },
        "commercial_gap_notes": [
            "These are extraction tasks, not completed movable assets.",
            "Commercial readiness still requires the tasks to be segmented, reconstructed, and packaged as object assets.",
        ],
    }


def image_movable_candidate_boxes(rgb: np.ndarray, *, max_candidates: int) -> list[dict[str, Any]]:
    height, width = rgb.shape[:2]
    if height < 8 or width < 8:
        return []
    gray = rgb[:, :, :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    saturation = rgb[:, :, :3].max(axis=2) - rgb[:, :, :3].min(axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2])
    gy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :])
    saliency = saturation * 0.55 + (gx + gy) * 0.45
    y_min = int(height * 0.42)
    saliency[:y_min, :] = 0.0
    threshold = float(np.percentile(saliency[y_min:, :], 92.0)) if y_min < height else float(np.percentile(saliency, 92.0))
    mask = saliency > max(threshold, 0.025)
    components = connected_components(mask)
    candidates: list[dict[str, Any]] = []
    for component in components:
        ys, xs = component
        if len(xs) < 12:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        bw = x1 - x0
        bh = y1 - y0
        area = len(xs)
        if bw < width * 0.025 or bh < height * 0.025:
            continue
        if bw > width * 0.7 or bh > height * 0.7:
            continue
        margin_x = max(3, int(bw * 0.25))
        margin_y = max(3, int(bh * 0.25))
        ex0 = max(0, x0 - margin_x)
        ey0 = max(0, y0 - margin_y)
        ex1 = min(width, x1 + margin_x)
        ey1 = min(height, y1 + margin_y)
        score = float(saliency[ys, xs].mean() * np.sqrt(area))
        candidates.append({"bbox_xyxy": [ex0, ey0, ex1, ey1], "score": round(score, 6)})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return non_overlapping_boxes(candidates, max_candidates=max_candidates)


def read_rgb_with_pillow_fallback(path: Path) -> np.ndarray:
    try:
        return read_rgb(path)
    except Exception:
        try:
            from PIL import Image
        except Exception:
            raise
        image = Image.open(path).convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return np.clip(arr, 0.0, 1.0)


def connected_components(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[np.ndarray, np.ndarray]] = []
    height, width = mask.shape
    for y in range(height):
        xs = np.where(mask[y] & ~visited[y])[0]
        for x_start in xs.tolist():
            if visited[y, x_start] or not mask[y, x_start]:
                continue
            stack = [(y, x_start)]
            visited[y, x_start] = True
            ys: list[int] = []
            xs_component: list[int] = []
            while stack:
                cy, cx = stack.pop()
                ys.append(cy)
                xs_component.append(cx)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            components.append((np.asarray(ys, dtype=np.int32), np.asarray(xs_component, dtype=np.int32)))
    return components


def non_overlapping_boxes(candidates: list[dict[str, Any]], *, max_candidates: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if all(box_iou(candidate["bbox_xyxy"], other["bbox_xyxy"]) < 0.35 for other in selected):
            selected.append(candidate)
        if len(selected) >= max_candidates:
            break
    return selected


def box_iou(a: list[int], b: list[int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def default_movable_physics_material(label: str) -> dict[str, Any]:
    semantic = label.lower()
    if "bottle" in semantic or "cup" in semantic:
        return {"movable": True, "mass": 0.08, "friction": [0.65, 0.02, 0.001], "restitution": 0.04, "contact_role": "graspable"}
    if "book" in semantic or "box" in semantic:
        return {"movable": True, "mass": 0.35, "friction": [0.8, 0.02, 0.001], "restitution": 0.02, "contact_role": "pushable"}
    return {"movable": True, "mass": 0.2, "friction": [0.7, 0.02, 0.001], "restitution": 0.03, "contact_role": "movable_asset"}


def readiness_gate(gate_id: str, passed: bool, weight: float, requirement: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": gate_id,
        "passed": bool(passed),
        "weight": float(weight),
        "requirement": requirement,
        "evidence": dict(evidence),
    }


def commercial_next_actions(blockers: list[str]) -> list[str]:
    actions = {
        "texture_bake": "Improve multi-view texture selection, seam handling, and visibility filtering before final visual export.",
        "movable_asset_instances": "Run or integrate instance segmentation/reconstruction so bottles, monitors, books, and tools become separate movable assets.",
        "room_completeness": "Capture missing floor/back-side views or promote verified completion proxies into scene geometry, then regenerate navigation surfaces.",
        "navigation_surface_hints": "Generate a real navmesh from floor/support surfaces and validate traversability.",
        "physical_material_hints": "Infer or assign task-specific friction, mass, and contact roles for every structure and movable asset.",
        "static_structure_layers": "Recover semantic walls/support surfaces before exporting to the simulator.",
        "physics_collision": "Generate collision proxies for semantic structure before Motrix import.",
        "dense_scene_geometry": "Re-run dense reconstruction or mesh cleanup until a sufficiently detailed scene mesh is available.",
        "metric_scale_plausibility": "Calibrate metric scale from COLMAP poses, known robot/object dimensions, or manual room measurements before commercial simulation export.",
        "navigation_acceptance": "Recover an observed floor or verified floor mesh, generate a walkable nav surface, and validate robot footprint clearance.",
        "material_readiness": "Export texture-backed OBJ/MTL or MJCF materials and include every referenced texture file in the MotrixSim package.",
        "task_affordance_targets": "Mark production movable assets as dynamic task targets with finite poses and actionable graspable or pushable contact roles.",
    }
    return [actions[item] for item in blockers if item in actions]


def metric_scale_diagnostics(blockers: list[str]) -> list[dict[str, str]]:
    messages = {
        "missing_metric_bounds": {
            "problem": "scene_structure_report did not expose metric p05/p95 bounds",
            "next_step": "write scene bounds during structure analysis or attach COLMAP/mesh-derived bounds before packaging",
        },
        "invalid_or_extreme_bounds": {
            "problem": "scene bounds contain zero, non-finite, or extreme extents",
            "next_step": "verify coordinate units and remove outlier geometry before exporting MJCF",
        },
        "implausible_horizontal_extent": {
            "problem": "horizontal room extent is outside the expected meter-scale envelope",
            "next_step": "calibrate reconstruction scale from camera poses, robot footprint, table width, or manual measurements",
        },
        "implausible_vertical_extent": {
            "problem": "vertical room extent is outside the expected meter-scale envelope",
            "next_step": "fix up-axis selection, floor/ceiling detection, or monocular scale before physics validation",
        },
        "missing_structure_proxy_sizes": {
            "problem": "static structure proxies have no usable sizes",
            "next_step": "regenerate semantic collision proxies from observed floor, walls, and support surfaces",
        },
        "implausible_structure_proxy_size": {
            "problem": "static collision proxy sizes are too small or too large for robot simulation",
            "next_step": "recompute proxy boxes after metric scale calibration and outlier cleanup",
        },
    }
    return [messages[item] for item in blockers if item in messages]


def navigation_acceptance_diagnostics(blockers: list[str]) -> list[dict[str, str]]:
    messages = {
        "missing_walkable_floor_surface": {
            "problem": "navigation layers contain no walkable floor surface",
            "next_step": "capture or reconstruct the floor plane and mark it as walkable_floor before robot navigation export",
        },
        "walkable_surface_too_small": {
            "problem": "largest walkable surface is too small for robot route planning",
            "next_step": "expand floor reconstruction, remove cropped bounds, or calibrate metric scale before navmesh generation",
        },
        "walkable_surface_inferred_only": {
            "problem": "all walkable surfaces are inferred completion proxies rather than directly observed geometry",
            "next_step": "verify the completed floor against additional views or promote a real reconstructed floor mesh before commercial acceptance",
        },
    }
    return [messages[item] for item in blockers if item in messages]


def material_readiness_diagnostics(blockers: list[str]) -> list[dict[str, str]]:
    messages = {
        "scene_mesh_missing_texture_material": {
            "problem": "the scene mesh is packaged but has no OBJ/MTL diffuse texture material",
            "next_step": "run texture baking or preserve COLMAP/OpenMVS texture sidecars before package export",
        },
        "missing_texture_backed_material": {
            "problem": "no MJCF or OBJ material references a real texture image",
            "next_step": "export UVs, MTL files, and map_Kd images, or add MJCF texture/material assets for simulator rendering",
        },
        "missing_texture_files": {
            "problem": "at least one material references a texture file that is not present in the package",
            "next_step": "copy texture sidecars into meshes/ and rewrite material paths to package-relative names",
        },
        "missing_mesh_assets": {
            "problem": "the package contains no mesh assets whose materials can be validated",
            "next_step": "export a scene mesh or production object meshes in addition to 3DGS assets",
        },
    }
    return [messages[item] for item in blockers if item in messages]


def task_affordance_diagnostics(blockers: list[str]) -> list[dict[str, str]]:
    messages = {
        "target_not_dynamic": {
            "problem": "a task target is not marked dynamic",
            "next_step": "enable dynamic body export for movable objects that should be manipulated or pushed",
        },
        "target_not_production_asset": {
            "problem": "a task target is not a production-grade reconstructed movable asset",
            "next_step": "replace candidate/provisional targets with complete SAM3D or multi-view object reconstructions",
        },
        "missing_actionable_contact_role": {
            "problem": "a task target has no graspable/pushable/movable contact role",
            "next_step": "assign contact roles from object semantics and intended robot task",
        },
        "invalid_target_position": {
            "problem": "a task target pose has missing, non-finite, or out-of-workspace position",
            "next_step": "run pose alignment or scale calibration before task export",
        },
        "invalid_target_scale": {
            "problem": "a task target scale is missing, non-positive, non-finite, or implausibly large",
            "next_step": "normalize object reconstruction scale and update pose metadata",
        },
        "invalid_target_orientation": {
            "problem": "a task target orientation quaternion is missing or non-finite",
            "next_step": "write a valid quat_wxyz pose during object alignment",
        },
    }
    return [messages[item] for item in blockers if item in messages]


def room_completeness_diagnostics(
    *,
    has_observed_floor: bool,
    has_floor_completion: bool,
    wall_count: int,
    support_count: int,
    navigation_count: int,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    if has_observed_floor:
        diagnostics.append(
            {
                "id": "floor",
                "status": "observed",
                "meaning": "a floor plane was detected directly from the reconstructed mesh",
                "input": "scene_structure_report.semantic_planes",
                "output": "walkable_floor navigation surface",
            }
        )
    elif has_floor_completion:
        diagnostics.append(
            {
                "id": "floor",
                "status": "inferred_completion",
                "meaning": "no direct floor plane was detected, so a bounded floor proxy was inferred for simulation stability",
                "input": "scene_structure_report.bounds plus observed walls/support surfaces",
                "output": "floor_completion collision and walkable_floor_inferred navigation surface",
            }
        )
    else:
        diagnostics.append(
            {
                "id": "floor",
                "status": "missing",
                "meaning": "the package has no directly observed or inferred floor, so room-level simulation is incomplete",
                "input": "missing low horizontal plane or usable room bounds",
                "output": "none",
            }
        )
    diagnostics.append(
        {
            "id": "walls",
            "status": "sufficient" if wall_count >= 2 else "partial",
            "meaning": "two or more wall planes are enough to anchor a coarse room frame; four walls are preferred for commercial scans",
            "input": "scene_structure_report.semantic_planes with wall labels",
            "output": f"{wall_count} wall structure surfaces",
        }
    )
    diagnostics.append(
        {
            "id": "support_surfaces",
            "status": "present" if support_count else "missing",
            "meaning": "support surfaces such as tables or shelves create candidate slots for movable object extraction",
            "input": "support_surface semantic planes",
            "output": f"{support_count} support surfaces",
        }
    )
    diagnostics.append(
        {
            "id": "navigation",
            "status": "present" if navigation_count else "missing",
            "meaning": "navigation surfaces are what MotrixSim and downstream planners can consume as walkable/placeable hints",
            "input": "floor/support surfaces and completion proxies",
            "output": f"{navigation_count} navigation surfaces",
        }
    )
    return diagnostics


def read_optional_json(path: Path) -> Any:
    return read_json(path) if path.is_file() else None


def numeric_triplet(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def vec(values: tuple[float, float, float]) -> str:
    return " ".join(f"{value:.6g}" for value in values)


def slug_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value).strip("_").lower() or "structure"


def indent_xml(element: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = indent + "  "
        for child in element:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indent


def _manifest_from_alignment_json(path: Path, fallback: Real2SimManifest) -> Real2SimManifest:
    data = read_json(path)
    if not isinstance(data, Mapping):
        raise StageArtifactError(f"aligned assets file must be a JSON object: {path}")
    # Alignment tools can either update object poses in-place using the same
    # manifest schema, or leave the file untouched. The conservative path keeps
    # existing artifacts and only applies pose overrides by name.
    overrides = {}
    for entry in data.get("objects", []):
        if isinstance(entry, Mapping) and "name" in entry and "pose" in entry:
            overrides[str(entry["name"])] = entry["pose"]
    if not overrides:
        return fallback

    objects = []
    for asset in fallback.objects:
        pose_raw = overrides.get(asset.name)
        if pose_raw is None:
            objects.append(asset)
            continue
        pose_file = path.parent / f".{asset.name}_pose_override.json"
        write_json(pose_file, pose_raw)
        pose = Pose.from_json(pose_file)
        pose_file.unlink(missing_ok=True)
        objects.append(
            Real2SimAsset(
                name=asset.name,
                label=asset.label,
                gaussian_ply=asset.gaussian_ply,
                mesh_obj=asset.mesh_obj,
                pose=pose,
                dynamic=asset.dynamic,
                source_instance_id=asset.source_instance_id,
            )
        )
    return Real2SimManifest(
        source=fallback.source,
        prompt=fallback.prompt,
        scene=fallback.scene,
        objects=tuple(objects),
        camera_artifacts=fallback.camera_artifacts,
        metadata={**fallback.metadata, "pose_alignment": path.as_posix()},
    )


def _resolve_path(base_dir: Path, value: Any) -> Path:
    if value is None:
        raise StageArtifactError("expected path value, got null")
    path = Path(str(value))
    return path if path.is_absolute() else (base_dir / path).resolve()


def _supported_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def instances_are_refined(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        raw = read_json(path)
    except Exception:
        return False
    metadata = raw.get("metadata", {}) if isinstance(raw, Mapping) else {}
    return isinstance(metadata, Mapping) and isinstance(metadata.get("instance_refinement"), Mapping)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value).strip("_").lower() or "object"
