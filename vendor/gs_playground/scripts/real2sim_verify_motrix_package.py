#!/usr/bin/env python3
"""Verify a MotrixSim-importable Real2Sim scene package.

The verifier accepts either an unpacked package directory or a ``.tar.gz``
archive. It checks the Motrix import entrypoints, required scene assets, and
quality metrics written by the one-image AnySplat workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


COMMERCIAL_ACCEPTANCE_FORMAT = "gs-playground-real2sim-commercial-acceptance-v1"
COMMERCIAL_ACCEPTANCE_REPORT_PATH = "artifacts/quality/commercial_acceptance.json"

REQUIRED_COMMERCIAL_ACCEPTANCE_ENTRYPOINTS = {
    "motrix_import": "motrix_import.json",
    "package_manifest": "package_manifest.json",
    "scene_xml": "mjcf/scene.xml",
    "viewer_config": "configs/go2_real2sim.json",
    "scene_gaussians": "3dgs/scene.ply",
}

REQUIRED_COMMERCIAL_ACCEPTANCE_REPORTS = {
    "commercial_readiness": "artifacts/quality/commercial_readiness.json",
    "room_completeness": "artifacts/quality/room_completeness.json",
    "metric_scale": "artifacts/quality/metric_scale.json",
    "navigation_acceptance": "artifacts/quality/navigation_acceptance.json",
    "material_readiness": "artifacts/quality/material_readiness.json",
    "task_affordance": "artifacts/quality/task_affordance.json",
    "movable_assets": "artifacts/assets/movable_assets.json",
    "static_structure_layers": "artifacts/scene_structure/static_structure.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a Real2Sim MotrixSim import package")
    parser.add_argument("--package", type=Path, required=True, help="Package directory or .tar.gz archive")
    parser.add_argument("--min-scene-vertices", type=int, default=100_000)
    parser.add_argument("--min-source-preview-psnr", type=float, default=18.0)
    parser.add_argument("--max-source-preview-mae", type=float, default=0.10)
    parser.add_argument("--visual-report-dir", type=Path, help="Write a visual acceptance JSON report and contact sheet")
    parser.add_argument("--motrix-viewer-render", type=Path, help="Optional Motrix viewer screenshot/render to include")
    parser.add_argument("--mujoco-offscreen-render-dir", type=Path, help="Render MJCF cameras with mujoco.Renderer into this directory")
    parser.add_argument("--mujoco-offscreen-cameras", default="overview,closeup", help="Comma-separated MJCF cameras to render")
    parser.add_argument("--mujoco-offscreen-width", type=int, default=640)
    parser.add_argument("--mujoco-offscreen-height", type=int, default=480)
    parser.add_argument("--mujoco-offscreen-max-geom", type=int, default=0, help="Override mujoco.Renderer max_geom; 0 derives it from model.ngeom")
    parser.add_argument(
        "--require-commercial-acceptance",
        action="store_true",
        help="Fail strict commercial visual quality unless the package declares and passes commercial_acceptance.json",
    )
    parser.add_argument(
        "--require-capture-acceptance",
        action="store_true",
        help="Fail strict commercial visual quality unless capture acceptance is present and accepted",
    )
    args = parser.parse_args()
    if (args.require_commercial_acceptance or args.require_capture_acceptance) and args.visual_report_dir is None:
        parser.error("--require-commercial-acceptance/--require-capture-acceptance requires --visual-report-dir")

    with package_root(args.package) as root:
        report = verify_package(
            root,
            min_scene_vertices=args.min_scene_vertices,
            min_source_preview_psnr=args.min_source_preview_psnr,
            max_source_preview_mae=args.max_source_preview_mae,
        )
        report["physics_task_readiness"] = physics_task_readiness_report(root, report)
        offscreen_report = None
        if args.mujoco_offscreen_render_dir is not None:
            cameras = [camera.strip() for camera in args.mujoco_offscreen_cameras.split(",") if camera.strip()]
            offscreen_report = render_mujoco_offscreen(
                root,
                report,
                args.mujoco_offscreen_render_dir,
                cameras=cameras,
                width=args.mujoco_offscreen_width,
                height=args.mujoco_offscreen_height,
                max_geom=args.mujoco_offscreen_max_geom,
            )
            report["mujoco_offscreen_render"] = offscreen_report
            if isinstance(report.get("physics_task_readiness"), dict):
                report["physics_task_readiness"]["mujoco_simulation"] = offscreen_report.get("simulation_stability")
                report["physics_task_readiness"]["initial_contact_diagnostics"] = offscreen_report.get(
                    "initial_contact_diagnostics"
                )
                report["physics_task_readiness"]["settle_drift_probe"] = offscreen_report.get("settle_drift_probe")
                report["physics_task_readiness"]["passed"] = physics_task_readiness_passed(
                    report["physics_task_readiness"]
                )
        report["robot_task_readiness"] = robot_task_readiness_report(root, report)
        report["task_execution_probe"] = task_execution_probe_report(root, report)
        if args.visual_report_dir is not None:
            viewer_render = args.motrix_viewer_render
            viewer_render_label = "Motrix viewer render"
            if viewer_render is None and offscreen_report is not None:
                renders = offscreen_report.get("renders")
                if isinstance(renders, list) and renders:
                    first_render = renders[0]
                    if isinstance(first_render, dict) and first_render.get("path"):
                        viewer_render = Path(str(first_render["path"]))
                        viewer_render_label = "MuJoCo offscreen render"
            report["visual_acceptance_report"] = write_visual_acceptance_report(
                root,
                report,
                args.visual_report_dir,
                motrix_viewer_render=viewer_render,
                motrix_viewer_render_label=viewer_render_label,
            )
            report["strict_commercial_visual_quality"] = write_strict_commercial_visual_quality_report(
                root,
                report,
                args.visual_report_dir,
                visual_acceptance=report["visual_acceptance_report"],
                require_commercial_acceptance=args.require_commercial_acceptance,
                require_capture_acceptance=args.require_capture_acceptance,
            )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.require_commercial_acceptance or args.require_capture_acceptance:
        strict_summary = report.get("strict_commercial_visual_quality", {}).get("summary", {})
        if not strict_summary.get("strict_commercial_visual_grade"):
            return 1
    return 0


def package_root(path: Path):
    class _PackageRoot:
        def __init__(self, source: Path) -> None:
            self.source = source
            self.tmp: tempfile.TemporaryDirectory[str] | None = None
            self.root: Path | None = None

        def __enter__(self) -> Path:
            if self.source.is_dir():
                self.root = self.source.resolve()
                return self.root
            if not self.source.is_file():
                raise SystemExit(f"package path does not exist: {self.source}")
            self.tmp = tempfile.TemporaryDirectory()
            tmp_root = Path(self.tmp.name)
            with tarfile.open(self.source, "r:gz") as tar:
                tar.extractall(tmp_root, filter="data")
            candidates = [path for path in tmp_root.iterdir() if path.is_dir()]
            if len(candidates) != 1:
                raise SystemExit(f"archive must contain exactly one package directory: {self.source}")
            self.root = candidates[0]
            return self.root

        def __exit__(self, exc_type, exc, tb) -> None:
            if self.tmp is not None:
                self.tmp.cleanup()

    return _PackageRoot(path)


def verify_package(
    root: Path,
    *,
    min_scene_vertices: int,
    min_source_preview_psnr: float,
    max_source_preview_mae: float,
) -> dict[str, Any]:
    motrix_import_path = require_file(root / "motrix_import.json", "Motrix import manifest")
    motrix_import = read_json(motrix_import_path)
    if motrix_import.get("format") != "gs-playground-real2sim-motrix-import-v1":
        raise SystemExit("motrix_import.json has an unsupported format")
    if motrix_import.get("import_type") != "mjcf_scene_with_3dgs":
        raise SystemExit("motrix_import.json import_type must be mjcf_scene_with_3dgs")

    scene_xml = require_file(root / str(motrix_import.get("scene_xml", "")), "MJCF scene")
    config_json = require_file(root / str(motrix_import.get("config", "")), "viewer config")
    ET.parse(scene_xml)
    config = read_json(config_json)

    scene_gaussians = motrix_import.get("scene_gaussians")
    if not isinstance(scene_gaussians, dict) or not scene_gaussians.get("scene"):
        raise SystemExit("motrix_import.json missing scene_gaussians.scene")
    scene_ply = require_file(root / str(scene_gaussians["scene"]), "scene 3DGS PLY")
    vertices = count_ply_vertices(scene_ply)
    if vertices < min_scene_vertices:
        raise SystemExit(f"scene vertex count too low: {vertices} < {min_scene_vertices}")

    config_gaussians = config.get("scene_gaussians")
    if not isinstance(config_gaussians, dict) or not config_gaussians.get("scene"):
        raise SystemExit("viewer config missing scene_gaussians.scene")
    require_file(resolve_config_path(config_json, config_gaussians["scene"]), "config scene 3DGS PLY")

    camera_artifacts = motrix_import.get("camera_artifacts")
    if not isinstance(camera_artifacts, dict):
        raise SystemExit("motrix_import.json missing camera_artifacts")
    for key in ("cameras_json", "source_gs_preview"):
        require_file(root / str(camera_artifacts.get(key, "")), f"camera artifact {key}")

    preview_path = require_file(root / str(motrix_import.get("preview", "")), "preview image")
    reconstruction_quality = load_reconstruction_quality(
        root,
        motrix_import,
        min_source_preview_psnr=min_source_preview_psnr,
        max_source_preview_mae=max_source_preview_mae,
    )

    viewer_args = motrix_import.get("viewer_args")
    if not isinstance(viewer_args, list) or "--show-source-gs" not in viewer_args:
        raise SystemExit("viewer_args must include --show-source-gs")
    wuying_launch_command = motrix_import.get("wuying_archive_launch")
    if not isinstance(wuying_launch_command, str) or "--with-gs" not in wuying_launch_command:
        raise SystemExit("wuying_archive_launch must be present and include --with-gs")
    viewer_verification = viewer_verification_summary(
        wuying_launch_command=wuying_launch_command,
        viewer_args=viewer_args,
    )

    synthetic_quality = None
    quality_path = root / "synthetic_multiview_quality.json"
    if quality_path.is_file():
        synthetic_quality = read_json(quality_path)

    return {
        "ok": True,
        "root": root.as_posix(),
        "scene_xml": scene_xml.relative_to(root).as_posix(),
        "config": config_json.relative_to(root).as_posix(),
        "scene_ply": scene_ply.relative_to(root).as_posix(),
        "scene_ply_vertices": vertices,
        "preview": preview_path.relative_to(root).as_posix(),
        "source_preview_metrics": reconstruction_quality["source_preview_metrics"],
        "source_preview_quality_gate": reconstruction_quality["source_preview_quality_gate"],
        "reconstruction_quality": reconstruction_quality,
        "motrix_viewer_verification": viewer_verification,
        "synthetic_multiview_quality": synthetic_quality,
    }


def require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or empty {label}: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"JSON file must contain an object: {path}")
    return data


def resolve_config_path(config_json: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (config_json.parent / path).resolve()


def count_ply_vertices(path: Path) -> int:
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise SystemExit(f"PLY vertex count not found: {path}")


def require_metrics(metadata: dict[str, Any], min_psnr: float, max_mae: float) -> dict[str, float]:
    metrics = metadata.get("source_preview_metrics")
    if not isinstance(metrics, dict):
        raise SystemExit("reconstruction metadata missing source_preview_metrics")
    psnr = float(metrics.get("psnr", 0.0))
    mae = float(metrics.get("mae", 1.0))
    if psnr < min_psnr:
        raise SystemExit(f"source preview PSNR too low: {psnr:.3f} < {min_psnr:.3f}")
    if mae > max_mae:
        raise SystemExit(f"source preview MAE too high: {mae:.4f} > {max_mae:.4f}")
    return {"psnr": psnr, "mae": mae, "mse": float(metrics.get("mse", 0.0))}


def load_reconstruction_quality(
    root: Path,
    motrix_import: dict[str, Any],
    *,
    min_source_preview_psnr: float,
    max_source_preview_mae: float,
) -> dict[str, Any]:
    metadata_value = motrix_import.get("reconstruction_metadata")
    if metadata_value:
        metadata_path = root / str(metadata_value)
        metadata = read_json(require_file(metadata_path, "reconstruction metadata"))
        metrics = require_metrics(metadata, min_source_preview_psnr, max_source_preview_mae)
        gate = metadata.get("source_preview_quality_gate")
        if not isinstance(gate, dict) or not gate.get("passed"):
            raise SystemExit("source preview quality gate did not pass")
        return {
            "mode": "source_preview_metadata",
            "metadata": metadata_path.relative_to(root).as_posix(),
            "source_preview_metrics": metrics,
            "source_preview_quality_gate": gate,
            "mesh_cleanup": None,
            "texture_bake": None,
        }

    manifest_path = require_file(root / "package_manifest.json", "package manifest")
    manifest = read_json(manifest_path)
    package_metadata = manifest.get("metadata")
    if not isinstance(package_metadata, dict):
        raise SystemExit("package_manifest.json missing metadata")

    mesh_cleanup = read_optional_json(root / "artifacts" / "mesh_cleanup" / "cleanup_report.json")
    texture_bake = read_optional_json(root / "artifacts" / "colmap_texture_bake" / "texture_bake_report.json")
    if mesh_cleanup is None and texture_bake is None:
        raise SystemExit("missing reconstruction metadata or scene reconstruction quality reports")
    if texture_bake is not None:
        face_count = int(texture_bake.get("face_count", 0) or 0)
        drawn_faces = int(texture_bake.get("drawn_face_count", 0) or 0)
        coverage = float(texture_bake.get("texture_coverage_ratio", 0.0) or 0.0)
        if face_count <= 0 or drawn_faces <= 0 or coverage <= 0.0:
            raise SystemExit("texture bake report does not show a textured scene mesh")

    gate = {
        "passed": True,
        "mode": "scene_reconstruction_reports",
        "reason": "Existing multi-image scene package verified with mesh/texture reconstruction reports.",
    }
    return {
        "mode": "scene_reconstruction_reports",
        "metadata": "package_manifest.json",
        "source_preview_metrics": None,
        "source_preview_quality_gate": gate,
        "mesh_cleanup": mesh_cleanup,
        "texture_bake": texture_bake,
        "package_metadata": package_metadata,
    }


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    return read_json(path)


def viewer_verification_summary(*, wuying_launch_command: str, viewer_args: list[Any]) -> dict[str, Any]:
    return {
        "wuying_launch_command": wuying_launch_command,
        "viewer_args": [str(arg) for arg in viewer_args],
        "required_log_markers": [
            "Loading 3DGS assets: scene",
            "3DGS renderer ready.",
            "Scene viewer ready.",
            "3DGS rendered source camera.",
            "3DGS static preview rendered.",
        ],
    }


def render_mujoco_offscreen(
    root: Path,
    verification_report: dict[str, Any],
    output_dir: Path,
    *,
    cameras: list[str],
    width: int,
    height: int,
    max_geom: int = 0,
) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("MuJoCo offscreen rendering requires mujoco, Pillow, and numpy") from exc

    scene_xml = root / str(verification_report["scene_xml"])
    model = mujoco.MjModel.from_xml_path(scene_xml.as_posix())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial_contact_diagnostics = mujoco_contact_diagnostics(mujoco, model, data)
    settle_drift_probe = mujoco_settle_drift_probe(
        mujoco,
        model,
        physics_task_body_names(verification_report),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    renderer_max_geom = mujoco_renderer_max_geom(int(model.ngeom), requested=max_geom)

    renders: list[dict[str, Any]] = []
    for camera_name in cameras:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            raise SystemExit(f"MJCF camera not found for offscreen render: {camera_name}")
        try:
            renderer = mujoco.Renderer(model, height=height, width=width, max_geom=renderer_max_geom)
        except TypeError:
            renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            renderer.update_scene(data, camera=camera_id)
            rgb = renderer.render()
        finally:
            renderer.close()
        image = Image.fromarray(rgb)
        output_path = output_dir / f"{camera_name}_mujoco_offscreen.png"
        image.save(output_path)
        array = np.asarray(rgb, dtype=np.float32)
        quality = offscreen_image_quality(array)
        renders.append(
            {
                "camera": camera_name,
                "path": output_path.as_posix(),
                "size": [int(width), int(height)],
                "mean": float(array.mean()),
                "std": float(array.std()),
                "nonblank": bool(array.std() > 1.0),
                "quality": quality,
            }
        )

    if not renders or not all(render["nonblank"] for render in renders):
        raise SystemExit("MuJoCo offscreen render was blank or missing")
    simulation_stability = mujoco_simulation_stability(mujoco, model, data)
    report = {
        "ok": True,
        "scene_xml": scene_xml.as_posix(),
        "model": {
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "nmesh": int(model.nmesh),
            "ntex": int(model.ntex),
            "nmat": int(model.nmat),
            "ncam": int(model.ncam),
            "renderer_max_geom": int(renderer_max_geom),
        },
        "renders": renders,
        "initial_contact_diagnostics": initial_contact_diagnostics,
        "settle_drift_probe": settle_drift_probe,
        "simulation_stability": simulation_stability,
    }
    report_path = output_dir / "mujoco_offscreen_render_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def mujoco_simulation_stability(mujoco: Any, model: Any, data: Any, *, step_count: int = 240) -> dict[str, Any]:
    import numpy as np

    if not hasattr(mujoco, "mj_step"):
        return {
            "passed": True,
            "skipped": True,
            "reason": "mujoco_mj_step_unavailable",
            "step_count": int(step_count),
            "finite_state": True,
            "max_abs_qpos": 0.0,
            "max_abs_qvel": 0.0,
            "max_abs_qpos_threshold": 10.0,
            "max_abs_qvel_threshold": 100.0,
            "ncon": int(getattr(data, "ncon", 0)),
        }
    max_abs_qpos = 0.0
    max_abs_qvel = 0.0
    finite = True
    for _ in range(max(int(step_count), 1)):
        mujoco.mj_step(model, data)
        qpos = np.asarray(data.qpos, dtype=np.float64)
        qvel = np.asarray(data.qvel, dtype=np.float64)
        finite = finite and bool(np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel)))
        max_abs_qpos = max(max_abs_qpos, float(np.max(np.abs(qpos), initial=0.0)))
        max_abs_qvel = max(max_abs_qvel, float(np.max(np.abs(qvel), initial=0.0)))
        if not finite:
            break
    max_qpos_threshold = 10.0
    max_qvel_threshold = 100.0
    passed = bool(finite and max_abs_qpos < max_qpos_threshold and max_abs_qvel < max_qvel_threshold)
    return {
        "passed": passed,
        "step_count": int(step_count),
        "finite_state": bool(finite),
        "max_abs_qpos": round(max_abs_qpos, 6),
        "max_abs_qvel": round(max_abs_qvel, 6),
        "max_abs_qpos_threshold": max_qpos_threshold,
        "max_abs_qvel_threshold": max_qvel_threshold,
        "ncon": int(getattr(data, "ncon", 0)),
    }


def mujoco_contact_diagnostics(mujoco: Any, model: Any, data: Any, *, max_allowed_penetration: float = 0.25) -> dict[str, Any]:
    contacts: list[dict[str, Any]] = []
    max_penetration = 0.0
    for index in range(int(getattr(data, "ncon", 0))):
        contact = data.contact[index]
        dist = float(contact.dist)
        penetration = max(0.0, -dist)
        max_penetration = max(max_penetration, penetration)
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        contacts.append(
            {
                "geom1": geom1,
                "geom2": geom2,
                "dist": round(dist, 6),
                "penetration": round(penetration, 6),
            }
        )
    contacts.sort(key=lambda item: float(item["penetration"]), reverse=True)
    return {
        "passed": bool(max_penetration <= max_allowed_penetration),
        "contact_count": int(getattr(data, "ncon", 0)),
        "max_penetration": round(max_penetration, 6),
        "max_allowed_penetration": float(max_allowed_penetration),
        "deepest_contacts": contacts[:16],
    }


def mujoco_settle_drift_probe(
    mujoco: Any,
    model: Any,
    body_names: list[str],
    *,
    step_count: int = 240,
    max_allowed_displacement: float = 0.15,
    max_allowed_qvel: float = 2.0,
) -> dict[str, Any]:
    import numpy as np

    if not hasattr(mujoco, "mj_step"):
        return {
            "passed": True,
            "skipped": True,
            "reason": "mujoco_mj_step_unavailable",
            "step_count": int(step_count),
            "body_count": 0,
            "missing_bodies": [],
            "max_displacement": 0.0,
            "max_allowed_displacement": float(max_allowed_displacement),
            "max_abs_qvel": 0.0,
            "max_allowed_qvel": float(max_allowed_qvel),
            "bodies": [],
        }
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_ids: list[tuple[str, int]] = []
    missing: list[str] = []
    for name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            missing.append(name)
        else:
            body_ids.append((name, int(body_id)))
    start_positions = {name: np.asarray(data.xpos[body_id], dtype=np.float64).copy() for name, body_id in body_ids}
    for _ in range(max(int(step_count), 1)):
        mujoco.mj_step(model, data)
    body_reports: list[dict[str, Any]] = []
    max_displacement = 0.0
    for name, body_id in body_ids:
        end = np.asarray(data.xpos[body_id], dtype=np.float64)
        displacement = float(np.linalg.norm(end - start_positions[name]))
        max_displacement = max(max_displacement, displacement)
        body_reports.append(
            {
                "body": name,
                "displacement": round(displacement, 6),
                "start": [round(float(value), 6) for value in start_positions[name]],
                "end": [round(float(value), 6) for value in end],
            }
        )
    qvel = np.asarray(data.qvel, dtype=np.float64)
    max_abs_qvel = float(np.max(np.abs(qvel), initial=0.0))
    passed = bool(body_ids and not missing and max_displacement <= max_allowed_displacement and max_abs_qvel <= max_allowed_qvel)
    return {
        "passed": passed,
        "step_count": int(step_count),
        "body_count": len(body_ids),
        "missing_bodies": missing,
        "max_displacement": round(max_displacement, 6),
        "max_allowed_displacement": float(max_allowed_displacement),
        "max_abs_qvel": round(max_abs_qvel, 6),
        "max_allowed_qvel": float(max_allowed_qvel),
        "bodies": body_reports,
    }


def physics_task_readiness_report(root: Path, verification_report: dict[str, Any]) -> dict[str, Any]:
    scene_xml = root / str(verification_report["scene_xml"])
    tree = ET.parse(scene_xml)
    dynamic_bodies: list[dict[str, Any]] = []
    for body in tree.getroot().iter("body"):
        free_joints = free_joint_elements(body)
        if not free_joints:
            continue
        body_name = body.get("name") or "<unnamed>"
        inertials = body.findall("inertial")
        collision_geoms = [
            geom
            for geom in body.findall("geom")
            if geom_is_collision_enabled(geom)
        ]
        masses = [parse_positive_float(inertial.get("mass")) for inertial in inertials]
        diaginertias = [parse_float_triplet(inertial.get("diaginertia")) for inertial in inertials]
        damping_values = [parse_positive_float(joint.get("damping")) for joint in free_joints if joint.tag == "joint"]
        collision_half_extents = [geom_half_extents(geom) for geom in collision_geoms]
        max_collision_half_extent = max(
            (max(extents) for extents in collision_half_extents if extents is not None),
            default=0.0,
        )
        friction_values = [geom_friction(geom) for geom in collision_geoms]
        damping_ready = bool(damping_values and all(value is not None and value > 0.0 for value in damping_values))
        friction_ready = bool(friction_values and all(value is not None and value[0] >= 0.2 for value in friction_values))
        bounded_collision_proxy = bool(collision_geoms and 0.0 < max_collision_half_extent <= 0.25)
        has_positive_mass = any(mass is not None and mass > 0.0 for mass in masses)
        plausible_mass = bool(masses and all(mass is not None and 0.02 <= mass <= 5.0 for mass in masses))
        plausible_diaginertia = bool(
            diaginertias
            and all(values is not None and all(1e-6 <= item <= 1.0 for item in values) for values in diaginertias)
        )
        dynamic_bodies.append(
            {
                "body": body_name,
                "freejoint_count": len(free_joints),
                "inertial_count": len(inertials),
                "positive_mass_count": sum(1 for mass in masses if mass is not None and mass > 0.0),
                "mass_values": [round(float(mass), 6) for mass in masses if mass is not None],
                "diaginertia_values": [list(values) for values in diaginertias if values is not None],
                "plausible_mass": plausible_mass,
                "plausible_diaginertia": plausible_diaginertia,
                "collision_geom_count": len(collision_geoms),
                "free_joint_damping": [joint.get("damping") for joint in free_joints if joint.tag == "joint"],
                "damping_ready": damping_ready,
                "friction_ready": friction_ready,
                "bounded_collision_proxy": bounded_collision_proxy,
                "max_collision_half_extent": round(float(max_collision_half_extent), 6),
                "collision_geoms": [
                    {
                        "name": geom.get("name"),
                        "type": geom.get("type", "sphere"),
                        "contype": geom.get("contype", "1"),
                        "conaffinity": geom.get("conaffinity", "1"),
                        "size": geom.get("size"),
                        "friction": geom.get("friction"),
                    }
                    for geom in collision_geoms
                ],
                "ready": bool(
                    inertials
                    and has_positive_mass
                    and plausible_mass
                    and plausible_diaginertia
                    and collision_geoms
                    and damping_ready
                    and friction_ready
                    and bounded_collision_proxy
                ),
            }
        )
    summary = {
        "dynamic_body_count": len(dynamic_bodies),
        "ready_dynamic_body_count": sum(1 for item in dynamic_bodies if item["ready"]),
        "freejoint_body_count": len(dynamic_bodies),
    }
    report = {
        "format": "gs-playground-real2sim-physics-task-readiness-v1",
        "scene_xml": scene_xml.as_posix(),
        "summary": summary,
        "dynamic_bodies": dynamic_bodies,
    }
    report["passed"] = physics_task_readiness_passed(report)
    return report


def physics_task_readiness_passed(report: dict[str, Any]) -> bool:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    dynamic_count = int(summary.get("dynamic_body_count") or 0)
    ready_count = int(summary.get("ready_dynamic_body_count") or 0)
    static_passed = bool(dynamic_count > 0 and ready_count == dynamic_count)
    simulation = report.get("mujoco_simulation")
    contact = report.get("initial_contact_diagnostics")
    settle = report.get("settle_drift_probe")
    if isinstance(simulation, dict):
        contact_passed = bool(contact.get("passed")) if isinstance(contact, dict) else True
        settle_passed = bool(settle.get("passed")) if isinstance(settle, dict) else True
        return bool(static_passed and simulation.get("passed") and contact_passed and settle_passed)
    return static_passed


def physics_task_body_names(verification_report: dict[str, Any]) -> list[str]:
    physics = verification_report.get("physics_task_readiness")
    physics = physics if isinstance(physics, dict) else {}
    bodies = physics.get("dynamic_bodies") if isinstance(physics.get("dynamic_bodies"), list) else []
    return [str(item["body"]) for item in bodies if isinstance(item, dict) and item.get("body")]


def robot_task_readiness_report(root: Path, verification_report: dict[str, Any]) -> dict[str, Any]:
    scene_xml = root / str(verification_report["scene_xml"])
    config_json = root / str(verification_report["config"])
    config = read_json(config_json)
    tree = ET.parse(scene_xml)
    cameras = sorted({camera.get("name") for camera in tree.getroot().iter("camera") if camera.get("name")})
    manifest = read_optional_json(root / "package_manifest.json") or {}
    objects = manifest.get("objects") if isinstance(manifest.get("objects"), list) else []
    dynamic_names = dynamic_body_names(tree)
    target_objects = [
        {
            "name": item.get("name"),
            "label": item.get("label"),
            "dynamic": bool(item.get("dynamic")),
            "source_instance_id": item.get("source_instance_id"),
        }
        for item in objects
        if isinstance(item, dict) and bool(item.get("dynamic"))
    ]
    if not target_objects:
        target_objects = [{"name": name, "label": None, "dynamic": True, "source_instance_id": None} for name in dynamic_names]
    body_positions = dynamic_body_positions(tree)
    finite_targets = all(all(isinstance(value, (int, float)) and abs(float(value)) < 10.0 for value in pos) for pos in body_positions.values())
    physics = verification_report.get("physics_task_readiness")
    physics = physics if isinstance(physics, dict) else {}
    render_summary = strict_render_quality_summary(verification_report)
    robot = config.get("robot")
    gates = [
        strict_gate(
            "robot_config_present",
            isinstance(robot, str) and bool(robot.strip()),
            "viewer config names the robot platform expected to consume the Real2Sim package",
            {"robot": robot},
        ),
        strict_gate(
            "task_targets_present",
            bool(target_objects),
            "package exposes at least one dynamic target object for downstream navigation, reach, or manipulation tasks",
            {"target_count": len(target_objects), "targets": target_objects},
        ),
        strict_gate(
            "task_targets_physics_ready",
            bool(physics.get("passed")),
            "all dynamic task targets have contact-ready physics parameters and stable MuJoCo simulation",
            physics,
        ),
        strict_gate(
            "observation_cameras_present",
            {"overview", "closeup"}.issubset(set(cameras)),
            "MJCF includes standard overview and closeup cameras for task observation and review",
            {"cameras": cameras, "required": ["overview", "closeup"]},
        ),
        strict_gate(
            "observation_render_ready",
            bool(render_summary.get("passed")),
            "offscreen task-observation renders are nonblank and balanced",
            render_summary,
        ),
        strict_gate(
            "finite_task_target_poses",
            bool(body_positions and finite_targets),
            "dynamic task target body poses are finite and inside the verifier workspace envelope",
            {"body_positions": body_positions, "max_abs_coordinate": 10.0},
        ),
    ]
    blockers = [gate["id"] for gate in gates if not gate["passed"]]
    return {
        "format": "gs-playground-real2sim-robot-task-readiness-v1",
        "passed": not blockers,
        "summary": {
            "robot": robot,
            "target_count": len(target_objects),
            "camera_count": len(cameras),
            "blockers": blockers,
        },
        "gates": gates,
        "task_targets": target_objects,
    }


def task_execution_probe_report(root: Path, verification_report: dict[str, Any]) -> dict[str, Any]:
    scene_xml = root / str(verification_report["scene_xml"])
    tree = ET.parse(scene_xml)
    probes = dynamic_task_execution_probes(tree)
    robot_task = verification_report.get("robot_task_readiness")
    robot_task = robot_task if isinstance(robot_task, dict) else {}
    gates = [
        strict_gate(
            "execution_targets_present",
            bool(probes),
            "at least one dynamic task target has an execution probe",
            {"probe_count": len(probes)},
        ),
        strict_gate(
            "collision_targets_present",
            bool(probes and all(probe["collision_geom_count"] > 0 for probe in probes)),
            "each execution target has an enabled collision proxy for task contact",
            {"probes": probes},
        ),
        strict_gate(
            "manipulation_proxy_scale",
            bool(probes and all(0.015 <= float(probe["max_collision_half_extent"]) <= 0.25 for probe in probes)),
            "target collision proxies are within the desktop manipulation scale envelope",
            {"min_half_extent": 0.015, "max_half_extent": 0.25, "probes": probes},
        ),
        strict_gate(
            "approach_waypoints_valid",
            bool(probes and all(probe["approach_waypoint_count"] >= 4 and probe["approach_waypoints_in_workspace"] for probe in probes)),
            "each target has four finite cardinal approach waypoints inside the verifier workspace",
            {"workspace_max_abs_coordinate": 10.0, "probes": probes},
        ),
        strict_gate(
            "robot_task_gate_already_passed",
            bool(robot_task.get("passed")),
            "task execution probes are only accepted after robot task readiness passes",
            robot_task,
        ),
    ]
    blockers = [gate["id"] for gate in gates if not gate["passed"]]
    return {
        "format": "gs-playground-real2sim-task-execution-probe-v1",
        "passed": not blockers,
        "summary": {
            "probe_count": len(probes),
            "blockers": blockers,
        },
        "gates": gates,
        "probes": probes,
    }


def dynamic_task_execution_probes(tree: ET.ElementTree) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for body in tree.getroot().iter("body"):
        if not free_joint_elements(body):
            continue
        body_name = body.get("name") or "<unnamed>"
        body_pos = parse_float_list(body.get("pos"), expected=3) or [0.0, 0.0, 0.0]
        collision_geoms = [geom for geom in body.findall("geom") if geom_is_collision_enabled(geom)]
        collision_centers: list[list[float]] = []
        half_extents: list[tuple[float, float, float]] = []
        for geom in collision_geoms:
            local_pos = parse_float_list(geom.get("pos"), expected=3) or [0.0, 0.0, 0.0]
            collision_centers.append([body_pos[idx] + local_pos[idx] for idx in range(3)])
            extents = geom_half_extents(geom)
            if extents is not None:
                half_extents.append(extents)
        if collision_centers:
            target_center = [
                sum(center[idx] for center in collision_centers) / float(len(collision_centers))
                for idx in range(3)
            ]
        else:
            target_center = [float(value) for value in body_pos]
        max_half = max((max(extents) for extents in half_extents), default=0.0)
        standoff = max(0.25, min(0.6, 2.5 * max_half + 0.1))
        z_clearance = min(max_half + 0.05, 0.35)
        waypoints = [
            [target_center[0] + standoff, target_center[1], target_center[2] + z_clearance],
            [target_center[0] - standoff, target_center[1], target_center[2] + z_clearance],
            [target_center[0], target_center[1] + standoff, target_center[2] + z_clearance],
            [target_center[0], target_center[1] - standoff, target_center[2] + z_clearance],
        ]
        in_workspace = all(all(abs(float(value)) <= 10.0 for value in waypoint) for waypoint in waypoints)
        probes.append(
            {
                "body": body_name,
                "target_center": [round(float(value), 6) for value in target_center],
                "collision_geom_count": len(collision_geoms),
                "max_collision_half_extent": round(float(max_half), 6),
                "standoff_distance": round(float(standoff), 6),
                "approach_waypoint_count": len(waypoints),
                "approach_waypoints_in_workspace": bool(in_workspace),
                "approach_waypoints": [[round(float(value), 6) for value in waypoint] for waypoint in waypoints],
            }
        )
    return probes


def dynamic_body_names(tree: ET.ElementTree) -> list[str]:
    names: list[str] = []
    for body in tree.getroot().iter("body"):
        if not free_joint_elements(body):
            continue
        if body.get("name"):
            names.append(str(body.get("name")))
    return names


def dynamic_body_positions(tree: ET.ElementTree) -> dict[str, list[float]]:
    positions: dict[str, list[float]] = {}
    for body in tree.getroot().iter("body"):
        if not free_joint_elements(body):
            continue
        name = body.get("name") or "<unnamed>"
        pos = parse_float_list(body.get("pos"), expected=3)
        if pos is not None:
            positions[name] = [round(float(value), 6) for value in pos]
    return positions


def free_joint_elements(body: ET.Element) -> list[ET.Element]:
    free_joints = list(body.findall("freejoint"))
    free_joints.extend(joint for joint in body.findall("joint") if joint.get("type") == "free")
    return free_joints


def geom_is_collision_enabled(geom: ET.Element) -> bool:
    contype = parse_int_attr(geom.get("contype"), default=1)
    conaffinity = parse_int_attr(geom.get("conaffinity"), default=1)
    return bool(contype != 0 or conaffinity != 0)


def parse_int_attr(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except ValueError:
        return default


def parse_positive_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(str(value).split()[0])
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def parse_float_triplet(value: str | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        values = [float(item) for item in str(value).split()]
    except ValueError:
        return None
    if len(values) != 3:
        return None
    return tuple(values)


def parse_float_list(value: str | None, *, expected: int) -> list[float] | None:
    if value is None:
        return None
    try:
        values = [float(item) for item in str(value).split()]
    except ValueError:
        return None
    if len(values) != expected:
        return None
    return values


def geom_half_extents(geom: ET.Element) -> tuple[float, float, float] | None:
    size = geom.get("size")
    if not size:
        return None
    try:
        values = [abs(float(item)) for item in str(size).split()]
    except ValueError:
        return None
    if not values:
        return None
    if len(values) == 1:
        values = values * 3
    elif len(values) == 2:
        values = [values[0], values[0], values[1]]
    return tuple(values[:3])


def geom_friction(geom: ET.Element) -> tuple[float, float, float] | None:
    value = geom.get("friction")
    if not value:
        return None
    try:
        values = [float(item) for item in str(value).split()]
    except ValueError:
        return None
    if len(values) < 3:
        return None
    return tuple(values[:3])


def mujoco_renderer_max_geom(model_ngeom: int, *, requested: int = 0) -> int:
    if requested > 0:
        return max(int(requested), 10_000)
    return max(int(model_ngeom) + 1_000, 10_000)


def offscreen_image_quality(array: Any) -> dict[str, float]:
    import numpy as np

    rgb = np.asarray(array, dtype=np.float32)
    if rgb.max(initial=0.0) > 1.0:
        rgb = rgb / 255.0
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise SystemExit("offscreen render must be an RGB image")
    rgb = np.clip(rgb[:, :, :3], 0.0, 1.0)
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    black_ratio = float(np.mean(luminance < 0.03))
    dark_ratio = float(np.mean(luminance < 0.08))
    mean_luminance = float(np.mean(luminance))
    std_luminance = float(np.std(luminance))
    high_saturation_ratio = float(np.mean(chroma > 0.35))
    score = 1.0
    score -= 1.5 * black_ratio
    score -= 2.0 * max(0.0, high_saturation_ratio - 0.75)
    score -= 0.8 * max(0.0, mean_luminance - 0.72)
    score -= 0.7 * max(0.0, 0.18 - std_luminance)
    return {
        "black_ratio": black_ratio,
        "dark_ratio": dark_ratio,
        "mean_luminance": mean_luminance,
        "std_luminance": std_luminance,
        "high_saturation_ratio": high_saturation_ratio,
        "score": float(min(max(score, 0.0), 1.0)),
    }


def write_visual_acceptance_report(
    root: Path,
    verification_report: dict[str, Any],
    output_dir: Path,
    *,
    motrix_viewer_render: Path | None = None,
    motrix_viewer_render_label: str = "Motrix viewer render",
) -> dict[str, Any]:
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise SystemExit("visual report generation requires Pillow and numpy") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    source_preview = require_file(root / str(verification_report["preview"]), "source GS preview")
    source_image, source_selection = select_visual_source_image(root, reference_image=source_preview)
    pointcloud_preview = root / "pointcloud_projection_preview.png"
    panels: list[tuple[str, Path]] = [
        ("source image", source_image),
        ("3DGS source render", source_preview),
    ]
    if pointcloud_preview.is_file():
        panels.append(("pointcloud projection", pointcloud_preview))
    if motrix_viewer_render is not None:
        panels.append((motrix_viewer_render_label, require_file(motrix_viewer_render, motrix_viewer_render_label)))

    panel_size = (360, 240)
    header_h = 34
    contact = Image.new("RGB", (panel_size[0] * len(panels), panel_size[1] + header_h), "white")
    draw = ImageDraw.Draw(contact)
    image_summaries: dict[str, dict[str, Any]] = {}
    for index, (label, path) in enumerate(panels):
        image = Image.open(path).convert("RGB")
        image_summaries[label] = {"path": path.as_posix(), "size": list(image.size)}
        thumb = fit_image(image, panel_size)
        x = index * panel_size[0]
        draw.text((x + 10, 10), label, fill=(20, 20, 20))
        contact.paste(thumb, (x + (panel_size[0] - thumb.width) // 2, header_h + (panel_size[1] - thumb.height) // 2))

    contact_sheet = output_dir / "visual_acceptance_contact_sheet.png"
    contact.save(contact_sheet)

    source = Image.open(source_image).convert("RGB")
    preview = Image.open(source_preview).convert("RGB").resize(source.size)
    diff = np.asarray(source, dtype=np.float32) / 255.0 - np.asarray(preview, dtype=np.float32) / 255.0
    resized_mae = float(np.mean(np.abs(diff)))

    report = {
        "ok": bool(verification_report["ok"]),
        "contact_sheet": contact_sheet.as_posix(),
        "images": image_summaries,
        "source_selection": source_selection,
        "source_to_preview_resized_mae": resized_mae,
        "quality_gate": verification_report["source_preview_quality_gate"],
        "source_preview_metrics": verification_report["source_preview_metrics"],
        "reconstruction_quality": verification_report.get("reconstruction_quality"),
        "notes": [
            "Use the contact sheet for human visual acceptance.",
            "The resized MAE is a coarse sanity check because source and preview cameras/aspect ratios can differ.",
        ],
    }
    report_path = output_dir / "visual_acceptance_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"json": report_path.as_posix(), "contact_sheet": contact_sheet.as_posix(), "resized_mae": resized_mae}


def write_strict_commercial_visual_quality_report(
    root: Path,
    verification_report: dict[str, Any],
    output_dir: Path,
    *,
    visual_acceptance: dict[str, Any] | None = None,
    require_commercial_acceptance: bool = False,
    require_capture_acceptance: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = strict_commercial_visual_quality_report(
        root,
        verification_report,
        visual_acceptance=visual_acceptance,
        require_commercial_acceptance=require_commercial_acceptance,
        require_capture_acceptance=require_capture_acceptance,
    )
    report_path = output_dir / "strict_commercial_visual_quality.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "json": report_path.as_posix(),
        "summary": report["summary"],
        "blockers": report["summary"]["blockers"],
    }


def strict_commercial_visual_quality_report(
    root: Path,
    verification_report: dict[str, Any],
    *,
    visual_acceptance: dict[str, Any] | None = None,
    require_commercial_acceptance: bool = False,
    require_capture_acceptance: bool = False,
) -> dict[str, Any]:
    commercial = read_optional_json(root / "artifacts" / "quality" / "commercial_readiness.json") or {}
    motrix_import = read_optional_json(root / "motrix_import.json") or {}
    commercial_acceptance = read_optional_json(root / "artifacts" / "quality" / "commercial_acceptance.json") or {}
    room = read_optional_json(root / "artifacts" / "quality" / "room_completeness.json") or {}
    metric_scale = read_optional_json(root / "artifacts" / "quality" / "metric_scale.json") or {}
    navigation_acceptance = read_optional_json(root / "artifacts" / "quality" / "navigation_acceptance.json") or {}
    material_readiness = read_optional_json(root / "artifacts" / "quality" / "material_readiness.json") or {}
    task_affordance = read_optional_json(root / "artifacts" / "quality" / "task_affordance.json") or {}
    movable = read_optional_json(root / "artifacts" / "assets" / "movable_assets.json") or {}
    commercial_summary = commercial.get("summary") if isinstance(commercial.get("summary"), dict) else {}
    commercial_acceptance_summary = commercial_acceptance.get("summary") if isinstance(commercial_acceptance.get("summary"), dict) else {}
    room_summary = room.get("summary") if isinstance(room.get("summary"), dict) else {}
    metric_scale_summary = metric_scale.get("summary") if isinstance(metric_scale.get("summary"), dict) else {}
    navigation_summary = navigation_acceptance.get("summary") if isinstance(navigation_acceptance.get("summary"), dict) else {}
    material_summary = material_readiness.get("summary") if isinstance(material_readiness.get("summary"), dict) else {}
    task_affordance_summary = task_affordance.get("summary") if isinstance(task_affordance.get("summary"), dict) else {}
    movable_quality = movable_asset_quality_summary(movable)
    reconstruction = verification_report.get("reconstruction_quality")
    reconstruction = reconstruction if isinstance(reconstruction, dict) else {}
    texture = reconstruction.get("texture_bake") if isinstance(reconstruction.get("texture_bake"), dict) else {}
    mesh_cleanup = reconstruction.get("mesh_cleanup") if isinstance(reconstruction.get("mesh_cleanup"), dict) else {}
    visual_detail = load_visual_acceptance_detail(visual_acceptance)
    render_summary = strict_render_quality_summary(verification_report)
    physics_task = verification_report.get("physics_task_readiness")
    physics_task = physics_task if isinstance(physics_task, dict) else {}
    robot_task = verification_report.get("robot_task_readiness")
    robot_task = robot_task if isinstance(robot_task, dict) else {}
    execution_probe = verification_report.get("task_execution_probe")
    execution_probe = execution_probe if isinstance(execution_probe, dict) else {}
    capture_acceptance = capture_space_acceptance_summary(root)
    capture_acceptance_required = bool(require_capture_acceptance or require_commercial_acceptance)
    material_required = material_readiness_required(commercial, motrix_import)
    task_affordance_required = task_affordance_required_for_package(commercial, motrix_import)
    commercial_acceptance_required = require_commercial_acceptance or commercial_acceptance_required_for_package(
        commercial,
        motrix_import,
    )
    commercial_acceptance_declaration = commercial_acceptance_declaration_report(
        motrix_import,
        required=commercial_acceptance_required,
    )
    commercial_acceptance_schema = commercial_acceptance_schema_report(
        commercial_acceptance,
        required=commercial_acceptance_required,
    )
    commercial_acceptance_integrity = commercial_acceptance_integrity_report(
        root,
        commercial_acceptance,
        required=commercial_acceptance_required,
    )
    source_to_preview_mae = visual_detail.get("source_to_preview_resized_mae")
    source_mae_target = 0.16
    source_mae_score = source_fidelity_score(source_to_preview_mae, target=source_mae_target)
    texture_face_ratio = texture_textured_face_ratio(texture)
    room_score = float(room_summary.get("score") or 0.0)
    engineering_score = 1.0 if commercial_summary.get("commercial_grade") else float(commercial_summary.get("score") or 0.0)
    score = round(
        min(
            1.0,
            0.25 * engineering_score
            + 0.25 * float(render_summary["average_score"])
            + 0.2 * source_mae_score
            + 0.15 * min(room_score, 1.0)
            + 0.1 * min(texture_face_ratio, 1.0)
            + 0.02 * min(float(mesh_cleanup.get("kept_area_ratio") or 0.0), 1.0)
            + 0.02 * (1.0 if physics_task.get("passed") else 0.0)
            + 0.005 * (1.0 if robot_task.get("passed") else 0.0)
            + 0.005 * (1.0 if execution_probe.get("passed") else 0.0),
        ),
        6,
    )
    gates = [
        strict_gate(
            "engineering_import_readiness",
            bool(commercial_summary.get("commercial_grade")),
            "package passes core Real2Sim import, structure, object, collision, navigation, and texture gates",
            {"commercial_readiness": commercial_summary},
        ),
        strict_gate(
            "commercial_acceptance_manifest",
            bool(
                (not commercial_acceptance_required and not commercial_acceptance_summary)
                or (
                    commercial_acceptance_declaration["passed"]
                    and commercial_acceptance_schema["passed"]
                    and commercial_acceptance_summary.get("accepted")
                    and commercial_acceptance_integrity["passed"]
                )
            ),
            "when the package declares a commercial acceptance manifest, the manifest must exist, be accepted, and match declared artifact hashes",
            {
                "evaluated": bool(commercial_acceptance_summary),
                "required": commercial_acceptance_required,
                "require_commercial_acceptance": bool(require_commercial_acceptance),
                "declaration": commercial_acceptance_declaration,
                "schema": commercial_acceptance_schema,
                "summary": commercial_acceptance_summary,
                "artifact_integrity": commercial_acceptance_integrity,
            },
        ),
        strict_gate(
            "render_nonblank_and_balanced",
            bool(render_summary["passed"]),
            "MuJoCo/Motrix renders are nonblank and visually balanced across requested cameras",
            render_summary,
        ),
        strict_gate(
            "source_view_fidelity",
            isinstance(source_to_preview_mae, (int, float)) and float(source_to_preview_mae) <= source_mae_target,
            "source image and reconstructed source-view preview remain close after resize",
            {"source_to_preview_resized_mae": source_to_preview_mae, "target_max": source_mae_target, "score": round(source_mae_score, 6)},
        ),
        strict_gate(
            "observed_room_floor",
            bool(room_summary.get("has_observed_floor")),
            "commercial room reconstruction should observe a real floor plane rather than only using an inferred simulation floor",
            {
                "has_observed_floor": room_summary.get("has_observed_floor"),
                "has_floor_completion": room_summary.get("has_floor_completion"),
                "room_completeness_score": room_summary.get("score"),
            },
        ),
        strict_gate(
            "texture_mesh_coverage",
            texture_face_ratio >= 0.98,
            "nearly all reconstructed mesh faces have projected/baked texture",
            {"textured_face_ratio": round(texture_face_ratio, 6)},
        ),
        strict_gate(
            "production_movable_assets",
            bool(movable_quality["commercial_asset_grade"]),
            "movable objects are production-grade reconstructed assets, not provisional extraction-task billboards",
            movable_quality,
        ),
        strict_gate(
            "physics_task_readiness",
            bool(physics_task.get("passed")),
            "dynamic assets have free joints, positive inertial mass, enabled collision geometry, and stable MuJoCo simulation when rendered",
            physics_task,
        ),
        strict_gate(
            "robot_task_readiness",
            bool(robot_task.get("passed")),
            "package exposes robot config, task targets, observation cameras, and render-ready task views",
            robot_task,
        ),
        strict_gate(
            "task_execution_probe",
            bool(execution_probe.get("passed")),
            "task targets expose finite approach waypoints and manipulation-scale collision proxies for execution probes",
            execution_probe,
        ),
        strict_gate(
            "task_affordance_targets",
            bool((not task_affordance_required and not task_affordance_summary) or task_affordance_summary.get("passed")),
            "when task affordance analysis is available, production movable assets must expose finite poses and actionable contact roles",
            {
                "evaluated": bool(task_affordance_summary),
                "required": task_affordance_required,
                "summary": task_affordance_summary,
            },
        ),
        strict_gate(
            "capture_space_acceptance",
            bool((not capture_acceptance_required and not capture_acceptance["evaluated"]) or capture_acceptance["accepted"]),
            "when SfM capture acceptance is available, camera path coverage must support space-level reconstruction",
            {
                **capture_acceptance,
                "required": capture_acceptance_required,
                "require_capture_acceptance": bool(require_capture_acceptance),
            },
        ),
        strict_gate(
            "metric_scale_plausibility",
            bool((not metric_scale_summary) or metric_scale_summary.get("passed")),
            "when metric-scale analysis is available, room bounds and structure proxies must be plausible for robot simulation",
            {
                "evaluated": bool(metric_scale_summary),
                "summary": metric_scale_summary,
            },
        ),
        strict_gate(
            "navigation_acceptance",
            bool((not navigation_summary) or navigation_summary.get("passed")),
            "when navigation acceptance is available, package must expose a robot-usable walkable surface",
            {
                "evaluated": bool(navigation_summary),
                "summary": navigation_summary,
            },
        ),
        strict_gate(
            "material_readiness",
            bool((not material_required and not material_summary) or material_summary.get("passed")),
            "when material readiness is available, package must include texture-backed material assets and referenced texture files",
            {
                "evaluated": bool(material_summary),
                "required": material_required,
                "summary": material_summary,
            },
        ),
    ]
    blockers = [gate["id"] for gate in gates if not gate["passed"]]
    return {
        "format": "gs-playground-real2sim-strict-commercial-visual-quality-v1",
        "summary": {
            "score": score,
            "target": 0.85,
            "strict_commercial_visual_grade": bool(score >= 0.85 and not blockers),
            "blocker_count": len(blockers),
            "blockers": blockers,
        },
        "gates": gates,
        "evidence": {
            "commercial_readiness": commercial_summary,
            "commercial_acceptance": commercial_acceptance_summary,
            "commercial_acceptance_declaration": commercial_acceptance_declaration,
            "commercial_acceptance_schema": commercial_acceptance_schema,
            "commercial_acceptance_integrity": commercial_acceptance_integrity,
            "room_completeness": room_summary,
            "metric_scale": metric_scale_summary,
            "navigation_acceptance": navigation_summary,
            "material_readiness": material_summary,
            "task_affordance": task_affordance_summary,
            "render_quality": render_summary,
            "source_to_preview_resized_mae": source_to_preview_mae,
            "texture_textured_face_ratio": round(texture_face_ratio, 6),
            "mesh_cleanup_kept_area_ratio": mesh_cleanup.get("kept_area_ratio"),
            "movable_asset_quality": movable_quality,
            "physics_task_readiness": physics_task,
            "robot_task_readiness": robot_task,
            "task_execution_probe": execution_probe,
            "capture_space_acceptance": capture_acceptance,
        },
        "next_actions": strict_visual_next_actions(blockers),
        "notes": [
            "This strict report is intentionally harder than package import readiness.",
            "It is meant to track the remaining gap to Polycam/RealityCapture-like visual and geometric quality.",
        ],
    }


def material_readiness_required(commercial: dict[str, Any], motrix_import: dict[str, Any]) -> bool:
    if isinstance(motrix_import, dict) and motrix_import.get("material_readiness_report"):
        return True
    gates = commercial.get("gates") if isinstance(commercial, dict) else None
    if not isinstance(gates, list):
        return False
    return any(isinstance(gate, dict) and gate.get("id") == "material_readiness" for gate in gates)


def task_affordance_required_for_package(commercial: dict[str, Any], motrix_import: dict[str, Any]) -> bool:
    if isinstance(motrix_import, dict) and motrix_import.get("task_affordance_report"):
        return True
    gates = commercial.get("gates") if isinstance(commercial, dict) else None
    if not isinstance(gates, list):
        return False
    return any(isinstance(gate, dict) and gate.get("id") == "task_affordance_targets" for gate in gates)


def commercial_acceptance_required_for_package(commercial: dict[str, Any], motrix_import: dict[str, Any]) -> bool:
    if isinstance(motrix_import, dict) and motrix_import.get("commercial_acceptance_report"):
        return True
    gates = commercial.get("gates") if isinstance(commercial, dict) else None
    if not isinstance(gates, list):
        return False
    return any(isinstance(gate, dict) and gate.get("id") == "commercial_acceptance_manifest" for gate in gates)


def commercial_acceptance_declaration_report(motrix_import: dict[str, Any], *, required: bool = False) -> dict[str, Any]:
    actual_path = motrix_import.get("commercial_acceptance_report") if isinstance(motrix_import, dict) else None
    if not actual_path:
        return {
            "evaluated": False,
            "passed": not required,
            "expected_path": COMMERCIAL_ACCEPTANCE_REPORT_PATH,
            "actual_path": None,
            "issues": [{"reason": "missing_commercial_acceptance_report"}] if required else [],
        }
    issues = []
    if actual_path != COMMERCIAL_ACCEPTANCE_REPORT_PATH:
        issues.append(
            {
                "reason": "commercial_acceptance_report_path_mismatch",
                "expected": COMMERCIAL_ACCEPTANCE_REPORT_PATH,
                "actual": actual_path,
            }
        )
    return {
        "evaluated": True,
        "passed": not issues,
        "expected_path": COMMERCIAL_ACCEPTANCE_REPORT_PATH,
        "actual_path": actual_path,
        "issues": issues,
    }


def commercial_acceptance_schema_report(manifest: dict[str, Any], *, required: bool = False) -> dict[str, Any]:
    if not isinstance(manifest, dict) or not manifest:
        return {
            "evaluated": False,
            "passed": not required,
            "expected_format": COMMERCIAL_ACCEPTANCE_FORMAT,
            "actual_format": None,
            "issues": [{"reason": "missing_commercial_acceptance_manifest"}] if required else [],
        }
    actual_format = manifest.get("format")
    issues = []
    if actual_format != COMMERCIAL_ACCEPTANCE_FORMAT:
        issues.append(
            {
                "reason": "unsupported_commercial_acceptance_format",
                "expected": COMMERCIAL_ACCEPTANCE_FORMAT,
                "actual": actual_format,
            }
        )
    if not isinstance(manifest.get("summary"), dict):
        issues.append({"reason": "missing_commercial_acceptance_summary"})
    return {
        "evaluated": True,
        "passed": not issues,
        "expected_format": COMMERCIAL_ACCEPTANCE_FORMAT,
        "actual_format": actual_format,
        "issues": issues,
    }


def commercial_acceptance_integrity_report(root: Path, manifest: dict[str, Any], *, required: bool = False) -> dict[str, Any]:
    artifact_integrity = manifest.get("artifact_integrity") if isinstance(manifest, dict) else None
    if not isinstance(artifact_integrity, dict):
        return {
            "evaluated": False,
            "passed": not required,
            "checked_count": 0,
            "issues": [{"reason": "missing_artifact_integrity"}] if required else [],
        }
    issues: list[dict[str, Any]] = []
    checked_count = 0
    groups: dict[str, Any] = {}
    required_groups: dict[str, dict[str, str]] = {
        "entrypoints": REQUIRED_COMMERCIAL_ACCEPTANCE_ENTRYPOINTS,
        "reports": REQUIRED_COMMERCIAL_ACCEPTANCE_REPORTS,
    }
    for group_name in ("entrypoints", "reports"):
        group = artifact_integrity.get(group_name)
        if not isinstance(group, dict):
            if required:
                issues.append({"group": group_name, "reason": "missing_integrity_group"})
            continue
        group_result: dict[str, Any] = {}
        if required:
            for artifact_id in sorted(set(required_groups[group_name]) - set(str(key) for key in group)):
                issue = {"group": group_name, "artifact": artifact_id, "reason": "missing_integrity_entry"}
                group_result[artifact_id] = {"passed": False, "reason": "missing_integrity_entry"}
                issues.append(issue)
        for artifact_id, declared in group.items():
            checked_count += 1
            expected_path = required_groups[group_name].get(str(artifact_id)) if required else None
            result = validate_declared_artifact(root, declared, expected_path=expected_path)
            group_result[str(artifact_id)] = result
            if not result["passed"]:
                issue = {"group": group_name, "artifact": str(artifact_id)}
                issue.update(result)
                issues.append(issue)
        groups[group_name] = group_result
    if checked_count == 0:
        issues.append({"reason": "empty_artifact_integrity"})
    return {
        "evaluated": True,
        "passed": not issues,
        "checked_count": checked_count,
        "issues": issues,
        "groups": groups,
    }


def validate_declared_artifact(root: Path, declared: Any, *, expected_path: str | None = None) -> dict[str, Any]:
    if not isinstance(declared, dict):
        return {"passed": False, "reason": "invalid_manifest_entry"}
    package_path = declared.get("path")
    if not isinstance(package_path, str) or not package_path:
        return {"passed": False, "reason": "missing_path"}
    if Path(package_path).is_absolute() or ".." in Path(package_path).parts:
        return {"path": package_path, "passed": False, "reason": "unsafe_package_path"}
    path = root / package_path
    if not path.is_file():
        return {"path": package_path, "passed": False, "reason": "missing_file"}
    actual_size = path.stat().st_size
    actual_sha256 = sha256_file(path) if actual_size > 0 else None
    expected_size = declared.get("size")
    expected_sha256 = declared.get("sha256")
    issues = []
    if expected_path is not None and package_path != expected_path:
        issues.append("path_mismatch")
    if not isinstance(expected_size, int) or expected_size != actual_size:
        issues.append("size_mismatch")
    if not isinstance(expected_sha256, str) or expected_sha256 != actual_sha256:
        issues.append("sha256_mismatch")
    return {
        "path": package_path,
        "passed": not issues,
        "issues": issues,
        "expected_path": expected_path,
        "expected_size": expected_size,
        "actual_size": actual_size,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_visual_acceptance_detail(visual_acceptance: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(visual_acceptance, dict):
        return {}
    value = visual_acceptance.get("json")
    if not value:
        return {"source_to_preview_resized_mae": visual_acceptance.get("resized_mae")}
    path = Path(str(value))
    if not path.is_file():
        return {"source_to_preview_resized_mae": visual_acceptance.get("resized_mae")}
    try:
        return read_json(path)
    except Exception:
        return {"source_to_preview_resized_mae": visual_acceptance.get("resized_mae")}


def capture_space_acceptance_summary(root: Path) -> dict[str, Any]:
    report = read_optional_json(root / "artifacts" / "capture_acceptance_report.json")
    if not isinstance(report, dict) or not report:
        return {
            "evaluated": False,
            "accepted": None,
            "decision": "not_evaluated",
            "blockers": [],
            "metrics": {},
        }
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    metric_keys = [
        "frame_count",
        "registered_image_count",
        "registered_ratio",
        "sparse_point_count",
        "verified_pair_count",
        "camera_path_length",
        "camera_planar_coverage_ratio",
        "camera_max_step_ratio",
        "camera_loop_closure_ratio",
    ]
    return {
        "evaluated": True,
        "accepted": bool(report.get("accepted")),
        "decision": report.get("decision"),
        "blockers": report.get("blockers", []),
        "metrics": {key: metrics.get(key) for key in metric_keys if key in metrics},
    }


def strict_render_quality_summary(verification_report: dict[str, Any]) -> dict[str, Any]:
    offscreen = verification_report.get("mujoco_offscreen_render")
    renders = offscreen.get("renders") if isinstance(offscreen, dict) else []
    renders = renders if isinstance(renders, list) else []
    scores: list[float] = []
    nonblank = True
    max_black = 1.0
    max_saturation = 1.0
    for render in renders:
        if not isinstance(render, dict):
            continue
        nonblank = nonblank and bool(render.get("nonblank"))
        quality = render.get("quality") if isinstance(render.get("quality"), dict) else {}
        scores.append(float(quality.get("score") or 0.0))
        max_black = min(max_black, float(quality.get("black_ratio") or 0.0)) if max_black == 1.0 else max(max_black, float(quality.get("black_ratio") or 0.0))
        max_saturation = min(max_saturation, float(quality.get("high_saturation_ratio") or 0.0)) if max_saturation == 1.0 else max(max_saturation, float(quality.get("high_saturation_ratio") or 0.0))
    average = sum(scores) / len(scores) if scores else 0.0
    minimum = min(scores) if scores else 0.0
    if not renders:
        max_black = 1.0
        max_saturation = 1.0
    return {
        "render_count": len(renders),
        "all_nonblank": bool(nonblank and renders),
        "average_score": round(float(average), 6),
        "min_score": round(float(minimum), 6),
        "max_black_ratio": round(float(max_black), 6),
        "max_high_saturation_ratio": round(float(max_saturation), 6),
        "passed": bool(renders and nonblank and average >= 0.78 and minimum >= 0.70 and max_black <= 0.16),
    }


def texture_textured_face_ratio(texture: dict[str, Any]) -> float:
    value = texture.get("textured_face_ratio") if isinstance(texture, dict) else None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(texture, dict):
        return 0.0
    drawn = int(texture.get("drawn_face_count") or 0)
    selected = int(texture.get("selected_face_count") or texture.get("face_count") or 0)
    return drawn / float(max(selected, 1))


def movable_asset_quality_summary(movable: dict[str, Any]) -> dict[str, Any]:
    quality = movable.get("quality") if isinstance(movable.get("quality"), dict) else {}
    assets = movable.get("movable_assets") if isinstance(movable.get("movable_assets"), list) else []
    asset_count = int(quality.get("movable_asset_count") or len(assets))
    production_count = int(quality.get("production_asset_count") or 0)
    provisional_count = int(quality.get("provisional_asset_count") or 0)
    incomplete_count = int(quality.get("incomplete_asset_count") or 0)
    if assets:
        asset_count = len(assets)
        production_count = 0
        provisional_count = 0
        incomplete_count = 0
        for asset in assets:
            if not isinstance(asset, dict):
                incomplete_count += 1
                continue
            rq = asset.get("reconstruction_quality") if isinstance(asset.get("reconstruction_quality"), dict) else {}
            completeness_blockers = rq.get("asset_completeness_blockers") if isinstance(rq.get("asset_completeness_blockers"), list) else []
            if completeness_blockers or not rq.get("has_gaussian_asset") or not rq.get("has_mesh_asset") or not rq.get("has_pose"):
                incomplete_count += 1
            if rq.get("commercial_asset_grade") and not completeness_blockers:
                production_count += 1
            elif str(asset.get("source_instance_id", "")).startswith("candidate_slot_"):
                provisional_count += 1
    if asset_count and production_count == 0 and provisional_count == 0:
        for asset in assets:
            rq = asset.get("reconstruction_quality") if isinstance(asset, dict) and isinstance(asset.get("reconstruction_quality"), dict) else {}
            if rq.get("commercial_asset_grade"):
                production_count += 1
            elif str(asset.get("source_instance_id", "")).startswith("candidate_slot_") if isinstance(asset, dict) else False:
                provisional_count += 1
    commercial_asset_grade = bool(asset_count > 0 and production_count == asset_count and provisional_count == 0 and incomplete_count == 0)
    return {
        "movable_asset_count": asset_count,
        "production_asset_count": production_count,
        "provisional_asset_count": provisional_count,
        "incomplete_asset_count": incomplete_count,
        "commercial_asset_grade": commercial_asset_grade,
        "required_upgrade": None
        if commercial_asset_grade
        else "replace provisional movable assets with SAM/SAM3D or multi-view object reconstructions and ensure gaussian, mesh, and pose assets are present",
    }


def source_fidelity_score(value: Any, *, target: float) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    mae = max(float(value), 0.0)
    target = max(float(target), 1e-6)
    if mae <= target:
        return max(0.0, min(1.0, 0.75 + 0.25 * (1.0 - mae / target)))
    return max(0.0, min(0.75, 0.75 * (1.0 - (mae - target) / max(target * 0.75, 1e-6))))


def strict_gate(gate_id: str, passed: bool, requirement: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": gate_id,
        "passed": bool(passed),
        "requirement": requirement,
        "evidence": evidence,
    }


def strict_visual_next_actions(blockers: list[str]) -> list[str]:
    actions = {
        "engineering_import_readiness": "Fix core package readiness before evaluating visual fidelity.",
        "commercial_acceptance_manifest": "Regenerate or attach artifacts/quality/commercial_acceptance.json and resolve every acceptance blocker before final handoff.",
        "render_nonblank_and_balanced": "Tune camera framing, scene completion, material colors, and render exposure until all verifier cameras are balanced.",
        "source_view_fidelity": "Improve texture projection, source-camera alignment, and mesh/3DGS registration to lower source-view MAE.",
        "observed_room_floor": "Capture more floor and room-corner views or reconstruct a real floor plane instead of relying on inferred floor completion.",
        "texture_mesh_coverage": "Increase projected/baked face coverage and inspect untextured regions before final export.",
        "production_movable_assets": "Replace provisional movable assets with SAM/SAM3D or multi-view object reconstructions before commercial export.",
        "physics_task_readiness": "Add task-ready collision proxies, inertial mass, free joints, and a stable MuJoCo simulation step check for every movable asset.",
        "robot_task_readiness": "Add robot config, dynamic task targets, standard observation cameras, and render-ready task views before task-level acceptance.",
        "task_execution_probe": "Add finite approach waypoints and manipulation-scale collision proxies for every dynamic task target.",
        "task_affordance_targets": "Assign dynamic task targets finite poses and actionable graspable or pushable contact roles before commercial export.",
        "capture_space_acceptance": "Recapture with a continuous room-scale camera path, stronger lateral parallax, and no discontinuous jumps before claiming space-level commercial reconstruction.",
        "metric_scale_plausibility": "Calibrate metric scale from COLMAP poses, known robot/object dimensions, or manual room measurements before claiming robot-ready commercial reconstruction.",
        "navigation_acceptance": "Generate and validate an observed walkable floor/nav surface before claiming robot-navigation-ready commercial reconstruction.",
        "material_readiness": "Export texture-backed mesh/MJCF materials and ensure every referenced texture file is included in the simulator package.",
    }
    return [actions[item] for item in blockers if item in actions]


def fit_image(image: Any, size: tuple[int, int]) -> Any:
    fitted = image.copy()
    fitted.thumbnail(size)
    return fitted


def select_visual_source_image(root: Path, *, reference_image: Path) -> tuple[Path, dict[str, Any]]:
    candidates = visual_source_candidates(root)
    if not candidates:
        return require_file(root / "artifacts" / "background_image.jpg", "source/background image"), {
            "strategy": "fallback_background_image",
            "candidate_count": 0,
        }
    explicit = [item for item in candidates if item["priority"] <= 1]
    if explicit:
        selected = require_file(explicit[0]["path"], "source/background image")
        return selected, {
            "strategy": "explicit_camera_artifact",
            "selected": selected.as_posix(),
            "candidate_count": len(candidates),
            "selected_source": explicit[0]["source"],
        }
    scored: list[dict[str, Any]] = []
    for item in candidates[:512]:
        path = item["path"]
        try:
            score = image_resized_mae(path, reference_image, size=(160, 120))
        except Exception:
            continue
        scored.append({**item, "resized_mae_to_preview": round(float(score), 6)})
    if not scored:
        selected = require_file(candidates[0]["path"], "source/background image")
        return selected, {
            "strategy": "first_readable_candidate",
            "selected": selected.as_posix(),
            "candidate_count": len(candidates),
        }
    scored.sort(key=lambda item: float(item["resized_mae_to_preview"]))
    selected_item = scored[0]
    selected = require_file(selected_item["path"], "source/background image")
    return selected, {
        "strategy": "best_resized_mae_to_source_gs_preview",
        "selected": selected.as_posix(),
        "candidate_count": len(candidates),
        "scored_candidate_count": len(scored),
        "selected_source": selected_item["source"],
        "selected_resized_mae_to_preview": selected_item["resized_mae_to_preview"],
        "top_candidates": [
            {
                "path": item["path"].as_posix(),
                "source": item["source"],
                "resized_mae_to_preview": item["resized_mae_to_preview"],
            }
            for item in scored[:5]
        ],
    }


def visual_source_candidates(root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    motrix_import_path = root / "motrix_import.json"
    if motrix_import_path.is_file():
        motrix_import = read_json(motrix_import_path)
        camera_artifacts = motrix_import.get("camera_artifacts")
        if isinstance(camera_artifacts, dict):
            for key in ("background_image", "source_image"):
                value = camera_artifacts.get(key)
                if value:
                    path = root / str(value)
                    if path.is_file():
                        candidates.append({"path": path, "source": f"camera_artifacts.{key}", "priority": 0})
    for pattern in ("background_image.*", "source_image.*"):
        matches = sorted((root / "artifacts").glob(pattern))
        for match in matches:
            candidates.append({"path": match, "source": f"artifacts/{pattern}", "priority": 1})
    manifest_path = root / "package_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        source = manifest.get("source")
        if source:
            source_path = Path(str(source))
            if source_path.is_dir():
                for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                    matches = sorted(source_path.glob(pattern))
                    for match in matches:
                        candidates.append({"path": match, "source": "package_manifest.source", "priority": 2})
    return dedupe_visual_candidates(candidates)


def dedupe_visual_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda value: (int(value["priority"]), value["path"].as_posix())):
        key = item["path"].resolve().as_posix()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def image_resized_mae(a: Path, b: Path, *, size: tuple[int, int]) -> float:
    from PIL import Image
    import numpy as np

    left = Image.open(a).convert("RGB").resize(size)
    right = Image.open(b).convert("RGB").resize(size)
    diff = np.asarray(left, dtype=np.float32) / 255.0 - np.asarray(right, dtype=np.float32) / 255.0
    return float(np.mean(np.abs(diff)))


if __name__ == "__main__":
    raise SystemExit(main())
