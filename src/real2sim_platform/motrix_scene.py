from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from copy import deepcopy
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable


SPATIAL_BUNDLE_SCHEMA = "real2sim.spatial_scene_bundle.v1"
SCENE_MANIFEST_SCHEMA = "real2sim.scene_manifest.v3"
MOTRIX_IMPORT_SCHEMA = "real2sim.motrixsim_scene.v2"
SUPPORTED_MESH_SUFFIXES = {".obj", ".stl", ".ply", ".glb", ".gltf", ".usd", ".usda", ".usdc", ".usdz"}
SUPPORTED_PRIMITIVES = {"box", "sphere", "cylinder", "capsule"}
SCENE_COMPOSITION_SCHEMA = "real2sim.scene_composition.v1"

COMPOSITION_ASSET_RECIPES: dict[str, dict[str, Any]] = {
    "recipe:coffee_table": {
        "id": "recipe:coffee_table",
        "name": "茶几",
        "description": "静态碰撞茶几，适用于桌面操作任务",
        "runtime_name": "coffee_table",
        "category": "task_fixture",
        "default_transform": {"translation": [0.65, 0.0, 0.25], "rotation_euler_degrees": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        "physics": {"dynamic": False, "mass": 8.0, "collision_mode": "visual_and_collision", "dimensions": [0.9, 0.64, 0.34], "friction": [0.8, 0.01, 0.001], "rgba": [0.45, 0.30, 0.18, 1.0]},
        "components": [
            {"name": "top", "primitive": "box", "position": [0.0, 0.0, 0.0], "dimensions": [0.9, 0.64, 0.06], "rgba": [0.45, 0.30, 0.18, 1.0]},
            {"name": "leg_1", "primitive": "box", "position": [0.36, 0.25, -0.14], "dimensions": [0.05, 0.05, 0.28], "rgba": [0.20, 0.13, 0.08, 1.0]},
            {"name": "leg_2", "primitive": "box", "position": [-0.36, 0.25, -0.14], "dimensions": [0.05, 0.05, 0.28], "rgba": [0.20, 0.13, 0.08, 1.0]},
            {"name": "leg_3", "primitive": "box", "position": [0.36, -0.25, -0.14], "dimensions": [0.05, 0.05, 0.28], "rgba": [0.20, 0.13, 0.08, 1.0]},
            {"name": "leg_4", "primitive": "box", "position": [-0.36, -0.25, -0.14], "dimensions": [0.05, 0.05, 0.28], "rgba": [0.20, 0.13, 0.08, 1.0]},
        ],
        "required_for_collection": True,
    },
    "recipe:pick_cube": {
        "id": "recipe:pick_cube",
        "name": "抓取方块",
        "description": "带自由关节和真实接触的动态抓取目标",
        "runtime_name": "pick_cube",
        "category": "task_object",
        "default_transform": {"translation": [0.65, 0.0, 0.33], "rotation_euler_degrees": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        "physics": {"dynamic": True, "mass": 0.05, "collision_mode": "visual_and_collision", "dimensions": [0.06, 0.06, 0.06], "friction": [0.9, 0.01, 0.001], "rgba": [0.9, 0.2, 0.12, 1.0]},
        "components": [{"name": "geom", "primitive": "box", "position": [0.0, 0.0, 0.0], "dimensions": [0.06, 0.06, 0.06], "rgba": [0.9, 0.2, 0.12, 1.0]}],
        "required_for_collection": True,
    },
    "recipe:place_target": {
        "id": "recipe:place_target",
        "name": "放置目标",
        "description": "不参与碰撞的半透明任务目标区域",
        "runtime_name": "place_target",
        "category": "task_marker",
        "default_transform": {"translation": [0.45, 0.18, 0.335], "rotation_euler_degrees": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        "physics": {"dynamic": False, "mass": 0.01, "collision_mode": "visual_only", "dimensions": [0.11, 0.11, 0.008], "friction": [0.0, 0.0, 0.0], "rgba": [0.1, 0.8, 0.6, 0.35]},
        "components": [{"name": "marker", "primitive": "cylinder", "position": [0.0, 0.0, 0.0], "dimensions": [0.11, 0.11, 0.008], "rgba": [0.1, 0.8, 0.6, 0.35], "collision_mode": "visual_only"}],
        "required_for_collection": True,
    },
    "recipe:storage_bin": {
        "id": "recipe:storage_bin",
        "name": "收纳盒",
        "description": "由底板和四面碰撞壁组成的静态容器",
        "runtime_name": "storage_bin",
        "category": "prop",
        "default_transform": {"translation": [0.45, -0.22, 0.37], "rotation_euler_degrees": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        "physics": {"dynamic": False, "mass": 0.6, "collision_mode": "visual_and_collision", "dimensions": [0.28, 0.20, 0.12], "friction": [0.75, 0.01, 0.001], "rgba": [0.72, 0.72, 0.64, 1.0]},
        "components": [
            {"name": "base", "primitive": "box", "position": [0.0, 0.0, -0.05], "dimensions": [0.28, 0.20, 0.02]},
            {"name": "wall_front", "primitive": "box", "position": [0.0, -0.095, 0.0], "dimensions": [0.28, 0.01, 0.12]},
            {"name": "wall_back", "primitive": "box", "position": [0.0, 0.095, 0.0], "dimensions": [0.28, 0.01, 0.12]},
            {"name": "wall_left", "primitive": "box", "position": [-0.135, 0.0, 0.0], "dimensions": [0.01, 0.18, 0.12]},
            {"name": "wall_right", "primitive": "box", "position": [0.135, 0.0, 0.0], "dimensions": [0.01, 0.18, 0.12]},
        ],
        "required_for_collection": False,
    },
}


def detect_spatial_bundle(root: Path) -> dict[str, Any]:
    root = root.resolve()
    files = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
    by_name: dict[str, list[Path]] = {}
    for path in files:
        by_name.setdefault(path.name.lower(), []).append(path)

    collider = _first_named(by_name, "environment_collider.usdz")
    shaded = _first_named(by_name, "scene_shaded.usdz")
    splat = _first_named(by_name, "full_res.spz")
    if collider is None:
        collider = next((path for path in files if path.suffix.lower() == ".usdz" and "collider" in path.stem.lower()), None)
    if shaded is None:
        shaded = next((path for path in files if path.suffix.lower() == ".usdz" and path != collider), None)
    if splat is None:
        splat = next((path for path in files if path.suffix.lower() in {".spz", ".splat", ".ksplat", ".sog"}), None)

    missing = [
        label
        for label, value in (("environment collider USDZ", collider), ("shaded scene USDZ", shaded), ("3DGS SPZ/SPLAT", splat))
        if value is None
    ]
    return {
        "schema": SPATIAL_BUNDLE_SCHEMA,
        "root": str(root),
        "ready": not missing,
        "missing": missing,
        "layers": {
            "collision": _file_record(root, collider),
            "visual_mesh": _file_record(root, shaded),
            "appearance_3dgs": _file_record(root, splat),
        },
        "file_count": len(files),
        "size_bytes": sum(path.stat().st_size for path in files),
    }


def build_spatial_scene_package(
    input_root: Path,
    package_dir: Path,
    *,
    scene_name: str,
    validation_steps: int,
    emit: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    emit = emit or (lambda _message: None)
    detected = detect_spatial_bundle(input_root)
    if not detected["ready"]:
        raise ValueError("MotrixSim spatial bundle is incomplete: " + ", ".join(detected["missing"]))

    package_dir.mkdir(parents=True, exist_ok=False)
    source_dir = package_dir / "source"
    mesh_dir = package_dir / "assets" / "meshes"
    viewer_dir = package_dir / "viewer"
    simulator_dir = package_dir / "simulator_assets"
    reports_dir = package_dir / "reports"
    for path in (source_dir, mesh_dir, viewer_dir, simulator_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    layers = detected["layers"]
    collision_source = Path(layers["collision"]["path"])
    visual_source = Path(layers["visual_mesh"]["path"])
    splat_source = Path(layers["appearance_3dgs"]["path"])
    copied_collision = source_dir / collision_source.name
    copied_visual = source_dir / visual_source.name
    copied_splat = viewer_dir / splat_source.name
    shutil.copy2(collision_source, copied_collision)
    shutil.copy2(visual_source, copied_visual)
    shutil.copy2(splat_source, copied_splat)

    emit("Reading collision USDZ and converting it to a MotrixSim mesh")
    collision_meshes, collision_meta = load_usd_meshes(copied_collision)
    if not collision_meshes:
        raise ValueError("environment collider USDZ contains no mesh prims")
    collision_obj = mesh_dir / "environment_collision.obj"
    collision_stats = export_combined_obj(collision_meshes, collision_obj)

    emit("Reading shaded USDZ and building a browser/MotrixSim mesh fallback")
    visual_meshes, visual_meta = load_usd_meshes(copied_visual)
    if not visual_meshes:
        raise ValueError("shaded scene USDZ contains no mesh prims")
    visual_obj = mesh_dir / "environment_visual.obj"
    visual_stats = export_combined_obj(visual_meshes, visual_obj)
    visual_glb = viewer_dir / "environment_visual.glb"
    glb_stats = export_glb(visual_meshes, visual_glb)

    runtime_scene = simulator_dir / "scene.xml"
    bounds = _combined_bounds(collision_meshes + visual_meshes)
    write_runtime_scene(runtime_scene, scene_name=scene_name, collision_obj=collision_obj, visual_obj=visual_obj)

    emit("Loading the generated MJCF with MuJoCo and MotrixSim")
    validation = validate_runtime_scene(runtime_scene, steps=validation_steps)
    validation_path = reports_dir / "runtime_validation.json"
    _write_json(validation_path, validation)
    if not validation["ok"]:
        failed = [name for name in ("mujoco", "motrixsim") if not validation[name]["ok"]]
        raise RuntimeError("Imported scene failed runtime validation: " + "; ".join(f"{name}: {validation[name].get('error', 'unknown error')}" for name in failed))

    descriptor = {
        "schema": MOTRIX_IMPORT_SCHEMA,
        "name": scene_name,
        "coordinate_frame": {"up_axis": visual_meta["up_axis"], "meters_per_unit": visual_meta["meters_per_unit"]},
        "runtime_scene": str(runtime_scene),
        "layers": {
            "collision": {"source": str(copied_collision), "runtime_mesh": str(collision_obj), "stats": collision_stats},
            "visual_mesh": {"source": str(copied_visual), "runtime_mesh": str(visual_obj), "browser_mesh": str(visual_glb), "stats": visual_stats},
            "appearance_3dgs": {"source": str(copied_splat), "format": copied_splat.suffix.lower().lstrip(".")},
        },
        "bounds_m": bounds,
        "runtime_validation": validation,
        "added_assets": [],
        "created_at": _utc_now(),
    }
    descriptor_path = package_dir / "motrix_import.json"
    _write_json(descriptor_path, descriptor)

    import_report = {
        "schema": "real2sim.motrixsim_scene_import_report.v1",
        "ok": True,
        "scene_name": scene_name,
        "source": detected,
        "usd": {"collision": collision_meta, "visual": visual_meta},
        "conversion": {"collision": collision_stats, "visual": visual_stats, "browser_glb": glb_stats},
        "runtime_validation": validation,
        "bounds_m": bounds,
        "next_actions": ["Add robots or props from the digital asset library", "Save edits as a new scene revision", "Run simulator validation before collection"],
    }
    report_path = reports_dir / "motrixsim_scene_import_report.json"
    _write_json(report_path, import_report)

    manifest = {
        "schema": SCENE_MANIFEST_SCHEMA,
        "name": scene_name,
        "source_type": "motrixsim_spatial_bundle",
        "renderer": "spark",
        "renderer_status": {"selected": "spark", "spark_available": True, "fallback": "environment_visual.glb"},
        "coordinate_frame": descriptor["coordinate_frame"],
        "bounds_m": bounds,
        "layers": {
            "appearance": "3DGS SPZ/SPLAT",
            "visual_mesh": "USDZ shaded mesh",
            "collision": "USDZ environment collider",
            "simulation": "MotrixSim MJCF",
        },
        "assets": [
            _manifest_asset(copied_splat, "scene_appearance", "splat_or_pointcloud", autoload=True),
            _manifest_asset(visual_glb, "visual_fallback", "mesh", autoload=False),
            _manifest_asset(collision_obj, "environment_collision", "mesh", autoload=False),
            _manifest_asset(runtime_scene, "motrixsim_runtime", "simulator", autoload=False),
            _manifest_asset(copied_visual, "source_visual_usdz", "source", autoload=False),
            _manifest_asset(copied_collision, "source_collision_usdz", "source", autoload=False),
            _manifest_asset(validation_path, "runtime_report", "metadata", autoload=False),
        ],
        "simulator_artifacts": {
            "motrixsim": {"path": str(descriptor_path), "runtime_scene": str(runtime_scene), "validated": validation["motrixsim"]["ok"]},
            "mujoco": {"path": str(runtime_scene), "validated": validation["mujoco"]["ok"]},
        },
        "asset_editing": {
            "enabled": True,
            "revisioned": True,
            "supported_primitives": sorted(SUPPORTED_PRIMITIVES),
            "supported_mesh_formats": sorted(SUPPORTED_MESH_SUFFIXES),
        },
        "metrics": {
            "mesh_objects": len(visual_meshes),
            "visual_vertices": visual_stats["vertices"],
            "visual_triangles": visual_stats["triangles"],
            "collision_vertices": collision_stats["vertices"],
            "collision_triangles": collision_stats["triangles"],
            "runtime_steps": validation_steps,
        },
        "created_at": _utc_now(),
    }
    manifest_path = package_dir / "scene_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "package_dir": str(package_dir),
        "manifest_path": str(manifest_path),
        "descriptor_path": str(descriptor_path),
        "runtime_scene": str(runtime_scene),
        "report_path": str(report_path),
        "validation_path": str(validation_path),
        "manifest": manifest,
        "report": import_report,
    }


def compose_scene_asset(
    package_dir: Path,
    *,
    asset_spec: dict[str, Any],
    source_path: Path | None,
    validation_steps: int,
    emit: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    emit = emit or (lambda _message: None)
    manifest_path = package_dir / "scene_manifest.json"
    manifest = _read_json(manifest_path)
    runtime_scene = _runtime_scene_from_manifest(package_dir, manifest)
    if not runtime_scene.is_file():
        raise FileNotFoundError(f"MotrixSim runtime scene not found: {runtime_scene}")

    name = str(asset_spec.get("name") or (source_path.stem if source_path else asset_spec.get("primitive")) or "digital_asset").strip()
    asset_key = _safe_name(f"{name}-{hashlib.sha1(json.dumps(asset_spec, sort_keys=True, default=str).encode()).hexdigest()[:8]}")
    transform = normalize_transform(asset_spec.get("transform") or {})
    physics = normalize_physics(asset_spec.get("physics") or {})
    primitive = str(asset_spec.get("primitive") or "").lower().strip()
    if primitive and primitive not in SUPPORTED_PRIMITIVES:
        raise ValueError(f"unsupported primitive: {primitive}")
    if not primitive and source_path is None:
        raise ValueError("scene asset requires a primitive or a registered source file")

    tree = ET.parse(runtime_scene)
    root = tree.getroot()
    if root.tag != "mujoco":
        raise ValueError("scene runtime file is not an MJCF <mujoco> document")
    asset_node = root.find("asset")
    if asset_node is None:
        asset_node = ET.Element("asset")
        root.insert(2, asset_node)
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    compiler = root.find("compiler")
    compiler_angle = str(compiler.attrib.get("angle", "degree") if compiler is not None else "degree").lower()
    euler = transform["rotation_euler_degrees"]
    if compiler_angle.startswith("rad"):
        euler = [math.radians(value) for value in euler]
    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": asset_key,
            "pos": _numbers(transform["translation"]),
            "euler": _numbers(euler),
        },
    )
    if physics["dynamic"]:
        ET.SubElement(body, "freejoint", {"name": f"{asset_key}_joint"})

    copied_asset: Path | None = None
    viewer_asset: Path | None = None
    geom_attributes = {
        "name": f"{asset_key}_geom",
        "friction": _numbers(physics["friction"]),
        "contype": "0" if physics["collision_mode"] == "visual_only" else "1",
        "conaffinity": "0" if physics["collision_mode"] == "visual_only" else "1",
        "group": "2" if physics["collision_mode"] == "visual_only" else "3",
        "rgba": _numbers(physics["rgba"]),
    }
    if physics["dynamic"]:
        geom_attributes["mass"] = f"{physics['mass']:.9g}"

    if primitive:
        scaled_dimensions = [
            dimension * scale
            for dimension, scale in zip(physics["dimensions"], transform["scale"], strict=True)
        ]
        geom_attributes.update(_primitive_geom(primitive, scaled_dimensions))
    else:
        emit(f"Converting {source_path.name} into a MotrixSim mesh")
        added_dir = package_dir / "assets" / "added" / asset_key
        added_dir.mkdir(parents=True, exist_ok=True)
        copied_asset = added_dir / source_path.name
        shutil.copy2(source_path, copied_asset)
        runtime_mesh = added_dir / f"{asset_key}.obj"
        convert_asset_to_obj(copied_asset, runtime_mesh)
        viewer_asset = copied_asset if copied_asset.suffix.lower() in {".glb", ".gltf", ".obj"} else runtime_mesh
        compiler = root.find("compiler")
        meshdir = Path((compiler.attrib.get("meshdir") if compiler is not None else "") or ".")
        resolved_meshdir = (runtime_scene.parent / meshdir).resolve()
        relative_mesh = os.path.relpath(runtime_mesh, resolved_meshdir).replace(os.sep, "/")
        mesh_name = f"{asset_key}_mesh"
        ET.SubElement(asset_node, "mesh", {"name": mesh_name, "file": relative_mesh, "scale": _numbers(transform["scale"])} )
        geom_attributes.update({"type": "mesh", "mesh": mesh_name})
    ET.SubElement(body, "geom", geom_attributes)
    _indent_xml(root)
    tree.write(runtime_scene, encoding="utf-8", xml_declaration=True)

    emit("Validating the revised scene with MuJoCo and MotrixSim")
    validation = validate_runtime_scene(runtime_scene, steps=validation_steps)
    validation_path = package_dir / "reports" / f"asset_{asset_key}_validation.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(validation_path, validation)
    if not validation["ok"]:
        failed = [name for name in ("mujoco", "motrixsim") if not validation[name]["ok"]]
        raise RuntimeError("Added asset failed runtime validation: " + "; ".join(f"{name}: {validation[name].get('error', 'unknown error')}" for name in failed))

    entry = {
        "name": name,
        "path": str(viewer_asset or ""),
        "kind": "primitive" if primitive else "mesh",
        "role": "added_asset",
        "format": primitive or (viewer_asset.suffix.lower().lstrip(".") if viewer_asset else ""),
        "autoload": True,
        "viewer_hint": "overlay",
        "viewer_primitive": primitive or None,
        "transform": transform,
        "physics": physics,
        "runtime": {"body": asset_key, "scene": str(runtime_scene), "validated": validation["motrixsim"]["ok"]},
        "added_at": _utc_now(),
    }
    if viewer_asset:
        entry["size_bytes"] = viewer_asset.stat().st_size

    assets = [item for item in manifest.get("assets", []) if not (isinstance(item, dict) and item.get("runtime", {}).get("body") == asset_key)]
    manifest["assets"] = [entry] + assets
    manifest["updated_at"] = _utc_now()
    manifest["asset_editing"] = {
        **(manifest.get("asset_editing") or {}),
        "enabled": True,
        "revisioned": True,
        "last_added_asset": name,
    }
    manifest.setdefault("simulator_artifacts", {}).setdefault("motrixsim", {})["validated"] = validation["motrixsim"]["ok"]
    manifest.setdefault("simulator_artifacts", {}).setdefault("mujoco", {})["validated"] = validation["mujoco"]["ok"]
    _write_json(manifest_path, manifest)

    descriptor_path = package_dir / "motrix_import.json"
    descriptor = _read_json(descriptor_path) if descriptor_path.is_file() else {"schema": MOTRIX_IMPORT_SCHEMA, "runtime_scene": str(runtime_scene), "added_assets": []}
    descriptor.setdefault("added_assets", []).append(entry)
    descriptor["runtime_validation"] = validation
    descriptor["updated_at"] = _utc_now()
    _write_json(descriptor_path, descriptor)
    return {
        "manifest_path": str(manifest_path),
        "runtime_scene": str(runtime_scene),
        "validation_path": str(validation_path),
        "validation": validation,
        "asset": entry,
    }


def composition_asset_catalog() -> list[dict[str, Any]]:
    return [deepcopy(COMPOSITION_ASSET_RECIPES[key]) for key in sorted(COMPOSITION_ASSET_RECIPES)]


def build_scene_composition(
    package_dir: Path,
    *,
    robot_source: Path,
    robot_spec: dict[str, Any],
    asset_specs: list[dict[str, Any]],
    source_paths: dict[str, Path] | None,
    environment_collision_mode: str = "isolated_task_workspace",
    validation_steps: int,
    emit: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    emit = emit or (lambda _message: None)
    source_paths = source_paths or {}
    manifest_path = package_dir / "scene_manifest.json"
    manifest = _read_json(manifest_path)
    runtime_scene = _runtime_scene_from_manifest(package_dir, manifest)
    if not runtime_scene.is_file():
        raise FileNotFoundError(f"MotrixSim runtime scene not found: {runtime_scene}")
    if not robot_source.is_file():
        raise FileNotFoundError(f"RM65 source MJCF not found: {robot_source}")

    tree = ET.parse(runtime_scene)
    root = tree.getroot()
    if root.tag != "mujoco":
        raise ValueError("scene runtime file is not an MJCF <mujoco> document")
    environment = _prepare_composition_environment(root, manifest, mode=environment_collision_mode)
    compiler = _ensure_radian_compiler(root)
    emit("Merging the real RM65 articulated model into the MotrixSim scene")
    robot_entry = _merge_rm65_robot(
        root,
        runtime_scene=runtime_scene,
        compiler=compiler,
        robot_source=robot_source,
        robot_spec=robot_spec,
    )

    entries: list[dict[str, Any]] = []
    seen_instances: set[str] = set()
    seen_runtime_names: set[str] = set()
    for index, spec in enumerate(asset_specs):
        instance_id = _safe_name(str(spec.get("instance_id") or f"asset_{index + 1}"))
        if instance_id in seen_instances:
            raise ValueError(f"duplicate composition asset instance_id: {instance_id}")
        seen_instances.add(instance_id)
        catalog_id = str(spec.get("catalog_id") or "").strip()
        if catalog_id in COMPOSITION_ASSET_RECIPES or catalog_id.startswith("primitive:"):
            entry = _append_composition_primitive(root, spec=spec, instance_id=instance_id, seen_runtime_names=seen_runtime_names)
        else:
            source = source_paths.get(instance_id)
            if source is None:
                raise ValueError(f"composition asset is not registered or has no source file: {catalog_id or instance_id}")
            entry = _append_composition_mesh(
                root,
                package_dir=package_dir,
                runtime_scene=runtime_scene,
                compiler=compiler,
                spec=spec,
                instance_id=instance_id,
                source_path=source,
                seen_runtime_names=seen_runtime_names,
                emit=emit,
            )
        entries.append(entry)

    runtime_names = {str(entry.get("runtime", {}).get("body") or "") for entry in entries}
    collection_ready = {"coffee_table", "pick_cube", "place_target"}.issubset(runtime_names)
    _ensure_overview_camera(root, target="pick_cube" if "pick_cube" in runtime_names else "base_link")
    _indent_xml(root)
    tree.write(runtime_scene, encoding="utf-8", xml_declaration=True)
    _write_home_keyframe(runtime_scene)

    robot_glb = package_dir / "viewer" / "robots" / "rm65.glb"
    emit("Exporting the articulated RM65 visual geometry for the browser editor")
    robot_preview = export_mujoco_body_glb(runtime_scene, robot_glb, root_body="base_link")
    robot_entry.update(
        {
            "path": str(robot_glb),
            "size_bytes": robot_glb.stat().st_size,
            "sha256": _sha256(robot_glb),
            "preview": robot_preview,
        }
    )

    emit("Validating the composed scene in MuJoCo and MotrixSim")
    validation = validate_runtime_scene(runtime_scene, steps=validation_steps)
    reports_dir = package_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    validation_path = reports_dir / "scene_composition_validation.json"
    _write_json(validation_path, validation)
    if not validation["ok"]:
        failed = [name for name in ("mujoco", "motrixsim") if not validation[name]["ok"]]
        raise RuntimeError("Composed scene failed runtime validation: " + "; ".join(f"{name}: {validation[name].get('error', 'unknown error')}" for name in failed))

    for entry in [robot_entry, *entries]:
        entry.setdefault("runtime", {})["validated"] = True
    composition = {
        "schema": SCENE_COMPOSITION_SCHEMA,
        "simulator": "motrixsim",
        "environment_collision_mode": environment_collision_mode,
        "environment": environment,
        "robot": robot_entry,
        "assets": entries,
        "controller": "rm65_pick_place_6d_ik" if collection_ready else None,
        "collection_ready": collection_ready,
        "required_collection_objects": ["coffee_table", "pick_cube", "place_target"],
        "validation_steps": validation_steps,
        "updated_at": _utc_now(),
    }
    report_path = reports_dir / "scene_composition_report.json"
    _write_json(
        report_path,
        {
            "schema": "real2sim.scene_composition_report.v1",
            "ok": True,
            "runtime_scene": str(runtime_scene),
            "robot_source": str(robot_source),
            "robot_body_count": robot_preview["body_count"],
            "robot_mesh_count": robot_preview["mesh_count"],
            "asset_count": len(entries),
            "environment": environment,
            "collection_ready": collection_ready,
            "runtime_validation": validation,
        },
    )
    existing_assets = [
        item
        for item in manifest.get("assets", [])
        if not (isinstance(item, dict) and item.get("role") in {"robot", "added_asset"})
    ]
    manifest["assets"] = [robot_entry, *entries, *existing_assets]
    manifest["composition"] = composition
    manifest["updated_at"] = _utc_now()
    manifest.setdefault("asset_editing", {}).update({"enabled": True, "revisioned": True, "composition": True})
    manifest.setdefault("simulator_artifacts", {}).setdefault("motrixsim", {}).update({"runtime_scene": str(runtime_scene), "validated": True})
    manifest.setdefault("simulator_artifacts", {}).setdefault("mujoco", {}).update({"mjcf": str(runtime_scene), "validated": True})
    _write_json(manifest_path, manifest)

    descriptor_path = package_dir / "motrix_import.json"
    descriptor = _read_json(descriptor_path) if descriptor_path.is_file() else {"schema": MOTRIX_IMPORT_SCHEMA}
    descriptor.update({"runtime_scene": str(runtime_scene), "composition": composition, "runtime_validation": validation, "updated_at": _utc_now()})
    _write_json(descriptor_path, descriptor)
    return {
        "manifest_path": str(manifest_path),
        "runtime_scene": str(runtime_scene),
        "validation_path": str(validation_path),
        "report_path": str(report_path),
        "validation": validation,
        "composition": composition,
        "collection_ready": collection_ready,
    }


def _prepare_composition_environment(root: ET.Element, manifest: dict[str, Any], *, mode: str) -> dict[str, Any]:
    if mode not in {"isolated_task_workspace", "reconstructed_collision"}:
        raise ValueError("environment collision mode must be isolated_task_workspace or reconstructed_collision")
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    legacy_bodies = {
        str((item.get("runtime") or {}).get("body") or "")
        for item in manifest.get("assets", [])
        if isinstance(item, dict) and item.get("role") == "added_asset"
    }
    legacy_bodies.discard("")
    removed_bodies = []
    for body in list(worldbody):
        if body.tag == "body" and str(body.get("name") or "") in legacy_bodies:
            removed_bodies.append(str(body.get("name")))
            worldbody.remove(body)

    disabled_geoms = 0
    if mode == "isolated_task_workspace":
        for geom in root.findall(".//geom"):
            if str(geom.get("contype") or "1") == "0" and str(geom.get("conaffinity") or "1") == "0":
                continue
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            disabled_geoms += 1
        existing_ground = worldbody.find("./geom[@name='composition_ground']")
        if existing_ground is not None:
            worldbody.remove(existing_ground)
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "composition_ground",
                "type": "plane",
                "size": "12 12 0.1",
                "pos": "0 0 0",
                "contype": "1",
                "conaffinity": "1",
                "group": "0",
                "friction": "0.9 0.01 0.001",
                "rgba": "0.18 0.2 0.22 1",
            },
        )
    return {
        "mode": mode,
        "removed_legacy_asset_bodies": removed_bodies,
        "disabled_reconstructed_collision_geoms": disabled_geoms,
        "task_ground": mode == "isolated_task_workspace",
    }


def export_mujoco_body_glb(scene_path: Path, output: Path, *, root_body: str, relative_to_root: bool = False) -> dict[str, Any]:
    try:
        import mujoco
        import numpy as np
        import trimesh
    except ImportError as exc:
        raise RuntimeError("RM65 GLB export requires mujoco, numpy and trimesh") from exc

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
    if root_id < 0:
        raise ValueError(f"robot root body is missing: {root_body}")
    root_position = np.asarray(data.xpos[root_id], dtype=np.float64)
    root_rotation = np.asarray(data.xmat[root_id], dtype=np.float64).reshape(3, 3)

    def is_descendant(body_id: int) -> bool:
        current = body_id
        while current > 0:
            if current == root_id:
                return True
            current = int(model.body_parentid[current])
        return current == root_id

    scene = trimesh.Scene()
    mesh_count = 0
    body_ids: set[int] = set()
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        mesh_id = int(model.geom_dataid[geom_id])
        if not is_descendant(body_id) or int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH) or mesh_id < 0:
            continue
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        face_start = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(model.mesh_vert[vertex_start : vertex_start + vertex_count], dtype=np.float64).copy()
        faces = np.asarray(model.mesh_face[face_start : face_start + face_count], dtype=np.int64).copy()
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        vertices = vertices @ rotation.T + np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        if relative_to_root:
            vertices = (vertices - root_position) @ root_rotation
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
        rgba = np.clip(np.asarray(model.geom_rgba[geom_id]) * 255, 0, 255).astype(np.uint8)
        mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"rm65_geom_{geom_id}"
        scene.add_geometry(mesh, geom_name=name, node_name=name)
        mesh_count += 1
        body_ids.add(body_id)
    if not mesh_count:
        raise ValueError("RM65 model contains no exportable visual mesh geometry")
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output, file_type="glb")
    return {"path": str(output), "mesh_count": mesh_count, "body_count": len(body_ids), "size_bytes": output.stat().st_size}


def _ensure_radian_compiler(root: ET.Element) -> ET.Element:
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler", {"angle": "radian", "autolimits": "true"})
        root.insert(0, compiler)
        return compiler
    if not str(compiler.attrib.get("angle", "degree")).lower().startswith("rad"):
        for element in root.iter():
            if "euler" not in element.attrib:
                continue
            values = [float(value) for value in element.attrib["euler"].split()]
            element.attrib["euler"] = _numbers([math.radians(value) for value in values])
    compiler.attrib["angle"] = "radian"
    compiler.attrib.setdefault("autolimits", "true")
    return compiler


def _merge_rm65_robot(
    root: ET.Element,
    *,
    runtime_scene: Path,
    compiler: ET.Element,
    robot_source: Path,
    robot_spec: dict[str, Any],
) -> dict[str, Any]:
    if str(robot_spec.get("robot_id") or "") != "rm65":
        raise ValueError("only the registered RM65 robot can be composed in this release")
    if root.find("./worldbody/body[@name='base_link']") is not None:
        raise ValueError("the scene already contains an RM65 base_link body")
    transform = normalize_transform(robot_spec.get("transform") or {})
    if any(abs(value - 1.0) > 1e-9 for value in transform["scale"]):
        raise ValueError("articulated RM65 scale must remain [1, 1, 1] to preserve joint and inertia calibration")

    robot_root = ET.parse(robot_source).getroot()
    robot_compiler = robot_root.find("compiler")
    asset_dir = (robot_source.parent / str(robot_compiler.attrib.get("assetdir", "assets") if robot_compiler is not None else "assets")).resolve()
    if not asset_dir.is_dir():
        raise FileNotFoundError(f"RM65 asset directory is missing: {asset_dir}")
    mesh_dir = (runtime_scene.parent / str(compiler.attrib.get("meshdir") or ".")).resolve()
    robot_mesh_dir = mesh_dir / "robots" / "rm65"
    if robot_mesh_dir.exists():
        shutil.rmtree(robot_mesh_dir)
    shutil.copytree(asset_dir, robot_mesh_dir)
    license_source = robot_source.parent / "LICENSE"
    if license_source.is_file():
        shutil.copy2(license_source, robot_mesh_dir / "LICENSE")

    robot_default = robot_root.find("default")
    if robot_default is not None:
        target_default = root.find("default")
        if target_default is None:
            target_default = ET.Element("default")
            _insert_before(root, target_default, {"asset", "worldbody"})
        for child in robot_default:
            target_default.append(deepcopy(child))

    target_asset = root.find("asset")
    if target_asset is None:
        target_asset = ET.Element("asset")
        _insert_before(root, target_asset, {"worldbody"})
    source_asset = robot_root.find("asset")
    if source_asset is not None:
        for child in source_asset:
            copied = deepcopy(child)
            if copied.get("file"):
                copied.set("file", f"robots/rm65/{copied.get('file')}")
            target_asset.append(copied)

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    robot_body = robot_root.find("./worldbody/body[@name='base_link']")
    if robot_body is None:
        raise ValueError("RM65 source MJCF does not contain base_link")
    copied_body = deepcopy(robot_body)
    copied_body.set("pos", _numbers(transform["translation"]))
    copied_body.set("euler", _numbers([math.radians(value) for value in transform["rotation_euler_degrees"]]))
    worldbody.append(copied_body)

    for section_name in ("equality", "actuator", "contact"):
        source_section = robot_root.find(section_name)
        if source_section is None:
            continue
        target_section = root.find(section_name)
        if target_section is None:
            target_section = ET.SubElement(root, section_name)
        for child in source_section:
            target_section.append(deepcopy(child))
    return {
        "name": "RM65 双指夹爪",
        "kind": "robot",
        "role": "robot",
        "format": "glb",
        "autoload": True,
        "viewer_hint": "overlay",
        "transform": {"translation": [0.0, 0.0, 0.0], "rotation_euler_degrees": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        "source_transform": transform,
        "robot_id": "rm65",
        "controller": "rm65_pick_place_6d_ik",
        "runtime": {"body": "base_link", "scene": str(runtime_scene), "validated": False},
        "source": {"path": str(robot_source), "repository": "Motphys/motrixsim-docs"},
        "added_at": _utc_now(),
    }


def _append_composition_primitive(
    root: ET.Element,
    *,
    spec: dict[str, Any],
    instance_id: str,
    seen_runtime_names: set[str],
) -> dict[str, Any]:
    catalog_id = str(spec.get("catalog_id") or "")
    if catalog_id.startswith("primitive:"):
        primitive = catalog_id.split(":", 1)[1]
        if primitive not in SUPPORTED_PRIMITIVES:
            raise ValueError(f"unsupported composition primitive: {primitive}")
        recipe = {
            "id": catalog_id,
            "name": str(spec.get("name") or primitive),
            "runtime_name": instance_id,
            "default_transform": {"translation": [0.0, 0.0, 0.1], "rotation_euler_degrees": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
            "physics": {"dynamic": False, "mass": 0.25, "collision_mode": "visual_and_collision", "dimensions": [0.1, 0.1, 0.1], "friction": [0.8, 0.01, 0.001], "rgba": [0.13, 0.76, 0.67, 1.0]},
            "components": [{"name": "geom", "primitive": primitive, "position": [0.0, 0.0, 0.0], "dimensions": [0.1, 0.1, 0.1]}],
        }
    else:
        recipe = COMPOSITION_ASSET_RECIPES[catalog_id]
    runtime_name = str(recipe["runtime_name"])
    if runtime_name in seen_runtime_names or root.find(f"./worldbody/body[@name='{runtime_name}']") is not None:
        raise ValueError(f"composition already contains runtime object: {runtime_name}")
    seen_runtime_names.add(runtime_name)
    transform = normalize_transform(spec.get("transform") or recipe.get("default_transform") or {})
    physics = normalize_physics({**(recipe.get("physics") or {}), **(spec.get("physics") or {})})
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    body_attributes = {
        "name": runtime_name,
        "pos": _numbers(transform["translation"]),
        "euler": _numbers([math.radians(value) for value in transform["rotation_euler_degrees"]]),
    }
    if runtime_name == "place_target":
        body_attributes["mocap"] = "true"
    body = ET.SubElement(worldbody, "body", body_attributes)
    if physics["dynamic"]:
        joint_name = "pick_cube_free" if runtime_name == "pick_cube" else f"{runtime_name}_free"
        ET.SubElement(body, "freejoint", {"name": joint_name})

    viewer_components: list[dict[str, Any]] = []
    collidable = [component for component in recipe["components"] if component.get("collision_mode", physics["collision_mode"]) != "visual_only"]
    mass_per_geom = physics["mass"] / max(1, len(collidable))
    for component in recipe["components"]:
        primitive = str(component["primitive"])
        dimensions = [float(value) for value in component.get("dimensions") or physics["dimensions"]]
        scaled_dimensions = [value * transform["scale"][index] for index, value in enumerate(dimensions)]
        local_position = [float(value) for value in component.get("position") or [0.0, 0.0, 0.0]]
        scaled_position = [value * transform["scale"][index] for index, value in enumerate(local_position)]
        collision_mode = str(component.get("collision_mode") or physics["collision_mode"])
        rgba = [float(value) for value in component.get("rgba") or physics["rgba"]]
        component_name = str(component.get("name") or "geom")
        if runtime_name == "pick_cube":
            geom_name = "pick_cube_geom"
        elif runtime_name == "place_target":
            geom_name = "place_target_marker"
        elif runtime_name == "coffee_table" and component_name == "top":
            geom_name = "coffee_table_top"
        else:
            geom_name = f"{runtime_name}_{_safe_name(component_name)}"
        attributes = {
            "name": geom_name,
            "pos": _numbers(scaled_position),
            "friction": _numbers(physics["friction"]),
            "contype": "0" if collision_mode == "visual_only" else "1",
            "conaffinity": "0" if collision_mode == "visual_only" else "1",
            "group": "2" if collision_mode == "visual_only" else "0",
            "rgba": _numbers(rgba),
            **_primitive_geom(primitive, scaled_dimensions),
        }
        if physics["dynamic"] and collision_mode != "visual_only":
            attributes["mass"] = f"{mass_per_geom:.9g}"
        ET.SubElement(body, "geom", attributes)
        viewer_components.append(
            {
                "name": component_name,
                "viewer_primitive": primitive,
                "translation": local_position,
                "rotation_euler_degrees": component.get("rotation_euler_degrees") or [0.0, 0.0, 0.0],
                "dimensions": dimensions,
                "rgba": rgba,
            }
        )
    return {
        "name": str(spec.get("name") or recipe["name"]),
        "path": "",
        "kind": "primitive" if len(viewer_components) == 1 else "compound_primitive",
        "role": "added_asset",
        "format": str(viewer_components[0]["viewer_primitive"]) if len(viewer_components) == 1 else "compound",
        "autoload": True,
        "viewer_hint": "overlay",
        "viewer_primitive": viewer_components[0]["viewer_primitive"] if len(viewer_components) == 1 else None,
        "viewer_components": viewer_components,
        "catalog_id": catalog_id,
        "instance_id": instance_id,
        "transform": transform,
        "physics": physics,
        "runtime": {"body": runtime_name, "validated": False},
        "added_at": _utc_now(),
    }


def _append_composition_mesh(
    root: ET.Element,
    *,
    package_dir: Path,
    runtime_scene: Path,
    compiler: ET.Element,
    spec: dict[str, Any],
    instance_id: str,
    source_path: Path,
    seen_runtime_names: set[str],
    emit: Callable[[str], None],
) -> dict[str, Any]:
    runtime_name = _safe_name(f"asset_{instance_id}")
    if runtime_name in seen_runtime_names or root.find(f"./worldbody/body[@name='{runtime_name}']") is not None:
        raise ValueError(f"composition already contains runtime object: {runtime_name}")
    seen_runtime_names.add(runtime_name)
    transform = normalize_transform(spec.get("transform") or {})
    physics = normalize_physics(spec.get("physics") or {})
    added_dir = package_dir / "assets" / "added" / instance_id
    added_dir.mkdir(parents=True, exist_ok=True)
    copied_asset = added_dir / source_path.name
    shutil.copy2(source_path, copied_asset)
    runtime_mesh = added_dir / f"{runtime_name}.obj"
    emit(f"Converting {source_path.name} into a MotrixSim composition mesh")
    convert_asset_to_obj(copied_asset, runtime_mesh)
    viewer_asset = copied_asset if copied_asset.suffix.lower() in {".glb", ".gltf", ".obj"} else runtime_mesh
    mesh_dir = (runtime_scene.parent / str(compiler.attrib.get("meshdir") or ".")).resolve()
    relative_mesh = os.path.relpath(runtime_mesh, mesh_dir).replace(os.sep, "/")
    asset_node = root.find("asset")
    if asset_node is None:
        asset_node = ET.Element("asset")
        _insert_before(root, asset_node, {"worldbody"})
    mesh_name = f"{runtime_name}_mesh"
    ET.SubElement(asset_node, "mesh", {"name": mesh_name, "file": relative_mesh, "scale": _numbers(transform["scale"])})
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(
        worldbody,
        "body",
        {"name": runtime_name, "pos": _numbers(transform["translation"]), "euler": _numbers([math.radians(value) for value in transform["rotation_euler_degrees"]])},
    )
    if physics["dynamic"]:
        ET.SubElement(body, "freejoint", {"name": f"{runtime_name}_free"})
    attributes = {
        "name": f"{runtime_name}_geom",
        "type": "mesh",
        "mesh": mesh_name,
        "friction": _numbers(physics["friction"]),
        "contype": "0" if physics["collision_mode"] == "visual_only" else "1",
        "conaffinity": "0" if physics["collision_mode"] == "visual_only" else "1",
        "group": "2" if physics["collision_mode"] == "visual_only" else "0",
        "rgba": _numbers(physics["rgba"]),
    }
    if physics["dynamic"]:
        attributes["mass"] = f"{physics['mass']:.9g}"
    ET.SubElement(body, "geom", attributes)
    return {
        "name": str(spec.get("name") or source_path.stem),
        "path": str(viewer_asset),
        "kind": "mesh",
        "role": "added_asset",
        "format": viewer_asset.suffix.lower().lstrip("."),
        "size_bytes": viewer_asset.stat().st_size,
        "autoload": True,
        "viewer_hint": "overlay",
        "catalog_id": str(spec.get("catalog_id") or ""),
        "instance_id": instance_id,
        "transform": transform,
        "physics": physics,
        "runtime": {"body": runtime_name, "validated": False},
        "added_at": _utc_now(),
    }


def _ensure_overview_camera(root: ET.Element, *, target: str) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    existing = worldbody.find("./camera[@name='overview']")
    if existing is None:
        ET.SubElement(worldbody, "camera", {"name": "overview", "mode": "targetbody", "target": target, "pos": "1.65 -1.25 1.15", "fovy": "50"})


def _write_home_keyframe(runtime_scene: Path) -> None:
    try:
        import mujoco
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("composition keyframe generation requires mujoco and numpy") from exc
    tree = ET.parse(runtime_scene)
    root = tree.getroot()
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    for child in list(keyframe):
        if child.tag == "key" and child.get("name") == "home":
            keyframe.remove(child)
    home = ET.Element("key", {"name": "home"})
    keyframe.insert(0, home)
    _indent_xml(root)
    tree.write(runtime_scene, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(runtime_scene))
    qpos = np.asarray(model.qpos0, dtype=float).copy()
    arm_home = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079]
    for index, value in enumerate(arm_home, start=1):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{index}")
        if joint_id >= 0:
            qpos[int(model.jnt_qposadr[joint_id])] = value
    gripper_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_Left_1_Joint")
    if gripper_joint >= 0:
        qpos[int(model.jnt_qposadr[gripper_joint])] = -0.7853
    controls = np.zeros(model.nu, dtype=float)
    controls[: min(6, model.nu)] = arm_home[: min(6, model.nu)]
    if model.nu >= 7:
        controls[6] = -0.7853
    home.set("qpos", _numbers(qpos.tolist()))
    if model.nu:
        home.set("ctrl", _numbers(controls.tolist()))
    _indent_xml(root)
    tree.write(runtime_scene, encoding="utf-8", xml_declaration=True)
    mujoco.MjModel.from_xml_path(str(runtime_scene))


def _insert_before(root: ET.Element, child: ET.Element, before_tags: set[str]) -> None:
    index = next((position for position, existing in enumerate(list(root)) if existing.tag in before_tags), len(root))
    root.insert(index, child)


def load_usd_meshes(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import numpy as np
        from pxr import Gf, Usd, UsdGeom
    except ImportError as exc:
        raise RuntimeError("USD conversion requires usd-core and numpy") from exc

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"unable to open USD stage: {path}")
    up_axis = str(stage.GetMetadata("upAxis") or "Z")
    meters_per_unit = float(stage.GetMetadata("metersPerUnit") or 1.0)
    cache = UsdGeom.XformCache()
    meshes: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        raw_points = mesh.GetPointsAttr().Get() or []
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
        if not raw_points or not len(counts) or not len(indices):
            continue
        points = np.asarray(raw_points, dtype=np.float64)
        matrix = cache.GetLocalToWorldTransform(prim)
        transformed = np.asarray([matrix.Transform(Gf.Vec3d(*point)) for point in points], dtype=np.float64)
        transformed *= meters_per_unit
        faces = triangulate_faces(counts, indices)
        meshes.append({"name": _safe_name(prim.GetName()), "vertices": transformed, "faces": faces})
    return meshes, {
        "path": str(path),
        "up_axis": up_axis,
        "meters_per_unit": meters_per_unit,
        "mesh_count": len(meshes),
        "vertices": sum(len(item["vertices"]) for item in meshes),
        "triangles": sum(len(item["faces"]) for item in meshes),
    }


def triangulate_faces(counts: Any, indices: Any) -> Any:
    import numpy as np

    if len(counts) and bool(np.all(counts == 3)):
        return np.asarray(indices, dtype=np.int64).reshape((-1, 3))
    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    for count in counts:
        polygon = indices[cursor : cursor + int(count)]
        cursor += int(count)
        if len(polygon) < 3:
            continue
        triangles.extend((int(polygon[0]), int(polygon[index]), int(polygon[index + 1])) for index in range(1, len(polygon) - 1))
    return np.asarray(triangles, dtype=np.int64)


def export_combined_obj(meshes: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    import numpy as np

    output.parent.mkdir(parents=True, exist_ok=True)
    vertices: list[Any] = []
    faces: list[Any] = []
    offset = 0
    for mesh in meshes:
        mesh_vertices = np.asarray(mesh["vertices"], dtype=np.float64)
        mesh_faces = np.asarray(mesh["faces"], dtype=np.int64)
        vertices.append(mesh_vertices)
        faces.append(mesh_faces + offset)
        offset += len(mesh_vertices)
    all_vertices = np.concatenate(vertices, axis=0)
    all_faces = np.concatenate(faces, axis=0)
    with output.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# Real2Sim USD mesh conversion\n")
        np.savetxt(handle, all_vertices, fmt="v %.9g %.9g %.9g")
        np.savetxt(handle, all_faces + 1, fmt="f %d %d %d")
    return {
        "path": str(output),
        "vertices": int(len(all_vertices)),
        "triangles": int(len(all_faces)),
        "size_bytes": output.stat().st_size,
        "sha256": _sha256(output),
    }


def export_glb(meshes: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    try:
        import numpy as np
        import trimesh
    except ImportError as exc:
        raise RuntimeError("GLB conversion requires trimesh and numpy") from exc

    scene = trimesh.Scene()
    palette = ([55, 170, 155, 255], [78, 132, 175, 255], [172, 132, 82, 255], [138, 112, 168, 255])
    for index, item in enumerate(meshes):
        mesh = trimesh.Trimesh(vertices=item["vertices"], faces=item["faces"], process=False, validate=False)
        mesh.visual.vertex_colors = np.tile(np.asarray(palette[index % len(palette)], dtype=np.uint8), (len(mesh.vertices), 1))
        scene.add_geometry(mesh, geom_name=f"{item['name']}_{index}", node_name=f"{item['name']}_{index}")
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output, file_type="glb")
    return {"path": str(output), "size_bytes": output.stat().st_size, "sha256": _sha256(output), "mesh_count": len(meshes)}


def convert_asset_to_obj(source: Path, output: Path) -> None:
    suffix = source.suffix.lower()
    if suffix == ".obj":
        shutil.copy2(source, output)
        return
    if suffix in {".usd", ".usda", ".usdc", ".usdz"}:
        meshes, _metadata = load_usd_meshes(source)
        if not meshes:
            raise ValueError(f"USD asset contains no meshes: {source.name}")
        export_combined_obj(meshes, output)
        return
    if suffix not in SUPPORTED_MESH_SUFFIXES:
        raise ValueError(f"unsupported MotrixSim mesh asset format: {suffix or 'unknown'}")
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("mesh conversion requires trimesh") from exc
    loaded = trimesh.load(source, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        mesh = loaded.to_geometry()
        if mesh is None or not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            raise ValueError(f"mesh asset contains no geometry: {source.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output, file_type="obj")


def write_runtime_scene(path: Path, *, scene_name: str, collision_obj: Path, visual_obj: Path) -> None:
    root = ET.Element("mujoco", {"model": _safe_name(scene_name)})
    ET.SubElement(root, "compiler", {"angle": "degree", "meshdir": "../assets/meshes", "autolimits": "true"})
    ET.SubElement(root, "option", {"timestep": "0.002", "gravity": "0 0 -9.81", "integrator": "implicitfast"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", {"name": "environment_collision", "file": collision_obj.name})
    ET.SubElement(asset, "mesh", {"name": "environment_visual", "file": visual_obj.name})
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"name": "key_light", "pos": "2 -2 6", "dir": "-0.2 0.2 -1", "diffuse": "0.9 0.9 0.9"})
    environment = ET.SubElement(worldbody, "body", {"name": "imported_environment"})
    ET.SubElement(environment, "geom", {"name": "environment_visual", "type": "mesh", "mesh": "environment_visual", "contype": "0", "conaffinity": "0", "group": "2", "rgba": "0.72 0.76 0.78 1"})
    ET.SubElement(environment, "geom", {"name": "environment_collision", "type": "mesh", "mesh": "environment_collision", "contype": "1", "conaffinity": "1", "group": "3", "friction": "0.9 0.01 0.001", "rgba": "0.2 0.7 0.65 0"})
    _indent_xml(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def validate_runtime_scene(path: Path, *, steps: int) -> dict[str, Any]:
    result = {"steps": steps, "mujoco": {"ok": False}, "motrixsim": {"ok": False}}
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        for _ in range(steps):
            mujoco.mj_step(model, data)
        result["mujoco"] = {"ok": True, "status": "loaded_and_stepped", "nbody": int(model.nbody), "ngeom": int(model.ngeom), "steps": steps}
    except Exception as exc:
        result["mujoco"] = {"ok": False, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    try:
        from motrixsim import SceneData, load_model, step

        model = load_model(str(path))
        data = SceneData(model)
        for _ in range(steps):
            step(model, data)
        result["motrixsim"] = {
            "ok": True,
            "status": "loaded_and_stepped",
            "steps": steps,
            "timestep": float(model.options.timestep),
            "nq": len(data.low.dof_positions),
            "nv": len(data.low.dof_velocities),
        }
    except Exception as exc:
        result["motrixsim"] = {"ok": False, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    result["ok"] = bool(result["motrixsim"]["ok"] and result["mujoco"]["ok"])
    return result


def normalize_transform(value: dict[str, Any]) -> dict[str, list[float]]:
    translation = _vector(value.get("translation"), 3, 0.0)
    rotation = _vector(value.get("rotation_euler_degrees"), 3, 0.0)
    if value.get("yaw_degrees") is not None and not any(rotation):
        rotation[2] = float(value["yaw_degrees"])
    scale_value = value.get("scale", 1.0)
    scale = [float(scale_value)] * 3 if isinstance(scale_value, (int, float)) else _vector(scale_value, 3, 1.0)
    if any(component <= 0 for component in scale):
        raise ValueError("asset scale must be positive")
    return {"translation": translation, "rotation_euler_degrees": rotation, "scale": scale}


def normalize_physics(value: dict[str, Any]) -> dict[str, Any]:
    collision_mode = str(value.get("collision_mode") or "visual_and_collision")
    if collision_mode not in {"visual_only", "collision_only", "visual_and_collision"}:
        raise ValueError("collision_mode must be visual_only, collision_only, or visual_and_collision")
    mass = float(value.get("mass", 0.25))
    if mass <= 0:
        raise ValueError("asset mass must be positive")
    dimensions = _vector(value.get("dimensions"), 3, 0.1)
    if any(component <= 0 for component in dimensions):
        raise ValueError("asset dimensions must be positive")
    rgba = _vector(value.get("rgba"), 4, 0.0)
    if not value.get("rgba"):
        rgba = [0.13, 0.76, 0.67, 1.0 if collision_mode != "collision_only" else 0.2]
    return {
        "dynamic": bool(value.get("dynamic", False)),
        "mass": mass,
        "collision_mode": collision_mode,
        "dimensions": dimensions,
        "friction": _vector(value.get("friction"), 3, 0.0) if value.get("friction") else [0.8, 0.01, 0.001],
        "rgba": rgba,
    }


def _primitive_geom(primitive: str, dimensions: list[float]) -> dict[str, str]:
    x, y, z = dimensions
    if primitive == "box":
        size = [x / 2, y / 2, z / 2]
    elif primitive == "sphere":
        size = [max(x, y, z) / 2]
    else:
        size = [max(x, y) / 2, z / 2]
    return {"type": primitive, "size": _numbers(size)}


def _runtime_scene_from_manifest(package_dir: Path, manifest: dict[str, Any]) -> Path:
    for relative in ("simulator_assets/scene.xml", "scene.xml", "scene_mujoco.xml"):
        path = package_dir / relative
        if path.is_file():
            return path
    motrix = (manifest.get("simulator_artifacts") or {}).get("motrixsim") or {}
    candidate = Path(str(motrix.get("runtime_scene") or ""))
    package_root = package_dir.resolve()
    if candidate.is_file():
        resolved = candidate.resolve()
        if resolved == package_root or package_root in resolved.parents:
            return resolved
    return package_dir / "simulator_assets" / "scene.xml"


def _manifest_asset(path: Path, role: str, kind: str, *, autoload: bool) -> dict[str, Any]:
    return {
        "name": path.name,
        "path": str(path),
        "role": role,
        "kind": kind,
        "format": path.suffix.lower().lstrip("."),
        "size_bytes": path.stat().st_size,
        "autoload": autoload,
        "viewer_hint": "autoload" if autoload else "manual_load",
        "sha256": _sha256(path),
    }


def _combined_bounds(meshes: list[dict[str, Any]]) -> dict[str, list[float]]:
    import numpy as np

    minimum = np.min(np.vstack([mesh["vertices"].min(axis=0) for mesh in meshes]), axis=0)
    maximum = np.max(np.vstack([mesh["vertices"].max(axis=0) for mesh in meshes]), axis=0)
    return {"min": [float(value) for value in minimum], "max": [float(value) for value in maximum]}


def _first_named(by_name: dict[str, list[Path]], name: str) -> Path | None:
    values = by_name.get(name.lower()) or []
    return values[0] if values else None


def _file_record(root: Path, path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {
        "path": str(path),
        "relative_path": path.relative_to(root).as_posix() if root.is_dir() else path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "format": path.suffix.lower().lstrip("."),
    }


def _safe_name(value: str) -> str:
    normalized = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in value.strip())
    return normalized.strip("_-")[:96] or "asset"


def _vector(value: Any, length: int, default: float) -> list[float]:
    if value is None:
        return [default] * length
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"expected a {length}-component vector")
    result = [float(component) for component in value]
    if any(not math.isfinite(component) for component in result):
        raise ValueError("vector components must be finite")
    return result


def _numbers(values: list[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return payload


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _indent_xml(root: ET.Element) -> None:
    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass


def safe_extract_zip(archive: zipfile.ZipFile, target: Path, *, max_total_bytes: int = 4 * 1024 * 1024 * 1024) -> None:
    total = 0
    target_resolved = target.resolve()
    for info in archive.infolist():
        pure = PurePosixPath(info.filename.replace("\\", "/"))
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"unsafe archive path: {info.filename}")
        total += int(info.file_size)
        if total > max_total_bytes:
            raise ValueError("archive expands beyond the configured safety limit")
        destination = (target / Path(*pure.parts)).resolve()
        if destination != target_resolved and target_resolved not in destination.parents:
            raise ValueError(f"archive path escapes target directory: {info.filename}")
    archive.extractall(target)
