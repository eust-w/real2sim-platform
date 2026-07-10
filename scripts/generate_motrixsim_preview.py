#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent
SRC_ROOT = APP_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from real2sim_platform.config import ensure_workspace, load_config  # noqa: E402
from real2sim_platform.store import PlatformStore  # noqa: E402


SCENE_ID = "1f2618bdfcac4e45456569980716405a"
PACKAGE_SLUG = "parlour_rm65_pick_place"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a browser GLB preview from the MotrixSim parlour + RM65 scaffold package.")
    parser.add_argument("--scene-id", default=SCENE_ID)
    parser.add_argument("--max-parlour-meshes", type=int, default=240, help="Limit parlour meshes to keep the browser preview responsive.")
    parser.add_argument("--skip-rm65", action="store_true")
    args = parser.parse_args()

    config = load_config()
    ensure_workspace(config)
    store = PlatformStore(config.db_path)
    scene = store.get_scene(args.scene_id)
    package_dir = Path(str((scene.get("metadata") or {}).get("package") or config.workspace_dir / "motrixsim_assets" / PACKAGE_SLUG))
    manifest_path = Path(str(scene.get("manifest_path") or package_dir / "motrixsim_scene_manifest.json"))
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")

    output_dir = package_dir / "viewer_assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "motrixsim_parlour_rm65_preview.glb"
    report_path = output_dir / "motrixsim_preview_report.json"

    scene_glb = trimesh.Scene()
    report: dict[str, Any] = {
        "schema": "real2sim.motrixsim_preview_report.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": [],
        "warnings": [],
    }

    parlour_xml = package_dir / "assets/parlour/parlour.xml"
    parlour_stats = add_mjcf_visuals(scene_glb, parlour_xml, max_meshes=args.max_parlour_meshes, name_prefix="parlour")
    report["inputs"].append({"role": "parlour", "xml": str(parlour_xml), **parlour_stats})

    if not args.skip_rm65:
        rm65_xml = package_dir / "assets/rm65_open_cabinet/mjx_rm65.xml"
        rm65_base = translation_matrix(np.array([0.8, 0.0, 0.55]))
        rm65_stats = add_mjcf_visuals(scene_glb, rm65_xml, root_transform=rm65_base, max_meshes=80, name_prefix="rm65")
        report["inputs"].append({"role": "rm65", "xml": str(rm65_xml), **rm65_stats})

    if not scene_glb.geometry:
        raise SystemExit("no mesh geometry was loaded from MotrixSim package")

    data = scene_glb.export(file_type="glb")
    output_path.write_bytes(data)
    report["output"] = {"path": str(output_path), "size_bytes": output_path.stat().st_size, "geometry_count": len(scene_glb.geometry)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    artifact = store.add_artifact(task_id=None, name=output_path.name, path=output_path, kind="mesh")
    report_artifact = store.add_artifact(task_id=None, name=report_path.name, path=report_path, kind="json")
    update_manifest(manifest_path, output_path, artifact["id"], report_path, report_artifact["id"], report)
    print(json.dumps({"ok": True, "scene_id": args.scene_id, "glb": str(output_path), "report": str(report_path), "geometry_count": len(scene_glb.geometry), "size_bytes": output_path.stat().st_size}, ensure_ascii=False, indent=2))
    return 0


def add_mjcf_visuals(
    scene: trimesh.Scene,
    xml_path: Path,
    *,
    root_transform: np.ndarray | None = None,
    max_meshes: int,
    name_prefix: str,
) -> dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    asset_dir = xml_path.parent / (root.find("compiler").attrib.get("assetdir", "") if root.find("compiler") is not None else "")
    mesh_defs = {
        item.attrib["name"]: asset_dir / item.attrib["file"]
        for item in root.findall("./asset/mesh")
        if item.attrib.get("name") and item.attrib.get("file")
    }
    stats = {"mesh_defs": len(mesh_defs), "loaded": 0, "missing": 0, "skipped": 0}
    world = root.find("worldbody")
    if world is None:
        return stats
    root_matrix = root_transform if root_transform is not None else np.eye(4)
    for node in list(world):
        traverse_mjcf_node(scene, node, mesh_defs, root_matrix, stats, max_meshes=max_meshes, name_prefix=name_prefix)
    return stats


def traverse_mjcf_node(
    scene: trimesh.Scene,
    node: ET.Element,
    mesh_defs: dict[str, Path],
    parent_transform: np.ndarray,
    stats: dict[str, int],
    *,
    max_meshes: int,
    name_prefix: str,
) -> None:
    current = parent_transform
    if node.tag == "body":
        current = parent_transform @ node_transform(node.attrib)
    elif node.tag == "geom":
        mesh_name = node.attrib.get("mesh")
        geom_class = node.attrib.get("class", "")
        if mesh_name and mesh_name in mesh_defs and (geom_class != "collision" or node.attrib.get("group") == "2"):
            if stats["loaded"] >= max_meshes:
                stats["skipped"] += 1
            else:
                mesh_path = mesh_defs[mesh_name]
                if mesh_path.is_file():
                    try:
                        mesh = load_mesh(mesh_path)
                        mesh.apply_transform(parent_transform @ node_transform(node.attrib))
                        scene.add_geometry(mesh, geom_name=f"{name_prefix}_{mesh_name}_{stats['loaded']}")
                        stats["loaded"] += 1
                    except Exception:
                        stats["skipped"] += 1
                else:
                    stats["missing"] += 1
    for child in list(node):
        traverse_mjcf_node(scene, child, mesh_defs, current, stats, max_meshes=max_meshes, name_prefix=name_prefix)


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        return trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return loaded


def node_transform(attrs: dict[str, str]) -> np.ndarray:
    transform = np.eye(4)
    if "pos" in attrs:
        transform = transform @ translation_matrix(parse_vec(attrs["pos"], 3))
    if "quat" in attrs:
        transform = transform @ quat_matrix(parse_vec(attrs["quat"], 4))
    elif "euler" in attrs:
        transform = transform @ euler_matrix(parse_vec(attrs["euler"], 3))
    if "scale" in attrs:
        scale = parse_vec(attrs["scale"], 3)
        transform = transform @ np.diag([scale[0], scale[1], scale[2], 1.0])
    return transform


def parse_vec(value: str, count: int) -> np.ndarray:
    parts = [float(item) for item in value.split()]
    if len(parts) == 1 and count == 3:
        parts = parts * 3
    while len(parts) < count:
        parts.append(0.0)
    return np.array(parts[:count], dtype=float)


def translation_matrix(vec: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = vec[:3]
    return matrix


def quat_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0:
        return np.eye(4)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )
    return matrix


def euler_matrix(euler: np.ndarray) -> np.ndarray:
    x, y, z = euler
    cx, cy, cz = math.cos(x), math.cos(y), math.cos(z)
    sx, sy, sz = math.sin(x), math.sin(y), math.sin(z)
    rx = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]])
    ry = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]])
    rz = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return rz @ ry @ rx


def update_manifest(manifest_path: Path, glb_path: Path, artifact_id: str, report_path: Path, report_artifact_id: str, report: dict[str, Any]) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = [item for item in manifest.get("assets", []) if not (isinstance(item, dict) and item.get("name") == glb_path.name)]
    glb_asset = {
        "name": glb_path.name,
        "role": "viewer_preview",
        "kind": "mesh",
        "path": str(glb_path),
        "relative_path": str(glb_path.relative_to(manifest_path.parent)),
        "url": f"/api/artifacts/{artifact_id}/content",
        "exists": True,
        "size_bytes": glb_path.stat().st_size,
        "size_label": format_bytes(glb_path.stat().st_size),
        "autoload": True,
        "generated": True,
    }
    report_asset = {
        "name": report_path.name,
        "role": "preview_report",
        "kind": "metadata",
        "path": str(report_path),
        "relative_path": str(report_path.relative_to(manifest_path.parent)),
        "url": f"/api/artifacts/{report_artifact_id}/content",
        "exists": True,
        "size_bytes": report_path.stat().st_size,
        "size_label": format_bytes(report_path.stat().st_size),
        "generated": True,
    }
    manifest["assets"] = [glb_asset, report_asset, *assets]
    manifest["viewer_assets"] = {"primary": glb_asset, "report": report_asset, "generator": "generate_motrixsim_preview.py"}
    manifest["preview_generation"] = report
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{value} B"


if __name__ == "__main__":
    raise SystemExit(main())
