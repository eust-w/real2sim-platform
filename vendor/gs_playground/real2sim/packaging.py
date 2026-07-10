"""GS-Playground package writer for Real2Sim assets."""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from math import atan2, cos, pi, sin, sqrt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .contracts import Real2SimAsset, Real2SimManifest, write_json
from .errors import StageArtifactError


MAX_MOTRIX_PREVIEW_MESH_FACES = 15_000
DEFAULT_MOTRIX_PREVIEW_PROXY_TEXTURE_SIZE = 1024
DEFAULT_SCENE_SURFEL_GEOMS = 9_000
DEFAULT_OBJECT_SURFEL_GEOMS = 3_000


@dataclass(frozen=True)
class PackageResult:
    root: Path
    scene_xml: Path
    config_json: Path
    manifest_json: Path


@dataclass(frozen=True)
class _ObjFace:
    tokens: tuple[str, ...]
    vertices: tuple[int, ...]
    texcoords: tuple[int | None, ...]
    normals: tuple[int | None, ...]
    material: str | None


@dataclass(frozen=True)
class _ProxyPartMaterial:
    key: str
    color: tuple[float, float, float]
    map_texture: bool
    specular: float
    shininess: float
    alpha: float = 1.0


class PackageWriter:
    def __init__(
        self,
        *,
        robot: str = "go2",
        dynamic_objects: bool = False,
        include_scene_mesh: bool = False,
    ) -> None:
        self.robot = robot
        self.dynamic_objects = dynamic_objects
        self.include_scene_mesh = include_scene_mesh

    def write(self, manifest: Real2SimManifest, output_dir: Path) -> PackageResult:
        root = output_dir.resolve()
        gs_dir = root / "3dgs"
        mesh_dir = root / "meshes"
        mjcf_dir = root / "mjcf"
        config_dir = root / "configs"
        artifact_dir = root / "artifacts"
        for directory in (gs_dir, mesh_dir, mjcf_dir, config_dir, artifact_dir):
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

        packaged_scene = self._copy_asset(
            manifest.scene,
            gs_dir,
            mesh_dir,
            "scene",
            include_mesh=self.include_scene_mesh,
            allow_preview_proxy=False,
        )
        packaged_objects = tuple(
            self._copy_asset(asset, gs_dir, mesh_dir, f"object_{idx:03d}_{asset.name}")
            for idx, asset in enumerate(manifest.objects)
        )
        packaged_camera_artifacts = self._copy_camera_artifacts(manifest.camera_artifacts, artifact_dir)
        packaged_manifest = Real2SimManifest(
            source=manifest.source,
            prompt=manifest.prompt,
            scene=packaged_scene,
            objects=packaged_objects,
            camera_artifacts=packaged_camera_artifacts,
            metadata={
                **manifest.metadata,
                "package_format": "gs-playground-real2sim-v1",
                "robot": self.robot,
                "motrix_preview_mesh_mode": motrix_preview_mesh_mode(),
                "motrix_preview_mesh_face_limit": motrix_preview_mesh_face_limit(),
                "motrix_preview_proxy_profile": motrix_preview_proxy_profile(),
                "motrix_preview_proxy_quality": motrix_preview_proxy_quality(),
                "motrix_preview_proxy_texture_size": motrix_preview_proxy_texture_size(),
                "motrix_scene_visual_mode": scene_visual_mode(),
                "motrix_scene_surfel_limit": scene_surfel_geom_limit(),
                "motrix_scene_surfel_saturation": scene_surfel_saturation(),
                "motrix_scene_surfel_contrast": scene_surfel_contrast(),
                "motrix_scene_surfel_highlight_cap": scene_surfel_highlight_cap(),
                "motrix_object_surfel_limit": object_surfel_geom_limit(),
                "motrix_visual_maxgeom": motrix_visual_maxgeom(len(packaged_objects)),
            },
        )

        scene_xml = mjcf_dir / "scene.xml"
        self._write_mjcf(packaged_manifest, scene_xml, mesh_dir)

        config_json = config_dir / f"{self.robot}_real2sim.json"
        write_json(
            config_json,
            {
                "robot": self.robot,
                "scene": _relative_posix(scene_xml, config_dir),
                "scene_gaussians": self._scene_gaussian_config(packaged_manifest, config_dir),
                "gaussian_links": self._gaussian_link_config(packaged_manifest),
                "camera_artifacts": self._camera_artifact_config(packaged_manifest, config_dir),
                "assets_dir": "assets",
                "robot_gs_dir": "",
                "save_dir": _relative_posix(root / "data", config_dir),
                "prompt": manifest.prompt or "Navigate in the reconstructed Real2Sim scene",
                "auto_reset": True,
            },
        )

        manifest_json = root / "package_manifest.json"
        write_json(manifest_json, packaged_manifest.to_json(root))
        self._write_motrix_import(root, packaged_manifest, scene_xml, config_json)
        return PackageResult(root=root, scene_xml=scene_xml, config_json=config_json, manifest_json=manifest_json)

    def _write_motrix_import(
        self,
        root: Path,
        manifest: Real2SimManifest,
        scene_xml: Path,
        config_json: Path,
    ) -> None:
        preview = manifest.camera_artifacts.get("source_gs_preview") or manifest.camera_artifacts.get("background_image")
        has_source_camera_preview = (
            manifest.camera_artifacts.get("cameras_json") is not None
            and manifest.camera_artifacts.get("source_gs_preview") is not None
        )
        scene_visual_mode_value = str(manifest.metadata.get("motrix_scene_visual_mode") or "")
        prefer_mjcf_scene = scene_visual_mode_value in {"mesh", "hybrid"}
        recommended_camera = "overview" if prefer_mjcf_scene else ("source_3dgs_preview" if has_source_camera_preview else "overview")
        viewer_camera = "closeup" if (has_source_camera_preview and not prefer_mjcf_scene) else "overview"
        viewer_args = [
            "--package",
            f"/path/to/{root.name}",
            "--camera",
            viewer_camera,
            "--gs-width",
            "900",
            "--gs-height",
            "700",
            "--static-gs-preview",
            "--show-source-gs",
        ]
        if has_source_camera_preview and not prefer_mjcf_scene:
            viewer_args.append("--source-gs-only")
        import_manifest = {
            "format": "gs-playground-real2sim-motrix-import-v1",
            "import_type": "mjcf_scene_with_3dgs",
            "scene_xml": _relative_posix(scene_xml, root),
            "config": _relative_posix(config_json, root),
            "scene_gaussians": {"scene": _relative_posix(manifest.scene.gaussian_ply, root)},
            "camera_artifacts": {
                key: _relative_posix(path, root)
                for key, path in sorted(manifest.camera_artifacts.items())
            },
            "preview": _relative_posix(preview, root) if preview is not None else None,
            "recommended_camera": recommended_camera,
            "viewer_args": viewer_args,
            "wuying_archive_launch": (
                "scripts/real2sim_wuying_launch_package.sh "
                f"--archive /path/to/{root.name}.tar.gz "
                f"--dest /tmp/{root.name}_import --with-gs"
            ),
            "verification": {
                "scene_ply_vertices": count_ply_vertices(manifest.scene.gaussian_ply),
                "object_count": len(manifest.objects),
                "pipeline": manifest.metadata.get("pipeline"),
                "recommended_preview": recommended_camera,
            },
        }
        if import_manifest["preview"] is None:
            import_manifest.pop("preview")

        write_json(root / "motrix_import.json", import_manifest)
        (root / "MOTRIX_IMPORT.md").write_text(
            "\n".join(
                [
                    "# MotrixSim Import",
                    "",
                    "Required entrypoints:",
                    "",
                    "- MJCF scene: `mjcf/scene.xml`",
                    "- Viewer config: `configs/*_real2sim.json`",
                    "- Scene Gaussian asset: `3dgs/scene.ply`",
                    "- Import manifest: `motrix_import.json`",
                    "",
                    "Recommended Wuying launch:",
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
                    f"- scene vertices: {import_manifest['verification']['scene_ply_vertices']}",
                    f"- object count: {import_manifest['verification']['object_count']}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _copy_asset(
        self,
        asset: Real2SimAsset,
        gs_dir: Path,
        mesh_dir: Path,
        prefix: str,
        *,
        include_mesh: bool = True,
        allow_preview_proxy: bool = True,
    ) -> Real2SimAsset:
        safe = _slug(prefix)
        ply_dst = gs_dir / f"{safe}.ply"
        shutil.copy2(asset.gaussian_ply, ply_dst)

        mesh_dst = None
        if include_mesh and asset.mesh_obj is not None:
            mesh_dst = mesh_dir / f"{safe}.obj"
            shutil.copy2(asset.mesh_obj, mesh_dst)
            copy_obj_material_sidecars(asset.mesh_obj, mesh_dst)
            clean_obj_floating_top_artifacts(mesh_dst)
            materialize_obj_materials(mesh_dst)
            auto_expose_obj_diffuse_textures(mesh_dst)
            write_obj_preview_lod(mesh_dst, max_faces=motrix_preview_mesh_face_limit(), allow_proxy=allow_preview_proxy)

        return Real2SimAsset(
            name=safe,
            label=asset.label,
            gaussian_ply=ply_dst,
            mesh_obj=mesh_dst,
            pose=asset.pose,
            dynamic=asset.dynamic,
            source_instance_id=asset.source_instance_id,
        )

    def _copy_camera_artifacts(self, camera_artifacts: dict[str, Path], artifact_dir: Path) -> dict[str, Path]:
        copied: dict[str, Path] = {}
        for key, path in sorted(camera_artifacts.items()):
            src = Path(path)
            if not src.is_file():
                raise StageArtifactError(f"missing camera artifact {key}: {src}")
            suffix = src.suffix
            dst = artifact_dir / f"{_slug(key)}{suffix}"
            shutil.copy2(src, dst)
            copied[key] = dst
        return copied

    def _scene_gaussian_config(self, manifest: Real2SimManifest, config_dir: Path) -> dict[str, str]:
        gaussians = {"scene": _relative_posix(manifest.scene.gaussian_ply, config_dir)}
        for asset in manifest.objects:
            gaussians[asset.name] = _relative_posix(asset.gaussian_ply, config_dir)
        return gaussians

    def _gaussian_link_config(self, manifest: Real2SimManifest) -> dict[str, str]:
        return {
            asset.name: asset.name
            for asset in manifest.objects
            if asset.dynamic or self.dynamic_objects
        }

    def _camera_artifact_config(self, manifest: Real2SimManifest, config_dir: Path) -> dict[str, str]:
        return {
            key: _relative_posix(path, config_dir)
            for key, path in sorted(manifest.camera_artifacts.items())
        }

    def _write_mjcf(self, manifest: Real2SimManifest, scene_xml: Path, mesh_dir: Path) -> None:
        root = ET.Element("mujoco", {"model": "real2sim_scene"})
        ET.SubElement(root, "option", {"gravity": "0 0 -9.81"})
        ET.SubElement(root, "compiler", {"meshdir": "../meshes", "texturedir": "../meshes"})
        visual = ET.SubElement(root, "visual")
        ET.SubElement(
            visual,
            "headlight",
            {
                "active": "1",
                "ambient": "0.42 0.42 0.42",
                "diffuse": "0.78 0.78 0.78",
                "specular": "0.18 0.18 0.18",
            },
        )
        asset_node = ET.SubElement(root, "asset")
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(
            world,
            "light",
            {
                "name": "key_light",
                "pos": "1.5 -2 2.8",
                "dir": "-0.45 0.55 -1",
                "diffuse": "0.9 0.88 0.82",
                "specular": "0.22 0.22 0.22",
            },
        )
        ET.SubElement(
            world,
            "light",
            {
                "name": "fill_light",
                "pos": "-2 -1 1.6",
                "dir": "0.8 0.35 -0.45",
                "diffuse": "0.55 0.58 0.62",
                "specular": "0.08 0.08 0.08",
            },
        )
        ET.SubElement(
            world,
            "light",
            {
                "name": "top_light",
                "pos": "0 0 3",
                "dir": "0 0 -1",
                "diffuse": "0.55 0.55 0.55",
                "specular": "0.12 0.12 0.12",
            },
        )
        ET.SubElement(world, "camera", {"name": "overview", **manifest_overview_camera_attrs(manifest)})
        ET.SubElement(world, "camera", {"name": "closeup", **manifest_closeup_camera_attrs(manifest)})
        if manifest.scene.mesh_obj is None:
            ET.SubElement(
                world,
                "geom",
                {
                    "name": "fallback_ground",
                    "type": "plane",
                    "pos": "0 0 -0.01",
                    "size": "5 5 0.01",
                    "rgba": "0.35 0.35 0.35 1",
                    "group": "3",
                },
            )

        completion_material = add_scene_completion_material(asset_node, mesh_dir)
        completion_backdrop_mesh = add_scene_completion_backdrop_mesh(asset_node, mesh_dir, completion_material)

        self._add_asset_mesh(asset_node, manifest.scene, mesh_dir)
        for obj in manifest.objects:
            self._add_asset_mesh(asset_node, obj, mesh_dir)

        scene_front_y = scene_preview_front_y(manifest.scene.mesh_obj)

        self._add_body(
            world,
            manifest.scene,
            mesh_dir,
            is_scene=True,
            scene_completion_material=completion_material,
            scene_completion_backdrop_mesh=completion_backdrop_mesh,
        )
        for obj in manifest.objects:
            self._add_body(world, obj, mesh_dir, is_scene=False, object_preview_y=scene_front_y)

        _indent_xml(root)
        tree = ET.ElementTree(root)
        scene_xml.parent.mkdir(parents=True, exist_ok=True)
        tree.write(scene_xml, encoding="utf-8", xml_declaration=True)

    def _add_asset_mesh(self, asset_node: ET.Element, asset: Real2SimAsset, mesh_dir: Path) -> None:
        if asset.mesh_obj is None:
            return
        attrs = {
            "name": f"{asset.name}_mesh",
            "file": asset.mesh_obj.resolve().relative_to(mesh_dir.resolve()).as_posix(),
        }
        if not _is_unit_scale(asset.pose.scale):
            attrs["scale"] = _vec(asset.pose.scale)
        ET.SubElement(asset_node, "mesh", attrs)
        material_attrs = {
            "name": f"{asset.name}_material",
            "rgba": _rgba(obj_average_color(asset.mesh_obj)),
            "specular": "0.15",
            "shininess": "0.25",
        }
        texture = obj_diffuse_texture(asset.mesh_obj)
        if texture is not None:
            texture_name = f"{asset.name}_texture"
            ET.SubElement(
                asset_node,
                "texture",
                {
                    "name": texture_name,
                    "type": "2d",
                    "file": texture.resolve().relative_to(mesh_dir.resolve()).as_posix(),
                },
            )
            material_attrs["texture"] = texture_name
            material_attrs["rgba"] = "1 1 1 1"
        ET.SubElement(asset_node, "material", material_attrs)

    def _add_body(
        self,
        world: ET.Element,
        asset: Real2SimAsset,
        mesh_dir: Path,
        *,
        is_scene: bool,
        object_preview_y: float | None = None,
        scene_completion_material: str | None = None,
        scene_completion_backdrop_mesh: str | None = None,
    ) -> None:
        body_attrs = {
            "name": asset.name,
            "pos": _vec(asset.pose.position),
            "quat": _vec(asset.pose.quat_wxyz),
        }
        is_dynamic_object = not is_scene and (asset.dynamic or self.dynamic_objects)
        bounds = None
        if asset.mesh_obj is not None:
            bounds = (
                object_dynamic_collision_bounds(asset, preview_world_y=object_preview_y)
                if is_dynamic_object
                else obj_bounds(asset.mesh_obj, scale=asset.pose.scale)
            )
        if is_dynamic_object:
            body_attrs["gravcomp"] = "1"
        body = ET.SubElement(world, "body", body_attrs)
        if is_dynamic_object:
            mass, diaginertia = dynamic_object_inertial(asset, bounds)
            ET.SubElement(body, "joint", {"name": f"{asset.name}_freejoint", "type": "free", "damping": "5"})
            ET.SubElement(body, "inertial", {"pos": "0 0 0", "mass": f"{mass:.8g}", "diaginertia": _vec(diaginertia)})

        if asset.mesh_obj is not None:
            visual_mode = scene_visual_mode() if is_scene else "surfels"
            color_surfels = 0
            if not is_scene or visual_mode in {"surfels", "hybrid"}:
                color_surfels = (
                    add_scene_color_surfel_geoms(body, asset)
                    if is_scene
                    else add_object_color_surfel_geoms(body, asset, preview_world_y=object_preview_y)
                )
            if color_surfels == 0 or visual_mode in {"mesh", "hybrid"}:
                self._add_mesh_visual_geom(body, asset, is_scene=is_scene)
            if is_scene:
                add_scene_completion_geoms(body)
                add_scene_completion_backdrop_geom(
                    body,
                    mesh_name=scene_completion_backdrop_mesh,
                    material_name=scene_completion_material,
                )
            if bounds is not None and not is_scene:
                center, half_size = bounds
                ET.SubElement(
                    body,
                    "geom",
                    {
                        "name": f"{asset.name}_collision_box",
                        "type": "box",
                        "pos": _vec(center),
                        "size": _vec(half_size),
                        "rgba": "0 1 0 0",
                        "group": "3",
                        "contype": "1",
                        "conaffinity": "1",
                        "friction": "0.8 0.01 0.001",
                    },
                )
        elif not is_scene:
            raise StageArtifactError(f"object asset requires a mesh for collision packaging: {asset.name}")

    def _add_mesh_visual_geom(self, body: ET.Element, asset: Real2SimAsset, *, is_scene: bool) -> None:
        geom_attrs = {
            "name": f"{asset.name}_visual",
            "type": "mesh",
            "mesh": f"{asset.name}_mesh",
            "group": "2",
            "contype": "0",
            "conaffinity": "0",
        }
        if asset.mesh_obj is not None and (
            not is_scene
            or obj_diffuse_texture(asset.mesh_obj) is not None
            or not obj_has_material_library(asset.mesh_obj)
        ):
            geom_attrs["material"] = f"{asset.name}_material"
        ET.SubElement(body, "geom", geom_attrs)


def obj_bounds(
    path: Path,
    *,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    vertices: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not vertices:
        return None

    mins = [min(v[i] for v in vertices) for i in range(3)]
    maxs = [max(v[i] for v in vertices) for i in range(3)]
    center = tuple(((mins[i] + maxs[i]) / 2.0) * scale[i] for i in range(3))
    half = tuple(max(((maxs[i] - mins[i]) / 2.0) * abs(scale[i]), 1e-4) for i in range(3))
    return center, half


def object_dynamic_collision_bounds(
    asset: Real2SimAsset,
    *,
    preview_world_y: float | None = None,
    max_half_extent: float = 0.12,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    bounds = obj_bounds(asset.mesh_obj, scale=asset.pose.scale) if asset.mesh_obj is not None else None
    if preview_world_y is None:
        return clamp_collision_bounds(bounds, max_half_extent=max_half_extent)

    points = read_gaussian_points(asset.gaussian_ply, max_points=20_000)
    if not points:
        return clamp_collision_bounds(bounds, max_half_extent=max_half_extent)
    xs = [point[0] * asset.pose.scale[0] for point in points]
    ys = [point[1] * asset.pose.scale[1] for point in points]
    zs = [point[2] * asset.pose.scale[2] for point in points]
    center = (
        (min(xs) + max(xs)) * 0.5,
        preview_world_y - asset.pose.position[1],
        (min(zs) + max(zs)) * 0.5,
    )
    half = (
        max((max(xs) - min(xs)) * 0.5, 1e-4),
        max((max(ys) - min(ys)) * 0.5 * 0.08, 1e-4),
        max((max(zs) - min(zs)) * 0.5, 1e-4),
    )
    return clamp_collision_bounds((center, half), max_half_extent=max_half_extent)


def clamp_collision_bounds(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    *,
    max_half_extent: float,
    min_half_extent: float = 0.015,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if bounds is None:
        return None
    center, half = bounds
    clamped_half = tuple(min(max(abs(value), min_half_extent), max_half_extent) for value in half)
    return center, clamped_half


def dynamic_object_inertial(
    asset: Real2SimAsset,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
) -> tuple[float, tuple[float, float, float]]:
    half = bounds[1] if bounds is not None else (0.05, 0.05, 0.05)
    dimensions = tuple(max(2.0 * abs(value), 0.03) for value in half)
    mass = dynamic_object_mass(asset, dimensions)
    x, y, z = dimensions
    inertia = (
        max(mass * (y * y + z * z) / 12.0, 1e-5),
        max(mass * (x * x + z * z) / 12.0, 1e-5),
        max(mass * (x * x + y * y) / 12.0, 1e-5),
    )
    return mass, inertia


def dynamic_object_mass(asset: Real2SimAsset, dimensions: tuple[float, float, float]) -> float:
    semantic = f"{asset.label} {asset.name}".lower()
    if any(token in semantic for token in ("bottle", "can", "cup", "mug")):
        return 0.2
    if any(token in semantic for token in ("book", "box", "cube", "block")):
        return 0.35
    volume = max(dimensions[0] * dimensions[1] * dimensions[2], 1e-6)
    return min(max(volume * 250.0, 0.05), 1.0)


def scene_preview_front_y(scene_mesh: Path | None) -> float | None:
    bounds = obj_bounds(scene_mesh) if scene_mesh is not None else None
    if bounds is None:
        return None
    center, half = bounds
    return center[1] - half[1] - 0.06


def count_ply_vertices(path: Path) -> int | None:
    try:
        with path.open("r", encoding="ascii", errors="ignore") as handle:
            for line in handle:
                if line.startswith("element vertex "):
                    return int(line.split()[-1])
                if line.strip() == "end_header":
                    break
    except OSError:
        return None
    return None


def overview_camera_attrs(scene_mesh: Path | None) -> dict[str, str]:
    bounds = obj_bounds(scene_mesh) if scene_mesh is not None else None
    return overview_camera_attrs_from_bounds(bounds)


def manifest_overview_camera_attrs(manifest: Real2SimManifest) -> dict[str, str]:
    bounds = manifest_world_bounds(manifest)
    return overview_camera_attrs_from_bounds(bounds)


def manifest_closeup_camera_attrs(manifest: Real2SimManifest) -> dict[str, str]:
    bounds = manifest_world_bounds(manifest)
    return closeup_camera_attrs_from_bounds(bounds)


def manifest_world_bounds(manifest: Real2SimManifest) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    mins: list[tuple[float, float, float]] = []
    maxs: list[tuple[float, float, float]] = []
    for asset in (manifest.scene, *manifest.objects):
        if asset.mesh_obj is None:
            continue
        bounds = obj_bounds(asset.mesh_obj, scale=asset.pose.scale)
        if bounds is None:
            continue
        center, half = bounds
        world_center = tuple(center[idx] + asset.pose.position[idx] for idx in range(3))
        mins.append(tuple(world_center[idx] - half[idx] for idx in range(3)))
        maxs.append(tuple(world_center[idx] + half[idx] for idx in range(3)))
    if not mins:
        return None
    world_min = tuple(min(values[idx] for values in mins) for idx in range(3))
    world_max = tuple(max(values[idx] for values in maxs) for idx in range(3))
    center = tuple((world_min[idx] + world_max[idx]) * 0.5 for idx in range(3))
    half = tuple(max((world_max[idx] - world_min[idx]) * 0.5, 1e-4) for idx in range(3))
    return center, half


def overview_camera_attrs_from_bounds(bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None) -> dict[str, str]:
    if bounds is None:
        return {
            "pos": "1.5 -2.5 1.4",
            "xyaxes": "0.866 0.5 0 -0.25 0.433 0.866",
        }
    center, half = bounds
    radius = max(max(half[0], half[2]), 0.35)
    target = (center[0], center[1], center[2])
    pos = (
        center[0],
        center[1] - max(radius * 2.4, half[1] + 1.35),
        center[2],
    )
    x_axis, y_axis = source_view_xyaxes(pos, target)
    return {"pos": _vec(pos), "xyaxes": f"{_vec(x_axis)} {_vec(y_axis)}"}


def closeup_camera_attrs_from_bounds(bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None) -> dict[str, str]:
    if bounds is None:
        return {
            "pos": "0 -0.75 1",
            "xyaxes": "1 0 0 0 0 1",
        }
    center, half = bounds
    radius = max(max(half[0], half[2]), 0.35)
    target = (center[0], center[1], center[2])
    pos = (
        center[0],
        center[1] - max(radius * 1.15, half[1] + 0.55, 0.55),
        center[2],
    )
    x_axis, y_axis = source_view_xyaxes(pos, target)
    return {"pos": _vec(pos), "xyaxes": f"{_vec(x_axis)} {_vec(y_axis)}"}


def source_view_xyaxes(
    pos: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    forward = _normalise_vec(tuple(target[idx] - pos[idx] for idx in range(3)), fallback=(0.0, 1.0, 0.0))
    x_axis = _normalise_vec(_cross(forward, (0.0, 0.0, 1.0)), fallback=(1.0, 0.0, 0.0))
    y_axis = _normalise_vec(_cross(x_axis, forward), fallback=(0.0, 0.0, 1.0))
    return x_axis, y_axis


def look_at_xyaxes(
    pos: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    forward = _normalise_vec(tuple(target[idx] - pos[idx] for idx in range(3)), fallback=(0.0, 1.0, -0.35))
    z_axis = tuple(-value for value in forward)
    up = (0.0, 0.0, 1.0)
    x_axis = _normalise_vec(_cross(up, z_axis), fallback=(1.0, 0.0, 0.0))
    y_axis = _normalise_vec(_cross(z_axis, x_axis), fallback=(0.0, 0.0, 1.0))
    return x_axis, y_axis


def _normalise_vec(
    values: tuple[float, float, float],
    *,
    fallback: tuple[float, float, float],
) -> tuple[float, float, float]:
    norm = sqrt(sum(value * value for value in values))
    if norm < 1e-8:
        return fallback
    return tuple(value / norm for value in values)


def _cross(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def obj_average_color(path: Path, default: tuple[float, float, float] = (0.72, 0.72, 0.72)) -> tuple[float, float, float]:
    count = 0
    sums = [0.0, 0.0, 0.0]
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            color = _normalise_rgb(parts[4:7])
            for idx, channel in enumerate(color):
                sums[idx] += channel
            count += 1
    if count == 0:
        return default
    return tuple(channel_sum / count for channel_sum in sums)


def obj_has_material_library(path: Path) -> bool:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return any(line.startswith("mtllib ") for line in f)


def add_scene_color_surfel_geoms(
    body: ET.Element,
    asset: Real2SimAsset,
    *,
    max_points: int | None = None,
) -> int:
    max_points = scene_surfel_geom_limit() if max_points is None else max_points
    mesh_obj = asset.mesh_obj
    if mesh_obj is None:
        points = []
    else:
        points = read_obj_vertex_color_points(mesh_obj, max_points=max_points)
        if not points:
            points = read_obj_texture_color_points(mesh_obj, max_points=max_points)
    if not points:
        points = read_gaussian_points(asset.gaussian_ply, max_points=max_points)
    return add_color_surfel_geoms(body, "scene_color_surfel", points, max_points=max_points)


def read_obj_vertex_color_points(
    mesh_obj: Path,
    *,
    max_points: int,
) -> list[tuple[float, float, float, float, float, float]]:
    color_mesh = mesh_obj.with_name(f"{mesh_obj.stem}_highres.obj")
    if not color_mesh.is_file():
        color_mesh = mesh_obj
    vertices: list[tuple[float, float, float, float, float, float]] = []
    with color_mesh.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            x, y, z = (float(parts[idx]) for idx in range(1, 4))
            r, g, b = _normalise_rgb(parts[4:7])
            vertices.append((x, y, z, r, g, b))
    if not vertices:
        return []
    stride = max(1, int(len(vertices) / max_points))
    return vertices[::stride][:max_points]


def read_obj_texture_color_points(
    mesh_obj: Path,
    *,
    max_points: int,
) -> list[tuple[float, float, float, float, float, float]]:
    texture = obj_diffuse_texture(mesh_obj)
    if texture is None:
        return []
    try:
        from PIL import Image
    except ImportError:
        return []
    vertices: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    uv_sums: dict[int, list[float]] = {}
    uv_counts: dict[int, int] = {}
    try:
        with mesh_obj.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif line.startswith("vt "):
                    parts = line.split()
                    if len(parts) >= 3:
                        texcoords.append((float(parts[1]), float(parts[2])))
                elif line.startswith("f "):
                    for token in line.split()[1:]:
                        vertex_index, texcoord_index = parse_obj_vertex_texcoord_token(
                            token,
                            vertex_count=len(vertices),
                            texcoord_count=len(texcoords),
                        )
                        if vertex_index is None or texcoord_index is None:
                            continue
                        if vertex_index < 0 or vertex_index >= len(vertices):
                            continue
                        if texcoord_index < 0 or texcoord_index >= len(texcoords):
                            continue
                        u, v = texcoords[texcoord_index]
                        if vertex_index not in uv_sums:
                            uv_sums[vertex_index] = [0.0, 0.0]
                            uv_counts[vertex_index] = 0
                        uv_sums[vertex_index][0] += u
                        uv_sums[vertex_index][1] += v
                        uv_counts[vertex_index] += 1
    except (OSError, ValueError):
        return []
    if not uv_sums:
        return []
    try:
        image = Image.open(texture).convert("RGB")
    except OSError:
        return []
    width, height = image.size
    pixels = image.load()
    vertex_indices = sorted(uv_sums)
    stride = max(1, int(len(vertex_indices) / max_points))
    points: list[tuple[float, float, float, float, float, float]] = []
    for vertex_index in vertex_indices[::stride]:
        if len(points) >= max_points:
            break
        count = max(uv_counts.get(vertex_index, 0), 1)
        u = uv_sums[vertex_index][0] / count
        v = uv_sums[vertex_index][1] / count
        x, y, z = vertices[vertex_index]
        px = min(max(int(round((u % 1.0) * (width - 1))), 0), width - 1)
        py = min(max(int(round((1.0 - (v % 1.0)) * (height - 1))), 0), height - 1)
        r, g, b = pixels[px, py]
        points.append((x, y, z, r / 255.0, g / 255.0, b / 255.0))
    return points


def parse_obj_vertex_texcoord_token(
    token: str,
    *,
    vertex_count: int,
    texcoord_count: int,
) -> tuple[int | None, int | None]:
    parts = token.split("/")
    if not parts or not parts[0]:
        return None, None
    try:
        vertex_index = int(parts[0])
    except ValueError:
        return None, None
    texcoord_index = None
    if len(parts) >= 2 and parts[1]:
        try:
            texcoord_index = int(parts[1])
        except ValueError:
            texcoord_index = None
    return resolve_obj_index(vertex_index, vertex_count), resolve_obj_index(texcoord_index, texcoord_count)


def resolve_obj_index(index: int | None, count: int) -> int | None:
    if index is None:
        return None
    if index > 0:
        return index - 1
    if index < 0:
        return count + index
    return None


def add_color_surfel_geoms(
    body: ET.Element,
    prefix: str,
    points: list[tuple[float, float, float, float, float, float]],
    *,
    max_points: int,
) -> int:
    if not points:
        return 0
    sampled = points[:max_points]
    xs = [point[0] for point in sampled]
    ys = [point[1] for point in sampled]
    zs = [point[2] for point in sampled]
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1e-3)
    radius = max(min(span / 180.0, 0.018), 0.003) * scene_surfel_radius_scale()
    size = (radius * 1.2, max(radius * 0.45, 0.0015), radius * 1.2)
    for idx, (x, y, z, r, g, b) in enumerate(sampled):
        r, g, b = motrix_scene_surfel_color((r, g, b))
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{prefix}_{idx:04d}",
                "type": "box",
                "pos": _vec((x, y, z)),
                "size": _vec(size),
                "rgba": _rgba((r, g, b), 1.0),
                "group": "2",
                "contype": "0",
                "conaffinity": "0",
            },
        )
    return len(sampled)


def add_object_color_surfel_geoms(
    body: ET.Element,
    asset: Real2SimAsset,
    *,
    max_points: int | None = None,
    preview_world_y: float | None = None,
) -> int:
    max_points = object_surfel_geom_limit() if max_points is None else max_points
    points = read_gaussian_points(asset.gaussian_ply, max_points=max_points)
    if len(points) < 50:
        return 0
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1e-3)
    center_y = (min(ys) + max(ys)) * 0.5
    radius = max(min(span / 80.0, 0.018), 0.004)
    size = (radius * 1.25, radius * 1.25, radius * 1.25)
    for idx, (x, y, z, r, g, b) in enumerate(points):
        r, g, b = motrix_scene_surfel_color((r, g, b))
        if preview_world_y is not None:
            y = (y - center_y) * 0.08 + (preview_world_y - asset.pose.position[1])
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{asset.name}_surfel_{idx:04d}",
                "type": "box",
                "pos": _vec((x, y, z)),
                "size": _vec(size),
                "rgba": _rgba((r, g, b), 1.0),
                "group": "2",
                "contype": "0",
                "conaffinity": "0",
            },
        )
    return len(points)


def read_gaussian_points(path: Path, *, max_points: int) -> list[tuple[float, float, float, float, float, float]]:
    try:
        with path.open("rb") as f:
            header_bytes: list[bytes] = []
            while True:
                line = f.readline()
                if not line:
                    return []
                header_bytes.append(line)
                if line.strip() == b"end_header":
                    break
            header = [line.decode("ascii", errors="ignore").strip() for line in header_bytes]
            if not header or header[0] != "ply":
                return []
            is_ascii = "format ascii 1.0" in header
            is_binary_little = "format binary_little_endian 1.0" in header
            if not is_ascii and not is_binary_little:
                return []
            vertex_count = 0
            properties: list[tuple[str, str]] = []
            reading_vertex = False
            for line in header:
                parts = line.split()
                if len(parts) >= 3 and parts[:2] == ["element", "vertex"]:
                    vertex_count = int(parts[2])
                    reading_vertex = True
                    continue
                if len(parts) >= 3 and parts[0] == "element" and parts[1] != "vertex":
                    reading_vertex = False
                if reading_vertex and len(parts) == 3 and parts[0] == "property":
                    properties.append((parts[1], parts[2]))
            names = [name for _kind, name in properties]
            indices = {name: idx for idx, name in enumerate(names)}
            required = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2")
            if not all(name in indices for name in required):
                return []
            stride = max(1, vertex_count // max(max_points, 1))
            points: list[tuple[float, float, float, float, float, float]] = []
            sh_c0 = 0.28209479177387814
            if is_ascii:
                for line_idx, raw_line in enumerate(f):
                    if line_idx % stride != 0:
                        continue
                    parts = raw_line.decode("ascii", errors="ignore").split()
                    if len(parts) < len(properties):
                        continue
                    points.append(_gaussian_point_from_values(parts, indices, sh_c0))
                    if len(points) >= max_points:
                        break
            else:
                readers, record_size = binary_ply_vertex_readers(properties)
                if record_size <= 0:
                    return []
                for vertex_idx in range(vertex_count):
                    record = f.read(record_size)
                    if len(record) != record_size:
                        break
                    if vertex_idx % stride != 0:
                        continue
                    values = [reader(record) for reader in readers]
                    points.append(_gaussian_point_from_values(values, indices, sh_c0))
                    if len(points) >= max_points:
                        break
            return points
    except (OSError, ValueError, struct.error):
        return []


def _gaussian_point_from_values(
    values: list[object],
    indices: dict[str, int],
    sh_c0: float,
) -> tuple[float, float, float, float, float, float]:
    x = float(values[indices["x"]])
    y = float(values[indices["y"]])
    z = float(values[indices["z"]])
    r = float(values[indices["f_dc_0"]]) * sh_c0 + 0.5
    g = float(values[indices["f_dc_1"]]) * sh_c0 + 0.5
    b = float(values[indices["f_dc_2"]]) * sh_c0 + 0.5
    return (x, y, z, r, g, b)


def binary_ply_vertex_readers(properties: list[tuple[str, str]]):
    offsets: list[tuple[str, int, int]] = []
    offset = 0
    sizes = {
        "char": 1,
        "int8": 1,
        "uchar": 1,
        "uint8": 1,
        "short": 2,
        "int16": 2,
        "ushort": 2,
        "uint16": 2,
        "int": 4,
        "int32": 4,
        "uint": 4,
        "uint32": 4,
        "float": 4,
        "float32": 4,
        "double": 8,
        "float64": 8,
    }
    formats = {
        "char": "b",
        "int8": "b",
        "uchar": "B",
        "uint8": "B",
        "short": "h",
        "int16": "h",
        "ushort": "H",
        "uint16": "H",
        "int": "i",
        "int32": "i",
        "uint": "I",
        "uint32": "I",
        "float": "f",
        "float32": "f",
        "double": "d",
        "float64": "d",
    }
    for kind, _name in properties:
        if kind not in sizes:
            return [], 0
        offsets.append((formats[kind], offset, sizes[kind]))
        offset += sizes[kind]

    def make_reader(fmt: str, start: int, size: int):
        return lambda record: struct.unpack_from("<" + fmt, record, start)[0]

    return [make_reader(fmt, start, size) for fmt, start, size in offsets], offset


def motrix_scene_surfel_color(color: tuple[float, float, float]) -> tuple[float, float, float]:
    """Keep single-image scene surfels legible under MotrixSim preview lights."""

    r, g, b = (min(max(float(channel), 0.0), 1.0) for channel in color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    saturation = scene_surfel_saturation()
    contrast = scene_surfel_contrast()
    channels = []
    for value in (r, g, b):
        saturated = luminance + (value - luminance) * saturation
        contrasted = 0.5 + (saturated - 0.5) * contrast
        compressed = min(contrasted, scene_surfel_highlight_cap())
        channels.append(min(max(compressed, 0.0), 1.0))
    return tuple(channels)


def scene_surfel_geom_limit() -> int:
    raw = os.environ.get("REAL2SIM_MOTRIX_SCENE_SURFELS")
    if raw is None:
        return DEFAULT_SCENE_SURFEL_GEOMS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_SCENE_SURFEL_GEOMS
    return min(max(value, 1_000), 30_000)


def object_surfel_geom_limit() -> int:
    raw = os.environ.get("REAL2SIM_MOTRIX_OBJECT_SURFELS")
    if raw is None:
        return DEFAULT_OBJECT_SURFEL_GEOMS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_OBJECT_SURFEL_GEOMS
    return min(max(value, 250), 12_000)


def motrix_visual_maxgeom(object_count: int = 0) -> int:
    raw = os.environ.get("REAL2SIM_MOTRIX_MAXGEOM")
    if raw is not None:
        try:
            return min(max(int(raw), 10_000), 200_000)
        except ValueError:
            pass
    scene_mode = scene_visual_mode()
    scene_geoms = scene_surfel_geom_limit() if scene_mode in {"surfels", "hybrid"} else 0
    object_geoms = max(int(object_count), 0) * object_surfel_geom_limit()
    mesh_and_completion_margin = 2_500
    return min(max((scene_geoms + object_geoms + mesh_and_completion_margin) * 2, 10_000), 200_000)


def scene_visual_mode() -> str:
    raw = os.environ.get("REAL2SIM_MOTRIX_SCENE_VISUAL", "surfels").strip().lower()
    if raw in {
        "mesh",
        "surface",
        "continuous",
        "textured_mesh",
        "textured-mesh",
        "photogrammetry",
        "high_fidelity",
        "high-fidelity",
    }:
        return "mesh"
    if raw in {"hybrid", "mesh+surfels", "surfels+mesh"}:
        return "hybrid"
    return "surfels"


def add_scene_completion_material(asset_node: ET.Element, mesh_dir: Path) -> str | None:
    texture_path = scene_completion_texture_path()
    if texture_path is None:
        return None
    suffix = texture_path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        return None
    mesh_dir.mkdir(parents=True, exist_ok=True)
    dest = mesh_dir / "scene_completion_texture.png"
    try:
        if suffix == ".png":
            shutil.copy2(texture_path, dest)
        else:
            from PIL import Image

            Image.open(texture_path).convert("RGB").save(dest)
    except OSError:
        return None
    except ImportError:
        return None
    ET.SubElement(
        asset_node,
        "texture",
        {
            "name": "scene_completion_texture",
            "type": "2d",
            "file": dest.name,
        },
    )
    ET.SubElement(
        asset_node,
        "material",
        {
            "name": "scene_completion_photo_material",
            "texture": "scene_completion_texture",
            "rgba": "1 1 1 0.88",
            "specular": "0.02",
            "shininess": "0.05",
        },
    )
    return "scene_completion_photo_material"


def scene_completion_texture_path() -> Path | None:
    raw = os.environ.get("REAL2SIM_SCENE_COMPLETION_TEXTURE")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    return path


def add_scene_completion_backdrop_mesh(
    asset_node: ET.Element,
    mesh_dir: Path,
    material_name: str | None,
) -> str | None:
    if material_name is None:
        return None
    report = scene_completion_report()
    if report is None:
        return None
    plane = scene_completion_back_plane(report)
    if plane is None:
        return None
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = mesh_dir / "scene_completion_backdrop.obj"
    vertices = plane
    mesh_path.write_text(
        "\n".join(
            [
                "o real2sim_scene_completion_backdrop",
                f"v {vertices[0][0]:.8f} {vertices[0][1]:.8f} {vertices[0][2]:.8f}",
                f"v {vertices[1][0]:.8f} {vertices[1][1]:.8f} {vertices[1][2]:.8f}",
                f"v {vertices[2][0]:.8f} {vertices[2][1]:.8f} {vertices[2][2]:.8f}",
                f"v {vertices[3][0]:.8f} {vertices[3][1]:.8f} {vertices[3][2]:.8f}",
                "vt 0.00000000 0.00000000",
                "vt 1.00000000 0.00000000",
                "vt 1.00000000 1.00000000",
                "vt 0.00000000 1.00000000",
                "f 1/1 2/2 3/3",
                "f 1/1 3/3 4/4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    ET.SubElement(
        asset_node,
        "mesh",
        {
            "name": "scene_completion_backdrop_mesh",
            "file": mesh_path.name,
            "inertia": "shell",
        },
    )
    return "scene_completion_backdrop_mesh"


def scene_completion_back_plane(report: dict[str, object]) -> list[tuple[float, float, float]] | None:
    low = report.get("bounds_low")
    high = report.get("bounds_high")
    if not (isinstance(low, list) and isinstance(high, list) and len(low) == 3 and len(high) == 3):
        return None
    try:
        bounds_low = tuple(float(value) for value in low)
        bounds_high = tuple(float(value) for value in high)
    except (TypeError, ValueError):
        return None
    up_axis = str(report.get("up_axis") or "z").lower()
    if up_axis not in {"x", "y", "z"}:
        up_axis = "z"
    axes = {"x": 0, "y": 1, "z": 2}
    up = axes[up_axis]
    horizontal = [axis for axis in (0, 1, 2) if axis != up]
    u_axis, depth_axis = horizontal
    span = [max(bounds_high[idx] - bounds_low[idx], 1e-4) for idx in range(3)]
    offset = max(max(span) / 1000.0, 0.005)
    depth = bounds_low[depth_axis] - offset
    backdrop_margin = scene_completion_backdrop_margin()
    u_margin = span[u_axis] * backdrop_margin
    up_margin = span[up] * backdrop_margin
    u_low = bounds_low[u_axis] - u_margin
    u_high = bounds_high[u_axis] + u_margin
    up_low = bounds_low[up] - up_margin
    up_high = bounds_high[up] + up_margin

    def vertex(u: float, v: float) -> tuple[float, float, float]:
        coords = [0.0, 0.0, 0.0]
        coords[u_axis] = u
        coords[depth_axis] = depth
        coords[up] = v
        return (coords[0], coords[1], coords[2])

    return [
        vertex(u_low, up_low),
        vertex(u_high, up_low),
        vertex(u_high, up_high),
        vertex(u_low, up_high),
    ]


def scene_completion_backdrop_margin() -> float:
    raw = os.environ.get("REAL2SIM_SCENE_COMPLETION_BACKDROP_MARGIN")
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return min(max(value, 0.0), 2.0)


def add_scene_completion_backdrop_geom(
    body: ET.Element,
    *,
    mesh_name: str | None,
    material_name: str | None,
) -> int:
    if mesh_name is None or material_name is None:
        return 0
    ET.SubElement(
        body,
        "geom",
        {
            "name": "scene_completion_photo_backdrop",
            "type": "mesh",
            "mesh": mesh_name,
            "material": material_name,
            "group": "2",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    return 1


def add_scene_completion_geoms(body: ET.Element) -> int:
    report = scene_completion_report()
    if report is None:
        return 0
    low = report.get("bounds_low")
    high = report.get("bounds_high")
    planes = report.get("planes")
    if not (isinstance(low, list) and isinstance(high, list) and len(low) == 3 and len(high) == 3):
        return 0
    if not isinstance(planes, list):
        return 0
    try:
        bounds_low = tuple(float(value) for value in low)
        bounds_high = tuple(float(value) for value in high)
    except (TypeError, ValueError):
        return 0
    up_axis = str(report.get("up_axis") or "z").lower()
    if up_axis not in {"x", "y", "z"}:
        up_axis = "z"
    axes = {"x": 0, "y": 1, "z": 2}
    up = axes[up_axis]
    horizontal = [axis for axis in (0, 1, 2) if axis != up]
    a0, a1 = horizontal
    span = [max(bounds_high[idx] - bounds_low[idx], 1e-4) for idx in range(3)]
    thickness = max(max(span) / 500.0, 0.01)
    added = 0
    for raw_name in planes:
        name = str(raw_name)
        pos = [(bounds_low[idx] + bounds_high[idx]) * 0.5 for idx in range(3)]
        size = [span[idx] * 0.5 for idx in range(3)]
        rgba = (0.68, 0.65, 0.59, 0.06)
        if name == "floor":
            pos[up] = bounds_low[up]
            size[up] = thickness
            rgba = (0.42, 0.41, 0.39, 0.12)
        elif name == "ceiling":
            pos[up] = bounds_high[up]
            size[up] = thickness
            rgba = (0.72, 0.70, 0.66, 0.04)
        elif name == "left":
            pos[a0] = bounds_low[a0]
            size[a0] = thickness
        elif name == "right":
            pos[a0] = bounds_high[a0]
            size[a0] = thickness
        elif name == "back":
            pos[a1] = bounds_low[a1]
            size[a1] = thickness
        elif name == "front":
            pos[a1] = bounds_high[a1]
            size[a1] = thickness
        else:
            continue
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"scene_completion_{name}",
                "type": "box",
                "pos": _vec(tuple(pos)),
                "size": _vec(tuple(size)),
                "rgba": _rgba(rgba[:3], rgba[3]),
                "group": "2",
                "contype": "1",
                "conaffinity": "1",
            },
        )
        added += 1
    return added


def scene_completion_report() -> dict[str, object] | None:
    path = os.environ.get("REAL2SIM_SCENE_COMPLETION_REPORT")
    if not path:
        return None
    report_path = Path(path)
    if not report_path.is_file():
        return None
    try:
        import json

        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def scene_surfel_saturation() -> float:
    return _env_float("REAL2SIM_MOTRIX_SCENE_SURFEL_SATURATION", 1.35, minimum=0.2, maximum=3.0)


def scene_surfel_radius_scale() -> float:
    return _env_float("REAL2SIM_MOTRIX_SCENE_SURFEL_RADIUS_SCALE", 1.0, minimum=0.2, maximum=4.0)


def scene_surfel_contrast() -> float:
    return _env_float("REAL2SIM_MOTRIX_SCENE_SURFEL_CONTRAST", 1.12, minimum=0.2, maximum=3.0)


def scene_surfel_highlight_cap() -> float:
    return _env_float("REAL2SIM_MOTRIX_SCENE_SURFEL_HIGHLIGHT_CAP", 0.88, minimum=0.45, maximum=1.0)


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return min(max(value, minimum), maximum)


def obj_diffuse_texture(path: Path) -> Path | None:
    obj_dir = path.parent
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("mtllib "):
            continue
        for raw_mtl in line.split()[1:]:
            texture = mtl_diffuse_texture(obj_dir / raw_mtl)
            if texture is not None:
                return texture
    return None


def mtl_diffuse_texture(path: Path) -> Path | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0] != "map_Kd":
            continue
        texture = (path.parent / parts[-1]).resolve()
        if texture.is_file():
            return texture
    return None


def clean_obj_floating_top_artifacts(
    path: Path,
    *,
    min_component_count: int = 50,
    min_faces: int = 100,
    top_ratio: float = 0.82,
    tip_ratio: float = 0.92,
    flat_y_ratio: float = 0.12,
    broad_ratio: float = 0.45,
    tip_broad_ratio: float = 0.25,
    desaturated_top_ratio: float = 0.88,
    desaturated_flat_y_ratio: float = 0.09,
    desaturated_broad_ratio: float = 0.08,
    preserve_saturated_top_saturation: float = 95.0,
) -> dict[str, int]:
    """Remove disconnected, wide, flat top sheets from SAM-3D OBJ exports.

    Transparent objects often produce thousands of tiny OBJ components. A common
    failure mode is a broad, very flat sheet near the top of the local object
    frame. It shows up as a floating "cloth" in MotrixSim after texture baking.
    The cleanup is deliberately gated on high component count so normal OBJ
    assets with intentional flat tops are left unchanged.
    """

    raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    parsed = _parse_obj(raw_lines)
    vertices = parsed["vertices"]
    texcoords = parsed["texcoords"]
    normals = parsed["normals"]
    faces: list[_ObjFace] = parsed["faces"]
    if not vertices or not faces:
        return {"removed_components": 0, "removed_faces": 0, "kept_faces": len(faces)}

    components = _obj_face_components(faces, len(vertices))
    if len(components) < min_component_count:
        return {"removed_components": 0, "removed_faces": 0, "kept_faces": len(faces)}

    used_vertices = sorted({vertex for face in faces for vertex in face.vertices})
    bounds = _vertex_bounds(vertices, used_vertices)
    if bounds is None:
        return {"removed_components": 0, "removed_faces": 0, "kept_faces": len(faces)}
    mins, maxs = bounds
    global_ext = tuple(max(maxs[idx] - mins[idx], 1e-8) for idx in range(3))
    texture_image = _load_obj_diffuse_texture(path) if texcoords else None

    removed_roots: set[int] = set()
    for root, face_indices in components.items():
        if len(face_indices) < min_faces:
            continue
        component_vertices = sorted({vertex for face_index in face_indices for vertex in faces[face_index].vertices})
        component_bounds = _vertex_bounds(vertices, component_vertices)
        if component_bounds is None:
            continue
        comp_mins, comp_maxs = component_bounds
        comp_ext = tuple(comp_maxs[idx] - comp_mins[idx] for idx in range(3))
        center_y = (comp_mins[1] + comp_maxs[1]) * 0.5
        top_position = (center_y - mins[1]) / global_ext[1]
        flatness = comp_ext[1] / global_ext[1]
        broadness = max(comp_ext[0] / global_ext[0], comp_ext[2] / global_ext[2])
        broad_top_sheet = top_position >= top_ratio and flatness <= flat_y_ratio and broadness >= broad_ratio
        high_tip_sheet = top_position >= tip_ratio and flatness <= flat_y_ratio and broadness >= tip_broad_ratio
        desaturated_top_sheet = False
        saturated_top_part = False
        texture_stats = None
        if texture_image is not None:
            texture_stats = _component_texture_stats(texture_image, texcoords, faces, face_indices)
            if texture_stats is not None:
                _, saturation = texture_stats
                saturated_top_part = saturation >= preserve_saturated_top_saturation
        if (
            texture_image is not None
            and top_position >= desaturated_top_ratio
            and flatness <= desaturated_flat_y_ratio
            and broadness >= desaturated_broad_ratio
        ):
            if texture_stats is not None:
                brightness, saturation = texture_stats
                desaturated_top_sheet = 30.0 <= brightness <= 155.0 and saturation <= 80.0
        if (not saturated_top_part and (broad_top_sheet or high_tip_sheet)) or desaturated_top_sheet:
            removed_roots.add(root)

    if not removed_roots:
        return {"removed_components": 0, "removed_faces": 0, "kept_faces": len(faces)}

    kept_faces = [face for idx, face in enumerate(faces) if parsed["face_roots"][idx] not in removed_roots]
    _write_compacted_obj(path, parsed["header"], vertices, texcoords, normals, kept_faces)
    removed_faces = len(faces) - len(kept_faces)
    return {"removed_components": len(removed_roots), "removed_faces": removed_faces, "kept_faces": len(kept_faces)}


def _load_obj_diffuse_texture(path: Path):
    texture = obj_diffuse_texture(path)
    if texture is None:
        return None
    try:
        from PIL import Image

        return Image.open(texture).convert("RGB")
    except Exception:
        return None


def _component_texture_stats(
    image,
    texcoords: list[str],
    faces: list[_ObjFace],
    face_indices: list[int],
    *,
    sample_limit: int = 300,
) -> tuple[float, float] | None:
    pixels: list[tuple[int, int, int]] = []
    width, height = image.size
    step = max(len(face_indices) // sample_limit, 1)
    for face_index in face_indices[::step]:
        face = faces[face_index]
        for texcoord in face.texcoords:
            if texcoord is None or texcoord <= 0 or texcoord > len(texcoords):
                continue
            parts = texcoords[texcoord - 1].split()
            if len(parts) < 3:
                continue
            u = float(parts[1])
            v = float(parts[2])
            x = min(max(int(u * (width - 1)), 0), width - 1)
            y = min(max(int((1.0 - v) * (height - 1)), 0), height - 1)
            pixels.append(image.getpixel((x, y)))
    if not pixels:
        return None
    channels = tuple(sum(pixel[idx] for pixel in pixels) / len(pixels) for idx in range(3))
    brightness = sum(channels) / 3.0
    saturation = max(channels) - min(channels)
    return brightness, saturation


def write_obj_preview_lod(path: Path, *, max_faces: int, allow_proxy: bool = True) -> dict[str, int]:
    """Write a lower-face OBJ for interactive MotrixSim preview.

    The SAM-3D textured meshes remain valuable as assets, but MotrixSim's
    interactive viewer stalls when every baked reconstruction triangle is loaded
    directly in MJCF. This keeps a packaged ``*_highres.obj`` copy and rewrites
    the MJCF-facing OBJ to a deterministic, component-aware face subset while
    preserving UVs, materials, and texture sidecars.
    """

    max_faces = max(int(max_faces), 1)
    raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    parsed = _parse_obj(raw_lines)
    vertices = parsed["vertices"]
    texcoords = parsed["texcoords"]
    normals = parsed["normals"]
    faces: list[_ObjFace] = parsed["faces"]
    if len(faces) <= max_faces:
        ensure_obj_vertex_normals(path)
        return {"original_faces": len(faces), "preview_faces": len(faces), "highres_written": 0}

    highres_path = path.with_name(f"{path.stem}_highres.obj")
    if not highres_path.exists():
        shutil.copy2(path, highres_path)

    if allow_proxy and motrix_preview_mesh_mode() == "proxy" and _write_revolved_proxy_preview_lod(
        path,
        vertices=vertices,
        texcoords=texcoords,
        faces=faces,
        max_faces=max_faces,
    ):
        ensure_obj_vertex_normals(path)
        return {
            "original_faces": len(faces),
            "preview_faces": _count_obj_faces(path),
            "highres_written": 1,
        }

    if _write_pymeshlab_preview_lod(path, max_faces=max_faces):
        ensure_obj_vertex_normals(path)
        return {
            "original_faces": len(faces),
            "preview_faces": _count_obj_faces(path),
            "highres_written": 1,
        }

    selected = _select_lod_faces(faces, len(vertices), max_faces=max_faces)
    _write_compacted_obj(path, parsed["header"], vertices, texcoords, normals, selected)
    ensure_obj_vertex_normals(path)
    return {"original_faces": len(faces), "preview_faces": len(selected), "highres_written": 1}


def motrix_preview_mesh_face_limit() -> int:
    raw = os.environ.get("REAL2SIM_MOTRIX_PREVIEW_MESH_FACES")
    if raw is None:
        return MAX_MOTRIX_PREVIEW_MESH_FACES
    try:
        value = int(raw)
    except ValueError:
        return MAX_MOTRIX_PREVIEW_MESH_FACES
    return max(value, 1_000)


def motrix_preview_mesh_mode() -> str:
    raw = os.environ.get("REAL2SIM_MOTRIX_PREVIEW_MODE", "proxy").strip().lower()
    return "lod" if raw in {"lod", "mesh", "decimate", "decimation"} else "proxy"


def motrix_preview_proxy_profile() -> str:
    raw = os.environ.get("REAL2SIM_MOTRIX_PREVIEW_PROXY_PROFILE", "radial").strip().lower()
    return "angular" if raw in {"angular", "silhouette", "non_symmetric", "asymmetric"} else "radial"


def motrix_preview_proxy_quality() -> str:
    raw = os.environ.get("REAL2SIM_MOTRIX_PREVIEW_PROXY_QUALITY", "high").strip().lower()
    if raw in {"fast", "low", "performance"}:
        return "fast"
    if raw in {"balanced", "medium"}:
        return "balanced"
    return "high"


def motrix_preview_proxy_texture_size() -> int:
    raw = os.environ.get("REAL2SIM_MOTRIX_PREVIEW_PROXY_TEXTURE_SIZE")
    if raw is None:
        return DEFAULT_MOTRIX_PREVIEW_PROXY_TEXTURE_SIZE
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MOTRIX_PREVIEW_PROXY_TEXTURE_SIZE
    return min(max(value, 256), 2048)


def _write_revolved_proxy_preview_lod(
    path: Path,
    *,
    vertices: list[tuple[str, tuple[float, float, float]]],
    texcoords: list[str],
    faces: list[_ObjFace],
    max_faces: int,
) -> bool:
    used_vertices = sorted({vertex for face in faces for vertex in face.vertices})
    bounds = _vertex_bounds(vertices, used_vertices)
    if bounds is None:
        return False
    mins, maxs = bounds
    ext = tuple(max(maxs[idx] - mins[idx], 1e-8) for idx in range(3))
    center_x = _quantile([vertices[index - 1][1][0] for index in used_vertices], 0.5)
    center_z = _quantile([vertices[index - 1][1][2] for index in used_vertices], 0.5)
    band_count, radial_segments = _proxy_resolution(max_faces)
    y_min, y_max = mins[1], maxs[1]
    y_span = max(y_max - y_min, 1e-8)
    min_radius = max(min(ext[0], ext[2]) * 0.035, 1e-4)
    if motrix_preview_proxy_profile() == "angular":
        radial_profile = _angular_radial_profile(
            vertices,
            used_vertices,
            band_count=band_count,
            radial_segments=radial_segments,
            center_x=center_x,
            center_z=center_z,
            y_min=y_min,
            y_span=y_span,
            min_radius=min_radius,
        )
    else:
        radial_profile = [
            [radius for _segment in range(radial_segments)]
            for radius in _robust_radial_profile(
                vertices,
                used_vertices,
                band_count=band_count,
                center_x=center_x,
                center_z=center_z,
                y_min=y_min,
                y_span=y_span,
                min_radius=min_radius,
            )
        ]
    texture_name = f"{path.stem}_proxy_texture.png"
    band_colors = _preview_band_colors(path, vertices, texcoords, faces, band_count, y_min, y_span)
    texture_size = motrix_preview_proxy_texture_size()
    if not _write_proxy_texture(
        path.with_name(texture_name),
        source_obj=path,
        vertices=vertices,
        texcoords=texcoords,
        faces=faces,
        width=texture_size,
        height=texture_size,
        center_x=center_x,
        center_z=center_z,
        y_min=y_min,
        y_span=y_span,
        fallback_band_colors=band_colors,
    ):
        texture_name = None

    cap_bands, cap_color = _top_red_cap_material(band_colors)
    part_band_keys, part_materials = _proxy_part_materials(
        band_colors,
        cap_bands=cap_bands,
        cap_color=cap_color,
        texture_name=texture_name,
    )
    vertex_records: list[tuple[float, float, float, float, float]] = []
    ring_indices: list[list[int]] = []
    for ring in range(band_count + 1):
        y = y_min + y_span * (ring / band_count)
        v_coord = ring / band_count
        if ring == 0:
            ring_radii = radial_profile[0]
        elif ring == band_count:
            ring_radii = radial_profile[-1]
        else:
            ring_radii = [
                (radial_profile[ring - 1][segment] + radial_profile[ring][segment]) * 0.5
                for segment in range(radial_segments)
            ]
        ring_vertices = []
        for segment in range(radial_segments + 1):
            theta = 2.0 * pi * segment / radial_segments
            radius = ring_radii[segment % radial_segments]
            vertex_records.append(
                (
                    center_x + radius * cos(theta),
                    y,
                    center_z + radius * sin(theta),
                    segment / radial_segments,
                    v_coord,
                )
            )
            ring_vertices.append(len(vertex_records))
        ring_indices.append(ring_vertices)

    vertex_records.append((center_x, y_min, center_z, 0.5, 0.0))
    bottom_center = len(vertex_records)
    vertex_records.append((center_x, y_max, center_z, 0.5, 1.0))
    top_center = len(vertex_records)

    material_name = f"{path.stem}_proxy_baked" if texture_name is not None else f"{path.stem}_proxy_band_00"
    obj_lines = [f"mtllib {path.stem}_proxy.mtl\n", f"o {_slug(path.stem)}_motrix_proxy\n"]
    for x, y, z, _u, _v in vertex_records:
        obj_lines.append(f"v {x:.8f} {y:.8f} {z:.8f}\n")
    for _x, _y, _z, u, v in vertex_records:
        obj_lines.append(f"vt {u:.8f} {v:.8f}\n")
    current_material: str | None = None
    for band in range(band_count):
        band_key = part_band_keys[band] if band < len(part_band_keys) else None
        band_material = f"{path.stem}_proxy_{band_key}" if band_key is not None else material_name
        if band_material != current_material:
            obj_lines.append(f"usemtl {band_material}\n")
            current_material = band_material
        lower = ring_indices[band]
        upper = ring_indices[band + 1]
        for segment in range(radial_segments):
            nxt = segment + 1
            obj_lines.append(f"f {lower[segment]}/{lower[segment]} {lower[nxt]}/{lower[nxt]} {upper[nxt]}/{upper[nxt]}\n")
            obj_lines.append(f"f {lower[segment]}/{lower[segment]} {upper[nxt]}/{upper[nxt]} {upper[segment]}/{upper[segment]}\n")

    if material_name != current_material:
        obj_lines.append(f"usemtl {material_name}\n")
        current_material = material_name
    for segment in range(radial_segments):
        nxt = segment + 1
        obj_lines.append(f"f {bottom_center}/{bottom_center} {ring_indices[0][nxt]}/{ring_indices[0][nxt]} {ring_indices[0][segment]}/{ring_indices[0][segment]}\n")
    top_key = part_band_keys[band_count - 1] if part_band_keys else None
    top_material = f"{path.stem}_proxy_{top_key}" if top_key is not None else material_name
    if top_material != current_material:
        obj_lines.append(f"usemtl {top_material}\n")
    for segment in range(radial_segments):
        nxt = segment + 1
        obj_lines.append(f"f {top_center}/{top_center} {ring_indices[-1][segment]}/{ring_indices[-1][segment]} {ring_indices[-1][nxt]}/{ring_indices[-1][nxt]}\n")

    path.write_text("".join(obj_lines), encoding="utf-8")
    _write_proxy_mtl(
        path.with_name(f"{path.stem}_proxy.mtl"),
        path.stem,
        band_colors,
        texture_name=texture_name,
        part_materials=part_materials,
    )
    return True


def _proxy_resolution(max_faces: int) -> tuple[int, int]:
    quality = motrix_preview_proxy_quality()
    if quality == "fast":
        band_count, radial_segments = 24, 32
    elif quality == "balanced":
        band_count, radial_segments = 48, 64
    else:
        band_count, radial_segments = 72, 96

    minimum_bands, minimum_segments = 8, 12
    while _proxy_face_count(band_count, radial_segments) > max_faces and (
        band_count > minimum_bands or radial_segments > minimum_segments
    ):
        if radial_segments / max(band_count, 1) > 1.35 and radial_segments > minimum_segments:
            radial_segments -= 4
        elif band_count > minimum_bands:
            band_count -= 2
        elif radial_segments > minimum_segments:
            radial_segments -= 2
    return band_count, radial_segments


def _proxy_face_count(band_count: int, radial_segments: int) -> int:
    return band_count * radial_segments * 2 + radial_segments * 2


def _robust_radial_profile(
    vertices: list[tuple[str, tuple[float, float, float]]],
    used_vertices: list[int],
    *,
    band_count: int,
    center_x: float,
    center_z: float,
    y_min: float,
    y_span: float,
    min_radius: float,
) -> list[float]:
    bands: list[list[float]] = [[] for _band in range(band_count)]
    for vertex_index in used_vertices:
        x, y, z = vertices[vertex_index - 1][1]
        band = min(max(int(((y - y_min) / y_span) * band_count), 0), band_count - 1)
        dx = x - center_x
        dz = z - center_z
        bands[band].append(max((dx * dx + dz * dz) ** 0.5, min_radius))

    profile: list[float | None] = []
    for values in bands:
        profile.append(max(_quantile(values, 0.74), min_radius) if values else None)
    _fill_missing_profile(profile, min_radius)
    smoothed = [float(value) for value in profile]
    for _ in range(3):
        smoothed = _smooth_profile(smoothed, min_radius)
    return smoothed


def _angular_radial_profile(
    vertices: list[tuple[str, tuple[float, float, float]]],
    used_vertices: list[int],
    *,
    band_count: int,
    radial_segments: int,
    center_x: float,
    center_z: float,
    y_min: float,
    y_span: float,
    min_radius: float,
) -> list[list[float]]:
    bins: list[list[list[float]]] = [[[] for _segment in range(radial_segments)] for _band in range(band_count)]
    band_values: list[list[float]] = [[] for _band in range(band_count)]
    for vertex_index in used_vertices:
        x, y, z = vertices[vertex_index - 1][1]
        band = min(max(int(((y - y_min) / y_span) * band_count), 0), band_count - 1)
        dx = x - center_x
        dz = z - center_z
        radius = max((dx * dx + dz * dz) ** 0.5, min_radius)
        theta = atan2(dz, dx)
        if theta < 0.0:
            theta += 2.0 * pi
        segment = min(max(int((theta / (2.0 * pi)) * radial_segments), 0), radial_segments - 1)
        bins[band][segment].append(radius)
        band_values[band].append(radius)

    rows: list[list[float]] = []
    last_fallback = min_radius
    for band in range(band_count):
        if band_values[band]:
            fallback = max(_quantile(band_values[band], 0.82), min_radius)
            last_fallback = fallback
        else:
            fallback = last_fallback
        row: list[float | None] = []
        for segment in range(radial_segments):
            values = bins[band][segment]
            row.append(max(_quantile(values, 0.88), min_radius) if values else None)
        rows.append(_fill_missing_angular_profile(row, fallback))

    if len(rows) >= 3:
        vertical = [row[:] for row in rows]
        for band in range(1, len(rows) - 1):
            for segment in range(radial_segments):
                vertical[band][segment] = (
                    rows[band - 1][segment] * 0.18
                    + rows[band][segment] * 0.64
                    + rows[band + 1][segment] * 0.18
                )
        rows = vertical

    return [_smooth_angular_profile(row, min_radius) for row in rows]


def _fill_missing_angular_profile(values: list[float | None], fallback: float) -> list[float]:
    if not values:
        return []
    if all(value is None for value in values):
        return [fallback for _value in values]
    filled: list[float] = []
    count = len(values)
    for idx, value in enumerate(values):
        if value is not None:
            filled.append(value)
            continue
        nearest: list[float] = []
        for step in range(1, count + 1):
            left = values[(idx - step) % count]
            right = values[(idx + step) % count]
            if left is not None:
                nearest.append(left)
            if right is not None:
                nearest.append(right)
            if nearest:
                break
        filled.append(sum(nearest) / len(nearest) if nearest else fallback)
    return filled


def _smooth_angular_profile(values: list[float], minimum: float) -> list[float]:
    if len(values) < 3:
        return [max(value, minimum) for value in values]
    smoothed = []
    count = len(values)
    for idx, value in enumerate(values):
        smoothed.append(max(value * 0.8 + values[(idx - 1) % count] * 0.1 + values[(idx + 1) % count] * 0.1, minimum))
    return smoothed


def _quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(max(int(round((len(ordered) - 1) * quantile)), 0), len(ordered) - 1)
    return ordered[index]


def _fill_missing_profile(values: list[float | None], minimum: float) -> None:
    last: float | None = None
    for idx, value in enumerate(values):
        if value is not None:
            last = value
        elif last is not None:
            values[idx] = last
    next_value: float | None = None
    for idx in range(len(values) - 1, -1, -1):
        if values[idx] is not None:
            next_value = values[idx]
        elif next_value is not None:
            values[idx] = next_value
    for idx, value in enumerate(values):
        if value is None:
            values[idx] = minimum


def _smooth_profile(values: list[float], minimum: float) -> list[float]:
    if len(values) < 3:
        return [max(value, minimum) for value in values]
    smoothed = values[:]
    for idx in range(1, len(values) - 1):
        smoothed[idx] = (values[idx - 1] + values[idx] * 2.0 + values[idx + 1]) / 4.0
    return [max(value, minimum) for value in smoothed]


def _preview_band_colors(
    path: Path,
    vertices: list[tuple[str, tuple[float, float, float]]],
    texcoords: list[str],
    faces: list[_ObjFace],
    band_count: int,
    y_min: float,
    y_span: float,
) -> list[tuple[float, float, float]]:
    texture_image = _load_obj_diffuse_texture(path) if texcoords else None
    bins: list[list[tuple[int, int, int]]] = [[] for _ in range(band_count)]
    if texture_image is not None:
        width, height = texture_image.size
        for face in faces:
            center_y = sum(vertices[vertex - 1][1][1] for vertex in face.vertices) / len(face.vertices)
            band = min(max(int(((center_y - y_min) / y_span) * band_count), 0), band_count - 1)
            for texcoord in face.texcoords:
                if texcoord is None or texcoord <= 0 or texcoord > len(texcoords):
                    continue
                parts = texcoords[texcoord - 1].split()
                if len(parts) < 3:
                    continue
                u = float(parts[1])
                v = float(parts[2])
                x = min(max(int(u * (width - 1)), 0), width - 1)
                y = min(max(int((1.0 - v) * (height - 1)), 0), height - 1)
                bins[band].append(texture_image.getpixel((x, y)))

    fallback = tuple(int(channel * 255) for channel in obj_average_color(path))
    colors: list[tuple[float, float, float]] = []
    for pixels in bins:
        if pixels:
            rgb = tuple(sum(pixel[idx] for pixel in pixels) / (255.0 * len(pixels)) for idx in range(3))
        else:
            rgb = tuple(channel / 255.0 for channel in fallback)
        colors.append(_visible_preview_color(rgb))
    return colors


def _visible_preview_color(color: tuple[float, float, float]) -> tuple[float, float, float]:
    brightness = max(sum(color) / 3.0, 1e-6)
    target = 0.32
    if brightness < target:
        scale = min(target / brightness, 4.5)
        color = tuple(min(channel * scale, 1.0) for channel in color)
    return tuple(min(max(channel, 0.03), 1.0) for channel in color)


def _top_red_cap_material(colors: list[tuple[float, float, float]]) -> tuple[set[int], tuple[float, float, float] | None]:
    if len(colors) < 4:
        return set(), None
    start = max(int(len(colors) * 0.72), 0)
    candidates = [
        idx
        for idx in range(start, len(colors))
        if _is_red_part_color(colors[idx])
    ]
    if not candidates:
        return set(), None
    first, last = min(candidates), max(candidates)
    if last - first > max(2, len(colors) // 5):
        first = max(candidates)
        while first > start and _is_red_part_color(colors[first - 1]):
            first -= 1
        last = max(candidates)
    bands = set(range(first, last + 1))
    selected = [colors[idx] for idx in sorted(bands)]
    color = tuple(sum(pixel[channel] for pixel in selected) / len(selected) for channel in range(3))
    return bands, _visible_preview_color(color)


def _is_red_part_color(color: tuple[float, float, float]) -> bool:
    r, g, b = color
    return r >= 0.28 and r - max(g, b) >= 0.08 and max(color) - min(color) >= 0.12


def _proxy_part_materials(
    colors: list[tuple[float, float, float]],
    *,
    cap_bands: set[int],
    cap_color: tuple[float, float, float] | None,
    texture_name: str | None,
) -> tuple[list[str | None], dict[str, _ProxyPartMaterial]]:
    band_keys: list[str | None] = [None for _color in colors]
    if cap_color is not None:
        for band in cap_bands:
            if 0 <= band < len(band_keys):
                band_keys[band] = "cap"

    for idx, color in enumerate(colors):
        if band_keys[idx] is not None:
            continue
        key = _proxy_part_key(color)
        if key is None:
            continue
        band_keys[idx] = key

    _promote_muted_label_neighbors(colors, band_keys)

    groups: dict[str, list[tuple[float, float, float]]] = {}
    for idx, key in enumerate(band_keys):
        if key is None:
            continue
        groups.setdefault(key, []).append(colors[idx])

    return band_keys, {
        key: _proxy_part_material(key, group, map_texture=(texture_name is not None and key != "cap"))
        for key, group in groups.items()
    }


def _proxy_part_key(color: tuple[float, float, float]) -> str | None:
    if _is_red_part_color(color):
        return "label_red"
    if _is_white_label_color(color):
        return "label_white"
    if _is_dark_part_color(color):
        return "dark_body"
    if _is_clear_plastic_color(color):
        return "clear_plastic"
    return None


def _is_white_label_color(color: tuple[float, float, float]) -> bool:
    brightness = sum(color) / 3.0
    saturation = max(color) - min(color)
    return brightness >= 0.64 and saturation <= 0.24


def _is_dark_part_color(color: tuple[float, float, float]) -> bool:
    brightness = sum(color) / 3.0
    return brightness <= 0.26


def _is_clear_plastic_color(color: tuple[float, float, float]) -> bool:
    brightness = sum(color) / 3.0
    saturation = max(color) - min(color)
    return 0.30 <= brightness <= 0.72 and saturation <= 0.16


def _promote_muted_label_neighbors(colors: list[tuple[float, float, float]], band_keys: list[str | None]) -> None:
    red_bands = [idx for idx, key in enumerate(band_keys) if key == "label_red"]
    if not red_bands:
        return
    for idx, key in enumerate(band_keys):
        if key not in {None, "clear_plastic"}:
            continue
        if not _is_muted_label_color(colors[idx]):
            continue
        if any(1 <= abs(idx - red_idx) <= 4 for red_idx in red_bands):
            band_keys[idx] = "label_white"


def _is_muted_label_color(color: tuple[float, float, float]) -> bool:
    brightness = sum(color) / 3.0
    saturation = max(color) - min(color)
    return brightness >= 0.38 and saturation <= 0.18


def _proxy_part_material(
    key: str,
    colors: list[tuple[float, float, float]],
    *,
    map_texture: bool,
) -> _ProxyPartMaterial:
    color = tuple(sum(pixel[channel] for pixel in colors) / len(colors) for channel in range(3))
    color = _visible_preview_color(color)
    if key == "clear_plastic":
        color = tuple(min(max(channel * 1.25, 0.48), 0.92) for channel in color)
        return _ProxyPartMaterial(key=key, color=color, map_texture=map_texture, specular=0.34, shininess=48.0, alpha=0.9)
    if key == "label_white":
        color = tuple(min(max(channel, 0.82), 1.0) for channel in color)
        return _ProxyPartMaterial(key=key, color=color, map_texture=map_texture, specular=0.12, shininess=18.0)
    if key == "label_red":
        return _ProxyPartMaterial(key=key, color=color, map_texture=map_texture, specular=0.18, shininess=24.0)
    if key == "dark_body":
        return _ProxyPartMaterial(key=key, color=color, map_texture=map_texture, specular=0.28, shininess=36.0)
    return _ProxyPartMaterial(key=key, color=color, map_texture=map_texture, specular=0.18, shininess=24.0)


def _write_proxy_texture(
    path: Path,
    *,
    source_obj: Path,
    vertices: list[tuple[str, tuple[float, float, float]]],
    texcoords: list[str],
    faces: list[_ObjFace],
    width: int,
    height: int,
    center_x: float,
    center_z: float,
    y_min: float,
    y_span: float,
    fallback_band_colors: list[tuple[float, float, float]],
) -> bool:
    try:
        from PIL import Image
    except Exception:
        return False

    texture_image = _load_obj_diffuse_texture(source_obj) if texcoords else None
    sums_r = [0.0] * (width * height)
    sums_g = [0.0] * (width * height)
    sums_b = [0.0] * (width * height)
    counts = [0] * (width * height)

    if texture_image is not None:
        for face in faces:
            color = _face_texture_color(texture_image, texcoords, face)
            if color is None:
                continue
            center = _face_center(vertices, face)
            if center is None:
                continue
            theta = atan2(center[2] - center_z, center[0] - center_x)
            if theta < 0.0:
                theta += 2.0 * pi
            u = theta / (2.0 * pi)
            v = min(max((center[1] - y_min) / y_span, 0.0), 1.0)
            x = min(max(int(u * (width - 1)), 0), width - 1)
            y = min(max(int((1.0 - v) * (height - 1)), 0), height - 1)
            for dy in (-1, 0, 1):
                yy = y + dy
                if yy < 0 or yy >= height:
                    continue
                for dx in (-1, 0, 1):
                    xx = (x + dx) % width
                    index = yy * width + xx
                    sums_r[index] += color[0]
                    sums_g[index] += color[1]
                    sums_b[index] += color[2]
                    counts[index] += 1

    data: list[tuple[int, int, int]] = []
    for y in range(height):
        row: list[tuple[int, int, int] | None] = []
        for x in range(width):
            index = y * width + x
            if counts[index]:
                row.append(
                    (
                        int(round(sums_r[index] / counts[index])),
                        int(round(sums_g[index] / counts[index])),
                        int(round(sums_b[index] / counts[index])),
                    )
                )
            else:
                row.append(None)
        _fill_proxy_texture_row(row, _proxy_texture_row_fallback(y, height, fallback_band_colors))
        data.extend([pixel if pixel is not None else (184, 184, 184) for pixel in row])

    data = _auto_expose_proxy_texture_data(data)
    image = Image.new("RGB", (width, height))
    image.putdata(data)
    image.save(path)
    return True


def _auto_expose_proxy_texture_data(data: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    if not data:
        return data
    lumas = sorted(_rgb_luminance(pixel) for pixel in data)
    p50 = lumas[len(lumas) // 2]
    p95 = lumas[min(max(int(len(lumas) * 0.95), 0), len(lumas) - 1)]
    if p50 >= 55.0 or p95 >= 95.0:
        return data
    gain = min(max(72.0 / max(p50, 1.0), 1.0), 4.0)
    gamma = 0.72 if p50 < 35.0 else 0.82
    lift = 8.0 if p50 < 35.0 else 4.0
    exposed: list[tuple[int, int, int]] = []
    for pixel in data:
        exposed.append(tuple(_auto_expose_channel(channel, gain, gamma, lift) for channel in pixel))
    return exposed


def _rgb_luminance(pixel: tuple[int, int, int]) -> float:
    r, g, b = pixel
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _auto_expose_channel(value: int, gain: float, gamma: float, lift: float) -> int:
    normalized = min(max((value * gain) / 255.0, 0.0), 1.0)
    return min(max(int(round((normalized ** gamma) * 255.0 + lift)), 0), 255)


def _face_center(
    vertices: list[tuple[str, tuple[float, float, float]]],
    face: _ObjFace,
) -> tuple[float, float, float] | None:
    if not face.vertices:
        return None
    coords = [vertices[index - 1][1] for index in face.vertices if 0 < index <= len(vertices)]
    if not coords:
        return None
    return tuple(sum(coord[idx] for coord in coords) / len(coords) for idx in range(3))


def _face_texture_color(image, texcoords: list[str], face: _ObjFace) -> tuple[int, int, int] | None:
    samples: list[tuple[int, int, int]] = []
    width, height = image.size
    for texcoord in face.texcoords:
        if texcoord is None or texcoord <= 0 or texcoord > len(texcoords):
            continue
        parts = texcoords[texcoord - 1].split()
        if len(parts) < 3:
            continue
        u = float(parts[1])
        v = float(parts[2])
        x = min(max(int(u * (width - 1)), 0), width - 1)
        y = min(max(int((1.0 - v) * (height - 1)), 0), height - 1)
        samples.append(image.getpixel((x, y)))
    if not samples:
        return None
    return tuple(int(round(sum(pixel[idx] for pixel in samples) / len(samples))) for idx in range(3))


def _proxy_texture_row_fallback(
    y: int,
    height: int,
    band_colors: list[tuple[float, float, float]],
) -> tuple[int, int, int]:
    if not band_colors:
        return (184, 184, 184)
    v = 1.0 - (y / max(height - 1, 1))
    band = min(max(int(v * len(band_colors)), 0), len(band_colors) - 1)
    return tuple(int(round(channel * 255.0)) for channel in band_colors[band])


def _fill_proxy_texture_row(row: list[tuple[int, int, int] | None], fallback: tuple[int, int, int]) -> None:
    last: tuple[int, int, int] | None = None
    for idx, value in enumerate(row):
        if value is not None:
            last = value
        elif last is not None:
            row[idx] = last
    next_value: tuple[int, int, int] | None = None
    for idx in range(len(row) - 1, -1, -1):
        if row[idx] is not None:
            next_value = row[idx]
        elif next_value is not None:
            row[idx] = next_value
    for idx, value in enumerate(row):
        if value is None:
            row[idx] = fallback


def _write_proxy_mtl(
    path: Path,
    stem: str,
    colors: list[tuple[float, float, float]],
    *,
    texture_name: str | None = None,
    part_materials: dict[str, _ProxyPartMaterial] | None = None,
) -> None:
    part_materials = part_materials or {}
    if texture_name is not None:
        lines = [
            "# Generated MotrixSim textured preview proxy material\n",
            f"newmtl {stem}_proxy_baked\n",
            "Ka 1.000000 1.000000 1.000000\n",
            "Kd 1.000000 1.000000 1.000000\n",
            "Ks 0.120000 0.120000 0.120000\n",
            "Ns 16.000000\n",
            "d 1.000000\n",
            "illum 2\n",
            f"map_Kd {texture_name}\n",
        ]
        _append_proxy_part_materials(lines, stem, part_materials, texture_name=texture_name)
        path.write_text("".join(lines), encoding="utf-8")
        return

    lines = ["# Generated MotrixSim solid preview proxy materials\n"]
    for idx, color in enumerate(colors):
        r, g, b = color
        lines.extend(
            [
                f"newmtl {stem}_proxy_band_{idx:02d}\n",
                f"Ka {r:.6f} {g:.6f} {b:.6f}\n",
                f"Kd {r:.6f} {g:.6f} {b:.6f}\n",
                "Ks 0.120000 0.120000 0.120000\n",
                "Ns 16.000000\n",
                "d 1.000000\n",
                "illum 2\n\n",
            ]
        )
    _append_proxy_part_materials(lines, stem, part_materials, texture_name=None)
    path.write_text("".join(lines), encoding="utf-8")


def _append_proxy_part_materials(
    lines: list[str],
    stem: str,
    materials: dict[str, _ProxyPartMaterial],
    *,
    texture_name: str | None,
) -> None:
    for key in sorted(materials):
        material = materials[key]
        r, g, b = material.color
        lines.extend(
            [
                "\n",
                f"newmtl {stem}_proxy_{material.key}\n",
                f"Ka {r:.6f} {g:.6f} {b:.6f}\n",
                f"Kd {r:.6f} {g:.6f} {b:.6f}\n",
                f"Ks {material.specular:.6f} {material.specular:.6f} {material.specular:.6f}\n",
                f"Ns {material.shininess:.6f}\n",
                f"d {material.alpha:.6f}\n",
                "illum 2\n",
            ]
        )
        if texture_name is not None and material.map_texture:
            lines.append(f"map_Kd {texture_name}\n")


def _write_pymeshlab_preview_lod(path: Path, *, max_faces: int) -> bool:
    tmp_obj: Path | None = None
    tmp_mtl: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}_lod_",
            suffix=".obj",
            dir=path.parent,
            delete=False,
        ) as tmp:
            tmp_obj = Path(tmp.name)
        code = r"""
import sys
import pymeshlab  # type: ignore

source, target, max_faces = sys.argv[1], sys.argv[2], int(sys.argv[3])
meshset = pymeshlab.MeshSet()
meshset.load_new_mesh(source)
meshset.meshing_decimation_quadric_edge_collapse_with_texture(
    targetfacenum=max_faces,
    targetperc=0,
    qualitythr=0.3,
    extratcoordw=1.0,
    preserveboundary=True,
    boundaryweight=1.0,
    optimalplacement=True,
    preservenormal=True,
    planarquadric=False,
    selected=False,
)
if int(meshset.current_mesh().face_number()) > int(max_faces * 1.15):
    raise SystemExit(2)
meshset.save_current_mesh(target)
"""
        completed = subprocess.run(
            [sys.executable, "-c", code, path.as_posix(), tmp_obj.as_posix(), str(int(max_faces))],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=90,
        )
        if completed.returncode != 0:
            if tmp_obj.exists():
                tmp_obj.unlink()
            return False
        tmp_mtl = tmp_obj.with_suffix(tmp_obj.suffix + ".mtl")
        if tmp_mtl.is_file():
            final_mtl = path.with_name(f"{path.stem}_lod.mtl")
            tmp_mtl.replace(final_mtl)
            _rewrite_obj_mtllib(tmp_obj, final_mtl.name)
        tmp_obj.replace(path)
        return True
    except Exception:
        if tmp_obj is not None and tmp_obj.exists():
            tmp_obj.unlink()
        if tmp_mtl is not None and tmp_mtl.exists():
            tmp_mtl.unlink()
        return False


def _rewrite_obj_mtllib(path: Path, mtl_name: str) -> None:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    changed = False
    output: list[str] = []
    for line in lines:
        if line.startswith("mtllib ") and not changed:
            output.append(f"mtllib {mtl_name}\n")
            changed = True
        else:
            output.append(line)
    if changed:
        path.write_text("".join(output), encoding="utf-8")


def _count_obj_faces(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("f "):
                count += 1
    return count


def ensure_obj_vertex_normals(path: Path) -> dict[str, int]:
    """Add smooth vertex normals to an OBJ when face tokens do not reference them."""

    raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    parsed = _parse_obj(raw_lines)
    vertices: list[tuple[str, tuple[float, float, float]]] = parsed["vertices"]
    texcoords: list[str] = parsed["texcoords"]
    faces: list[_ObjFace] = parsed["faces"]
    if not vertices or not faces:
        return {"vertices": len(vertices), "faces": len(faces), "normals_written": 0}
    if all(all(normal is not None for normal in face.normals) for face in faces):
        return {"vertices": len(vertices), "faces": len(faces), "normals_written": 0}

    accum = [(0.0, 0.0, 0.0) for _ in range(len(vertices) + 1)]
    for face in faces:
        if len(face.vertices) < 3:
            continue
        origin = vertices[face.vertices[0] - 1][1]
        for idx in range(1, len(face.vertices) - 1):
            a = vertices[face.vertices[idx] - 1][1]
            b = vertices[face.vertices[idx + 1] - 1][1]
            normal = _cross(_sub(a, origin), _sub(b, origin))
            for vertex_index in (face.vertices[0], face.vertices[idx], face.vertices[idx + 1]):
                accum[vertex_index] = _add(accum[vertex_index], normal)

    normal_lines: list[str] = []
    for idx in range(1, len(vertices) + 1):
        nx, ny, nz = _normalise_vector(accum[idx])
        normal_lines.append(f"vn {nx:.8f} {ny:.8f} {nz:.8f}\n")

    output: list[str] = []
    header: list[str] = parsed["header"]
    if header:
        output.extend(header)
    else:
        output.append(f"o {_slug(path.stem)}\n")
    output.extend(line for line, _ in vertices)
    output.extend(texcoords)
    output.extend(normal_lines)
    current_material: str | None = None
    for face in faces:
        if face.material != current_material:
            if face.material:
                output.append(f"usemtl {face.material}\n")
            current_material = face.material
        output.append("f " + " ".join(_face_token_with_vertex_normal(face, idx) for idx in range(len(face.vertices))) + "\n")
    path.write_text("".join(output), encoding="utf-8")
    return {"vertices": len(vertices), "faces": len(faces), "normals_written": len(vertices)}


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalise_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
    if length <= 1e-12:
        return (0.0, 0.0, 1.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _face_token_with_vertex_normal(face: _ObjFace, token_index: int) -> str:
    vertex = face.vertices[token_index]
    texcoord = face.texcoords[token_index]
    normal = vertex
    if texcoord is None:
        return f"{vertex}//{normal}"
    return f"{vertex}/{texcoord}/{normal}"


def _parse_obj(lines: list[str]) -> dict[str, object]:
    header: list[str] = []
    vertices: list[tuple[str, tuple[float, float, float]]] = []
    texcoords: list[str] = []
    normals: list[str] = []
    faces: list[_ObjFace] = []
    current_material: str | None = None
    for line in lines:
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                vertices.append((line, (float(parts[1]), float(parts[2]), float(parts[3]))))
        elif line.startswith("vt "):
            texcoords.append(line)
        elif line.startswith("vn "):
            normals.append(line)
        elif line.startswith("usemtl "):
            current_material = line.split(maxsplit=1)[1].strip()
        elif line.startswith("f "):
            tokens = tuple(line.split()[1:])
            face = _parse_obj_face(tokens, len(vertices), len(texcoords), len(normals), current_material)
            if face is not None:
                faces.append(face)
        elif line.startswith(("mtllib ", "o ", "g ", "s ")):
            header.append(line)
    face_roots = _obj_face_roots(faces, len(vertices))
    return {
        "header": header,
        "vertices": vertices,
        "texcoords": texcoords,
        "normals": normals,
        "faces": faces,
        "face_roots": face_roots,
    }


def _parse_obj_face(
    tokens: tuple[str, ...],
    vertex_count: int,
    texcoord_count: int,
    normal_count: int,
    material: str | None,
) -> _ObjFace | None:
    vertices: list[int] = []
    texcoords: list[int | None] = []
    normals: list[int | None] = []
    for token in tokens:
        parts = token.split("/")
        vertex = _parse_obj_index(parts[0], vertex_count)
        if vertex is None:
            return None
        texcoord = _parse_obj_index(parts[1], texcoord_count) if len(parts) >= 2 and parts[1] else None
        normal = _parse_obj_index(parts[2], normal_count) if len(parts) >= 3 and parts[2] else None
        vertices.append(vertex)
        texcoords.append(texcoord)
        normals.append(normal)
    return _ObjFace(tokens=tokens, vertices=tuple(vertices), texcoords=tuple(texcoords), normals=tuple(normals), material=material)


def _select_lod_faces(faces: list[_ObjFace], vertex_count: int, *, max_faces: int) -> list[_ObjFace]:
    components = _obj_face_components(faces, vertex_count)
    component_faces = sorted(components.values(), key=len, reverse=True)
    if len(component_faces) >= max_faces:
        return [faces[indices[0]] for indices in component_faces[:max_faces]]

    allocations = [1] * len(component_faces)
    remaining = max_faces - len(component_faces)
    weights = [max(len(indices) - 1, 0) for indices in component_faces]
    total_weight = sum(weights)
    if remaining > 0 and total_weight > 0:
        fractional: list[tuple[float, int]] = []
        for idx, weight in enumerate(weights):
            exact = remaining * weight / total_weight
            extra = int(exact)
            allocations[idx] += extra
            fractional.append((exact - extra, idx))
        assigned = sum(allocations)
        for _, idx in sorted(fractional, reverse=True):
            if assigned >= max_faces:
                break
            allocations[idx] += 1
            assigned += 1

    selected_indices: list[int] = []
    for indices, count in zip(component_faces, allocations):
        selected_indices.extend(_evenly_spaced_indices(indices, min(count, len(indices))))
    return [faces[idx] for idx in sorted(selected_indices[:max_faces])]


def _evenly_spaced_indices(values: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= len(values):
        return values
    if count == 1:
        return [values[len(values) // 2]]
    last = len(values) - 1
    selected = []
    for idx in range(count):
        selected.append(values[round(idx * last / (count - 1))])
    return selected


def _parse_obj_index(raw: str, count: int) -> int | None:
    try:
        value = int(raw)
    except ValueError:
        return None
    if value < 0:
        value = count + value + 1
    if value <= 0 or value > count:
        return None
    return value


def _obj_face_roots(faces: list[_ObjFace], vertex_count: int) -> list[int]:
    components = _obj_face_components(faces, vertex_count)
    root_by_face: dict[int, int] = {}
    for root, face_indices in components.items():
        for face_index in face_indices:
            root_by_face[face_index] = root
    return [root_by_face[idx] for idx in range(len(faces))]


def _obj_face_components(faces: list[_ObjFace], vertex_count: int) -> dict[int, list[int]]:
    parent = list(range(vertex_count + 1))
    size = [1] * (vertex_count + 1)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if size[root_a] < size[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        size[root_a] += size[root_b]

    for face in faces:
        first = face.vertices[0]
        for vertex in face.vertices[1:]:
            union(first, vertex)

    components: dict[int, list[int]] = {}
    for face_index, face in enumerate(faces):
        components.setdefault(find(face.vertices[0]), []).append(face_index)
    return components


def _vertex_bounds(
    vertices: list[tuple[str, tuple[float, float, float]]],
    vertex_indices: list[int],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if not vertex_indices:
        return None
    coords = [vertices[index - 1][1] for index in vertex_indices]
    mins = tuple(min(coord[idx] for coord in coords) for idx in range(3))
    maxs = tuple(max(coord[idx] for coord in coords) for idx in range(3))
    return mins, maxs


def _write_compacted_obj(
    path: Path,
    header: list[str],
    vertices: list[tuple[str, tuple[float, float, float]]],
    texcoords: list[str],
    normals: list[str],
    faces: list[_ObjFace],
) -> None:
    used_vertices = sorted({vertex for face in faces for vertex in face.vertices})
    used_texcoords = sorted({texcoord for face in faces for texcoord in face.texcoords if texcoord is not None})
    used_normals = sorted({normal for face in faces for normal in face.normals if normal is not None})
    vertex_map = {old: new for new, old in enumerate(used_vertices, start=1)}
    texcoord_map = {old: new for new, old in enumerate(used_texcoords, start=1)}
    normal_map = {old: new for new, old in enumerate(used_normals, start=1)}

    output: list[str] = []
    if header:
        output.extend(header)
    else:
        output.append(f"o {_slug(path.stem)}\n")
    for old_index in used_vertices:
        output.append(vertices[old_index - 1][0])
    for old_index in used_texcoords:
        output.append(texcoords[old_index - 1])
    for old_index in used_normals:
        output.append(normals[old_index - 1])
    current_material: str | None = None
    for face in faces:
        if face.material != current_material:
            if face.material:
                output.append(f"usemtl {face.material}\n")
            current_material = face.material
        output.append("f " + " ".join(_rewrite_face_token(face, idx, vertex_map, texcoord_map, normal_map) for idx in range(len(face.vertices))) + "\n")
    path.write_text("".join(output), encoding="utf-8")


def _rewrite_face_token(
    face: _ObjFace,
    token_index: int,
    vertex_map: dict[int, int],
    texcoord_map: dict[int, int],
    normal_map: dict[int, int],
) -> str:
    vertex = str(vertex_map[face.vertices[token_index]])
    texcoord = face.texcoords[token_index]
    normal = face.normals[token_index]
    if texcoord is None and normal is None:
        return vertex
    if normal is None:
        return f"{vertex}/{texcoord_map[texcoord]}" if texcoord is not None else vertex
    if texcoord is None:
        return f"{vertex}//{normal_map[normal]}"
    return f"{vertex}/{texcoord_map[texcoord]}/{normal_map[normal]}"


def materialize_obj_materials(path: Path, *, color_levels: int = 6) -> Path | None:
    """Convert OBJ vertex colors into simple MTL face materials.

    SAM-3D meshes are exported with non-standard ``v x y z r g b`` vertex
    colors. Some mesh viewers use those colors directly, but Motrix currently
    reports zero loaded materials for such OBJ files. This helper preserves the
    original OBJ geometry and adds a small MTL palette so Motrix has explicit
    materials to load.
    """

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    if not lines or any(line.startswith("mtllib ") for line in lines):
        return None

    vertex_colors: dict[int, tuple[float, float, float]] = {}
    vertex_count = 0
    for line in lines:
        if not line.startswith("v "):
            continue
        vertex_count += 1
        parts = line.split()
        if len(parts) >= 7:
            vertex_colors[vertex_count] = _normalise_rgb(parts[4:7])

    face_materials: dict[int, str] = {}
    materials: dict[str, tuple[float, float, float]] = {}
    default_name = f"{_slug(path.stem)}_mat_default"
    default_color = (0.72, 0.72, 0.72)

    for idx, line in enumerate(lines):
        if not line.startswith("f "):
            continue
        color = _face_average_color(line, vertex_colors, vertex_count) or default_color
        name, quantized = _quantized_material(path.stem, color, color_levels)
        face_materials[idx] = name
        materials.setdefault(name, quantized)

    if not face_materials:
        return None
    if not materials:
        materials[default_name] = default_color

    mtl_path = path.with_suffix(".mtl")
    _write_mtl(mtl_path, materials)

    output = [f"mtllib {mtl_path.name}\n"]
    current_material = None
    for idx, line in enumerate(lines):
        material = face_materials.get(idx)
        if material is not None and material != current_material:
            output.append(f"usemtl {material}\n")
            current_material = material
        output.append(line)
    path.write_text("".join(output), encoding="utf-8")
    return mtl_path


def auto_expose_obj_diffuse_textures(path: Path) -> dict[str, int]:
    """Brighten very dark baked diffuse textures referenced by an OBJ.

    SAM-3D baked textures for glossy black objects can be near-black under
    MotrixSim's default preview lighting. The adjustment is gated by luminance
    percentiles, so normal-bright textures are left unchanged.
    """

    changed = 0
    checked = 0
    for texture in _iter_obj_diffuse_textures(path):
        checked += 1
        if _auto_expose_texture_file(texture):
            changed += 1
    return {"textures_checked": checked, "textures_changed": changed}


def _iter_obj_diffuse_textures(path: Path) -> list[Path]:
    textures: list[Path] = []
    seen: set[Path] = set()
    obj_dir = path.parent
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("mtllib "):
            continue
        for raw_mtl in line.split()[1:]:
            mtl_path = obj_dir / raw_mtl
            if not mtl_path.is_file():
                continue
            for mtl_line in mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = mtl_line.split()
                if len(parts) < 2 or parts[0] != "map_Kd":
                    continue
                texture = (mtl_path.parent / parts[-1]).resolve()
                if texture.is_file() and texture not in seen:
                    textures.append(texture)
                    seen.add(texture)
    return textures


def _auto_expose_texture_file(path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path).convert("RGB") as image:
            pixel_data = getattr(image, "get_flattened_data", image.getdata)
            data = list(pixel_data())
            exposed = _auto_expose_proxy_texture_data(data)
            if exposed == data:
                return False
            output = Image.new("RGB", image.size)
            output.putdata(exposed)
            output.save(path)
            return True
    except Exception:
        return False


def copy_obj_material_sidecars(source_obj: Path, packaged_obj: Path) -> None:
    lines = packaged_obj.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    output: list[str] = []
    changed = False
    for line in lines:
        if not line.startswith("mtllib "):
            output.append(line)
            continue
        names = line.split()[1:]
        copied_names = []
        for raw_name in names:
            src = (source_obj.parent / raw_name).resolve()
            if not src.is_file():
                copied_names.append(raw_name)
                continue
            dst_name = namespaced_sidecar_name(packaged_obj.stem, raw_name)
            dst = packaged_obj.parent / dst_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            copy_mtl_with_texture_sidecars(src, dst, namespace=packaged_obj.stem)
            copied_names.append(dst_name)
            changed = True
        output.append(f"mtllib {' '.join(copied_names)}\n")
    if changed:
        packaged_obj.write_text("".join(output), encoding="utf-8")


def copy_mtl_with_texture_sidecars(source_mtl: Path, packaged_mtl: Path, *, namespace: str) -> None:
    texture_keys = {
        "map_Ka",
        "map_Kd",
        "map_Ks",
        "map_Ke",
        "map_Ns",
        "map_d",
        "map_bump",
        "bump",
        "disp",
        "decal",
        "norm",
    }
    output: list[str] = []
    for line in source_mtl.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True):
        parts = line.split()
        if len(parts) < 2 or parts[0] not in texture_keys:
            output.append(line)
            continue
        raw_name = parts[-1]
        if raw_name.startswith("-"):
            output.append(line)
            continue
        src = (source_mtl.parent / raw_name).resolve()
        if not src.is_file():
            output.append(line)
            continue
        dst_name = namespaced_sidecar_name(namespace, raw_name)
        dst = packaged_mtl.parent / dst_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        parts[-1] = dst_name
        output.append(" ".join(parts) + "\n")
    packaged_mtl.write_text("".join(output), encoding="utf-8")


def namespaced_sidecar_name(namespace: str, raw_name: str) -> str:
    name = Path(raw_name).name
    if name.startswith(f"{namespace}_"):
        return name
    return f"{namespace}_{name}"


def _face_average_color(
    face_line: str,
    vertex_colors: dict[int, tuple[float, float, float]],
    vertex_count: int,
) -> tuple[float, float, float] | None:
    colors = []
    for token in face_line.split()[1:]:
        raw_index = token.split("/", 1)[0]
        if not raw_index:
            continue
        try:
            vertex_index = int(raw_index)
        except ValueError:
            continue
        if vertex_index < 0:
            vertex_index = vertex_count + vertex_index + 1
        color = vertex_colors.get(vertex_index)
        if color is not None:
            colors.append(color)
    if not colors:
        return None
    return tuple(sum(color[channel] for color in colors) / len(colors) for channel in range(3))


def _normalise_rgb(values: list[str]) -> tuple[float, float, float]:
    rgb = [float(value) for value in values[:3]]
    if max(rgb) > 1.0:
        rgb = [value / 255.0 for value in rgb]
    return tuple(min(max(value, 0.0), 1.0) for value in rgb)


def _quantized_material(
    stem: str,
    color: tuple[float, float, float],
    levels: int,
) -> tuple[str, tuple[float, float, float]]:
    levels = max(int(levels), 2)
    quantized = tuple(round(channel * (levels - 1)) / (levels - 1) for channel in color)
    suffix = "_".join(f"{int(round(channel * 255)):02x}" for channel in quantized)
    return f"{_slug(stem)}_mat_{suffix}", quantized


def _write_mtl(path: Path, materials: dict[str, tuple[float, float, float]]) -> None:
    lines = ["# Generated from OBJ vertex colors for MotrixSim preview\n"]
    for name, color in sorted(materials.items()):
        r, g, b = color
        lines.extend(
            [
                f"newmtl {name}\n",
                f"Ka {r:.6f} {g:.6f} {b:.6f}\n",
                f"Kd {r:.6f} {g:.6f} {b:.6f}\n",
                "Ks 0.050000 0.050000 0.050000\n",
                "Ns 12.000000\n",
                "d 1.000000\n",
                "illum 2\n",
                "\n",
            ]
        )
    path.write_text("".join(lines), encoding="utf-8")


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return slug or "asset"


def _relative_posix(path: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base_dir.resolve())).as_posix()


def _vec(values: Iterable[float]) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def _rgba(color: Iterable[float], alpha: float = 1.0) -> str:
    return _vec([*color, alpha])


def _is_unit_scale(values: Iterable[float]) -> bool:
    return all(abs(float(v) - 1.0) < 1e-8 for v in values)


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent
