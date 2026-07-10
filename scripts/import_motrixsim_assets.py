#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent
SRC_ROOT = APP_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from real2sim_platform.config import ensure_workspace, load_config  # noqa: E402
from real2sim_platform.store import PlatformStore  # noqa: E402


DEFAULT_SOURCE = Path("/Users/d-robotics/workSpace/motrixsim-docs")
PACKAGE_SLUG = "parlour_rm65_pick_place"
SCENE_ID = hashlib.sha1(f"motrixsim:{PACKAGE_SLUG}".encode("utf-8")).hexdigest()[:32]


REQUIRED_FILES = {
    "parlour_scene": Path("examples/assets/parlour/parlour.xml"),
    "rm65_robot": Path("examples/assets/rm65_open_cabinet/mjx_rm65.xml"),
    "rm65_reference_scene": Path("examples/assets/rm65_open_cabinet/scene.xml"),
    "rm65_reference_controller": Path("examples/control/rm65_open_cabinet.py"),
    "pick_place_template": Path("examples/assets/franka_emika_panda/scene_pick_cube.xml"),
    "parlour_locomotion_example": Path("examples/control/robot_locomotion.py"),
}

DIRECTORIES_TO_COPY = {
    "parlour": Path("examples/assets/parlour"),
    "rm65_open_cabinet": Path("examples/assets/rm65_open_cabinet"),
    "franka_emika_panda": Path("examples/assets/franka_emika_panda"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Import MotrixSim parlour + RM65 pick-place scaffold into Real2Sim Platform.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Local MotrixSim docs repository path.")
    parser.add_argument("--name", default="MotrixSim 客厅 RM65 Pick-Place", help="Scene name shown in Real2Sim.")
    parser.add_argument("--replace-index", action="store_true", help="Replace the existing deterministic scene index row.")
    parser.add_argument("--skip-copy", action="store_true", help="Only refresh manifest/index using existing package files.")
    parser.add_argument("--allow-missing-source", action="store_true", help="Allow registering an already-copied package when the MotrixSim repo is not present.")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    config = load_config()
    ensure_workspace(config)
    store = PlatformStore(config.db_path)
    package_dir = config.workspace_dir / "motrixsim_assets" / PACKAGE_SLUG
    if source.is_dir():
        _assert_source(source)
        source_commit = _git_commit(source)
    elif args.skip_copy and args.allow_missing_source:
        source_commit = _existing_source_commit(package_dir) or ""
    else:
        raise SystemExit(f"MotrixSim docs repo not found: {source}")

    if not args.skip_copy:
        _copy_package(source, package_dir)
    _assert_package(package_dir)

    scene_descriptor = package_dir / f"{PACKAGE_SLUG}.scene.json"
    manifest_path = package_dir / "motrixsim_scene_manifest.json"
    scene_descriptor.write_text(
        json.dumps(
            {
                "schema": "real2sim.scene_descriptor.v1",
                "name": args.name,
                "package": str(package_dir),
                "manifest_path": str(manifest_path),
                "simulator": "motrixsim",
                "status": "asset_imported_adapter_pending",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = _register_file_artifacts(store, package_dir)
    manifest = _build_manifest(
        name=args.name,
        source=source,
        source_commit=source_commit,
        package_dir=package_dir,
        scene_descriptor=scene_descriptor,
        artifact_urls=artifacts,
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    scene = _upsert_scene(
        store=store,
        name=args.name,
        package_dir=package_dir,
        scene_descriptor=scene_descriptor,
        manifest_path=manifest_path,
        manifest=manifest,
        replace_index=args.replace_index,
    )

    print(json.dumps({"ok": True, "scene_id": scene["id"], "scene_name": scene["name"], "package_dir": str(package_dir), "manifest_path": str(manifest_path)}, ensure_ascii=False, indent=2))
    return 0


def _assert_source(source: Path) -> None:
    if not source.is_dir():
        raise SystemExit(f"MotrixSim docs repo not found: {source}")
    missing = [str(source / rel) for rel in REQUIRED_FILES.values() if not (source / rel).is_file()]
    missing.extend(str(source / rel) for rel in DIRECTORIES_TO_COPY.values() if not (source / rel).is_dir())
    if missing:
        raise SystemExit("Missing required MotrixSim assets:\n" + "\n".join(missing))


def _copy_package(source: Path, package_dir: Path) -> None:
    if package_dir.exists():
        shutil.rmtree(package_dir)
    (package_dir / "assets").mkdir(parents=True, exist_ok=True)
    (package_dir / "examples" / "control").mkdir(parents=True, exist_ok=True)
    for name, rel in DIRECTORIES_TO_COPY.items():
        shutil.copytree(source / rel, package_dir / "assets" / name)
    for rel in [Path("examples/control/rm65_open_cabinet.py"), Path("examples/control/robot_locomotion.py")]:
        shutil.copy2(source / rel, package_dir / "examples" / "control" / rel.name)


def _assert_package(package_dir: Path) -> None:
    required = [
        package_dir / "assets/parlour/parlour.xml",
        package_dir / "assets/rm65_open_cabinet/mjx_rm65.xml",
        package_dir / "assets/rm65_open_cabinet/scene.xml",
        package_dir / "assets/franka_emika_panda/scene_pick_cube.xml",
        package_dir / "examples/control/rm65_open_cabinet.py",
        package_dir / "examples/control/robot_locomotion.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Package is incomplete:\n" + "\n".join(missing))


def _register_file_artifacts(store: PlatformStore, package_dir: Path) -> dict[str, str]:
    files = {
        "scene_descriptor": package_dir / f"{PACKAGE_SLUG}.scene.json",
        "parlour_scene": package_dir / "assets/parlour/parlour.xml",
        "rm65_robot": package_dir / "assets/rm65_open_cabinet/mjx_rm65.xml",
        "rm65_reference_scene": package_dir / "assets/rm65_open_cabinet/scene.xml",
        "rm65_controller": package_dir / "examples/control/rm65_open_cabinet.py",
        "pick_place_template": package_dir / "assets/franka_emika_panda/scene_pick_cube.xml",
        "franka_preview": package_dir / "assets/franka_emika_panda/panda.png",
    }
    urls: dict[str, str] = {}
    for role, path in files.items():
        if not path.is_file():
            continue
        artifact = store.add_artifact(task_id=None, name=path.name, path=path, kind=_artifact_kind(path))
        urls[role] = f"/api/artifacts/{artifact['id']}/content"
    return urls


def _build_manifest(
    *,
    name: str,
    source: Path,
    source_commit: str,
    package_dir: Path,
    scene_descriptor: Path,
    artifact_urls: dict[str, str],
) -> dict[str, Any]:
    ready_assets = [
        _asset(package_dir, "assets/parlour/parlour.xml", "parlour_scene", "simulator", "MotrixSim parlour 客厅场景", artifact_urls.get("parlour_scene")),
        _asset(package_dir, "assets/rm65_open_cabinet/mjx_rm65.xml", "rm65_robot", "robot", "RM65 机械臂 MJCF", artifact_urls.get("rm65_robot")),
        _asset(package_dir, "assets/rm65_open_cabinet/scene.xml", "rm65_reference_scene", "simulator", "RM65 开柜门参考场景", artifact_urls.get("rm65_reference_scene")),
        _asset(package_dir, "examples/control/rm65_open_cabinet.py", "rm65_controller", "controller", "RM65 参考控制脚本", artifact_urls.get("rm65_controller")),
        _asset(package_dir, "assets/franka_emika_panda/scene_pick_cube.xml", "pick_place_template", "simulator", "Franka pick-cube 任务模板", artifact_urls.get("pick_place_template")),
    ]
    preview = _asset(package_dir, "assets/franka_emika_panda/panda.png", "reference_preview", "image", "机械臂参考预览图", artifact_urls.get("franka_preview"))
    assets = [preview, *ready_assets] if preview["exists"] else ready_assets
    total_size = _directory_size(package_dir)
    return {
        "schema": "real2sim.motrixsim_asset_package.v1",
        "name": name,
        "status": "asset_imported_adapter_pending",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package_dir": str(package_dir),
        "scene_descriptor": str(scene_descriptor),
        "source": {
            "repo": "https://github.com/Motphys/motrixsim-docs",
            "docs": "https://motrixsim.readthedocs.io/zh-cn/latest/user_guide/overview/examples.html",
            "local_path": str(source),
            "commit": source_commit,
        },
        "ready_assets": ready_assets,
        "assets": assets,
        "adapter_plan": {
            "goal": "将 parlour 客厅场景中的人形机器人示例替换为 RM65 机械臂，并在茶几区域完成 pick-place。",
            "base_scene": "assets/parlour/parlour.xml",
            "robot_asset": "assets/rm65_open_cabinet/mjx_rm65.xml",
            "reference_controller": "examples/control/rm65_open_cabinet.py",
            "pick_place_template": "assets/franka_emika_panda/scene_pick_cube.xml",
            "recommended_steps": [
                "在 parlour 中定位茶几 mesh/collision，并确定表面高度与可达工作区。",
                "把 RM65 base 固定到茶几旁或茶几边缘，完成 z-up 坐标、尺度和碰撞组检查。",
                "把 Franka pick-cube 的目标物、home/grasp/lift 阶段迁移到 RM65 关节空间。",
                "将 RM65 开柜门示例中的 7 维动作和 gripper 控制改成 cube pick-place controller。",
                "生成 combined MJCF 后再进入 simulator_validation runtime_required。",
            ],
        },
        "runbook": {
            "source_parlour_demo": "uv run examples/control/robot_locomotion.py --robot g1 --scene parlour",
            "source_rm65_demo": "uv run examples/control/rm65_open_cabinet.py",
            "source_pick_template": "uv run examples/bench/grasp/shake_test_mx.py --object cube",
        },
        "known_gaps": [
            "当前包是资产接入 scaffold，不是可直接运行的 combined MJCF。",
            "尚未完成 parlour 茶几 mesh 的语义定位和 RM65 base 位姿标定。",
            "尚未把 Franka pick-cube keyframe/controller 迁移到 RM65。",
        ],
        "metrics": {
            "asset_count": sum(1 for path in package_dir.rglob("*") if path.is_file()),
            "total_size_bytes": total_size,
            "copied_directories": sorted(DIRECTORIES_TO_COPY),
        },
    }


def _asset(package_dir: Path, rel: str, role: str, kind: str, name: str, url: str | None = None) -> dict[str, Any]:
    path = package_dir / rel
    return {
        "name": name,
        "role": role,
        "kind": kind,
        "path": str(path),
        "relative_path": rel,
        "url": url or "",
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def _upsert_scene(
    *,
    store: PlatformStore,
    name: str,
    package_dir: Path,
    scene_descriptor: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    replace_index: bool,
) -> dict[str, Any]:
    try:
        existing = store.get_scene(SCENE_ID)
    except KeyError:
        existing = None
    if existing and not replace_index:
        store.update_scene(SCENE_ID, name=name, tags=_tags(), version=str(manifest["source"]["commit"]))
        return store.get_scene(SCENE_ID)
    if existing and replace_index:
        store.delete_scene(SCENE_ID)
    return store.create_scene(
        item_id=SCENE_ID,
        name=name,
        path=scene_descriptor,
        metadata={
            "package": str(package_dir),
            "manifest_path": str(manifest_path),
            "simulator": "motrixsim",
            "status": "asset_imported_adapter_pending",
            "adapter_pending": True,
        },
        source_type="motrixsim_docs",
        source_uri="https://github.com/Motphys/motrixsim-docs",
        version=str(manifest["source"]["commit"]),
        manifest_path=manifest_path,
        metrics=manifest["metrics"],
        tags=_tags(),
    )


def _tags() -> list[str]:
    return ["motrixsim", "parlour", "rm65", "pick-place", "adapter-pending"]


def _git_commit(source: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _existing_source_commit(package_dir: Path) -> str:
    manifest_path = package_dir / "motrixsim_scene_manifest.json"
    if not manifest_path.is_file():
        return ""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    return str((manifest.get("source") or {}).get("commit") or "")


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    if suffix == ".xml":
        return "simulator"
    if suffix == ".py":
        return "controller"
    if suffix == ".json":
        return "metadata"
    return suffix.lstrip(".") or "file"


if __name__ == "__main__":
    raise SystemExit(main())
