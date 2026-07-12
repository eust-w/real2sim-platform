#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path
from typing import Any

from acceptance_poc_cluster import (
    DEFAULT_KUBECONFIG,
    NAMESPACE,
    AcceptanceError,
    AcceptanceRun,
    ApiClient,
    authenticate_admin,
    finalize,
    free_port,
    kubectl,
    start_port_forward,
    upload_with_resume,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = Path("/Users/d-robotics/Downloads/demo_2.zip")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_for_checksum(run: AcceptanceRun, checksum: str) -> dict[str, Any] | None:
    for dataset in run.client.get("/api/datasets"):
        if str(dataset.get("checksum") or "").lower() == checksum.lower():
            return dataset
    return None


def verify_spatial_dataset(run: AcceptanceRun, dataset: dict[str, Any], expected_checksum: str) -> None:
    scan = dataset.get("scan") or {}
    run.check("空间包 SHA-256 与原文件一致", dataset.get("checksum") == expected_checksum, dataset.get("checksum"))
    run.check("识别为 MotrixSim 空间包", scan.get("input_kind") == "motrixsim_spatial_bundle", scan)
    run.check("USDZ 碰撞、视觉和 3DGS 三层完整", bool(scan.get("spatial_bundle_ready")), scan)
    run.check("原始空间包已持久化到受控 BOS 前缀", str(dataset.get("storage_uri") or "").startswith("bos://testxyz/real2sim-poc/uploads/"), dataset.get("storage_uri"))


def import_scene(run: AcceptanceRun, dataset_id: str) -> dict[str, Any]:
    created = run.client.post(f"/api/datasets/{dataset_id}/import-motrixsim")
    task = run.wait_task(
        created["id"],
        required_artifacts=(
            "scene_manifest.json",
            "motrixsim_scene_import_report.json",
            "runtime_validation.json",
            "scene.xml",
            "environment_visual.glb",
            "full_res.spz",
        ),
        timeout_seconds=7200,
    )
    result = task.get("result") or {}
    runtime = result.get("runtime_validation") or {}
    run.check("MuJoCo 真实加载并步进导入场景", bool((runtime.get("mujoco") or {}).get("ok")), runtime)
    run.check("MotrixSim 真实加载并步进导入场景", bool((runtime.get("motrixsim") or {}).get("ok")), runtime)
    scene_id = str(result.get("scene_id") or "")
    run.check("空间包已登记为平台场景", bool(scene_id), result)
    manifest = run.client.get(f"/api/scenes/{scene_id}/manifest")
    formats = {str(item.get("format") or "").lower() for item in manifest.get("assets", [])}
    names = {str(item.get("name") or "") for item in manifest.get("assets", [])}
    run.check("场景保留 SPZ 高保真外观层", "spz" in formats or "full_res.spz" in names, sorted(names))
    run.check("场景生成可交互 GLB 回退预览", "glb" in formats or "environment_visual.glb" in names, sorted(names))
    run.report["scenes"].append({"id": scene_id, "role": "spatial_bundle_import", "manifest": manifest})
    return {"scene_id": scene_id, "manifest": manifest, "task": task}


def add_asset_revision(
    run: AcceptanceRun,
    *,
    scene_id: str,
    primitive: str,
    name: str,
    translation: list[float],
    dimensions: list[float],
    dynamic: bool,
) -> dict[str, Any]:
    created = run.client.post(
        f"/api/scenes/{scene_id}/assets",
        {
            "source": {"type": "primitive"},
            "primitive": primitive,
            "name": name,
            "transform": {
                "translation": translation,
                "rotation_euler_degrees": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            },
            "physics": {
                "dynamic": dynamic,
                "mass": 0.35 if dynamic else 1.0,
                "collision_mode": "visual_and_collision",
                "dimensions": dimensions,
                "friction": [0.8, 0.01, 0.001],
                "rgba": [0.12, 0.78, 0.68, 1.0],
            },
            "validation_steps": 120,
            "revision_name": f"空间场景 · {name}",
        },
    )
    task_id = str((created.get("task") or {}).get("id") or "")
    run.check(f"{name} 合成任务已创建", bool(task_id), created)
    task = run.wait_task(
        task_id,
        required_artifacts=("scene_manifest.json", "scene.xml"),
        timeout_seconds=7200,
    )
    result = task.get("result") or {}
    runtime = result.get("runtime_validation") or {}
    run.check(f"{name} 修订通过 MuJoCo", bool((runtime.get("mujoco") or {}).get("ok")), runtime)
    run.check(f"{name} 修订通过 MotrixSim", bool((runtime.get("motrixsim") or {}).get("ok")), runtime)
    if dynamic:
        run.check(f"{name} 在 MotrixSim 中具有真实自由关节", int((runtime.get("motrixsim") or {}).get("nq") or 0) >= 7, runtime)
    revision_id = str(result.get("scene_id") or "")
    run.check(f"{name} 生成独立场景版本", bool(revision_id) and revision_id != scene_id, result)
    run.check(
        f"{name} 使用修订包内独立运行时文件",
        "/scene_asset_compose/" in str(result.get("scene_path") or ""),
        result.get("scene_path"),
    )
    manifest = run.client.get(f"/api/scenes/{revision_id}/manifest")
    added = [item for item in manifest.get("assets", []) if item.get("role") == "added_asset"]
    run.check(f"{name} 已写入场景清单", any(item.get("name") == name for item in added), added)
    run.report["scenes"].append(
        {
            "id": revision_id,
            "role": f"asset_revision_{primitive}",
            "parent_scene_id": scene_id,
            "added_assets": [item.get("name") for item in added],
            "runtime_validation": runtime,
        }
    )
    return {"scene_id": revision_id, "manifest": manifest, "task": task}


def run_acceptance(args: argparse.Namespace) -> int:
    bundle = args.bundle.expanduser().resolve()
    if not bundle.is_file():
        raise AcceptanceError(f"spatial bundle does not exist: {bundle}")
    checksum = sha256_file(bundle)
    build_id = args.build_id or kubectl(
        args.kubeconfig,
        "-n",
        NAMESPACE,
        "get",
        "deployment/real2sim-api",
        "-o",
        "jsonpath={.metadata.labels.app\\.kubernetes\\.io/version}",
    ).stdout.strip()
    if not build_id:
        raise AcceptanceError("unable to determine deployed build id")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.out or ROOT / "acceptance_runs" / "poc" / f"{build_id}-spatial-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    port = args.port or free_port()
    forward = start_port_forward(args.kubeconfig, port, out_dir)
    run = AcceptanceRun(
        client=ApiClient(f"https://127.0.0.1:{port}"),
        out_dir=out_dir,
        kubeconfig=args.kubeconfig,
        build_id=build_id,
        timeout_seconds=args.timeout_seconds,
    )
    run.report["schema"] = "real2sim.spatial_bundle_acceptance.v1"
    run.report["source"] = {
        "path": str(bundle),
        "filename": bundle.name,
        "size_bytes": bundle.stat().st_size,
        "sha256": checksum,
    }
    try:
        if run.stage("管理员认证", lambda: authenticate_admin(run)) is None:
            return finalize(run)
        version = run.client.get("/api/version")
        doctor = run.client.get("/api/doctor")
        run.report["version"] = version
        run.report["doctor"] = doctor
        run.check("部署版本与验收目标一致", version.get("build_id") == build_id, version)
        run.check("生产运行时 doctor 通过", bool(doctor.get("ok")), doctor)

        dataset = dataset_for_checksum(run, checksum) if args.reuse else None
        if dataset is None:
            dataset = run.stage("真实空间包分片上传", lambda: upload_with_resume(run, bundle, checksum))
        else:
            run.report["warnings"].append(f"Reused dataset {dataset['id']} with identical SHA-256.")
        if dataset is None:
            return finalize(run)
        verify_spatial_dataset(run, dataset, checksum)

        imported = run.stage("USDZ/SPZ 转 MotrixSim 场景", lambda: import_scene(run, dataset["id"]))
        if not imported:
            return finalize(run)
        cube = run.stage(
            "添加动态抓取方块并生成修订",
            lambda: add_asset_revision(
                run,
                scene_id=imported["scene_id"],
                primitive="box",
                name="动态抓取方块",
                translation=[0.0, 0.0, 1.0],
                dimensions=[0.6, 0.6, 0.6],
                dynamic=True,
            ),
        )
        if not cube:
            return finalize(run)
        marker = run.stage(
            "添加静态目标圆柱并保留上一资产",
            lambda: add_asset_revision(
                run,
                scene_id=cube["scene_id"],
                primitive="cylinder",
                name="静态放置目标",
                translation=[1.2, 0.0, 0.6],
                dimensions=[0.8, 0.8, 1.2],
                dynamic=False,
            ),
        )
        if marker:
            added_names = {
                str(item.get("name") or "")
                for item in marker["manifest"].get("assets", [])
                if item.get("role") == "added_asset"
            }
            run.check("连续场景修订保留全部数字资产", {"动态抓取方块", "静态放置目标"}.issubset(added_names), sorted(added_names))
            parent_after = run.client.get(f"/api/scenes/{imported['scene_id']}/manifest")
            parent_added = [item.get("name") for item in parent_after.get("assets", []) if item.get("role") == "added_asset"]
            first_after = run.client.get(f"/api/scenes/{cube['scene_id']}/manifest")
            first_added = [item.get("name") for item in first_after.get("assets", []) if item.get("role") == "added_asset"]
            run.check("父场景未被资产修订污染", not parent_added, parent_added)
            run.check("第一版修订只包含首个新增资产", first_added == ["动态抓取方块"], first_added)
            run.report["browser"]["target_scene_id"] = marker["scene_id"]
            run.report["browser"]["viewer_url"] = f"/static/scene-viewer.html?scene_id={marker['scene_id']}"
        return finalize(run)
    finally:
        forward.terminate()
        try:
            forward.wait(timeout=10)
        except Exception:
            forward.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a real USDZ/SPZ spatial bundle, validate MotrixSim, and compose digital assets in the isolated POC namespace.")
    parser.add_argument("--bundle", type=Path, default=Path(os.environ.get("REAL2SIM_SPATIAL_BUNDLE", DEFAULT_BUNDLE)))
    parser.add_argument("--kubeconfig", type=Path, default=Path(os.environ.get("REAL2SIM_POC_KUBECONFIG", DEFAULT_KUBECONFIG)))
    parser.add_argument("--build-id", default=os.environ.get("REAL2SIM_POC_BUILD_ID", ""))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--no-reuse", dest="reuse", action="store_false")
    parser.set_defaults(reuse=True)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run_acceptance(parse_args()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"spatial bundle acceptance failed: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(2)
