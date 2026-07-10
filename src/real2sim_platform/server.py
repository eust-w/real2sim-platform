from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .adapters import SPARK_VENDOR_PATH, collect_task_reports, enrich_scene_manifest_assets, run_doctor, scan_dataset
from .config import PlatformConfig, ensure_workspace, is_path_allowed, load_config, write_default_config
from .store import PlatformStore
from .worker import TaskWorker


class ImportPathRequest(BaseModel):
    path: str
    name: str | None = None


class ImportSceneRequest(BaseModel):
    path: str
    name: str | None = None
    metadata: dict[str, Any] = {}


class SceneAssetRequest(BaseModel):
    path: str
    name: str | None = None
    role: str = "added_asset"
    kind: str | None = None
    transform: dict[str, Any] = {}


class CreateTaskRequest(BaseModel):
    type: str
    params: dict[str, Any] = {}


class UpdateEntityRequest(BaseModel):
    name: str | None = None
    tags: list[str] | None = None
    version: str | None = None


class UpdateTaskRequest(BaseModel):
    params: dict[str, Any]


class BulkActionRequest(BaseModel):
    action: str
    ids: list[str]
    params: dict[str, Any] = {}


def create_app(config: PlatformConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    ensure_workspace(cfg)
    store = PlatformStore(cfg.db_path)
    worker = TaskWorker(store, cfg)
    app = FastAPI(title="Real2Sim Platform", version="0.1.0")
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.mount("/datasets", StaticFiles(directory=cfg.datasets_dir), name="datasets")

    @app.on_event("startup")
    def _startup() -> None:
        store.recover_interrupted_tasks()
        worker.start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        worker.stop()

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "workspace": str(cfg.workspace_dir)}

    @app.get("/api/version")
    def version() -> dict[str, Any]:
        spark_available = SPARK_VENDOR_PATH.is_file()
        return {
            "version": __version__,
            "build_time": os.environ.get("REAL2SIM_PLATFORM_BUILD_TIME", "dev"),
            "app_root": str(cfg.app_root),
            "workspace": str(cfg.workspace_dir),
            "deploy_path": str(cfg.app_root),
            "build_id": os.environ.get("REAL2SIM_PLATFORM_BUILD_ID", os.environ.get("REAL2SIM_PLATFORM_BUILD_TIME", "dev")),
            "config_digest": _config_digest(cfg),
            "git_commit": _git_commit(cfg.app_root),
            "source_snapshot": _source_snapshot(cfg.app_root),
            "runtime_paths": _runtime_paths(cfg),
            "renderer_backend": "spark" if spark_available else "three_fallback",
            "features": {
                "hf_import": True,
                "media_prepare": True,
                "reconstruction_preflight": True,
                "scene_reconstruction": True,
                "spark_viewer": spark_available,
                "three_fallback_viewer": True,
                "aholo_optional": True,
                "mujoco": True,
                "motrixsim": True,
            },
        }

    @app.get("/api/doctor")
    def doctor() -> dict[str, Any]:
        return run_doctor(cfg)

    @app.get("/api/datasets")
    def list_datasets() -> list[dict[str, Any]]:
        return [_with_view_model(item, "dataset") for item in store.list_datasets()]

    @app.post("/api/datasets/import-path")
    def import_dataset(payload: ImportPathRequest) -> dict[str, Any]:
        path = Path(payload.path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="path not found")
        if not is_path_allowed(path, cfg):
            raise HTTPException(status_code=403, detail="path is outside configured import whitelist")
        scan = scan_dataset(path)
        return store.create_dataset(name=payload.name or path.name, path=path, scan=scan, source_type="local_path", source_uri=str(path))

    @app.post("/api/datasets/upload")
    def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
        target = cfg.uploads_dir / Path(file.filename or "upload.bin").name
        with target.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        dataset_dir = cfg.datasets_dir / target.stem
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() == ".zip":
            with zipfile.ZipFile(target) as archive:
                _safe_extract(archive, dataset_dir)
            path = dataset_dir
        else:
            shutil.copy2(target, dataset_dir / target.name)
            path = dataset_dir
        scan = scan_dataset(path)
        manifest_path = dataset_dir / "dataset_manifest.json"
        manifest_path.write_text(json.dumps({"schema": "real2sim.dataset_manifest.v1", "name": dataset_dir.name, "source_type": "upload", "scan": scan}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return store.create_dataset(name=dataset_dir.name, path=path, scan=scan, source_type="upload", source_uri=str(target), manifest_path=manifest_path)

    @app.get("/api/scenes")
    def list_scenes() -> list[dict[str, Any]]:
        return [_with_view_model(item, "scene") for item in store.list_scenes()]

    @app.post("/api/scenes/import")
    def import_scene(payload: ImportSceneRequest) -> dict[str, Any]:
        path = Path(payload.path)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="scene file not found")
        if not is_path_allowed(path, cfg):
            raise HTTPException(status_code=403, detail="scene path is outside configured import whitelist")
        return store.create_scene(name=payload.name or path.stem, path=path, metadata=payload.metadata, source_type="local_path", source_uri=str(path), manifest_path=Path(str(payload.metadata.get("manifest_path", ""))) if payload.metadata.get("manifest_path") else None)

    @app.get("/api/datasets/{dataset_id}/manifest")
    def dataset_manifest(dataset_id: str) -> dict[str, Any]:
        try:
            dataset = store.get_dataset(dataset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset not found") from exc
        return _entity_manifest(dataset, fallback_kind="dataset")

    @app.patch("/api/datasets/{dataset_id}")
    def update_dataset(dataset_id: str, payload: UpdateEntityRequest) -> dict[str, Any]:
        try:
            return store.update_dataset(dataset_id, name=payload.name, tags=payload.tags, version=payload.version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset not found") from exc

    @app.delete("/api/datasets/{dataset_id}")
    def delete_dataset(dataset_id: str) -> dict[str, Any]:
        try:
            store.delete_dataset(dataset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset not found") from exc
        return {"ok": True}

    @app.get("/api/scenes/{scene_id}/manifest")
    def scene_manifest(scene_id: str) -> dict[str, Any]:
        try:
            scene = store.get_scene(scene_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="scene not found") from exc
        return _entity_manifest(scene, fallback_kind="scene")

    @app.post("/api/scenes/{scene_id}/assets")
    def add_scene_asset(scene_id: str, payload: SceneAssetRequest) -> dict[str, Any]:
        try:
            scene = store.get_scene(scene_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="scene not found") from exc
        asset_path = Path(payload.path).expanduser()
        if not asset_path.is_file():
            raise HTTPException(status_code=404, detail="asset file not found")
        if not is_path_allowed(asset_path, cfg):
            raise HTTPException(status_code=403, detail="asset path is outside configured import whitelist")
        manifest_path = _scene_manifest_path_for_write(scene)
        manifest = _read_manifest_or_default(scene, manifest_path)
        artifact = store.add_artifact(task_id=None, name=payload.name or asset_path.name, path=asset_path, kind=(payload.kind or _scene_asset_kind(asset_path)))
        asset = _scene_asset_entry(
            asset_path,
            url=f"/api/artifacts/{artifact['id']}/content",
            name=payload.name,
            kind=payload.kind,
            role=payload.role,
            transform=payload.transform,
        )
        assets = [item for item in manifest.get("assets", []) if not (isinstance(item, dict) and str(item.get("path") or "") == str(asset_path))]
        manifest["assets"] = [asset] + assets
        manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest["asset_editing"] = {"supports_added_assets": True, "last_added_asset": asset["name"]}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tags = list(dict.fromkeys([*(scene.get("tags") or []), "added_asset"]))
        store.update_scene(scene_id, tags=tags)
        return {"ok": True, "scene": store.get_scene(scene_id), "asset": asset, "manifest_path": str(manifest_path)}

    @app.patch("/api/scenes/{scene_id}")
    def update_scene(scene_id: str, payload: UpdateEntityRequest) -> dict[str, Any]:
        try:
            return store.update_scene(scene_id, name=payload.name, tags=payload.tags, version=payload.version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="scene not found") from exc

    @app.delete("/api/scenes/{scene_id}")
    def delete_scene(scene_id: str) -> dict[str, Any]:
        try:
            store.delete_scene(scene_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="scene not found") from exc
        return {"ok": True}

    @app.get("/api/models")
    def list_models() -> list[dict[str, Any]]:
        return [_with_view_model(item, "model") for item in store.list_models()]

    @app.get("/api/models/{model_id}/card")
    def model_card(model_id: str) -> dict[str, Any]:
        try:
            model = store.get_model(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model not found") from exc
        return _entity_manifest(model, fallback_kind="model")

    @app.patch("/api/models/{model_id}")
    def update_model(model_id: str, payload: UpdateEntityRequest) -> dict[str, Any]:
        try:
            return store.update_model(model_id, name=payload.name, tags=payload.tags, version=payload.version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model not found") from exc

    @app.delete("/api/models/{model_id}")
    def delete_model(model_id: str) -> dict[str, Any]:
        try:
            store.delete_model(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model not found") from exc
        return {"ok": True}

    def _queue_task(task_type: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = CreateTaskRequest(type=task_type, params=params)
        _validate_task_payload(payload)
        _validate_task_resources(store, payload)
        if payload.type not in {"scene_reconstruction", "reconstruction_preflight", "media_prepare", "expert_collection", "training", "inference", "renderer_benchmark", "scene_cover_render", "scene_splat_optimize", "hf_import", "simulator_validation"}:
            raise ValueError(f"unknown task type: {payload.type}")
        run_dir = cfg.runs_dir / payload.type / _next_run_name(cfg.runs_dir / payload.type, payload.type)
        return store.create_task(task_type=payload.type, params=payload.params, run_dir=run_dir)

    def _bulk_results(ids: list[str], handler) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for item_id in ids:
            try:
                results.append({"id": item_id, "ok": True, "result": handler(item_id)})
            except Exception as exc:
                results.append({"id": item_id, "ok": False, "error": str(exc)})
        return {"ok": all(item["ok"] for item in results), "results": results}

    @app.post("/api/tasks/bulk")
    def bulk_tasks(payload: BulkActionRequest) -> dict[str, Any]:
        if payload.action not in {"retry", "cancel", "delete"}:
            raise HTTPException(status_code=400, detail="tasks bulk action must be retry, cancel, or delete")

        def handle(task_id: str) -> dict[str, Any]:
            if payload.action == "retry":
                return store.retry_task(task_id)
            if payload.action == "cancel":
                return store.cancel_task(task_id)
            store.delete_task(task_id)
            return {"deleted": True}

        return _bulk_results(payload.ids, handle)

    @app.post("/api/datasets/bulk")
    def bulk_datasets(payload: BulkActionRequest) -> dict[str, Any]:
        if payload.action not in {"prepare", "preflight", "delete"}:
            raise HTTPException(status_code=400, detail="datasets bulk action must be prepare, preflight, or delete")

        def handle(dataset_id: str) -> dict[str, Any]:
            dataset = store.get_dataset(dataset_id)
            if payload.action == "prepare":
                params = {
                    "dataset_id": dataset_id,
                    "camera_label": payload.params.get("camera_label", "observation.images.front"),
                    "sample_fps": payload.params.get("sample_fps", 2),
                    "max_frames_per_video": payload.params.get("max_frames_per_video", 80),
                    "max_videos": payload.params.get("max_videos", 1),
                    "target_frame_count": payload.params.get("target_frame_count", 0),
                    "min_total_frames": payload.params.get("min_total_frames", 0),
                }
                return _queue_task("media_prepare", params)
            if payload.action == "preflight":
                return _queue_task("reconstruction_preflight", {"dataset_id": dataset_id, "min_frames": payload.params.get("min_frames", 30)})
            store.delete_dataset(dataset["id"])
            return {"deleted": True}

        return _bulk_results(payload.ids, handle)

    @app.post("/api/scenes/bulk")
    def bulk_scenes(payload: BulkActionRequest) -> dict[str, Any]:
        if payload.action not in {"validate", "export", "delete"}:
            raise HTTPException(status_code=400, detail="scenes bulk action must be validate, export, or delete")

        def handle(scene_id: str) -> dict[str, Any]:
            scene = store.get_scene(scene_id)
            if payload.action == "validate":
                return _queue_task(
                    "simulator_validation",
                    {
                        "scene_id": scene_id,
                        "simulators": payload.params.get("simulators", ["mujoco", "motrixsim"]),
                        "level": payload.params.get("level", "runtime_if_available"),
                    },
                )
            if payload.action == "export":
                return {"scene_path": scene["path"], "manifest_path": scene.get("manifest_path", ""), "package_path": scene.get("metadata", {}).get("package", str(Path(scene["path"]).parent))}
            store.delete_scene(scene["id"])
            return {"deleted": True}

        return _bulk_results(payload.ids, handle)

    @app.post("/api/models/bulk")
    def bulk_models(payload: BulkActionRequest) -> dict[str, Any]:
        if payload.action not in {"compare", "export", "delete"}:
            raise HTTPException(status_code=400, detail="models bulk action must be compare, export, or delete")

        def handle(model_id: str) -> dict[str, Any]:
            model = store.get_model(model_id)
            if payload.action == "compare":
                return {"model_id": model_id, "name": model["name"], "metrics": model.get("metrics", {})}
            if payload.action == "export":
                return {"model_path": model["path"], "card_path": model.get("manifest_path", "")}
            store.delete_model(model["id"])
            return {"deleted": True}

        return _bulk_results(payload.ids, handle)

    @app.get("/api/tasks")
    def list_tasks() -> list[dict[str, Any]]:
        return [_with_task_view_model(item) for item in store.list_tasks()]

    @app.post("/api/tasks")
    def create_task(payload: CreateTaskRequest) -> dict[str, Any]:
        try:
            return _queue_task(payload.type, payload.params)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str) -> dict[str, Any]:
        try:
            return store.get_task(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    @app.get("/api/tasks/{task_id}/logs")
    def get_task_logs(task_id: str):
        try:
            task = store.get_task(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        path = Path(task["run_dir"]) / "logs" / "task.log"
        return PlainTextResponse(path.read_text(encoding="utf-8") if path.is_file() else "")

    @app.get("/api/tasks/{task_id}/artifacts")
    def get_task_artifacts(task_id: str) -> list[dict[str, Any]]:
        try:
            store.get_task(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        return [{**item, "url": f"/api/artifacts/{item['id']}/content"} for item in store.list_task_artifacts(task_id)]

    @app.get("/api/tasks/{task_id}/reports")
    def get_task_reports(task_id: str) -> dict[str, Any]:
        try:
            return collect_task_reports(store, task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    @app.post("/api/tasks/{task_id}/cancel")
    def cancel_task(task_id: str) -> dict[str, Any]:
        try:
            return store.cancel_task(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    @app.post("/api/tasks/{task_id}/retry")
    def retry_task(task_id: str) -> dict[str, Any]:
        try:
            return store.retry_task(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    @app.patch("/api/tasks/{task_id}")
    def update_task(task_id: str, payload: UpdateTaskRequest) -> dict[str, Any]:
        try:
            return store.update_task_params(task_id, params=payload.params)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.delete("/api/tasks/{task_id}")
    def delete_task(task_id: str) -> dict[str, Any]:
        try:
            store.delete_task(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True}

    @app.get("/api/artifacts/{artifact_id}/content")
    def artifact_content(artifact_id: str):
        try:
            artifact = store.get_artifact(artifact_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="artifact not found") from exc
        path = Path(artifact["path"])
        if not path.is_file():
            raise HTTPException(status_code=404, detail="artifact file not found")
        return FileResponse(path, filename=path.name)

    @app.get("/api/config/default")
    def default_config() -> dict[str, Any]:
        return {"default_renderer": cfg.default_renderer, "simulators": ["mujoco", "motrixsim"], "port": cfg.port}

    app.mount("/runs", StaticFiles(directory=cfg.runs_dir), name="runs")
    return app


def _entity_manifest(entity: dict[str, Any], *, fallback_kind: str) -> dict[str, Any]:
    manifest_path = Path(entity.get("manifest_path") or "")
    if manifest_path.is_file():
        return _postprocess_manifest(json.loads(manifest_path.read_text(encoding="utf-8")), fallback_kind=fallback_kind)
    path = Path(entity["path"])
    candidates = []
    if fallback_kind == "dataset":
        candidates = [path / "dataset_manifest.json", path / "manifest.json"]
    elif fallback_kind == "scene":
        candidates = [path.parent / "scene_manifest.json", path.parent / "package_manifest.json", path]
    elif fallback_kind == "model":
        candidates = [path / "model_card.json", path / "manifest.json"]
    for candidate in candidates:
        if candidate.is_file():
            return _postprocess_manifest(json.loads(candidate.read_text(encoding="utf-8")), fallback_kind=fallback_kind)
    return _postprocess_manifest({"schema": f"real2sim.{fallback_kind}_manifest.synthetic.v1", "id": entity["id"], "name": entity["name"], "path": entity["path"], "source_type": entity.get("source_type"), "source_uri": entity.get("source_uri"), "metrics": entity.get("metrics", {})}, fallback_kind=fallback_kind)


def _scene_manifest_path_for_write(scene: dict[str, Any]) -> Path:
    manifest_path = Path(scene.get("manifest_path") or "")
    if manifest_path.is_file():
        return manifest_path
    scene_path = Path(scene["path"])
    package_dir = scene_path if scene_path.is_dir() else scene_path.parent
    return package_dir / "scene_manifest.json"


def _read_manifest_or_default(scene: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload.setdefault("assets", [])
            return payload
    return {
        "schema": "real2sim.scene_manifest.v1",
        "name": scene.get("name") or scene.get("id"),
        "input_path": scene.get("source_uri") or "",
        "renderer": scene.get("metadata", {}).get("renderer", "auto"),
        "renderer_status": {},
        "simulators": scene.get("metadata", {}).get("simulators", []),
        "dry_run": scene.get("metadata", {}).get("dry_run", False),
        "assets": [],
        "simulator_artifacts": {},
    }


def _scene_asset_entry(path: Path, *, url: str, name: str | None, kind: str | None, role: str, transform: dict[str, Any]) -> dict[str, Any]:
    size_bytes = path.stat().st_size
    asset_kind = kind or _scene_asset_kind(path)
    autoload = asset_kind in {"splat_or_pointcloud", "mesh"} and size_bytes <= 25 * 1024 * 1024
    return {
        "name": name or path.name,
        "path": str(path),
        "url": url,
        "kind": asset_kind,
        "role": role or "added_asset",
        "format": path.suffix.lower().lstrip("."),
        "size_bytes": size_bytes,
        "autoload": autoload,
        "viewer_hint": "autoload" if autoload else "manual_load",
        "transform": transform or {},
        "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _scene_asset_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".ply", ".splat", ".spz", ".ksplat", ".sog", ".zip", ".rad"}:
        return "splat_or_pointcloud"
    if suffix in {".glb", ".gltf", ".obj"}:
        return "mesh"
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return "texture"
    if suffix in {".xml"}:
        return "simulator"
    if suffix in {".json", ".jsonl"}:
        return "metadata"
    return "file"


def _postprocess_manifest(manifest: dict[str, Any], *, fallback_kind: str) -> dict[str, Any]:
    if fallback_kind == "scene":
        return enrich_scene_manifest_assets(manifest)
    return manifest


def _with_view_model(entity: dict[str, Any], kind: str) -> dict[str, Any]:
    item = dict(entity)
    manifest = _safe_entity_manifest(item, kind)
    primary_asset = _primary_asset(manifest)
    thumbnail = _thumbnail_from_manifest(item, manifest, primary_asset)
    metrics = item.get("metrics") or {}
    scan = item.get("scan") or {}
    item["view"] = {
        "display_name": item.get("name") or Path(str(item.get("path") or "")).stem or item.get("id", ""),
        "thumbnail_url": thumbnail,
        "preview_url": _preview_url(kind, item, primary_asset),
        "primary_asset": primary_asset,
        "metrics_summary": _metrics_summary(kind, metrics, scan, manifest),
        "subtitle": _resource_subtitle(kind, item, manifest),
    }
    return item


def _with_task_view_model(task: dict[str, Any]) -> dict[str, Any]:
    item = dict(task)
    progress = int(float(item.get("progress") or 0))
    if item.get("status") == "succeeded":
        progress = 100
    item["view"] = {
        "display_name": _task_display_name(item),
        "stage_label": item.get("stage") or _task_type_label(str(item.get("type") or "")),
        "progress": max(0, min(100, progress)),
        "message": item.get("message") or item.get("error") or "",
        "updated_label": time.strftime("%m/%d %H:%M", time.localtime(float(item.get("updated_at") or item.get("created_at") or time.time()))),
    }
    return item


def _safe_entity_manifest(entity: dict[str, Any], kind: str) -> dict[str, Any]:
    try:
        return _entity_manifest(entity, fallback_kind=kind)
    except Exception:
        return {}


def _primary_asset(manifest: dict[str, Any]) -> dict[str, Any] | None:
    assets = manifest.get("assets") if isinstance(manifest, dict) else []
    if not isinstance(assets, list):
        return None
    candidates = [asset for asset in assets if isinstance(asset, dict)]
    if not candidates:
        return None
    return sorted(candidates, key=_asset_view_rank)[0]


def _asset_view_rank(asset: dict[str, Any]) -> tuple[int, int, str]:
    name = str(asset.get("name") or asset.get("path") or "").lower()
    path_text = str(asset.get("path") or "").lower()
    kind = str(asset.get("kind") or "").lower()
    size = int(asset.get("size_bytes") or 0)
    if asset.get("cover") or name == "scene_cover.png":
        rank = 0
    elif name.endswith("_threejs_rgb.ply"):
        rank = 1
    elif name.endswith((".splat", ".spz", ".ksplat", ".sog")):
        rank = 2
    elif name == "pose_init_trained_scene.ply":
        rank = 3
    elif name == "scene.ply" and "/3dgs/" in path_text:
        rank = 4
    elif name in {"pointcloud_projection_preview.png", "source_gs_preview.png"}:
        rank = 5
    elif name.endswith((".png", ".jpg", ".jpeg", ".webp")) and ("preview" in name or "render" in name):
        rank = 6
    elif name.startswith("thumb_") and name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        rank = 7
    elif name == "scene.ply":
        rank = 8
    elif kind in {"mesh", "splat_or_pointcloud"} and "sparse" not in name:
        rank = 9
    elif name in {"background_image.png"}:
        rank = 10
    elif name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        rank = 11
    elif "colmap_sparse" in name or "sparse_points" in name:
        rank = 12
    else:
        rank = 13
    return (rank, -size, name)


def _thumbnail_from_manifest(entity: dict[str, Any], manifest: dict[str, Any], primary_asset: dict[str, Any] | None) -> str:
    assets = manifest.get("assets") if isinstance(manifest, dict) else []
    if isinstance(assets, list):
        image_assets = [asset for asset in assets if isinstance(asset, dict) and str(asset.get("name") or asset.get("path") or "").lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        image_assets.sort(key=_asset_view_rank)
        for asset in image_assets:
            if asset.get("url"):
                return str(asset["url"])
    dataset_thumb = _dataset_thumbnail_url(entity)
    if dataset_thumb:
        return dataset_thumb
    return str(primary_asset.get("url") or "") if primary_asset and str(primary_asset.get("kind") or "") == "image" else ""


def _dataset_thumbnail_url(entity: dict[str, Any]) -> str:
    root = Path(str(entity.get("path") or ""))
    candidates = [
        root / "thumbnails" / "thumb_01.jpg",
        root / "thumbnails" / "thumb_01.png",
        root / "thumbnail.jpg",
        root / "preview.jpg",
        root / "preview.png",
    ]
    for path in candidates:
        if path.is_file():
            return _workspace_public_url(path)
    return ""


def _workspace_public_url(path: Path) -> str:
    parts = path.parts
    for marker in ("runs", "datasets"):
        if marker in parts:
            index = parts.index(marker)
            return "/" + "/".join(parts[index:])
    return ""


def _preview_url(kind: str, entity: dict[str, Any], primary_asset: dict[str, Any] | None) -> str:
    if kind == "scene":
        return f"/static/scene-viewer.html?scene_id={entity['id']}"
    return str(primary_asset.get("url") or "") if primary_asset else ""


def _metrics_summary(kind: str, metrics: dict[str, Any], scan: dict[str, Any], manifest: dict[str, Any]) -> str:
    if kind == "dataset":
        frames = metrics.get("frame_count") or manifest.get("frame_count") or scan.get("image_count")
        videos = scan.get("video_count")
        samples = metrics.get("samples") or scan.get("samples")
        parts = []
        if frames:
            parts.append(f"{frames} 帧")
        if videos:
            parts.append(f"{videos} 视频")
        if samples:
            parts.append(f"{samples} 样本")
        return " · ".join(parts) or "等待扫描"
    if kind == "scene":
        assets = manifest.get("assets") if isinstance(manifest.get("assets"), list) else []
        return f"{len(assets)} 个资产" if assets else "暂无资产"
    score = metrics.get("success_rate") or metrics.get("score") or metrics.get("accuracy")
    loss = metrics.get("loss") or metrics.get("val_loss")
    parts = []
    if score is not None:
        parts.append(f"成功率 {float(score) * 100:.1f}%" if float(score) <= 1 else f"成功率 {float(score):.1f}%")
    if loss is not None:
        parts.append(f"loss {float(loss):.3f}")
    return " · ".join(parts) or "暂无指标"


def _resource_subtitle(kind: str, entity: dict[str, Any], manifest: dict[str, Any]) -> str:
    source = str(entity.get("source_type") or manifest.get("source_type") or "")
    version = str(entity.get("version") or "")
    label = {"prepared_media": "已准备帧数据", "expert_collection": "专家采集数据", "scene_reconstruction": "场景重建", "huggingface": "HuggingFace", "upload": "上传文件", "local_path": "本地导入"}.get(source, source or kind)
    return f"{label}{' · ' + version if version else ''}"


def _task_display_name(task: dict[str, Any]) -> str:
    params = task.get("params") or {}
    return str(params.get("scene_name") or params.get("output_name") or _task_type_label(str(task.get("type") or "")))


def _task_type_label(task_type: str) -> str:
    return {
        "hf_import": "数据导入",
        "media_prepare": "媒体准备",
        "reconstruction_preflight": "重建预检",
        "scene_reconstruction": "场景重建",
        "simulator_validation": "仿真校验",
        "expert_collection": "专家采集",
        "training": "模型训练",
        "inference": "在线推理",
        "renderer_benchmark": "渲染测试",
        "scene_splat_optimize": "3DGS 优化",
    }.get(task_type, task_type)


def _validate_task_payload(payload: CreateTaskRequest) -> None:
    params = payload.params
    if payload.type == "hf_import":
        repo_type = params.get("repo_type", "dataset")
        if repo_type != "dataset":
            raise HTTPException(status_code=400, detail="hf_import only supports repo_type='dataset'")
        if not params.get("repo_id"):
            raise HTTPException(status_code=400, detail="hf_import requires repo_id")
    if payload.type == "media_prepare":
        if not (params.get("dataset_id") or params.get("input_path")):
            raise HTTPException(status_code=400, detail="media_prepare requires dataset_id or input_path")
        if float(params.get("sample_fps", 2.0)) <= 0:
            raise HTTPException(status_code=400, detail="sample_fps must be > 0")
        if int(params.get("max_frames_per_video", 80)) <= 0:
            raise HTTPException(status_code=400, detail="max_frames_per_video must be > 0")
        if int(params.get("target_frame_count", 0) or 0) < 0:
            raise HTTPException(status_code=400, detail="target_frame_count must be >= 0")
        if int(params.get("min_total_frames", 0) or 0) < 0:
            raise HTTPException(status_code=400, detail="min_total_frames must be >= 0")
    if payload.type == "reconstruction_preflight":
        if not (params.get("dataset_id") or params.get("input_path")):
            raise HTTPException(status_code=400, detail="reconstruction_preflight requires dataset_id or input_path")
        if int(params.get("min_frames", 30)) <= 0:
            raise HTTPException(status_code=400, detail="min_frames must be > 0")
    if payload.type == "scene_reconstruction":
        renderer = params.get("renderer", "auto")
        if renderer not in {"auto", "spark", "aholo"}:
            raise HTTPException(status_code=400, detail="renderer must be auto, spark, or aholo")
        simulators = set(params.get("simulators", ["mujoco", "motrixsim"]))
        if not simulators <= {"mujoco", "motrixsim"}:
            raise HTTPException(status_code=400, detail="simulators must be mujoco and/or motrixsim")
        if not (params.get("dataset_id") or params.get("input_path")):
            raise HTTPException(status_code=400, detail="scene_reconstruction requires dataset_id or input_path")
        if int(params.get("min_frames", 30)) <= 0:
            raise HTTPException(status_code=400, detail="min_frames must be > 0")
    if payload.type == "simulator_validation":
        if not (params.get("scene_id") or params.get("scene_path")):
            raise HTTPException(status_code=400, detail="simulator_validation requires scene_id or scene_path")
        if params.get("level", "runtime_if_available") not in {"smoke", "runtime_if_available", "runtime_required"}:
            raise HTTPException(status_code=400, detail="level must be smoke, runtime_if_available, or runtime_required")
    if payload.type == "expert_collection":
        if not (params.get("scene_id") or params.get("scene_path")):
            raise HTTPException(status_code=400, detail="expert_collection requires scene_id or scene_path")
        if params.get("policy_preset", "pick_and_place_v1") != "pick_and_place_v1":
            raise HTTPException(status_code=400, detail="当前仅支持抓取放置策略；导航和精细操作策略尚未接入执行器")
        if int(params.get("workers", 1)) != 1:
            raise HTTPException(status_code=400, detail="当前专家采集执行器仅支持 workers=1")
        if int(params.get("episodes", 4)) <= 0 or int(params.get("steps", 24)) <= 0:
            raise HTTPException(status_code=400, detail="episodes and steps must be > 0")
    if payload.type == "training":
        if not (params.get("dataset_id") or params.get("dataset_path")):
            raise HTTPException(status_code=400, detail="training requires dataset_id or dataset_path")
        if params.get("recipe", "bc_v2") not in {"bc_v2", "bc_fast"}:
            raise HTTPException(status_code=400, detail="当前仅支持行为克隆 v2 和快速验证；扩散策略尚未接入训练器")
        if int(params.get("epochs", 100)) <= 0:
            raise HTTPException(status_code=400, detail="epochs must be > 0")
    if payload.type == "inference":
        if not (params.get("model_id") or params.get("model_path")):
            raise HTTPException(status_code=400, detail="inference requires model_id or model_path")
        if not (params.get("scene_id") or params.get("scene_path")):
            raise HTTPException(status_code=400, detail="inference requires scene_id or scene_path")
        if int(params.get("workers", 1)) != 1:
            raise HTTPException(status_code=400, detail="当前推理执行器仅支持 workers=1")
        if int(params.get("steps", 80)) <= 0:
            raise HTTPException(status_code=400, detail="steps must be > 0")


def _validate_task_resources(store: PlatformStore, payload: CreateTaskRequest) -> None:
    params = payload.params
    try:
        if payload.type == "expert_collection" and params.get("scene_id"):
            store.get_scene(str(params["scene_id"]))
        if payload.type == "training" and params.get("dataset_id"):
            dataset = store.get_dataset(str(params["dataset_id"]))
            source = str(dataset.get("source_type") or "")
            input_kind = str(dataset.get("scan", {}).get("input_kind") or "")
            if source != "expert_collection" and input_kind != "expert_rollouts":
                raise HTTPException(status_code=400, detail="训练只能使用专家采集产生的 rollout 数据集")
        if payload.type == "inference":
            if params.get("model_id"):
                store.get_model(str(params["model_id"]))
            if params.get("scene_id"):
                store.get_scene(str(params["scene_id"]))
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"任务引用的资源不存在: {exc.args[0]}") from exc


def _runtime_paths(cfg: PlatformConfig) -> dict[str, str]:
    return {
        "gs_playground_root": str(cfg.gs_playground_root),
        "runtime_root": str(cfg.runtime_root),
        "anysplat_root": str(cfg.anysplat_root),
        "anysplat_python": str(cfg.anysplat_python),
        "gsplat_python": str(cfg.gsplat_python),
        "cuda_home": str(cfg.cuda_home),
        "runtime_env_script": str(cfg.runtime_env_script),
    }


def _config_digest(cfg: PlatformConfig) -> str:
    payload = json.dumps(_runtime_paths(cfg) | {"workspace": str(cfg.workspace_dir), "renderer": cfg.default_renderer}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _source_snapshot(root: Path) -> str:
    files = [root / "pyproject.toml", root / "src/real2sim_platform/adapters.py", root / "src/real2sim_platform/server.py", root / "src/real2sim_platform/config.py"]
    digest = hashlib.sha256()
    for path in files:
        if path.is_file():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _safe_extract(archive: zipfile.ZipFile, target_dir: Path) -> None:
    root = target_dir.resolve()
    for member in archive.infolist():
        destination = (target_dir / member.filename).resolve()
        try:
            destination.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"unsafe zip member: {member.filename}") from exc
    archive.extractall(target_dir)


def _next_run_name(root: Path, prefix: str) -> str:
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (root / f"{prefix}_{index:04d}").exists():
        index += 1
    return f"{prefix}_{index:04d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--write-config", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.host:
        cfg = PlatformConfig(**{**cfg.__dict__, "host": args.host})
    if args.port:
        cfg = PlatformConfig(**{**cfg.__dict__, "port": args.port})
    if args.write_config:
        write_default_config(args.write_config, cfg)
        print(args.write_config)
        return 0
    import uvicorn

    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
