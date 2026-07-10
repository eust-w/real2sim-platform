from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict, deque
import hashlib
import json
import os
import math
import secrets
import shutil
import subprocess
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .adapters import SPARK_VENDOR_PATH, collect_task_reports, enrich_scene_manifest_assets, run_doctor, scan_dataset
from .auth import AuthManager, AuthenticationError, SESSION_COOKIE, validate_password, validate_role, validate_username
from .config import PlatformConfig, ensure_workspace, is_path_allowed, load_config, write_default_config
from .storage import BosStorage, StorageBoundaryError, StorageUnavailable, safe_object_name, sha256_file
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


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str = ""
    new_password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "engineer"


class UpdateUserRequest(BaseModel):
    role: str | None = None
    disabled: bool | None = None
    reset_password: str | None = None


class CreateUploadRequest(BaseModel):
    filename: str
    content_type: str = "application/octet-stream"
    total_size: int
    chunk_size: int = 8 * 1024 * 1024
    checksum: str = ""


class ImportBosRequest(BaseModel):
    prefix: str
    name: str | None = None
    max_objects: int = 1000


def create_app(config: PlatformConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    ensure_workspace(cfg)
    store = PlatformStore(cfg.db_path)
    worker = TaskWorker(store, cfg)
    auth = AuthManager(store, cfg)
    storage = BosStorage(cfg)
    auth.bootstrap_admin()
    login_failures: dict[str, deque[float]] = defaultdict(deque)
    login_failure_lock = threading.Lock()
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if cfg.worker_mode == "embedded":
            worker.start()
        try:
            yield
        finally:
            if cfg.worker_mode == "embedded":
                worker.stop()

    app = FastAPI(title="Real2Sim Platform", version="0.1.0", lifespan=lifespan)
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.mount("/datasets", StaticFiles(directory=cfg.datasets_dir), name="datasets")

    public_paths = {"/", "/api/health", "/api/version", "/api/capabilities", "/api/auth/login"}

    @app.middleware("http")
    async def authenticate_request(request: Request, call_next):
        path = request.url.path
        if not cfg.auth_enabled or path in public_paths or path.startswith("/static/"):
            request.state.user = auth.authenticate(None)
            return await call_next(request)
        token = request.cookies.get(SESSION_COOKIE)
        user = auth.authenticate(token)
        if user is None:
            return JSONResponse(status_code=401, content={"detail": "请先登录"})
        request.state.user = user
        if user.get("must_change_password") and path not in {"/api/auth/me", "/api/auth/change-password", "/api/auth/logout"}:
            return JSONResponse(status_code=428, content={"detail": "首次登录必须先修改密码", "code": "password_change_required"})
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            self_service_paths = {"/api/auth/change-password", "/api/auth/logout"}
            if user.get("role") == "viewer" and path not in self_service_paths:
                return JSONResponse(status_code=403, content={"detail": "只读用户不能执行修改操作"})
            if path.startswith("/api/users") and user.get("role") != "admin":
                return JSONResponse(status_code=403, content={"detail": "仅管理员可以管理用户"})
            origin = request.headers.get("origin")
            origin_host = urlparse(origin).hostname if origin else None
            if origin_host and origin_host != request.url.hostname:
                return JSONResponse(status_code=403, content={"detail": "请求来源校验失败"})
        if path.startswith("/api/users") and user.get("role") != "admin":
            return JSONResponse(status_code=403, content={"detail": "仅管理员可以管理用户"})
        return await call_next(request)

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "worker_mode": cfg.worker_mode, "production_mode": cfg.production_mode}

    @app.post("/api/auth/login")
    def login(payload: LoginRequest, request: Request, response: Response) -> dict[str, Any]:
        if not cfg.auth_enabled:
            return {"user": auth.authenticate(None), "auth_enabled": False}
        login_key = f"{request.client.host if request.client else 'unknown'}:{payload.username.strip().lower()}"
        now = time.time()
        with login_failure_lock:
            attempts = login_failures[login_key]
            while attempts and now - attempts[0] > 900:
                attempts.popleft()
            if len(attempts) >= 8:
                raise HTTPException(status_code=429, detail="登录失败次数过多，请 15 分钟后再试")
        try:
            user, token = auth.login(payload.username, payload.password)
        except AuthenticationError as exc:
            with login_failure_lock:
                login_failures[login_key].append(time.time())
            time.sleep(0.2)
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        with login_failure_lock:
            login_failures.pop(login_key, None)
        response.set_cookie(
            SESSION_COOKIE,
            token,
            max_age=cfg.session_ttl_seconds,
            httponly=True,
            secure=cfg.secure_cookies,
            samesite="lax",
            path="/",
        )
        return {"user": user, "auth_enabled": True}

    @app.post("/api/auth/logout")
    def logout(request: Request, response: Response) -> dict[str, bool]:
        auth.logout(request.cookies.get(SESSION_COOKIE))
        response.delete_cookie(SESSION_COOKIE, path="/")
        return {"ok": True}

    @app.get("/api/auth/me")
    def auth_me(request: Request) -> dict[str, Any]:
        return {"user": request.state.user, "auth_enabled": cfg.auth_enabled}

    @app.post("/api/auth/change-password")
    def change_password(payload: ChangePasswordRequest, request: Request, response: Response) -> dict[str, Any]:
        try:
            user = auth.change_password(
                request.state.user["id"],
                current_password=payload.current_password,
                new_password=payload.new_password,
            )
        except (AuthenticationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        response.delete_cookie(SESSION_COOKIE, path="/")
        return {"ok": True, "user": user, "login_required": True}

    @app.get("/api/users")
    def list_users() -> list[dict[str, Any]]:
        return store.list_users()

    @app.post("/api/users")
    def create_user(payload: CreateUserRequest) -> dict[str, Any]:
        try:
            return auth.create_user(username=payload.username, password=payload.password, role=payload.role)
        except (ValueError, __import__("sqlite3").IntegrityError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.patch("/api/users/{user_id}")
    def update_user(user_id: str, payload: UpdateUserRequest, request: Request) -> dict[str, Any]:
        try:
            target = store.get_user(user_id)
            if payload.role is not None:
                validate_role(payload.role)
            if target["id"] == request.state.user["id"] and payload.disabled:
                raise ValueError("不能禁用当前登录账号")
            if target["role"] == "admin" and (payload.disabled or payload.role not in {None, "admin"}) and _active_admin_count(store) <= 1:
                raise ValueError("系统至少需要一个可用管理员")
            if payload.reset_password:
                validate_password(payload.reset_password)
                return auth.change_password(user_id, current_password=None, new_password=payload.reset_password, admin_reset=True)
            result = store.update_user(user_id, role=payload.role, disabled=payload.disabled)
            if payload.disabled:
                store.delete_user_sessions(user_id)
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="用户不存在") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/users/{user_id}")
    def delete_user(user_id: str, request: Request) -> dict[str, bool]:
        try:
            target = store.get_user(user_id)
            if target["id"] == request.state.user["id"]:
                raise ValueError("不能删除当前登录账号")
            if target["role"] == "admin" and not target["disabled"] and _active_admin_count(store) <= 1:
                raise ValueError("系统至少需要一个可用管理员")
            store.delete_user(user_id)
            return {"ok": True}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="用户不存在") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        result = run_doctor(cfg)
        bos_health = storage.health()
        sqlite_wal = _sqlite_journal_mode(store) == "wal"
        result["checks"]["bos_storage"] = bos_health
        result["checks"]["sqlite_wal"] = sqlite_wal
        result["checks"]["worker_mode"] = {"ready": cfg.worker_mode in {"embedded", "api", "worker"}, "value": cfg.worker_mode}
        heartbeats = store.list_worker_heartbeats()
        latest = heartbeats[0] if heartbeats else None
        heartbeat_age = time.time() - float(latest["updated_at"]) if latest else None
        worker_ready = bool(latest and heartbeat_age is not None and heartbeat_age < 30 and latest["payload"].get("ready"))
        result["worker"] = {"ready": worker_ready, "age_seconds": heartbeat_age, "latest": latest}
        if cfg.production_mode and cfg.worker_mode == "api":
            result["ok"] = bool(bos_health.get("ready") and sqlite_wal and worker_ready)
            result["core_checks"] = ["bos_storage", "sqlite_wal", "worker"]
        return result

    @app.get("/api/capabilities")
    def capabilities() -> dict[str, Any]:
        return _capabilities(cfg, storage, store)

    @app.get("/api/recipes")
    def recipes() -> dict[str, Any]:
        capabilities_payload = _capabilities(cfg, storage, store)
        act_ready = bool(capabilities_payload["runtimes"]["discoverse_act"]["ready"])
        rm65_ready = bool(capabilities_payload["runtimes"]["rm65_controller"]["ready"])
        return {
            "collection": [
                {
                    "id": "rm65_pick_place_6d_ik",
                    "name": "RM65 茶几抓取放置",
                    "enabled": rm65_ready,
                    "engine": "MuJoCo / MotrixSim",
                    "requirements": ["rm65", "parallel_gripper", "pick_place_target", "contact_sensors"],
                    "reason": "" if rm65_ready else "RM65 物理控制器或运行时尚未安装",
                }
            ],
            "training": [
                {
                    "id": "discoverse_act",
                    "name": "DISCOVERSE ACT",
                    "enabled": act_ready,
                    "commit": cfg.act_commit,
                    "input_format": "discoverse_act_hdf5",
                    "reason": "" if act_ready else "DISCOVERSE ACT 固定版本运行时尚未就绪",
                },
                {"id": "bc_v2", "name": "历史岭回归模型", "enabled": False, "read_only": True, "reason": "仅用于查看历史模型，禁止新建"},
            ],
            "inference": [
                {"id": "discoverse_act", "name": "DISCOVERSE ACT 在线推理", "enabled": act_ready, "requires_real_simulator": True}
            ],
        }

    @app.post("/api/uploads")
    def create_upload(payload: CreateUploadRequest, request: Request) -> dict[str, Any]:
        if not storage.configured:
            raise HTTPException(status_code=503, detail="BOS 存储尚未配置")
        if payload.total_size <= 0:
            raise HTTPException(status_code=400, detail="total_size must be > 0")
        if not 1024 * 1024 <= payload.chunk_size <= 64 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="chunk_size must be between 1 MiB and 64 MiB")
        if payload.checksum and (len(payload.checksum) != 64 or any(c not in "0123456789abcdefABCDEF" for c in payload.checksum)):
            raise HTTPException(status_code=400, detail="checksum must be a SHA-256 hex digest")
        filename = safe_object_name(payload.filename)
        upload_id = secrets.token_hex(16)
        local_dir = cfg.upload_parts_dir / upload_id
        local_dir.mkdir(parents=True, exist_ok=False)
        return store.create_upload(
            item_id=upload_id,
            owner_user_id=request.state.user.get("id"),
            filename=filename,
            content_type=payload.content_type,
            total_size=payload.total_size,
            chunk_size=payload.chunk_size,
            total_parts=math.ceil(payload.total_size / payload.chunk_size),
            checksum_expected=payload.checksum.lower(),
            local_dir=local_dir,
        )

    @app.get("/api/uploads/{upload_id}")
    def get_upload(upload_id: str, request: Request) -> dict[str, Any]:
        try:
            upload = store.get_upload(upload_id)
            _check_upload_owner(upload, request.state.user)
            return upload
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="上传任务不存在") from exc

    @app.put("/api/uploads/{upload_id}/parts/{part_number}")
    async def upload_part(upload_id: str, part_number: int, request: Request) -> dict[str, Any]:
        try:
            upload = store.get_upload(upload_id)
            _check_upload_owner(upload, request.state.user)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="上传任务不存在") from exc
        if upload["status"] != "uploading":
            raise HTTPException(status_code=409, detail="上传任务已进入终态")
        if not 1 <= part_number <= upload["total_parts"]:
            raise HTTPException(status_code=400, detail="分片序号超出范围")
        expected_size = upload["chunk_size"]
        if part_number == upload["total_parts"]:
            expected_size = upload["total_size"] - upload["chunk_size"] * (upload["total_parts"] - 1)
        target = Path(upload["local_dir"]) / f"part-{part_number:08d}"
        temporary = target.with_suffix(".tmp")
        digest = hashlib.sha256()
        written = 0
        try:
            with temporary.open("wb") as handle:
                async for chunk in request.stream():
                    written += len(chunk)
                    if written > expected_size:
                        raise HTTPException(status_code=413, detail="分片大小超过声明值")
                    handle.write(chunk)
                    digest.update(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if written != expected_size:
                raise HTTPException(status_code=400, detail=f"分片大小不匹配，期望 {expected_size} 字节，实际 {written} 字节")
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
        result = store.record_upload_part(upload_id, part_number)
        return {"upload": result, "part_number": part_number, "size_bytes": written, "checksum": digest.hexdigest()}

    @app.post("/api/uploads/{upload_id}/complete")
    def complete_upload(upload_id: str, request: Request) -> dict[str, Any]:
        try:
            upload = store.get_upload(upload_id)
            _check_upload_owner(upload, request.state.user)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="上传任务不存在") from exc
        if upload["status"] == "completed":
            return {"upload": upload, "dataset": _dataset_for_storage_uri(store, upload["storage_uri"])}
        missing = sorted(set(range(1, upload["total_parts"] + 1)) - set(upload["received_parts"]))
        if missing:
            raise HTTPException(status_code=409, detail={"message": "仍有分片未上传", "missing_parts": missing[:100]})
        local_dir = Path(upload["local_dir"])
        assembled = local_dir / upload["filename"]
        digest = hashlib.sha256()
        with assembled.open("wb") as output:
            for part_number in range(1, upload["total_parts"] + 1):
                part = local_dir / f"part-{part_number:08d}"
                with part.open("rb") as handle:
                    while chunk := handle.read(8 * 1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        checksum = digest.hexdigest()
        if assembled.stat().st_size != upload["total_size"] or (upload["checksum_expected"] and checksum != upload["checksum_expected"]):
            store.fail_upload(upload_id, "assembled file checksum or size mismatch")
            raise HTTPException(status_code=422, detail="完整文件校验失败，请续传错误分片后重试")
        try:
            uploaded = storage.upload_file(
                assembled,
                area="uploads",
                relative_path=f"{upload_id}/{upload['filename']}",
                metadata={"sha256": checksum, "upload-id": upload_id},
            )
            dataset_id = secrets.token_hex(16)
            dataset_dir = cfg.datasets_dir / dataset_id
            dataset_dir.mkdir(parents=True, exist_ok=False)
            if assembled.suffix.lower() == ".zip":
                with zipfile.ZipFile(assembled) as archive:
                    _safe_extract(archive, dataset_dir)
            else:
                shutil.copy2(assembled, dataset_dir / upload["filename"])
            scan = scan_dataset(dataset_dir)
            manifest_path = dataset_dir / "dataset_manifest.json"
            manifest = {
                "schema": "real2sim.dataset_manifest.v2",
                "id": dataset_id,
                "name": Path(upload["filename"]).stem,
                "source_type": "browser_upload",
                "storage_uri": uploaded["storage_uri"],
                "checksum": checksum,
                "version": checksum[:12],
                "scan": scan,
            }
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            storage.upload_file(manifest_path, area="datasets", relative_path=f"{dataset_id}/dataset_manifest.json", metadata={"sha256": sha256_file(manifest_path)})
            dataset = store.create_dataset(
                item_id=dataset_id,
                name=manifest["name"],
                path=dataset_dir,
                scan=scan,
                source_type="browser_upload",
                source_uri=uploaded["storage_uri"],
                version=manifest["version"],
                manifest_path=manifest_path,
                storage_uri=uploaded["storage_uri"],
                checksum=checksum,
                cache_state="local",
            )
            completed = store.complete_upload(upload_id, checksum_actual=checksum, storage_uri=uploaded["storage_uri"])
            return {"upload": completed, "dataset": _with_view_model(dataset, "dataset")}
        except (StorageUnavailable, StorageBoundaryError) as exc:
            store.fail_upload(upload_id, str(exc))
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.delete("/api/uploads/{upload_id}")
    def delete_upload(upload_id: str, request: Request) -> dict[str, bool]:
        try:
            upload = store.get_upload(upload_id)
            _check_upload_owner(upload, request.state.user)
            shutil.rmtree(Path(upload["local_dir"]), ignore_errors=True)
            store.delete_upload(upload_id)
            return {"ok": True}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="上传任务不存在") from exc

    @app.get("/api/storage/objects")
    def list_storage_objects(area: str = "datasets", prefix: str = "", max_keys: int = 200) -> dict[str, Any]:
        try:
            return {"bucket": storage.bucket, "prefix": storage.prefix, "objects": storage.list_objects(area=area, relative_prefix=prefix, max_keys=max_keys)}
        except (StorageUnavailable, StorageBoundaryError) as exc:
            raise HTTPException(status_code=503 if isinstance(exc, StorageUnavailable) else 400, detail=str(exc)) from exc

    @app.post("/api/datasets/import-bos")
    def import_bos_dataset(payload: ImportBosRequest) -> dict[str, Any]:
        return _queue_task("bos_import", {"prefix": payload.prefix, "name": payload.name or Path(payload.prefix.rstrip("/")).name, "max_objects": payload.max_objects})

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
        if cfg.production_mode:
            raise HTTPException(status_code=410, detail="生产模式请使用 /api/uploads 分片上传，旧上传接口已禁用")
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
        revision_id = secrets.token_hex(16)
        revision_dir = cfg.scenes_dir / "revisions" / revision_id
        source_scene_path = Path(scene["path"])
        source_package = source_scene_path if source_scene_path.is_dir() else source_scene_path.parent
        _copy_scene_revision(source_package, revision_dir)
        revision_scene_path = revision_dir / source_scene_path.name if source_scene_path.is_file() else revision_dir
        source_manifest_path = _scene_manifest_path_for_write(scene)
        manifest_path = revision_dir / source_manifest_path.name
        manifest = _read_manifest_or_default(scene, source_manifest_path)
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
        manifest["revision"] = {"id": revision_id, "parent_scene_id": scene_id, "reason": "asset_added"}
        manifest["asset_editing"] = {"supports_added_assets": True, "last_added_asset": asset["name"]}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tags = list(dict.fromkeys([*(scene.get("tags") or []), "added_asset", "scene_revision"]))
        metadata = dict(scene.get("metadata") or {})
        metadata.update({"parent_scene_id": scene_id, "revision_reason": "asset_added", "package": str(revision_dir)})
        revision = store.create_scene(
            item_id=revision_id,
            name=f"{scene['name']} · 资产修订",
            path=revision_scene_path,
            metadata=metadata,
            source_type="scene_revision",
            source_uri=scene.get("storage_uri") or scene.get("source_uri") or str(source_scene_path),
            version=_next_revision_label(str(scene.get("version") or "")),
            parent_task_id=scene.get("parent_task_id"),
            manifest_path=manifest_path,
            metrics=scene.get("metrics") or {},
            tags=tags,
            cache_state="local",
        )
        return {"ok": True, "scene": revision, "parent_scene_id": scene_id, "asset": asset, "manifest_path": str(manifest_path)}

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
        _validate_task_payload(payload, cfg)
        _validate_task_resources(store, payload)
        if payload.type not in {"scene_reconstruction", "reconstruction_preflight", "media_prepare", "expert_collection", "training", "inference", "renderer_benchmark", "scene_cover_render", "scene_splat_optimize", "hf_import", "bos_import", "motrixsim_asset_import", "simulator_validation"}:
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

    @app.get("/api/events/tasks")
    async def task_events(request: Request):
        async def stream():
            previous = ""
            while not await request.is_disconnected():
                tasks = [_with_task_view_model(item) for item in store.list_tasks()]
                fingerprint = hashlib.sha256(json.dumps([(item["id"], item["status"], item["progress"], item["stage"], item["message"], item["updated_at"]) for item in tasks], ensure_ascii=False).encode("utf-8")).hexdigest()
                if fingerprint != previous:
                    yield f"event: tasks\ndata: {json.dumps(tasks, ensure_ascii=False)}\n\n"
                    previous = fingerprint
                else:
                    yield ": keepalive\n\n"
                await asyncio.sleep(1.5)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/tasks")
    def create_task(payload: CreateTaskRequest, request: Request) -> dict[str, Any]:
        if payload.type == "motrixsim_asset_import" and request.state.user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="仅管理员可以导入平台内置仿真资产")
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


def _copy_scene_revision(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)

    def link_or_copy(src: str, dst: str) -> str:
        try:
            os.link(src, dst)
            return dst
        except OSError:
            return shutil.copy2(src, dst)

    shutil.copytree(source, target, copy_function=link_or_copy)


def _next_revision_label(current: str) -> str:
    import re

    match = re.search(r"(?:^|[-_.])r(\d+)$", current, re.IGNORECASE)
    number = int(match.group(1)) + 1 if match else 2
    base = re.sub(r"(?:[-_.])r\d+$", "", current, flags=re.IGNORECASE).strip("-_.") or "v1"
    return f"{base}-r{number}"


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
        "motrixsim_asset_import": "MotrixSim 资产导入",
        "simulator_validation": "仿真校验",
        "expert_collection": "专家采集",
        "training": "模型训练",
        "inference": "在线推理",
        "renderer_benchmark": "渲染测试",
        "scene_splat_optimize": "3DGS 优化",
    }.get(task_type, task_type)


def _validate_task_payload(payload: CreateTaskRequest, cfg: PlatformConfig) -> None:
    params = payload.params
    if cfg.production_mode and bool(params.get("dry_run")):
        raise HTTPException(status_code=400, detail="生产模式禁止 dry-run 任务")
    if payload.type == "bos_import":
        if not str(params.get("prefix") or "").strip():
            raise HTTPException(status_code=400, detail="bos_import requires prefix")
        if not 1 <= int(params.get("max_objects", 1000)) <= 1000:
            raise HTTPException(status_code=400, detail="max_objects must be between 1 and 1000")
    if payload.type == "motrixsim_asset_import":
        if not 1 <= int(params.get("validation_steps", 120)) <= 5000:
            raise HTTPException(status_code=400, detail="validation_steps must be between 1 and 5000")
        if not 1 <= int(params.get("max_parlour_meshes", 240)) <= 1000:
            raise HTTPException(status_code=400, detail="max_parlour_meshes must be between 1 and 1000")
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
        supported_policy = "rm65_pick_place_6d_ik" if cfg.production_mode else "pick_and_place_v1"
        if params.get("policy_preset", supported_policy) != supported_policy:
            raise HTTPException(status_code=400, detail=f"当前运行模式仅允许真实控制器 {supported_policy}")
        if int(params.get("workers", 1)) != 1:
            raise HTTPException(status_code=400, detail="当前专家采集执行器仅支持 workers=1")
        if int(params.get("episodes", 4)) <= 0 or int(params.get("steps", 24)) <= 0:
            raise HTTPException(status_code=400, detail="episodes and steps must be > 0")
    if payload.type == "training":
        if not (params.get("dataset_id") or params.get("dataset_path")):
            raise HTTPException(status_code=400, detail="training requires dataset_id or dataset_path")
        allowed_recipes = {"discoverse_act"} if cfg.production_mode else {"bc_v2", "bc_fast", "discoverse_act"}
        if params.get("recipe", "discoverse_act" if cfg.production_mode else "bc_v2") not in allowed_recipes:
            raise HTTPException(status_code=400, detail="生产模式只允许 DISCOVERSE ACT；历史岭回归配方仅可查看")
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
                model = store.get_model(str(params["model_id"]))
                recipe = str((model.get("metadata") or {}).get("training", {}).get("recipe") or (model.get("metadata") or {}).get("recipe") or "")
                if params.get("recipe") == "discoverse_act" and recipe and recipe != "discoverse_act":
                    raise HTTPException(status_code=400, detail="所选模型不是 DISCOVERSE ACT 模型")
            if params.get("scene_id"):
                store.get_scene(str(params["scene_id"]))
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"任务引用的资源不存在: {exc.args[0]}") from exc


def _active_admin_count(store: PlatformStore) -> int:
    return sum(1 for user in store.list_users() if user["role"] == "admin" and not user["disabled"])


def _check_upload_owner(upload: dict[str, Any], user: dict[str, Any]) -> None:
    if user.get("role") != "admin" and upload.get("owner_user_id") != user.get("id"):
        raise HTTPException(status_code=403, detail="无权访问其他用户的上传任务")


def _dataset_for_storage_uri(store: PlatformStore, storage_uri: str) -> dict[str, Any] | None:
    return next((item for item in store.list_datasets() if item.get("storage_uri") == storage_uri), None)


def _sqlite_journal_mode(store: PlatformStore) -> str:
    with store._connect() as db:
        return str(db.execute("pragma journal_mode").fetchone()[0]).lower()


def _capabilities(cfg: PlatformConfig, storage: BosStorage, store: PlatformStore | None = None) -> dict[str, Any]:
    anysplat_commit = _git_commit_full(cfg.anysplat_root)
    discoverse_commit = _git_commit_full(cfg.discoverse_root)
    act_root = cfg.discoverse_root / "policies" / "act"
    act_commit = _git_commit_full(act_root)
    ffmpeg_ready = shutil.which("ffmpeg") is not None
    colmap_ready = cfg.colmap_binary.is_file() or shutil.which("colmap") is not None
    mujoco_ready = _module_available("mujoco")
    motrixsim_ready = _module_available("motrixsim") or bool(shutil.which("motrixsim"))
    torch_ready = _module_available("torch")
    rm65_controller = Path(__file__).with_name("rm65_controller.py")
    local_runtime = {
        "ffmpeg": ffmpeg_ready,
        "colmap": colmap_ready,
        "anysplat": cfg.anysplat_root.is_dir() and cfg.anysplat_python.is_file() and anysplat_commit == cfg.anysplat_commit,
        "mujoco": mujoco_ready,
        "motrixsim": motrixsim_ready,
        "discoverse_act": cfg.discoverse_root.is_dir() and act_root.is_dir() and discoverse_commit == cfg.discoverse_commit and act_commit == cfg.act_commit and torch_ready,
        "rm65_controller": rm65_controller.is_file() and mujoco_ready,
    }
    worker_status: dict[str, Any] = {"connected": cfg.worker_mode != "api", "age_seconds": 0 if cfg.worker_mode != "api" else None}
    if store is not None and cfg.worker_mode == "api":
        heartbeats = store.list_worker_heartbeats()
        latest = heartbeats[0] if heartbeats else None
        age = time.time() - float(latest["updated_at"]) if latest else None
        connected = bool(latest and age is not None and age < 30)
        worker_status = {"connected": connected, "age_seconds": age, "worker_id": latest.get("worker_id") if latest else None}
        if connected:
            checks = latest["payload"].get("checks", {})
            local_runtime.update(
                ffmpeg=bool(checks.get("ffmpeg")),
                colmap=bool(checks.get("colmap")),
                anysplat=all(bool(checks.get(name)) for name in ("anysplat_root", "anysplat_python", "anysplat_pinned")),
                mujoco=bool(checks.get("mujoco_python")),
                motrixsim=bool(checks.get("motrixsim_python")),
                discoverse_act=all(bool(checks.get(name)) for name in ("discoverse_act", "torch_cuda", "act_import")),
                rm65_controller=bool(checks.get("rm65_controller")) and bool(checks.get("mujoco_python")),
            )
        else:
            for name in local_runtime:
                local_runtime[name] = False
    return {
        "schema": "real2sim.capabilities.v1",
        "production_mode": cfg.production_mode,
        "auth_enabled": cfg.auth_enabled,
        "worker_mode": cfg.worker_mode,
        "worker": worker_status,
        "storage": {
            "bos": {
                "configured": storage.configured,
                "bucket": storage.bucket,
                "prefix": storage.prefix,
                "resumable_upload": True,
                "sha256": True,
            }
        },
        "runtimes": {
            "ffmpeg": {"ready": local_runtime["ffmpeg"]},
            "colmap": {"ready": local_runtime["colmap"]},
            "anysplat": {
                "ready": local_runtime["anysplat"],
                "commit": anysplat_commit,
                "required_commit": cfg.anysplat_commit,
                "model_revision": cfg.anysplat_model_revision,
                "vggt_model_revision": cfg.vggt_model_revision,
            },
            "mujoco": {"ready": local_runtime["mujoco"]},
            "motrixsim": {"ready": local_runtime["motrixsim"]},
            "discoverse_act": {
                "ready": local_runtime["discoverse_act"],
                "discoverse_commit": discoverse_commit,
                "required_discoverse_commit": cfg.discoverse_commit,
                "act_commit": act_commit,
                "required_act_commit": cfg.act_commit,
            },
            "rm65_controller": {"ready": local_runtime["rm65_controller"]},
            "spark": {"ready": SPARK_VENDOR_PATH.is_file()},
        },
    }


def _git_commit_full(root: Path) -> str:
    if not root.is_dir():
        return ""
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=3).strip()
    except Exception:
        return ""


def _module_available(name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _runtime_paths(cfg: PlatformConfig) -> dict[str, str]:
    return {
        "gs_playground_root": str(cfg.gs_playground_root),
        "runtime_root": str(cfg.runtime_root),
        "anysplat_root": str(cfg.anysplat_root),
        "anysplat_python": str(cfg.anysplat_python),
        "gsplat_python": str(cfg.gsplat_python),
        "cuda_home": str(cfg.cuda_home),
        "runtime_env_script": str(cfg.runtime_env_script),
        "discoverse_commit": cfg.discoverse_commit,
        "act_commit": cfg.act_commit,
        "anysplat_commit": cfg.anysplat_commit,
        "anysplat_model_revision": cfg.anysplat_model_revision,
        "vggt_model_revision": cfg.vggt_model_revision,
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
