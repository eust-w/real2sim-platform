from __future__ import annotations

import json
import hashlib
from functools import lru_cache
import math
import os
import shutil
import struct
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote, urlparse
from urllib.request import urlopen

from .config import PlatformConfig
from .source_revision import read_source_revision
from .storage import ALLOWED_AREAS, BosStorage, normalize_relative_path
from .store import PlatformStore


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".ppm"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v"}
PARQUET_SUFFIXES = {".parquet"}
SCENE_ASSET_SUFFIXES = {".ply", ".splat", ".spz", ".ksplat", ".sog", ".zip", ".rad", ".obj", ".glb", ".gltf", ".xml", ".json", ".png", ".jpg", ".jpeg"}
SPARK_ASSET_SUFFIXES = {".ply", ".splat", ".spz", ".ksplat", ".sog", ".zip", ".rad"}
SPARK_VENDOR_PATH = Path(__file__).parent / "static" / "vendor" / "spark.module.min.js"
THREE_VENDOR_PATH = Path(__file__).parent / "static" / "vendor" / "three" / "three.module.js"
VIEWER_AUTOLOAD_MAX_BYTES = 25 * 1024 * 1024
MEDIA_PREPARE_DEFAULT_CAMERA = "observation.images.front"
MEDIA_PREPARE_DEFAULT_SAMPLE_FPS = 2.0
MEDIA_PREPARE_DEFAULT_MAX_FRAMES = 80
RECONSTRUCTION_DEFAULT_MIN_FRAMES = 30
RECONSTRUCTION_DEFAULT_TARGET_FRAMES = 80
SIM_VALIDATION_LEVELS = {"smoke", "runtime_if_available", "runtime_required"}


class TaskCanceled(RuntimeError):
    pass


def scan_dataset(path: Path) -> dict[str, Any]:
    root = path.resolve()
    files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
    images = [p for p in files if p.suffix.lower() in IMAGE_SUFFIXES and "depth" not in {part.lower() for part in p.parts}]
    depths = [p for p in files if p.suffix.lower() in {".exr", ".npy", ".png"} and "depth" in {part.lower() for part in p.parts}]
    videos = [p for p in files if p.suffix.lower() in VIDEO_SUFFIXES]
    parquets = [p for p in files if p.suffix.lower() in PARQUET_SUFFIXES]
    metadata = [p for p in files if p.name in {"metadata.json", "transforms.json", "calibration.json"} or p.suffix.lower() == ".jsonl"]
    prepared_manifest = root / "prepared_manifest.json" if root.is_dir() else None
    input_kind = "prepared_frames" if prepared_manifest and prepared_manifest.is_file() else "video" if videos or root.suffix.lower() in VIDEO_SUFFIXES else "parquet_video" if parquets else "rgbd" if depths else "images"
    return {
        "path": str(root),
        "image_count": len(images),
        "depth_count": len(depths),
        "video_count": len(videos),
        "parquet_count": len(parquets),
        "metadata_files": [str(p.relative_to(root)) if root.is_dir() else p.name for p in metadata[:20]],
        "prepared_manifest": str(prepared_manifest) if prepared_manifest and prepared_manifest.is_file() else "",
        "input_kind": input_kind,
    }


def run_doctor(config: PlatformConfig) -> dict[str, Any]:
    core_checks = (
        [
            "workspace_writable",
            "rm65_controller",
            "discoverse_act",
            "torch_cuda",
            "act_import",
            "media_prepare_deps",
            "ffmpeg",
            "colmap",
            "gs_video_reconstruct",
            "anysplat_pinned",
            "mujoco_python",
            "motrixsim_python",
            "nvidia_smi",
        ]
        if config.production_mode
        else ["workspace_writable", "expert_policy", "training", "inference", "media_prepare_deps", "ffmpeg", "spark_bundle", "three_bundle"]
    )
    act_root = config.discoverse_root / "policies" / "act"
    checks = {
        "workspace_writable": _is_writable(config.workspace_dir),
        "expert_policy": (config.expert_policy_root / "src/real2sim_expert/collect.py").is_file(),
        "training": (config.training_root / "src/real2sim_train/train.py").is_file(),
        "inference": (config.inference_root / "src/real2sim_infer/run.py").is_file(),
        "gs_video_reconstruct": (config.gs_playground_root / "scripts/real2sim_video_motrix_scene.py").is_file(),
        "images23dgs": (config.images23dgs_root / "images23dgs/product/server.py").is_file(),
        "colmap": shutil.which(str(config.colmap_binary)) is not None or config.colmap_binary.exists(),
        "gsplat_python": config.gsplat_python.exists(),
        "anysplat_root": config.anysplat_root.is_dir(),
        "anysplat_python": config.anysplat_python.exists(),
        "runtime_env_script": config.runtime_env_script.is_file(),
        "cuda_home": config.cuda_home.is_dir(),
        "media_prepare_deps": all(_module_available(name) for name in ["huggingface_hub", "pyarrow", "PIL", "imageio", "cv2"]),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "spark_bundle": SPARK_VENDOR_PATH.is_file(),
        "three_bundle": THREE_VENDOR_PATH.is_file(),
        "mujoco_python": _module_available("mujoco"),
        "service_process": os.getpid() > 0,
        "nvidia_smi": shutil.which("nvidia-smi") is not None,
        "rm65_controller": Path(__file__).with_name("rm65_controller.py").is_file(),
        "discoverse_act": _git_head(config.discoverse_root) == config.discoverse_commit and _git_head(act_root) == config.act_commit,
        "torch_cuda": _torch_cuda_ready(),
        "act_import": _act_runtime_import_ready(str(config.discoverse_root), config.discoverse_commit, config.act_commit),
        "anysplat_pinned": _git_head(config.anysplat_root) == config.anysplat_commit and config.anysplat_python.exists(),
        "motrixsim_python": _module_available("motrixsim"),
    }
    return {
        "ok": all(bool(checks[name]) for name in core_checks),
        "core_checks": core_checks,
        "warnings": [
            name
            for name in [
                "images23dgs",
                "mujoco_python",
                "gs_video_reconstruct",
                "gsplat_python",
                "anysplat_root",
                "anysplat_python",
                "runtime_env_script",
                "cuda_home",
                "colmap",
                "nvidia_smi",
                "torch_cuda",
                "act_import",
                "three_bundle",
            ]
            if not checks.get(name)
        ],
        "checks": checks,
        "paths": {
            "workspace": str(config.workspace_dir),
            "gs_playground_root": str(config.gs_playground_root),
            "images23dgs_root": str(config.images23dgs_root),
            "colmap_binary": str(config.colmap_binary),
            "gsplat_python": str(config.gsplat_python),
            "anysplat_root": str(config.anysplat_root),
            "anysplat_python": str(config.anysplat_python),
            "runtime_root": str(config.runtime_root),
            "runtime_env_script": str(config.runtime_env_script),
            "cuda_home": str(config.cuda_home),
            "spark_bundle": str(SPARK_VENDOR_PATH),
            "three_bundle": str(THREE_VENDOR_PATH),
        },
        "renderers": renderer_decision(),
        "simulators": ["mujoco", "motrixsim"],
    }


def execute_task(config: PlatformConfig, store: PlatformStore, task: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(task["run_dir"])
    log_path = run_dir / "logs" / "task.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"task {task['id']} type={task['type']} started\n", encoding="utf-8")
    emit = lambda message: _append_log(log_path, message)
    _task_progress(store, task, 8, "准备执行", "正在校验任务参数和运行目录")
    task_type = task["type"]
    if task_type == "scene_reconstruction":
        return run_scene_reconstruction(config, store, task, emit)
    if task_type == "media_prepare":
        return run_media_prepare(config, store, task, emit)
    if task_type == "reconstruction_preflight":
        return run_reconstruction_preflight(config, store, task, emit)
    if task_type == "hf_import":
        return run_hf_import(config, store, task, emit)
    if task_type == "bos_import":
        return run_bos_import(config, store, task, emit)
    if task_type == "motrixsim_asset_import":
        return run_motrixsim_asset_import(config, store, task, emit)
    if task_type == "simulator_validation":
        return run_simulator_validation(config, store, task, emit)
    if task_type == "expert_collection":
        return run_expert_collection(config, store, task, emit)
    if task_type == "training":
        return run_training(config, store, task, emit)
    if task_type == "inference":
        return run_inference(config, store, task, emit)
    if task_type == "renderer_benchmark":
        return run_renderer_benchmark(config, store, task, emit)
    if task_type == "scene_cover_render":
        return run_scene_cover_render(config, store, task, emit)
    if task_type == "scene_splat_optimize":
        return run_scene_splat_optimize(config, store, task, emit)
    raise ValueError(f"unknown task type: {task_type}")


def sync_task_outputs(config: PlatformConfig, store: PlatformStore, task: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    storage = BosStorage(config)
    if not storage.configured:
        if config.production_mode:
            raise RuntimeError("生产任务产物无法持久化：BOS 尚未配置")
        return result
    _task_progress(store, task, 96, "持久化产物", "正在将任务报告和业务产物上传到 BOS")
    run_dir = Path(task["run_dir"])
    if run_dir.is_dir():
        _upload_directory(storage, run_dir, area="runs", relative_root=task["id"])
    resource_specs = [
        ("dataset_id", "datasets", store.get_dataset),
        ("scene_id", "scenes", store.get_scene),
        ("model_id", "models", store.get_model),
    ]
    persisted: dict[str, Any] = {}
    for result_key, area, getter in resource_specs:
        entity_id = str(result.get(result_key) or "")
        if not entity_id:
            continue
        entity = getter(entity_id)
        path = Path(str(entity["path"]))
        root = path if path.is_dir() else path.parent
        if not root.is_dir():
            raise FileNotFoundError(f"resource output directory not found: {root}")
        _upload_directory(storage, root, area=area, relative_root=entity_id)
        checksum = _directory_digest(root)
        uri = storage.uri_for_key(storage.object_key(area, f"{entity_id}/"))
        store.update_entity_storage(area, entity_id, storage_uri=uri, checksum=checksum, cache_state="local")
        persisted[result_key] = {"storage_uri": uri, "checksum": checksum}
    result = dict(result)
    result["persistent_outputs"] = persisted
    result["run_storage_uri"] = storage.uri_for_key(storage.object_key("runs", f"{task['id']}/"))
    return result


def _task_progress(store: PlatformStore, task: dict[str, Any] | str, progress: int, stage: str, message: str = "") -> None:
    task_id = task if isinstance(task, str) else task["id"]
    store.update_task(task_id, progress=progress, stage=stage, message=message)


def run_scene_reconstruction(config: PlatformConfig, store: PlatformStore, task: dict[str, Any], emit: Callable[[str], None]) -> dict[str, Any]:
    params = task["params"]
    run_dir = Path(task["run_dir"])
    package_dir = run_dir / "scene_package"
    work_dir = run_dir / "work"
    dry_run = bool(params.get("dry_run", False))
    _task_progress(store, task, 12, "解析输入", "正在解析数据集和重建参数")
    requested_input = _task_input_path(store, params)
    input_path = requested_input if dry_run else _resolve_reconstruction_input(config, store, task, emit)
    renderer = _select_renderer(params.get("renderer", config.default_renderer), package_dir)
    simulators = params.get("simulators", ["mujoco", "motrixsim"])
    if dry_run:
        _task_progress(store, task, 35, "生成样例场景", "正在创建可预览场景包")
        external_assets = _add_preview_asset_artifacts(store, task["id"], input_path)
        scene_path = _write_dry_run_scene(package_dir, input_path, renderer, simulators)
        manifest_path = _write_scene_manifest(package_dir, input_path=input_path, renderer=renderer, simulators=simulators, dry_run=True, external_assets=external_assets)
        scene = store.create_scene(
            name=params.get("scene_name", "dry-run scene"),
            path=scene_path,
            metadata={"renderer": renderer, "simulators": simulators, "dry_run": True, "manifest_path": str(manifest_path)},
            source_type="scene_reconstruction",
            source_uri=str(input_path),
            parent_task_id=task["id"],
            manifest_path=manifest_path,
            tags=["dry_run", renderer, *simulators],
        )
        _index_artifacts(store, task["id"], package_dir)
        _task_progress(store, task, 95, "索引产物", "场景包和预览产物已写入")
        return {"scene_id": scene["id"], "scene_path": str(scene_path), "manifest_path": str(manifest_path), "dry_run": True, "renderer": renderer, "simulators": simulators}

    script = config.gs_playground_root / "scripts/real2sim_video_motrix_scene.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    work_dir.mkdir(parents=True, exist_ok=True)
    _task_progress(store, task, 22, "重建预检", "正在检查帧数、清晰度、视角变化和运行时")
    preflight = build_reconstruction_preflight_report(config, store, params=params, input_path=input_path, run_dir=work_dir, emit=emit)
    preflight_required = bool(params.get("preflight_required", True))
    if preflight_required and not preflight.get("ok") and not bool(params.get("allow_low_quality", False)):
        readable = preflight.get("readable_error") or "reconstruction preflight failed"
        acceptance = {
            "schema": "real2sim.capture_acceptance.v1",
            "status": "not_run",
            "reason": readable,
            "preflight_report": str(work_dir / "capture_preflight_report.json"),
        }
        (work_dir / "capture_acceptance_report.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report = {
            "schema": "real2sim.reconstruction_report.v1",
            "status": "failed",
            "error": readable,
            "readable_error": readable,
            "requested_input": str(requested_input),
            "resolved_input": str(input_path),
            "package_dir": str(package_dir),
            "work_dir": str(work_dir),
            "preflight_report": str(work_dir / "capture_preflight_report.json"),
        }
        (work_dir / "reconstruction_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise ValueError(readable)
    _task_progress(store, task, 35, "准备重建输入", "正在写入输入清单、环境信息和命令参数")
    input_report = {
        "schema": "real2sim.reconstruction_input.v1",
        "requested_input": str(requested_input),
        "resolved_input": str(input_path),
        "scan": scan_dataset(input_path),
        "auto_prepare": bool(params.get("auto_prepare", False)),
        "prepared_manifest": _read_optional_json(_prepared_manifest_path_for_input(input_path)),
    }
    (work_dir / "reconstruction_input.json").write_text(json.dumps(input_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    env_report = {
        "schema": "real2sim.reconstruction_environment.v1",
        "python": sys.version,
        "executable": sys.executable,
        "ffmpeg": shutil.which("ffmpeg") or "",
        "colmap": str(config.colmap_binary),
        "gsplat_python": str(config.gsplat_python),
        "anysplat_root": str(config.anysplat_root),
        "anysplat_python": str(config.anysplat_python),
        "runtime_root": str(config.runtime_root),
        "runtime_env_script": str(config.runtime_env_script),
        "cuda_home": str(config.cuda_home),
        "gs_playground_root": str(config.gs_playground_root),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (work_dir / "environment.json").write_text(json.dumps(env_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reconstruction_python = config.anysplat_python if config.anysplat_python.exists() else config.gsplat_python if config.gsplat_python.exists() else Path(sys.executable)
    quality_preset = _normalize_reconstruction_quality(str(params.get("quality_preset", "discoverse-high")))
    command = [
        str(reconstruction_python),
        str(script),
        "--output",
        str(package_dir),
        "--work-dir",
        str(work_dir),
        "--quality-preset",
        quality_preset,
        "--sfm-report",
        "--capture-acceptance",
    ]
    anysplat_root = Path(str(params.get("anysplat_root") or config.anysplat_root))
    if str(anysplat_root):
        command.extend(["--anysplat-root", str(anysplat_root)])
    if str(config.anysplat_python):
        command.extend(["--anysplat-python", str(config.anysplat_python)])
    if Path(input_path).is_file() and Path(input_path).suffix.lower() in VIDEO_SUFFIXES:
        command.extend(["--video", str(input_path)])
    else:
        command.extend(["--frames-dir", str(input_path)])
    if params.get("pose_init_train", True):
        command.append("--pose-init-train")
    _append_discoverse_reconstruction_flags(command, params=params, quality_preset=quality_preset)
    if params.get("allow_low_quality", False):
        command.append("--allow-low-quality")
    (work_dir / "reconstruction_command.json").write_text(json.dumps({"command": command, "cwd": str(config.gs_playground_root)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    env = _env_with_paths(config)
    _task_progress(store, task, 48, "运行真实重建", "AnySplat/上游重建命令正在执行")
    try:
        run_logged(command, cwd=config.gs_playground_root, log_path=run_dir / "logs" / "task.log", env=env, store=store, task_id=task["id"])
    except Exception as exc:
        log_tail = _tail_text(run_dir / "logs" / "task.log")
        report = {
            "schema": "real2sim.reconstruction_report.v1",
            "status": "failed",
            "error": str(exc),
            "readable_error": _readable_failure_from_log(log_tail) or str(exc),
            "log_tail": log_tail,
            "requested_input": str(requested_input),
            "resolved_input": str(input_path),
            "package_dir": str(package_dir),
            "work_dir": str(work_dir),
        }
        (work_dir / "reconstruction_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    _task_progress(store, task, 82, "封装场景", "正在校验重建产物并生成统一场景包")
    postprocess = _postprocess_discoverse_scene_assets(package_dir, work_dir, emit)
    scene_path = _find_scene_json(package_dir) or (package_dir / "motrix_import.json")
    manifest_path = _write_scene_manifest(package_dir, input_path=input_path, renderer=renderer, simulators=simulators, dry_run=False, external_assets=[])
    report = {
        "schema": "real2sim.reconstruction_report.v1",
        "status": "succeeded",
        "requested_input": str(requested_input),
        "resolved_input": str(input_path),
        "scene_path": str(scene_path),
        "manifest_path": str(manifest_path),
        "package_dir": str(package_dir),
        "preflight_report": str(work_dir / "capture_preflight_report.json"),
        "discoverse_postprocess": postprocess,
        "asset_count": len(_candidate_asset_files(package_dir)),
    }
    (work_dir / "reconstruction_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    scene = store.create_scene(
        name=params.get("scene_name", Path(input_path).stem),
        path=scene_path,
        metadata={"renderer": renderer, "simulators": simulators, "package": str(package_dir), "manifest_path": str(manifest_path)},
        source_type="scene_reconstruction",
        source_uri=str(input_path),
        parent_task_id=task["id"],
        manifest_path=manifest_path,
        tags=[renderer, *simulators],
    )
    auto_cover: dict[str, Any] = {"status": "skipped"}
    if bool(params.get("auto_cover", True)):
        _task_progress(store, task, 90, "生成预览封面", "正在用 Scene Viewer 自动生成场景封面")
        try:
            auto_cover = _generate_scene_cover(config, store, task["id"], scene["id"], manifest_path, emit=emit)
        except Exception as exc:
            auto_cover = {"status": "warning", "error": str(exc)}
            emit(f"auto scene cover skipped: {type(exc).__name__}: {exc}")
    _index_artifacts(store, task["id"], run_dir)
    _task_progress(store, task, 95, "索引产物", "重建报告、场景包和仿真产物已登记")
    return {"scene_id": scene["id"], "scene_path": str(scene_path), "manifest_path": str(manifest_path), "package_dir": str(package_dir), "renderer": renderer, "simulators": simulators, "auto_cover": auto_cover}


def run_media_prepare(config: PlatformConfig, store: PlatformStore, task: dict[str, Any], emit: Callable[[str], None]) -> dict[str, Any]:
    _task_progress(store, task, 18, "准备媒体", "正在解析视频、图片和数据清单")
    prepared = prepare_media_dataset(config, store, params=task["params"], parent_task_id=task["id"], task_id=task["id"], run_dir=Path(task["run_dir"]), emit=emit)
    _task_progress(store, task, 88, "登记数据集", "正在写入 prepared manifest 和平台索引")
    _index_artifacts(store, task["id"], Path(prepared["dataset_path"]))
    _task_progress(store, task, 96, "索引产物", "抽帧结果和数据清单已登记")
    return prepared


def run_reconstruction_preflight(config: PlatformConfig, store: PlatformStore, task: dict[str, Any], emit: Callable[[str], None]) -> dict[str, Any]:
    params = task["params"]
    run_dir = Path(task["run_dir"])
    _task_progress(store, task, 20, "解析输入", "正在读取已准备帧数据")
    input_path = _resolve_preflight_input(store, params)
    _task_progress(store, task, 45, "质量预检", "正在检查帧数、分辨率、模糊度和视角变化")
    report = build_reconstruction_preflight_report(config, store, params=params, input_path=input_path, run_dir=run_dir, emit=emit)
    store.add_artifact(task_id=task["id"], name="capture_preflight_report.json", path=run_dir / "capture_preflight_report.json", kind="json")
    _task_progress(store, task, 95, "生成报告", "预检报告已生成")
    return {"ok": report["ok"], "report_path": str(run_dir / "capture_preflight_report.json"), "report": report}


def build_reconstruction_preflight_report(
    config: PlatformConfig,
    store: PlatformStore,
    *,
    params: dict[str, Any],
    input_path: Path,
    run_dir: Path,
    emit: Callable[[str], None],
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    min_frames = int(params.get("min_frames", RECONSTRUCTION_DEFAULT_MIN_FRAMES))
    target_frame_count = int(params.get("target_frame_count", RECONSTRUCTION_DEFAULT_TARGET_FRAMES))
    prepared_manifest = _read_optional_json(_prepared_manifest_path_for_input(input_path))
    frame_paths = _frame_paths_for_preflight(input_path)
    scan = scan_dataset(input_path) if input_path.exists() else {"path": str(input_path), "input_kind": "missing"}
    runtime_checks = {
        "gs_video_reconstruct": (config.gs_playground_root / "scripts/real2sim_video_motrix_scene.py").is_file(),
        "anysplat_root": Path(str(params.get("anysplat_root") or config.anysplat_root)).is_dir(),
        "anysplat_python": config.anysplat_python.exists(),
        "colmap": shutil.which(str(config.colmap_binary)) is not None or config.colmap_binary.exists(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }
    quality = _frame_quality_summary(frame_paths)
    failed_gates: list[str] = []
    warnings: list[str] = []
    if input_path.is_file() and input_path.suffix.lower() in VIDEO_SUFFIXES:
        warnings.append("input is a video file; run media_prepare first for frame-level quality gates")
    if len(frame_paths) < min_frames:
        failed_gates.append("enough_frames")
        warnings.append(f"only {len(frame_paths)} frame(s), expected at least {min_frames}; run media_prepare with more videos or choose a reconstruction capture")
    if not runtime_checks["gs_video_reconstruct"]:
        failed_gates.append("gs_video_reconstruct")
    if not runtime_checks["anysplat_root"]:
        failed_gates.append("anysplat_root")
    if not runtime_checks["anysplat_python"]:
        failed_gates.append("anysplat_python")
    if not runtime_checks["ffmpeg"]:
        failed_gates.append("ffmpeg")
    if quality["width"] and quality["height"] and (quality["width"] < 320 or quality["height"] < 240):
        warnings.append("input frames are below 320x240; reconstruction quality may be poor")
    if quality["mean_frame_delta"] is not None and quality["mean_frame_delta"] < 2.0 and len(frame_paths) > 1:
        warnings.append("adjacent frames are very similar; capture may lack camera motion/parallax")

    allow_low_quality = bool(params.get("allow_low_quality", False))
    ok = not failed_gates
    readable_error = "" if ok else "reconstruction preflight failed: " + ", ".join(failed_gates)
    readiness_score = _preflight_readiness_score(frame_count=len(frame_paths), min_frames=min_frames, runtime_checks=runtime_checks, warnings=warnings)
    next_action = _preflight_next_action(ok=ok, allow_low_quality=allow_low_quality, failed_gates=failed_gates, warnings=warnings)
    report = {
        "schema": "real2sim.capture_preflight.v1",
        "status": "succeeded" if ok else "failed",
        "ok": ok,
        "can_run": ok or allow_low_quality,
        "allow_low_quality": allow_low_quality,
        "readiness_score": readiness_score,
        "next_action": next_action,
        "readable_error": readable_error,
        "input_path": str(input_path),
        "scan": scan,
        "prepared_manifest": prepared_manifest,
        "frame_count": len(frame_paths),
        "min_frames": min_frames,
        "target_frame_count": target_frame_count,
        "runtime_checks": runtime_checks,
        "quality": quality,
        "failed_gates": failed_gates,
        "warnings": warnings,
        "recommendations": _preflight_recommendations(failed_gates, warnings),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_dir / "capture_preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    emit(f"reconstruction preflight {'passed' if ok else 'failed'}: {len(frame_paths)} frame(s)")
    return report


def run_hf_import(config: PlatformConfig, store: PlatformStore, task: dict[str, Any], emit: Callable[[str], None]) -> dict[str, Any]:
    params = task["params"]
    repo_id = str(params.get("repo_id", "")).strip()
    repo_type = str(params.get("repo_type", "dataset")).strip() or "dataset"
    if not repo_id or "/" not in repo_id:
        raise ValueError("hf_import requires repo_id like namespace/name")
    if repo_type != "dataset":
        raise ValueError("hf_import only supports repo_type='dataset'")
    revision = str(params.get("revision", "")).strip() or None
    allow_patterns = _normalize_patterns(params.get("allow_patterns"))
    dataset_id = uuid.uuid4().hex
    local_name = _safe_name(str(params.get("local_name") or repo_id.split("/")[-1]))
    target = config.datasets_dir / dataset_id
    target.mkdir(parents=True, exist_ok=True)
    emit(f"downloading {repo_id} to {target}\n")
    _task_progress(store, task, 18, "下载数据", f"正在从 HuggingFace 下载 {repo_id}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for hf_import") from exc
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            allow_patterns=allow_patterns or None,
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )
    except TypeError:
        snapshot_download(repo_id=repo_id, repo_type=repo_type, revision=revision, allow_patterns=allow_patterns or None, local_dir=str(target))
    scan = scan_dataset(target)
    _task_progress(store, task, 78, "扫描数据", "正在统计图片、视频和元数据文件")
    manifest = {
        "schema": "real2sim.dataset_manifest.v1",
        "id": dataset_id,
        "name": local_name,
        "source_type": "huggingface",
        "source_uri": f"https://huggingface.co/datasets/{repo_id}",
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision or "main",
        "allow_patterns": allow_patterns,
        "scan": scan,
        "files": _file_inventory(target, limit=200),
    }
    manifest_path = target / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset = store.create_dataset(
        item_id=dataset_id,
        name=local_name,
        path=target,
        scan=scan,
        source_type="huggingface",
        source_uri=manifest["source_uri"],
        version=revision or "main",
        parent_task_id=task["id"],
        manifest_path=manifest_path,
        tags=["huggingface", repo_id],
    )
    _index_artifacts(store, task["id"], target)
    _task_progress(store, task, 95, "登记数据集", "数据集清单和平台索引已生成")
    return {"dataset_id": dataset["id"], "dataset_path": str(target), "manifest_path": str(manifest_path), "scan": scan}


def run_bos_import(config: PlatformConfig, store: PlatformStore, task: dict[str, Any], emit: Callable[[str], None]) -> dict[str, Any]:
    params = task["params"]
    storage = BosStorage(config)
    raw_prefix = normalize_relative_path(str(params.get("prefix") or ""))
    prefix_parts = raw_prefix.split("/", 1)
    area = prefix_parts[0] if prefix_parts[0] in ALLOWED_AREAS else "datasets"
    relative_prefix = prefix_parts[1] if prefix_parts[0] in ALLOWED_AREAS and len(prefix_parts) > 1 else raw_prefix
    _task_progress(store, task, 18, "读取 BOS", "正在列举授权前缀内的对象")
    objects = storage.list_objects(area=area, relative_prefix=relative_prefix, max_keys=int(params.get("max_objects", 1000)))
    objects = [item for item in objects if not item["key"].endswith("/")]
    if not objects:
        raise ValueError("BOS 前缀中没有可导入对象")
    dataset_id = uuid.uuid4().hex
    out_dir = config.datasets_dir / dataset_id
    out_dir.mkdir(parents=True, exist_ok=False)
    common_prefix = storage.object_key(area, relative_prefix)
    _task_progress(store, task, 30, "下载数据", f"正在下载 {len(objects)} 个真实 BOS 对象")
    for index, item in enumerate(objects, start=1):
        relative = item["key"][len(common_prefix):].lstrip("/") or Path(item["key"]).name
        storage.download_file(item["storage_uri"], out_dir / relative)
        if index % max(1, len(objects) // 10) == 0:
            _task_progress(store, task, 30 + int(45 * index / len(objects)), "下载数据", f"已下载 {index}/{len(objects)} 个对象")
    scan = scan_dataset(out_dir)
    digest = _directory_digest(out_dir)
    source_uri = f"bos://{storage.bucket}/{common_prefix}"
    manifest = {
        "schema": "real2sim.dataset_manifest.v2",
        "id": dataset_id,
        "name": params.get("name") or Path(relative_prefix.rstrip("/")).name,
        "source_type": "bos_prefix",
        "storage_uri": source_uri,
        "checksum": digest,
        "version": digest[:12],
        "object_count": len(objects),
        "objects": objects,
        "scan": scan,
    }
    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset = store.create_dataset(
        item_id=dataset_id,
        name=str(manifest["name"]),
        path=out_dir,
        scan=scan,
        source_type="bos_prefix",
        source_uri=source_uri,
        version=digest[:12],
        parent_task_id=task["id"],
        manifest_path=manifest_path,
        metrics={"object_count": len(objects), "size_bytes": sum(int(item["size_bytes"]) for item in objects)},
        tags=["bos", "imported"],
        storage_uri=source_uri,
        checksum=digest,
        cache_state="local",
    )
    _index_artifacts(store, task["id"], out_dir)
    _task_progress(store, task, 95, "登记数据集", "BOS 数据及校验清单已登记")
    return {"dataset_id": dataset["id"], "dataset_path": str(out_dir), "storage_uri": source_uri, "object_count": len(objects), "checksum": digest}


def run_motrixsim_asset_import(config: PlatformConfig, store: PlatformStore, task: dict[str, Any], emit: Callable[[str], None]) -> dict[str, Any]:
    """Import the pinned MotrixSim package and prove both simulator runtimes load it."""
    params = task["params"]
    run_dir = Path(task["run_dir"])
    report_path = run_dir / "motrixsim_asset_import_report.json"
    source = Path(str(params.get("source") or "/opt/motrixsim-docs"))
    scene_id = hashlib.sha1(b"motrixsim:parlour_rm65_pick_place").hexdigest()[:32]
    scripts = config.app_root / "scripts"
    import_script = scripts / "import_motrixsim_assets.py"
    runtime_script = scripts / "generate_motrixsim_runtime_scene.py"
    preview_script = scripts / "generate_motrixsim_preview.py"
    missing = [str(path) for path in (import_script, runtime_script, preview_script) if not path.is_file()]
    if missing:
        raise FileNotFoundError("MotrixSim asset pipeline scripts are missing: " + ", ".join(missing))
    if not source.is_dir():
        raise FileNotFoundError(f"pinned MotrixSim source checkout not found: {source}")

    env = _env_with_paths(config)
    log_path = run_dir / "logs" / "task.log"
    _task_progress(store, task, 15, "导入官方资产", "正在复制固定版本的 MotrixSim 客厅、RM65 与 pick-place 资产")
    run_logged(
        [
            sys.executable,
            str(import_script),
            "--source",
            str(source),
            "--name",
            str(params.get("name") or "MotrixSim 客厅 RM65 Pick-Place"),
            "--replace-index",
        ],
        cwd=config.app_root,
        log_path=log_path,
        env=env,
        store=store,
        task_id=task["id"],
    )

    _task_progress(store, task, 48, "验证仿真运行时", "正在使用 MuJoCo 与 MotrixSim 真实加载并执行场景 step")
    run_logged(
        [sys.executable, str(runtime_script), "--scene-id", scene_id, "--steps", str(int(params.get("validation_steps", 120)))],
        cwd=config.app_root,
        log_path=log_path,
        env=env,
        store=store,
        task_id=task["id"],
    )

    _task_progress(store, task, 72, "生成浏览器预览", "正在从官方网格资产生成可交互 GLB")
    run_logged(
        [sys.executable, str(preview_script), "--scene-id", scene_id, "--max-parlour-meshes", str(int(params.get("max_parlour_meshes", 240)))],
        cwd=config.app_root,
        log_path=log_path,
        env=env,
        store=store,
        task_id=task["id"],
    )

    scene = store.get_scene(scene_id)
    package_dir = Path(str((scene.get("metadata") or {}).get("package") or ""))
    manifest_path = Path(str(scene.get("manifest_path") or ""))
    if not package_dir.is_dir() or not manifest_path.is_file():
        raise RuntimeError("MotrixSim asset pipeline completed without a valid scene package")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime_validation = (scene.get("metadata") or {}).get("runtime_validation") or {}
    if runtime_validation.get("mujoco") is not True or runtime_validation.get("motrixsim") is not True:
        raise RuntimeError(f"MotrixSim scene did not pass both runtimes: {runtime_validation}")
    primary = ((manifest.get("viewer_assets") or {}).get("primary") or {})
    preview_path = Path(str(primary.get("path") or ""))
    if not preview_path.is_file() or preview_path.stat().st_size == 0:
        raise RuntimeError("MotrixSim GLB preview was not generated")

    report = {
        "schema": "real2sim.motrixsim_asset_import_report.v1",
        "status": "succeeded",
        "scene_id": scene_id,
        "source": str(source),
        "source_commit": _git_head(source),
        "package_dir": str(package_dir),
        "manifest_path": str(manifest_path),
        "preview_path": str(preview_path),
        "preview_size_bytes": preview_path.stat().st_size,
        "runtime_validation": runtime_validation,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    store.add_artifact(task_id=task["id"], name=report_path.name, path=report_path, kind="json")
    _index_artifacts(store, task["id"], package_dir)
    _task_progress(store, task, 94, "登记场景资产", "官方资产、运行时报告和 GLB 预览已登记")
    emit(f"MotrixSim asset package ready: scene_id={scene_id} preview={preview_path}")
    return {
        "scene_id": scene_id,
        "scene_path": scene["path"],
        "manifest_path": str(manifest_path),
        "package_dir": str(package_dir),
        "preview_path": str(preview_path),
        "runtime_validation": runtime_validation,
    }


def run_simulator_validation(config: PlatformConfig, store: PlatformStore, task: dict[str, Any], emit: Callable[[str], None]) -> dict[str, Any]:
    params = task["params"]
    run_dir = Path(task["run_dir"])
    _task_progress(store, task, 20, "解析场景", "正在读取场景包和仿真产物")
    scene_path = _scene_path(store, params)
    if scene_path is None:
        raise ValueError("simulator_validation requires scene_id or scene_path")
    simulators = params.get("simulators", ["mujoco", "motrixsim"])
    level = str(params.get("level") or "runtime_if_available")
    if level not in SIM_VALIDATION_LEVELS:
        raise ValueError("simulator_validation level must be smoke, runtime_if_available, or runtime_required")
    root = scene_path.parent if scene_path.is_file() else scene_path
    report = {
        "schema": "real2sim.simulator_validation.v1",
        "scene_path": str(scene_path),
        "level": level,
        "simulators": {},
        "ok": True,
    }
    if "mujoco" in simulators:
        _task_progress(store, task, 45, "校验 MuJoCo", "正在检查 MJCF 并尝试运行时加载")
        candidates = [
            root / "simulator_assets" / "rm65_table_pick_place.xml",
            root / "scene_mujoco.xml",
            root / "mujoco.xml",
            root / "scene.xml",
            root / "mjcf" / "scene.xml",
            root / "mjcf" / "scene_mujoco.xml",
            root / "mujoco" / "scene.xml",
            *root.glob("*mujoco*.xml"),
            *root.glob("mjcf/*.xml"),
            *root.glob("mujoco/*.xml"),
            *root.glob("simulator_assets/*.xml"),
        ]
        report["simulators"]["mujoco"] = _validate_mujoco_asset(candidates, level=level)
    if "motrixsim" in simulators:
        _task_progress(store, task, 68, "校验 MotrixSim", "正在检查 MotrixSim 导入产物和运行时")
        candidates = [
            root / "motrix_scene.json",
            root / "motrix_import.json",
            root / "scene.json",
            root / "simulator_assets" / "rm65_table_pick_place.xml",
            root / "motrixsim" / "motrix_import.json",
            root / "motrixsim" / "scene.json",
            *root.glob("*motrix*.json"),
            *root.glob("motrixsim/*.json"),
            *root.glob("simulator_assets/*.xml"),
        ]
        report["simulators"]["motrixsim"] = _validate_motrixsim_asset(candidates, level=level)
    report["ok"] = all(item["ok"] for item in report["simulators"].values())
    out_dir = run_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "validation_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    store.add_artifact(task_id=task["id"], name="validation_report.json", path=path, kind="json")
    _task_progress(store, task, 95, "生成报告", "仿真校验报告已生成")
    return {"ok": report["ok"], "report_path": str(path), "report": report}


def run_expert_collection(config: PlatformConfig, store: PlatformStore, task: dict[str, Any], emit: Callable[[str], None]) -> dict[str, Any]:
    params = task["params"]
    run_dir = Path(task["run_dir"])
    scene_path = _scene_path(store, params) or config.expert_policy_root / "examples/retail_single_arm_scene.json"
    out_dir = run_dir / "dataset"
    policy_preset = str(params.get("policy_preset") or ("rm65_pick_place_6d_ik" if config.production_mode else "pick_and_place_v1"))
    if policy_preset == "rm65_pick_place_6d_ik":
        scene_path = _resolve_rm65_runtime_scene(store, params, scene_path)
        required_successes = int(params.get("episodes", 50))
        max_attempts = int(params.get("max_attempts", max(required_successes * 3, required_successes)))
        command = [
            sys.executable,
            "-m",
            "real2sim_platform.rm65_controller",
            "--scene",
            str(scene_path),
            "--out",
            str(out_dir),
            "--required-successes",
            str(required_successes),
            "--max-attempts",
            str(max_attempts),
            "--steps",
            str(int(params.get("steps", 200))),
            "--seed",
            str(int(params.get("seed", 7))),
            "--width",
            str(int(params.get("image_width", 320))),
            "--height",
            str(int(params.get("image_height", 240))),
        ]
        _task_progress(store, task, 28, "物理采集", "RM65 正在通过 6D IK、双指接触和真实物理步进采集成功 episode")
        run_logged(command, cwd=config.app_root, log_path=run_dir / "logs" / "task.log", env=_env_with_paths(config, [config.app_root / "src"]), store=store, task_id=task["id"])
        manifest = _read_json(out_dir / "manifest.json")
        dataset = store.create_dataset(
            name=params.get("name", f"RM65 ACT dataset {task['id'][:8]}"),
            path=out_dir,
            scan={"input_kind": "discoverse_act_hdf5", "samples": manifest.get("successful_episodes"), "manifest": str(out_dir / "manifest.json")},
            source_type="expert_collection",
            source_uri=str(scene_path),
            parent_task_id=task["id"],
            manifest_path=out_dir / "manifest.json",
            metrics={"samples": manifest.get("successful_episodes"), "success_rate": manifest.get("success_rate"), "policy_preset": policy_preset},
            tags=["discoverse_act_hdf5", "rm65", "physical_contact", policy_preset],
        )
        _index_artifacts(store, task["id"], out_dir)
        _task_progress(store, task, 95, "登记数据集", "真实采集 HDF5、接触记录和成功报告已登记")
        return {"dataset_id": dataset["id"], "dataset_path": str(out_dir), "samples": manifest.get("successful_episodes"), "success_rate": manifest.get("success_rate")}
    if policy_preset != "pick_and_place_v1" or config.production_mode:
        raise ValueError(f"unsupported expert policy preset: {policy_preset}")
    _task_progress(store, task, 18, "准备采集", "正在加载场景、专家策略和机器人配置")
    command = [
        sys.executable,
        "-m",
        "real2sim_expert.collect",
        "--scene",
        str(scene_path),
        "--out",
        str(out_dir),
        "--episodes",
        str(int(params.get("episodes", 8))),
        "--steps",
        str(int(params.get("steps", 48))),
        "--seed",
        str(int(params.get("seed", 7))),
    ]
    robot_config = params.get("robot_config") or str(config.expert_policy_root / "examples/single_arm_vla_robot.json")
    task_config = params.get("task_config") or str(config.expert_policy_root / "examples/retail_pick_task.json")
    command.extend(["--robot-config", robot_config, "--task-config", task_config])
    _task_progress(store, task, 38, "专家采集", "专家策略正在生成 rollout 数据")
    run_logged(command, cwd=config.expert_policy_root, log_path=run_dir / "logs" / "task.log", env=_env_with_paths(config, [config.expert_policy_root / "src"]), store=store, task_id=task["id"])
    _task_progress(store, task, 82, "登记数据集", "正在读取采集 manifest 并生成训练数据集")
    manifest = _read_json(out_dir / "manifest.json")
    manifest["policy_preset"] = policy_preset
    manifest["workers"] = 1
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset = store.create_dataset(
        name=params.get("name", f"expert dataset {task['id'][:8]}"),
        path=out_dir,
        scan={"input_kind": "expert_rollouts", "samples": manifest.get("samples"), "manifest": str(out_dir / "manifest.json")},
        source_type="expert_collection",
        source_uri=str(scene_path),
        parent_task_id=task["id"],
        manifest_path=out_dir / "manifest.json",
        metrics={"samples": manifest.get("samples"), "policy_preset": policy_preset},
        tags=["expert_rollouts", policy_preset],
    )
    _index_artifacts(store, task["id"], out_dir)
    _task_progress(store, task, 95, "索引产物", "采集数据和 manifest 已登记")
    return {"dataset_id": dataset["id"], "dataset_path": str(out_dir), "samples": manifest.get("samples")}


def run_training(config: PlatformConfig, store: PlatformStore, task: dict[str, Any], emit: Callable[[str], None]) -> dict[str, Any]:
    params = task["params"]
    run_dir = Path(task["run_dir"])
    _task_progress(store, task, 18, "准备训练", "正在加载训练数据集和训练配方")
    dataset_path = _dataset_path(store, params)
    out_dir = run_dir / "model"
    recipe = str(params.get("recipe") or ("discoverse_act" if config.production_mode else "bc_v2"))
    if recipe == "discoverse_act":
        return _run_discoverse_act_training(config, store, task, dataset_path, out_dir)
    recipe_config = {
        "bc_v2": {"regularization": 1e-3, "base_model": "facebook/dinov2-small"},
        "bc_fast": {"regularization": 1e-2, "base_model": "facebook/dinov2-small"},
    }.get(recipe)
    if recipe_config is None:
        raise ValueError(f"unsupported training recipe: {recipe}")
    command = [
        sys.executable,
        "-m",
        "real2sim_train.train",
        "--dataset",
        str(dataset_path),
        "--out",
        str(out_dir),
        "--epochs",
        str(int(params.get("epochs", 100))),
        "--regularization",
        str(recipe_config["regularization"]),
        "--base-model",
        str(recipe_config["base_model"]),
        "--split-seed",
        str(int(params.get("split_seed", 17))),
    ]
    _task_progress(store, task, 36, "模型训练", "训练进程正在运行")
    run_logged(command, cwd=config.training_root, log_path=run_dir / "logs" / "task.log", env=_env_with_paths(config, [config.training_root / "src"]), store=store, task_id=task["id"])
    _task_progress(store, task, 82, "生成模型卡", "正在读取训练指标和模型产物")
    model_card = _read_json(out_dir / "model_card.json")
    model_card["training"] = {
        "recipe": recipe,
        "epochs_requested": int(params.get("epochs", 100)),
        "regularization": recipe_config["regularization"],
        "base_model": recipe_config["base_model"],
    }
    (out_dir / "model_card.json").write_text(json.dumps(model_card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model = store.create_model(
        name=params.get("name", f"policy {task['id'][:8]}"),
        path=out_dir,
        metadata=model_card,
        source_type="training",
        source_uri=str(dataset_path),
        parent_task_id=task["id"],
        manifest_path=out_dir / "model_card.json",
        metrics=model_card.get("metrics", {}),
        tags=["policy", recipe],
    )
    _index_artifacts(store, task["id"], out_dir)
    _task_progress(store, task, 95, "索引产物", "模型卡和训练产物已登记")
    return {"model_id": model["id"], "model_path": str(out_dir), "metrics": model_card.get("metrics", {})}


def run_inference(config: PlatformConfig, store: PlatformStore, task: dict[str, Any], emit: Callable[[str], None]) -> dict[str, Any]:
    params = task["params"]
    run_dir = Path(task["run_dir"])
    _task_progress(store, task, 18, "准备推理", "正在加载模型和场景")
    scene_path = _scene_path(store, params) or config.expert_policy_root / "examples/retail_single_arm_scene.json"
    model_path = _model_path(store, params)
    out_dir = run_dir / "inference"
    model = store.get_model(str(params["model_id"])) if params.get("model_id") else None
    model_recipe = str(((model or {}).get("metadata") or {}).get("training", {}).get("recipe") or params.get("recipe") or "")
    if model_recipe == "discoverse_act":
        scene_path = _resolve_rm65_runtime_scene(store, params, scene_path)
        act_root = config.discoverse_root / "policies" / "act"
        if _git_head(act_root) != config.act_commit:
            raise RuntimeError(f"DISCOVERSE ACT commit mismatch; required {config.act_commit}")
        command = [
            sys.executable,
            "-m",
            "real2sim_platform.act_inference",
            "--scene",
            str(scene_path),
            "--model",
            str(model_path),
            "--act-root",
            str(act_root),
            "--out",
            str(out_dir),
            "--episodes",
            str(int(params.get("episodes", 10))),
            "--steps",
            str(int(params.get("steps", 200))),
            "--seed",
            str(int(params.get("seed", 11))),
            "--minimum-success-rate",
            str(float(params.get("minimum_success_rate", 0.5))),
        ]
        _task_progress(store, task, 38, "ACT 在线推理", "DISCOVERSE ACT 正在真实 MuJoCo 场景中闭环控制 RM65")
        run_logged(command, cwd=config.app_root, log_path=run_dir / "logs" / "task.log", env=_env_with_paths(config, [config.app_root / "src", act_root, config.discoverse_root]), store=store, task_id=task["id"])
        report = _read_json(out_dir / "inference_report.json")
        if not report.get("actions_nonconstant"):
            raise RuntimeError("ACT 推理动作是常量，拒绝登记为有效推理")
        _index_artifacts(store, task["id"], out_dir)
        _task_progress(store, task, 95, "索引推理产物", "真实 rollout 视频、轨迹和成功率报告已登记")
        return {"episodes": len(report.get("episodes", [])), "success_rate": report.get("success_rate"), "report": str(out_dir / "inference_report.json")}
    if config.production_mode:
        raise ValueError("生产模式只允许 DISCOVERSE ACT 模型在线推理")
    command = [
        sys.executable,
        "-m",
        "real2sim_infer.run",
        "--scene",
        str(scene_path),
        "--model",
        str(model_path),
        "--out",
        str(out_dir),
        "--steps",
        str(int(params.get("steps", 80))),
    ]
    _task_progress(store, task, 42, "在线推理", "模型正在生成 rollout 轨迹")
    run_logged(command, cwd=config.inference_root, log_path=run_dir / "logs" / "task.log", env=_env_with_paths(config, [config.inference_root / "src"]), store=store, task_id=task["id"])
    _task_progress(store, task, 86, "生成轨迹", "正在读取推理结果和 viewer")
    output = _read_json(out_dir / "prediction_rollout.json")
    _index_artifacts(store, task["id"], out_dir)
    _task_progress(store, task, 95, "索引产物", "推理轨迹和 viewer 已登记")
    return {"rollout_steps": len(output.get("rollout", [])), "viewer": str(out_dir / "viewer.html")}


def _run_discoverse_act_training(config: PlatformConfig, store: PlatformStore, task: dict[str, Any], dataset_path: Path, out_dir: Path) -> dict[str, Any]:
    params = task["params"]
    run_dir = Path(task["run_dir"])
    act_root = config.discoverse_root / "policies" / "act"
    if _git_head(act_root) != config.act_commit:
        raise RuntimeError(f"DISCOVERSE ACT commit mismatch; required {config.act_commit}")
    dataset_info = _inspect_act_dataset(dataset_path)
    if dataset_info["episode_count"] < 2:
        raise ValueError("DISCOVERSE ACT 至少需要 2 个真实 HDF5 episode")
    out_dir.mkdir(parents=True, exist_ok=True)
    config_root = run_dir / "generated_config"
    task_name = f"real2sim_{task['id'][:12]}"
    config_path = config_root / "configurations" / "task_configs" / f"{task_name}.py"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = int(params.get("epochs", 100))
    validate_every = max(1, min(int(params.get("validate_every", max(1, epochs // 5))), epochs))
    save_every = max(1, min(int(params.get("save_every", validate_every)), epochs))
    batch_size = max(1, min(int(params.get("batch_size", 8)), dataset_info["episode_count"] - 1))
    chunk_size = max(2, min(int(params.get("chunk_size", 32)), dataset_info["episode_length"]))
    core_dir = out_dir / "core"
    checkpoint_dir = out_dir / "checkpoints"
    config_path.write_text(
        _act_task_config_source(
            task_name=task_name,
            dataset_path=dataset_path,
            checkpoint_dir=checkpoint_dir,
            stats_path=core_dir / "dataset_stats.pkl",
            camera_names=dataset_info["camera_names"],
            state_dim=dataset_info["state_dim"],
            action_dim=dataset_info["action_dim"],
            epochs=epochs,
            batch_size=batch_size,
            chunk_size=chunk_size,
            validate_every=validate_every,
            save_every=save_every,
        ),
        encoding="utf-8",
    )
    command = [sys.executable, "policy_train.py", "-tn", task_name, "-ne", str(epochs), "-ve", str(validate_every), "-se", str(save_every), "--seed", str(int(params.get("seed", 1)))]
    env = _env_with_paths(config, [config_root, act_root, config.discoverse_root])
    env["MPLBACKEND"] = "Agg"
    _task_progress(store, task, 34, "ACT 训练", f"正在使用 {dataset_info['episode_count']} 个真实 episode 训练 DISCOVERSE ACT")
    run_logged(command, cwd=act_root, log_path=run_dir / "logs" / "task.log", env=env, store=store, task_id=task["id"])
    checkpoint = core_dir / "policy_best.ckpt"
    stats_path = core_dir / "dataset_stats.pkl"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0 or not stats_path.is_file():
        raise RuntimeError("DISCOVERSE ACT 训练结束但未生成非空 policy_best.ckpt 或 dataset_stats.pkl")
    description = (core_dir / "description.txt").read_text(encoding="utf-8").strip() if (core_dir / "description.txt").is_file() else ""
    best_loss = _parse_best_loss(description)
    if best_loss is None or not math.isfinite(best_loss):
        raise RuntimeError("ACT 训练指标不是有限数值")
    model_card = {
        "schema": "real2sim.model_card.v2",
        "name": params.get("name", f"RM65 ACT {task['id'][:8]}"),
        "framework": "DISCOVERSE ACT",
        "training": {
            "recipe": "discoverse_act",
            "discoverse_commit": config.discoverse_commit,
            "act_commit": config.act_commit,
            "dataset_path": str(dataset_path),
            "epochs": epochs,
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "camera_names": dataset_info["camera_names"],
            "state_dim": dataset_info["state_dim"],
            "action_dim": dataset_info["action_dim"],
            "image_color_order": "RGB (OpenCV-decoded BGR is converted by the generated ACT augmentor)",
            "action_alignment": dataset_info["action_alignment"],
            "control_hold_steps": dataset_info["control_hold_steps"],
            "config_path": str(config_path),
            "runtime_compatibility": "HDF5 ACT training; AIRBOT MCAP/hardware codecs remain optional and fail explicitly when requested",
        },
        "metrics": {"best_validation_loss": best_loss},
        "artifacts": {"checkpoint": str(checkpoint), "dataset_stats": str(stats_path)},
        "description": description,
    }
    card_path = out_dir / "model_card.json"
    card_path.write_text(json.dumps(model_card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(config_path, out_dir / "task_config.py")
    model = store.create_model(
        name=str(model_card["name"]),
        path=out_dir,
        metadata=model_card,
        source_type="training",
        source_uri=str(dataset_path),
        parent_task_id=task["id"],
        manifest_path=card_path,
        metrics=model_card["metrics"],
        tags=["policy", "discoverse_act", "rm65"],
    )
    _index_artifacts(store, task["id"], out_dir)
    _task_progress(store, task, 95, "索引 ACT 模型", "checkpoint、损失曲线、模型卡和数据血缘已登记")
    return {"model_id": model["id"], "model_path": str(out_dir), "metrics": model_card["metrics"], "checkpoint": str(checkpoint)}


def run_renderer_benchmark(config: PlatformConfig, store: PlatformStore, task: dict[str, Any], emit: Callable[[str], None]) -> dict[str, Any]:
    run_dir = Path(task["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    _task_progress(store, task, 45, "检测渲染器", "正在检查 Spark、Three.js 和 Aholo 可用性")
    decision = renderer_decision()
    path = run_dir / "renderer_decision.json"
    path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    store.add_artifact(task_id=task["id"], name="renderer_decision.json", path=path, kind="json")
    _task_progress(store, task, 95, "生成报告", "渲染器对比报告已生成")
    return decision


def run_scene_cover_render(config: PlatformConfig, store: PlatformStore, task: dict[str, Any], emit: Callable[[str], None]) -> dict[str, Any]:
    params = task["params"]
    scene_id = str(params.get("scene_id") or "").strip()
    if not scene_id:
        raise ValueError("scene_cover_render requires scene_id")
    scene = store.get_scene(scene_id)
    manifest_path = _scene_manifest_path_for_scene(scene)
    manifest = _read_optional_json(manifest_path)
    if not manifest:
        raise ValueError(f"scene manifest not found for scene {scene_id}")
    _task_progress(store, task, 20, "准备封面", "正在解析场景清单和预览资源")
    report = _generate_scene_cover(config, store, task["id"], scene_id, manifest_path, emit=emit)
    _task_progress(store, task, 92, "封面完成", "场景封面已生成并写入清单")
    return report


def run_scene_splat_optimize(config: PlatformConfig, store: PlatformStore, task: dict[str, Any], emit: Callable[[str], None]) -> dict[str, Any]:
    params = task["params"]
    scene_id = str(params.get("scene_id") or "").strip()
    if not scene_id:
        raise ValueError("scene_splat_optimize requires scene_id")
    scene = store.get_scene(scene_id)
    manifest_path = _scene_manifest_path_for_scene(scene)
    manifest = _read_optional_json(manifest_path)
    if not manifest:
        raise ValueError(f"scene manifest not found for scene {scene_id}")
    _task_progress(store, task, 18, "解析3DGS", "正在查找完整 3DGS 资产")
    full_asset = _find_full_3dgs_manifest_asset(manifest)
    if not full_asset:
        raise ValueError("scene has no full 3DGS scene.ply asset to optimize")
    run_dir = Path(task["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "splat_optimization_report.json"
    chrome = _find_chromium_binary()
    _task_progress(store, task, 36, "检测打包能力", "正在检查 Spark/WebGL 解析能力和可落盘打包能力")
    report = {
        "schema": "real2sim.splat_optimization_report.v1",
        "scene_id": scene_id,
        "source_asset": full_asset,
        "source_size_bytes": int(full_asset.get("size_bytes") or 0),
        "source_size_label": full_asset.get("size_label") or "",
        "viewer_probe": {},
        "packed_output": None,
        "pack_status": "not_available",
        "recommendation": "Spark can load full 3DGS PLY manually. A native server-side SPZ/KSPLAT exporter is still required before making it the default asset.",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _task_progress(store, task, 46, "导出SPZ", "正在把完整 3DGS PLY 转换成轻量 SPZ")
    export = _export_scene_spz(config, store, task["id"], manifest_path, full_asset, emit)
    report["spz_export"] = export
    if export.get("ok"):
        report["packed_output"] = export.get("asset")
        report["pack_status"] = "spz_exported"
        report["recommendation"] = "SPZ export succeeded. The scene manifest now includes a lightweight SPZ asset that should be preferred over the raw 3DGS PLY for browser preview."
    if chrome and _scene_viewer_http_ready(config):
        _task_progress(store, task, 55, "验证预览资产", "正在用 headless Scene Viewer 真实加载优化后的 SPZ/完整 3DGS")
        report["viewer_probe"] = _probe_full_3dgs_with_chromium(config, chrome, scene_id, emit)
    else:
        report["viewer_probe"] = {"status": "skipped", "reason": "Chromium/Chrome binary or Scene Viewer HTTP endpoint not available"}
    if report["pack_status"] == "spz_exported":
        pass
    elif report["viewer_probe"].get("loaded"):
        report["pack_status"] = "viewer_load_succeeded_pack_export_unavailable"
        report["recommendation"] = "Full 3DGS is valid and loaded by Spark. Keep it as manual full-quality asset until a server-side SPZ/KSPLAT exporter is installed."
    else:
        report["pack_status"] = "headless_viewer_probe_failed_pack_export_unavailable"
        report["recommendation"] = "Server-side headless viewer could not reliably load this 118MB 3DGS within the probe budget. Keep pose-init preview as default and use manual full 3DGS loading until a native SPZ/KSPLAT exporter is installed."
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    store.add_artifact(task_id=task["id"], name="splat_optimization_report.json", path=report_path, kind="json")
    _task_progress(store, task, 92, "生成报告", "3DGS 优化能力报告已生成")
    return report


def _export_scene_spz(config: PlatformConfig, store: PlatformStore, task_id: str, manifest_path: Path, source_asset: dict[str, Any], emit: Callable[[str], None]) -> dict[str, Any]:
    source_path = Path(str(source_asset.get("path") or ""))
    if not source_path.is_file():
        return {"ok": False, "error": f"source 3DGS PLY not found: {source_path}"}
    node = shutil.which("node")
    if not node:
        for candidate in [Path("/opt/images23dgs_app/node/bin/node"), Path("/usr/local/bin/node"), Path("/usr/bin/node")]:
            if candidate.is_file():
                node = str(candidate)
                break
    if not node:
        return {"ok": False, "error": "node executable not found; cannot run Spark SPZ exporter"}
    package_dir = manifest_path.parent
    out_dir = package_dir / "optimized"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "scene.spz"
    export_report_path = out_dir / "spz_export_report.json"
    script = config.app_root / "scripts" / "convert_3dgs_ply_to_spz.mjs"
    command = [
        node,
        str(script),
        "--input",
        str(source_path),
        "--output",
        str(output_path),
        "--vendor-root",
        str(config.app_root / "src" / "real2sim_platform" / "static" / "vendor"),
        "--report",
        str(export_report_path),
    ]
    try:
        completed = subprocess.run(command, cwd=str(config.app_root), text=True, capture_output=True, timeout=240)
    except Exception as exc:
        return {"ok": False, "error": f"SPZ exporter failed to run: {type(exc).__name__}: {exc}"}
    if completed.returncode != 0 or not output_path.is_file():
        return {"ok": False, "error": (completed.stderr or completed.stdout or f"SPZ exporter exited {completed.returncode}")[-2000:]}
    export_report = _read_optional_json(export_report_path)
    asset = _viewer_asset(output_path, url=_run_static_url(output_path), role="optimized")
    asset["optimized_from"] = str(source_path)
    asset["autoload"] = output_path.stat().st_size <= VIEWER_AUTOLOAD_MAX_BYTES
    asset["viewer_hint"] = "autoload" if asset["autoload"] else f"manual_load_large_asset_over_{VIEWER_AUTOLOAD_MAX_BYTES}_bytes"
    manifest = _read_optional_json(manifest_path)
    assets = [item for item in manifest.get("assets", []) if not (isinstance(item, dict) and item.get("path") == str(output_path))]
    manifest["assets"] = [asset] + assets
    manifest["optimized_assets"] = {"spz": asset, "source": str(source_path), "report": str(export_report_path)}
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest_path.write_text(json.dumps(enrich_scene_manifest_assets(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    store.add_artifact(task_id=task_id, name="scene.spz", path=output_path, kind="spz")
    store.add_artifact(task_id=task_id, name="spz_export_report.json", path=export_report_path, kind="json")
    emit(f"SPZ exported: {output_path.name} ({asset['size_label']})")
    return {"ok": True, "asset": asset, "report": export_report}


def _find_full_3dgs_manifest_asset(manifest: dict[str, Any]) -> dict[str, Any] | None:
    assets = [asset for asset in manifest.get("assets", []) if isinstance(asset, dict)]
    candidates = []
    for asset in assets:
        name = str(asset.get("name") or "").lower()
        path = str(asset.get("path") or asset.get("url") or "").lower()
        size = int(asset.get("size_bytes") or 0)
        if name == "scene.ply" and ("/3dgs/" in path or size > 50 * 1024 * 1024):
            candidates.append(asset)
    candidates.sort(key=lambda item: int(item.get("size_bytes") or 0), reverse=True)
    return dict(candidates[0]) if candidates else None


def _probe_full_3dgs_with_chromium(config: PlatformConfig, chrome: Path, scene_id: str, emit: Callable[[str], None]) -> dict[str, Any]:
    command = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--window-size=960,640",
        "--virtual-time-budget=110000",
        "--dump-dom",
        f"http://127.0.0.1:{config.port}/static/scene-viewer.html?scene_id={scene_id}&probe_full=1",
    ]
    emit("splat/SPZ probe command: " + " ".join(command[:6]) + " ...")
    completed = subprocess.run(command, text=True, capture_output=True, timeout=125)
    text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    start = text.find('<pre id="probeResult"')
    if start >= 0:
        json_start = text.find("{", start)
        json_end = text.find("</pre>", json_start)
        raw = text[json_start:json_end if json_end >= 0 else None].strip()
        try:
            raw = raw.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            return json.loads(raw)
        except Exception:
            emit("splat probe parse failed: " + raw[:300])
    return {"loaded": False, "error": "probe did not return structured result", "stdout_tail": text[-800:]}


def _generate_scene_cover(config: PlatformConfig, store: PlatformStore, task_id: str, scene_id: str, manifest_path: Path, *, emit: Callable[[str], None]) -> dict[str, Any]:
    manifest = _read_optional_json(manifest_path)
    if not manifest:
        raise ValueError(f"scene manifest not found for scene {scene_id}")
    package_dir = manifest_path.parent
    artifacts_dir = package_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cover_path = artifacts_dir / "scene_cover.png"
    report_path = artifacts_dir / "scene_cover_report.json"

    method = "chromium_scene_viewer"
    warning = ""
    chrome = _find_chromium_binary()
    if chrome and _scene_viewer_http_ready(config):
        candidates = _render_scene_cover_candidates(config, chrome, scene_id, artifacts_dir, emit)
        if candidates:
            best = max(candidates, key=lambda item: float(item.get("score") or 0))
            shutil.copy2(best["path"], cover_path)
            method = "chromium_scene_viewer_multiview"
        else:
            warning = "all Chromium cover candidates failed"
            emit("chromium cover failed: " + warning)
            method = "fallback_existing_preview"
            _copy_existing_scene_preview(manifest, cover_path)
    else:
        warning = "Chromium/Chrome binary or Scene Viewer HTTP endpoint not available on this host; reused existing scene preview artifact."
        method = "fallback_existing_preview"
        _copy_existing_scene_preview(manifest, cover_path)

    if not cover_path.is_file():
        raise RuntimeError("scene cover render failed and no existing preview image was available")
    cover_asset = _viewer_asset(cover_path, url=_run_static_url(cover_path), role="cover")
    cover_asset["cover"] = True
    assets = [item for item in manifest.get("assets", []) if not (isinstance(item, dict) and (item.get("cover") or item.get("name") == cover_asset["name"]))]
    manifest["assets"] = [cover_asset] + assets
    manifest["cover_asset"] = cover_asset
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest_path.write_text(json.dumps(enrich_scene_manifest_assets(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema": "real2sim.scene_cover_report.v1",
        "scene_id": scene_id,
        "method": method,
        "cover_path": str(cover_path),
        "cover_url": _run_static_url(cover_path),
        "candidates": candidates if chrome else [],
        "warning": warning,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    store.add_artifact(task_id=task_id, name="scene_cover.png", path=cover_path, kind="image")
    for candidate in report.get("candidates", []):
        candidate_path = Path(str(candidate.get("path") or ""))
        if candidate_path.is_file():
            store.add_artifact(task_id=task_id, name=candidate_path.name, path=candidate_path, kind="image")
    store.add_artifact(task_id=task_id, name="scene_cover_report.json", path=report_path, kind="json")
    return report


def _scene_viewer_http_ready(config: PlatformConfig) -> bool:
    url = f"http://127.0.0.1:{config.port}/static/scene-viewer.html"
    try:
        with urlopen(url, timeout=0.75) as response:
            return int(getattr(response, "status", 200)) < 500
    except Exception:
        return False


def renderer_decision() -> dict[str, Any]:
    return {
        "default": "spark",
        "fallback": "aholo",
        "reason": "Spark integrates directly with Three.js and mixed mesh/splat scenes, which fits simulator overlays. The platform falls back to a plain Three.js preview when the Spark bundle or input format is unavailable. Aholo is kept for huge 3DGS / chunked streaming LOD after assets are converted to its supported packed formats.",
        "spark": {
            "package": "@sparkjsdev/spark",
            "bundle": str(SPARK_VENDOR_PATH),
            "available": SPARK_VENDOR_PATH.is_file(),
            "strengths": ["Three.js integration", "mixed splat and mesh rendering", "PLY/SPZ/SPLAT/KSPLAT/SOG support", "broad WebGL2 portability"],
        },
        "three_fallback": {
            "bundle": str(THREE_VENDOR_PATH),
            "available": THREE_VENDOR_PATH.is_file(),
            "strengths": ["PLY point-cloud preview", "mesh preview", "small dependency surface"],
            "constraint": "not a full 3DGS renderer",
        },
        "aholo": {
            "package": "@manycore/aholo-viewer",
            "strengths": ["high performance 3DGS", "mesh support", "chunked streaming LOD for large scenes"],
            "constraint": "needs Aholo packing/LOD workflow for best results",
        },
    }


def _select_renderer(value: Any, package_dir: Path) -> str:
    renderer = str(value or "spark").lower()
    if renderer == "auto":
        aholo_lod = package_dir / "aholo" / "manifest.json"
        return "aholo" if aholo_lod.is_file() else "spark"
    if renderer not in {"spark", "aholo"}:
        raise ValueError(f"unsupported renderer: {renderer}")
    return renderer


def run_logged(command: list[str], *, cwd: Path, log_path: Path, env: dict[str, str] | None = None, store: PlatformStore | None = None, task_id: str | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        if store and task_id:
            store.set_task_pid(task_id, process.pid)
            store.update_task(task_id, message=f"子进程 {process.pid} 正在运行")
        assert process.stdout is not None
        while True:
            line = process.stdout.readline()
            if line:
                log.write(line)
                log.flush()
            if store and task_id and store.get_task(task_id)["cancel_requested"] and process.poll() is None:
                log.write("cancel requested; terminating subprocess\n")
                log.flush()
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                if store and task_id:
                    store.set_task_pid(task_id, None)
                raise TaskCanceled("task canceled by user")
            if process.poll() is not None:
                break
        for line in process.stdout:
            log.write(line)
        code = process.wait()
        if store and task_id:
            store.set_task_pid(task_id, None)
        if code != 0:
            raise RuntimeError(f"command failed with exit code {code}: {command[0]}")


def _resolve_preflight_input(store: PlatformStore, params: dict[str, Any]) -> Path:
    input_path = _task_input_path(store, params)
    dataset = store.get_dataset(str(params["dataset_id"])) if params.get("dataset_id") else None
    prepared_manifest = _prepared_manifest_for_dataset(dataset) if dataset else _read_optional_json(_prepared_manifest_path_for_input(input_path))
    if prepared_manifest:
        frames_dir = Path(str(prepared_manifest.get("frames_dir") or input_path / "frames"))
        if frames_dir.exists():
            return frames_dir
    return input_path


def _frame_paths_for_preflight(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if not path.is_dir():
        return []
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def _frame_quality_summary(frame_paths: list[Path], *, sample_limit: int = 12) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "width": None,
        "height": None,
        "sampled_frames": 0,
        "mean_luma": None,
        "mean_blur_score": None,
        "mean_frame_delta": None,
    }
    if not frame_paths:
        return summary
    try:
        from PIL import Image, ImageChops, ImageFilter, ImageStat
    except ImportError:
        return {**summary, "warning": "Pillow is required for frame quality metrics"}
    if len(frame_paths) <= sample_limit:
        samples = frame_paths
    else:
        step = max(1, math.floor(len(frame_paths) / sample_limit))
        samples = frame_paths[::step][:sample_limit]
    luma_values: list[float] = []
    blur_values: list[float] = []
    delta_values: list[float] = []
    previous = None
    for frame in samples:
        try:
            with Image.open(frame) as image:
                rgb = image.convert("RGB")
                if summary["width"] is None:
                    summary["width"], summary["height"] = rgb.size
                gray = rgb.convert("L")
                luma_values.append(float(ImageStat.Stat(gray).mean[0]))
                edges = gray.filter(ImageFilter.FIND_EDGES)
                blur_values.append(float(ImageStat.Stat(edges).var[0]))
                small = gray.resize((64, 64))
                if previous is not None:
                    diff = ImageChops.difference(previous, small)
                    delta_values.append(float(ImageStat.Stat(diff).mean[0]))
                previous = small
        except OSError:
            continue
    summary["sampled_frames"] = len(luma_values)
    summary["mean_luma"] = round(sum(luma_values) / len(luma_values), 3) if luma_values else None
    summary["mean_blur_score"] = round(sum(blur_values) / len(blur_values), 3) if blur_values else None
    summary["mean_frame_delta"] = round(sum(delta_values) / len(delta_values), 3) if delta_values else None
    return summary


def _preflight_recommendations(failed_gates: list[str], warnings: list[str]) -> list[str]:
    recommendations: list[str] = []
    if "enough_frames" in failed_gates:
        recommendations.append("Run media_prepare with target_frame_count=80 and max_videos high enough to produce at least 30 reconstructable frames.")
        recommendations.append("Use a reconstruction-oriented capture with camera motion/parallax; xarm policy videos may be too short or too static.")
    if "anysplat_root" in failed_gates:
        recommendations.append("Set anysplat_root or REAL2SIM_ANYSPLAT_ROOT to /opt/real2sim_paper_runtime/AnySplat on Wuying.")
    if "anysplat_python" in failed_gates:
        recommendations.append("Set anysplat_python to /opt/real2sim_paper_runtime/envs/anysplat/bin/python.")
    if "gs_video_reconstruct" in failed_gates:
        recommendations.append("Set gs_playground_root to the checkout containing scripts/real2sim_video_motrix_scene.py.")
    if not recommendations and warnings:
        recommendations.append("Review capture warnings before starting a long reconstruction job.")
    return recommendations


def _preflight_readiness_score(*, frame_count: int, min_frames: int, runtime_checks: dict[str, bool], warnings: list[str]) -> int:
    score = 100
    if frame_count < min_frames:
        score -= min(60, int((min_frames - frame_count) / max(min_frames, 1) * 60) + 20)
    missing_runtime = [name for name, ok in runtime_checks.items() if not ok]
    score -= 15 * len(missing_runtime)
    score -= min(20, 5 * len(warnings))
    return max(0, min(100, score))


def _preflight_next_action(*, ok: bool, allow_low_quality: bool, failed_gates: list[str], warnings: list[str]) -> str:
    if ok:
        return "ready_for_scene_reconstruction"
    if "enough_frames" in failed_gates:
        return "run_media_prepare_or_recapture_more_frames"
    if any(name in failed_gates for name in ["anysplat_root", "anysplat_python", "gs_video_reconstruct"]):
        return "fix_reconstruction_runtime_configuration"
    if allow_low_quality:
        return "can_run_with_allow_low_quality_but_expect_poor_reconstruction"
    if warnings:
        return "review_capture_warnings_before_reconstruction"
    return "fix_failed_preflight_gates"


def _task_input_path(store: PlatformStore, params: dict[str, Any]) -> Path:
    if params.get("input_path"):
        return Path(str(params["input_path"]))
    if params.get("dataset_id"):
        return Path(store.get_dataset(str(params["dataset_id"]))["path"])
    raise ValueError("scene_reconstruction requires input_path or dataset_id")


def _resolve_reconstruction_input(config: PlatformConfig, store: PlatformStore, task: dict[str, Any], emit: Callable[[str], None]) -> Path:
    params = task["params"]
    dataset = store.get_dataset(str(params["dataset_id"])) if params.get("dataset_id") else None
    input_path = Path(dataset["path"]) if dataset else Path(str(params.get("input_path", "")))
    prepared_manifest = _prepared_manifest_for_dataset(dataset) if dataset else _read_optional_json(_prepared_manifest_path_for_input(input_path))
    if prepared_manifest:
        frames_dir = Path(str(prepared_manifest.get("frames_dir") or input_path / "frames"))
        if _has_reconstruction_images(frames_dir):
            return frames_dir
        raise ValueError(f"prepared dataset has no reconstructable frames: {frames_dir}")
    if params.get("auto_prepare"):
        emit("auto_prepare enabled; preparing media before reconstruction")
        prepared = prepare_media_dataset(config, store, params=params, parent_task_id=task["id"], task_id=task["id"], run_dir=Path(task["run_dir"]) / "prepared_media", emit=emit)
        return Path(prepared["frames_dir"])
    if input_path.is_file() and input_path.suffix.lower() in VIDEO_SUFFIXES:
        return input_path
    if _has_reconstruction_images(input_path):
        return input_path
    scan = scan_dataset(input_path) if input_path.exists() else {"input_kind": "missing", "path": str(input_path)}
    if scan.get("video_count") or scan.get("parquet_count") or scan.get("input_kind") in {"video", "parquet_video"}:
        raise ValueError("scene_reconstruction non-dry input contains video/parquet data; run media_prepare first or set auto_prepare=true")
    raise ValueError("scene_reconstruction non-dry requires a video file or a prepared/images frame directory; use dry_run=true for PLY/3DGS-only preview assets")


def prepare_media_dataset(
    config: PlatformConfig,
    store: PlatformStore,
    *,
    params: dict[str, Any],
    parent_task_id: str,
    task_id: str,
    run_dir: Path,
    emit: Callable[[str], None],
) -> dict[str, Any]:
    source_dataset = store.get_dataset(str(params["dataset_id"])) if params.get("dataset_id") else None
    source_path = Path(source_dataset["path"]) if source_dataset else Path(str(params.get("input_path", "")))
    if not source_path.exists() and not (str(source_path).startswith("http://") or str(source_path).startswith("https://")):
        raise FileNotFoundError(source_path)
    sample_fps = float(params.get("sample_fps", MEDIA_PREPARE_DEFAULT_SAMPLE_FPS))
    max_frames_per_video = int(params.get("max_frames_per_video", MEDIA_PREPARE_DEFAULT_MAX_FRAMES))
    max_videos = int(params.get("max_videos", 1))
    target_frame_count = int(params.get("target_frame_count", 0) or 0)
    min_total_frames = int(params.get("min_total_frames", 0) or 0)
    camera_label = str(params.get("camera_label") or MEDIA_PREPARE_DEFAULT_CAMERA)
    source_column = str(params.get("source_column") or "").strip()
    if sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")
    if max_frames_per_video <= 0:
        raise ValueError("max_frames_per_video must be > 0")
    if max_videos <= 0:
        raise ValueError("max_videos must be > 0")
    if target_frame_count < 0:
        raise ValueError("target_frame_count must be >= 0")
    if min_total_frames < 0:
        raise ValueError("min_total_frames must be >= 0")

    dataset_id = uuid.uuid4().hex
    output_name = _safe_name(str(params.get("output_name") or f"prepared-{source_path.stem or 'media'}"))
    target = config.datasets_dir / dataset_id
    frames_dir = target / "frames"
    thumbnails_dir = target / "thumbnails"
    target.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = _read_optional_json(source_path / "dataset_manifest.json") if source_path.is_dir() else {}
    warnings: list[str] = []
    effective_max_videos = max_videos
    if target_frame_count:
        effective_max_videos = max(max_videos, math.ceil(target_frame_count / max_frames_per_video))
    video_sources = _collect_video_sources(source_path, source_manifest, camera_label=camera_label, source_column=source_column, limit=effective_max_videos)
    copied_images = 0
    video_frame_counts: list[dict[str, Any]] = []
    if video_sources:
        emit(f"media_prepare selected {len(video_sources)} video source(s)")
        for index, source in enumerate(video_sources):
            remaining = target_frame_count - copied_images if target_frame_count else max_frames_per_video
            per_video_limit = max(1, min(max_frames_per_video, remaining)) if target_frame_count else max_frames_per_video
            frame_count = _extract_video_frames(source, frames_dir, video_index=index, sample_fps=sample_fps, max_frames=per_video_limit, store=store, task_id=task_id, log_path=run_dir / "logs" / "task.log")
            copied_images += frame_count
            video_frame_counts.append({"source": str(source), "frame_count": frame_count})
            if target_frame_count and copied_images >= target_frame_count:
                break
    else:
        copied_images = _copy_image_frames(source_path, frames_dir, max_frames=target_frame_count or max_frames_per_video * max_videos)
        if copied_images == 0:
            warnings.append("no local videos or image frames found")
    if copied_images == 0:
        raise ValueError("media_prepare did not produce any frames; check camera_label/source_column or input media")
    if min_total_frames and copied_images < min_total_frames:
        raise ValueError(f"media_prepare produced {copied_images} frame(s), below min_total_frames={min_total_frames}")

    width, height = _image_size(next(iter(sorted(frames_dir.glob("*"))), None))
    _write_thumbnails(frames_dir, thumbnails_dir)
    manifest = {
        "schema": "real2sim.prepared_manifest.v1",
        "id": dataset_id,
        "name": output_name,
        "source_dataset_id": source_dataset["id"] if source_dataset else "",
        "source_uri": source_dataset["source_uri"] if source_dataset else str(source_path),
        "source_path": str(source_path),
        "video_urls": [str(source) for source in video_sources],
        "video_frame_counts": video_frame_counts,
        "frames_dir": str(frames_dir),
        "frame_count": copied_images,
        "target_frame_count": target_frame_count,
        "min_total_frames": min_total_frames,
        "camera_label": camera_label,
        "source_column": source_column,
        "sample_fps": sample_fps,
        "fps": sample_fps,
        "width": width,
        "height": height,
        "parent_task_id": parent_task_id,
        "warnings": warnings,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = target / "prepared_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / "source_manifest.json").write_text(json.dumps(source_manifest or {}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    scan = scan_dataset(target)
    dataset = store.create_dataset(
        item_id=dataset_id,
        name=output_name,
        path=target,
        scan=scan,
        source_type="prepared_media",
        source_uri=manifest["source_uri"],
        version=str(params.get("revision") or source_dataset["version"] if source_dataset else ""),
        parent_task_id=parent_task_id,
        manifest_path=manifest_path,
        metrics={"frame_count": copied_images, "sample_fps": sample_fps, "video_count": len(video_sources)},
        tags=["prepared_media", camera_label],
    )
    return {
        "dataset_id": dataset["id"],
        "dataset_path": str(target),
        "manifest_path": str(manifest_path),
        "frames_dir": str(frames_dir),
        "frame_count": copied_images,
        "video_urls": [str(source) for source in video_sources],
        "video_frame_counts": video_frame_counts,
        "warnings": warnings,
    }


def _collect_video_sources(source_path: Path, source_manifest: dict[str, Any], *, camera_label: str, source_column: str, limit: int) -> list[str]:
    sources: list[str] = []
    if str(source_path).startswith(("http://", "https://")):
        sources.append(str(source_path))
    elif source_path.is_file() and source_path.suffix.lower() in VIDEO_SUFFIXES:
        sources.append(str(source_path))
    elif source_path.is_dir():
        sources.extend(str(path) for path in sorted(source_path.rglob("*")) if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
    manifest_urls = _extract_video_url_candidates(source_manifest)
    sources.extend(manifest_urls)
    sources.extend(_video_urls_from_parquet(source_path, camera_label=camera_label, source_column=source_column))
    repo_id = str(source_manifest.get("repo_id") or "").strip()
    revision = str(source_manifest.get("revision") or "main").strip() or "main"
    if repo_id:
        sources.extend(_video_urls_from_hf_files(repo_id, revision=revision, camera_label=camera_label, limit=limit))
        sources.extend(_video_urls_from_hf_viewer(repo_id, camera_label=camera_label, limit=limit))
    deduped: list[str] = []
    for source in sources:
        if camera_label and camera_label not in source and source.startswith(("http://", "https://")):
            continue
        if source not in deduped:
            deduped.append(source)
        if len(deduped) >= limit:
            break
    return deduped


def _video_urls_from_hf_files(repo_id: str, *, revision: str, camera_label: str, limit: int) -> list[str]:
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    except Exception:
        return []
    urls: list[str] = []
    for filename in sorted(files):
        lower = filename.lower()
        if not any(lower.endswith(suffix) for suffix in VIDEO_SUFFIXES):
            continue
        if camera_label and camera_label not in filename:
            continue
        urls.append(f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}")
        if len(urls) >= limit:
            break
    return urls


def _video_urls_from_hf_viewer(repo_id: str, *, camera_label: str, limit: int) -> list[str]:
    dataset = quote(repo_id, safe="")
    try:
        with urlopen(f"https://datasets-server.huggingface.co/splits?dataset={dataset}", timeout=20) as response:
            splits_payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    urls: list[str] = []
    for split in splits_payload.get("splits", []):
        config = quote(str(split.get("config") or "default"), safe="")
        split_name = quote(str(split.get("split") or "train"), safe="")
        try:
            with urlopen(f"https://datasets-server.huggingface.co/first-rows?dataset={dataset}&config={config}&split={split_name}", timeout=30) as response:
                rows_payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            continue
        for row in rows_payload.get("rows", []):
            for candidate in _extract_video_url_candidates(row.get("row", {})):
                if camera_label and camera_label not in candidate:
                    continue
                if candidate not in urls:
                    urls.append(candidate)
                if len(urls) >= limit:
                    return urls
    return urls


def _video_urls_from_parquet(source_path: Path, *, camera_label: str, source_column: str) -> list[str]:
    parquet_files: list[Path] = []
    if source_path.is_file() and source_path.suffix.lower() in PARQUET_SUFFIXES:
        parquet_files = [source_path]
    elif source_path.is_dir():
        parquet_files = sorted(source_path.rglob("*.parquet"))
    if not parquet_files:
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read HuggingFace parquet video manifests") from exc
    urls: list[str] = []
    for parquet in parquet_files:
        table = pq.read_table(parquet)
        for row in table.to_pylist():
            values = [_nested_value(row, source_column)] if source_column else [row]
            for value in values:
                for candidate in _extract_video_url_candidates(value):
                    if camera_label and camera_label not in candidate:
                        continue
                    if candidate not in urls:
                        urls.append(candidate)
    return urls


def _extract_video_url_candidates(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        lower = value.lower().split("?", 1)[0]
        if value.startswith(("http://", "https://")) and any(lower.endswith(suffix) for suffix in VIDEO_SUFFIXES):
            found.append(value)
    elif isinstance(value, dict):
        if "src" in value:
            found.extend(_extract_video_url_candidates(value["src"]))
        for nested in value.values():
            found.extend(_extract_video_url_candidates(nested))
    elif isinstance(value, list):
        for item in value:
            found.extend(_extract_video_url_candidates(item))
    return found


def _nested_value(row: dict[str, Any], source_column: str) -> Any:
    value: Any = row
    for part in source_column.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def _extract_video_frames(source: str, frames_dir: Path, *, video_index: int, sample_fps: float, max_frames: int, store: PlatformStore, task_id: str, log_path: Path) -> int:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for media_prepare video extraction")
    ffmpeg_source = _materialize_video_source(source, frames_dir.parent / "sources", log_path)
    pattern = frames_dir / f"video{video_index:02d}_%06d.jpg"
    before = {path.name for path in frames_dir.glob("*.jpg")}
    command = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-i",
        ffmpeg_source,
        "-vf",
        f"fps={sample_fps}",
        "-frames:v",
        str(max_frames),
        "-q:v",
        "2",
        str(pattern),
    ]
    run_logged(command, cwd=frames_dir.parent, log_path=log_path, store=store, task_id=task_id)
    after = {path.name for path in frames_dir.glob("*.jpg")}
    return len(after - before)


def _materialize_video_source(source: str, source_dir: Path, log_path: Path) -> str:
    if not source.startswith(("http://", "https://")):
        return source
    source_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(source)
    filename = _safe_name(Path(parsed.path).name or "video.mp4")
    target = source_dir / filename
    if target.is_file() and target.stat().st_size > 0:
        return str(target)
    _append_log(log_path, f"downloading remote video source to {target}")
    hf_info = _parse_hf_resolve_url(parsed)
    if hf_info:
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id=hf_info["repo_id"],
                repo_type="dataset",
                revision=hf_info["revision"],
                filename=hf_info["filename"],
                local_dir=str(source_dir),
                local_dir_use_symlinks=False,
            )
            downloaded_path = Path(downloaded)
            if downloaded_path != target and downloaded_path.is_file():
                shutil.copy2(downloaded_path, target)
            return str(target if target.is_file() else downloaded_path)
        except Exception as exc:
            _append_log(log_path, f"huggingface_hub download failed; falling back to urlopen: {exc}")
    with urlopen(source, timeout=120) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return str(target)


def _parse_hf_resolve_url(parsed) -> dict[str, str] | None:
    if parsed.netloc != "huggingface.co":
        return None
    parts = [unquote(part) for part in parsed.path.strip("/").split("/")]
    if len(parts) < 6 or parts[0] != "datasets" or parts[3] != "resolve":
        return None
    return {
        "repo_id": f"{parts[1]}/{parts[2]}",
        "revision": parts[4],
        "filename": "/".join(parts[5:]),
    }


def _copy_image_frames(source_path: Path, frames_dir: Path, *, max_frames: int) -> int:
    if source_path.is_file() and source_path.suffix.lower() in IMAGE_SUFFIXES:
        image_paths = [source_path]
    elif source_path.is_dir():
        image_paths = sorted(path for path in source_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    else:
        image_paths = []
    count = 0
    for count, image_path in enumerate(image_paths[:max_frames], start=1):
        suffix = image_path.suffix.lower() if image_path.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
        shutil.copy2(image_path, frames_dir / f"frame_{count:06d}{suffix}")
    return count


def _write_thumbnails(frames_dir: Path, thumbnails_dir: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        return
    for index, frame in enumerate(sorted(frames_dir.glob("*"))[:6], start=1):
        try:
            with Image.open(frame) as image:
                image.thumbnail((320, 180))
                image.convert("RGB").save(thumbnails_dir / f"thumb_{index:02d}.jpg", quality=80)
        except OSError:
            continue


def _image_size(path: Path | None) -> tuple[int | None, int | None]:
    if path is None:
        return None, None
    try:
        from PIL import Image
        with Image.open(path) as image:
            return image.width, image.height
    except Exception:
        return None, None


def _has_reconstruction_images(path: Path) -> bool:
    if path.is_file():
        return path.suffix.lower() in IMAGE_SUFFIXES
    if not path.is_dir():
        return False
    return any(item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES for item in path.iterdir())


def _prepared_manifest_for_dataset(dataset: dict[str, Any] | None) -> dict[str, Any]:
    if not dataset:
        return {}
    manifest = _read_optional_json(Path(str(dataset.get("manifest_path") or "")))
    if manifest.get("schema") == "real2sim.prepared_manifest.v1":
        return manifest
    path = Path(dataset["path"])
    return _read_optional_json(path / "prepared_manifest.json")


def _prepared_manifest_path_for_input(path: Path) -> Path:
    return path / "prepared_manifest.json" if path.is_dir() else path.parent / "prepared_manifest.json"


def _dataset_path(store: PlatformStore, params: dict[str, Any]) -> Path:
    if params.get("dataset_path"):
        return Path(str(params["dataset_path"]))
    if params.get("dataset_id"):
        return Path(store.get_dataset(str(params["dataset_id"]))["path"])
    raise ValueError("training requires dataset_id or dataset_path")


def _scene_path(store: PlatformStore, params: dict[str, Any]) -> Path | None:
    if params.get("scene_path"):
        return Path(str(params["scene_path"]))
    if params.get("scene_id"):
        return Path(store.get_scene(str(params["scene_id"]))["path"])
    return None


def _resolve_rm65_runtime_scene(store: PlatformStore, params: dict[str, Any], fallback: Path) -> Path:
    candidates: list[Path] = []
    if fallback.suffix.lower() == ".xml":
        candidates.append(fallback)
    if params.get("scene_id"):
        scene = store.get_scene(str(params["scene_id"]))
        metadata = scene.get("metadata") or {}
        package = Path(str(metadata.get("package") or ""))
        manifest_path = Path(str(scene.get("manifest_path") or ""))
        manifest = _read_optional_json(manifest_path)
        for asset in manifest.get("assets", []) if isinstance(manifest.get("assets"), list) else []:
            if not isinstance(asset, dict):
                continue
            name = str(asset.get("name") or asset.get("path") or "").lower()
            if "rm65" in name and name.endswith(".xml"):
                candidates.append(Path(str(asset.get("path") or "")))
        if str(package):
            candidates.extend([
                package / "simulator_assets" / "rm65_table_pick_place.xml",
                package / "scene_mujoco.xml",
                package / "mjcf" / "scene.xml",
            ])
    candidates.extend([fallback.parent / "simulator_assets" / "rm65_table_pick_place.xml", fallback.parent / "scene_mujoco.xml"])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("所选场景没有可执行的 RM65 pick-place MJCF；请先在场景管理中绑定 RM65 控制器和碰撞资产")


def _model_path(store: PlatformStore, params: dict[str, Any]) -> Path:
    if params.get("model_path"):
        return Path(str(params["model_path"]))
    if params.get("model_id"):
        return Path(store.get_model(str(params["model_id"]))["path"])
    raise ValueError("inference requires model_id or model_path")


def _directory_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _upload_directory(storage: BosStorage, root: Path, *, area: str, relative_root: str) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        storage.upload_file(
            path,
            area=area,
            relative_path=f"{relative_root}/{path.relative_to(root).as_posix()}",
            metadata={"sha256": _file_sha256(path)},
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    return read_source_revision(root)


def _inspect_act_dataset(dataset_path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("DISCOVERSE ACT training requires h5py") from exc
    episodes = sorted(dataset_path.glob("episode_*.hdf5"))
    if not episodes:
        raise ValueError("训练数据目录中没有 episode_*.hdf5")
    lengths: list[int] = []
    cameras: list[str] = []
    state_dim = 0
    action_dim = 0
    action_alignment = ""
    control_hold_steps = 0
    for path in episodes:
        with h5py.File(path, "r") as root:
            if "/observations/qpos" not in root or "/action" not in root or "/observations/images" not in root:
                raise ValueError(f"ACT HDF5 缺少 qpos/action/images: {path.name}")
            current_cameras = sorted(str(name) for name in root["/observations/images"].keys())
            current_state_dim = int(root["/observations/qpos"].shape[1])
            current_action_dim = int(root["/action"].shape[1])
            current_alignment = str(root.attrs.get("action_alignment", ""))
            current_hold_steps = int(root.attrs.get("control_hold_steps", 0))
            if current_alignment != "observation_t_to_action_t" or current_hold_steps <= 0:
                raise ValueError(f"ACT HDF5 不是因果 observation_t -> action_t 数据: {path.name}")
            if not cameras:
                cameras = current_cameras
                state_dim = current_state_dim
                action_dim = current_action_dim
                action_alignment = current_alignment
                control_hold_steps = current_hold_steps
            if (
                current_cameras != cameras
                or current_state_dim != state_dim
                or current_action_dim != action_dim
                or current_alignment != action_alignment
                or current_hold_steps != control_hold_steps
            ):
                raise ValueError("ACT 数据集的相机、状态维度、动作维度或控制周期不一致")
            length = int(root["/action"].shape[0])
            if length < 2:
                raise ValueError(f"ACT episode 过短: {path.name}")
            lengths.append(length)
    return {
        "episode_count": len(episodes),
        "episode_length": min(lengths),
        "camera_names": cameras,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_alignment": action_alignment,
        "control_hold_steps": control_hold_steps,
    }


def _act_task_config_source(
    *,
    task_name: str,
    dataset_path: Path,
    checkpoint_dir: Path,
    stats_path: Path,
    camera_names: list[str],
    state_dim: int,
    action_dim: int,
    epochs: int,
    batch_size: int,
    chunk_size: int,
    validate_every: int,
    save_every: int,
) -> str:
    return f'''from configurations.task_configs.template import TASK_CONFIG_DEFAULT, activator, set_paths


def policy_maker(config: dict, stage=None):
    from policies.act.act import ACTPolicy
    from policies.common.maker import post_init_policies
    policy = ACTPolicy(config)
    post_init_policies([policy], stage, [config["ckpt_path"]])
    return policy


@activator(True)
def augment_images(image):
    return image[:, [2, 1, 0], ...]


set_paths({str(dataset_path)!r}, {str(checkpoint_dir)!r}, {str(stats_path)!r})
TASK_CONFIG_DEFAULT["common"]["camera_names"] = {camera_names!r}
TASK_CONFIG_DEFAULT["common"]["state_dim"] = {state_dim}
TASK_CONFIG_DEFAULT["common"]["action_dim"] = {action_dim}
TASK_CONFIG_DEFAULT["common"]["policy_config"]["policy_maker"] = policy_maker
TASK_CONFIG_DEFAULT["common"]["policy_config"]["temporal_agg"] = True
TASK_CONFIG_DEFAULT["common"]["policy_config"]["chunk_size"] = {chunk_size}
TASK_CONFIG_DEFAULT["common"]["policy_config"]["num_queries"] = {chunk_size}
TASK_CONFIG_DEFAULT["train"]["load_data"]["data_type"] = "hdf5"
TASK_CONFIG_DEFAULT["train"]["load_data"]["num_episodes"] = "ALL"
TASK_CONFIG_DEFAULT["train"]["load_data"]["batch_size_train"] = {batch_size}
TASK_CONFIG_DEFAULT["train"]["load_data"]["batch_size_validate"] = {batch_size}
TASK_CONFIG_DEFAULT["train"]["load_data"]["num_workers_train"] = 2
TASK_CONFIG_DEFAULT["train"]["load_data"]["num_workers_validate"] = 2
TASK_CONFIG_DEFAULT["train"]["load_data"]["observation_slice"] = None
TASK_CONFIG_DEFAULT["train"]["load_data"]["action_slice"] = None
TASK_CONFIG_DEFAULT["train"]["num_epochs"] = {epochs}
TASK_CONFIG_DEFAULT["train"]["validate_every"] = {validate_every}
TASK_CONFIG_DEFAULT["train"]["save_every"] = {save_every}
TASK_CONFIG_DEFAULT["train"]["eval_every"] = 0
TASK_CONFIG = TASK_CONFIG_DEFAULT
'''


def _parse_best_loss(description: str) -> float | None:
    import re

    match = re.search(r"val loss\s+([0-9.eE+-]+)", description)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _write_dry_run_scene(package_dir: Path, input_path: Path, renderer: str, simulators: list[str]) -> Path:
    package_dir.mkdir(parents=True, exist_ok=True)
    scene = {
        "schema": "discoverse.real2sim.scene_builder.v1",
        "id": "dry-run-scene",
        "name": "dry-run scene",
        "coordinate_frame": "gs_metric_zup",
        "source": str(input_path),
        "renderer": renderer,
        "simulators": simulators,
        "assets": [],
        "simulator_artifacts": _simulator_artifact_plan(package_dir, simulators),
    }
    path = package_dir / "scene.json"
    path.write_text(json.dumps(scene, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if "mujoco" in simulators:
        (package_dir / "scene_mujoco.xml").write_text('<mujoco model="dry_run"><worldbody /></mujoco>\n', encoding="utf-8")
    if "motrixsim" in simulators:
        (package_dir / "motrix_import.json").write_text(json.dumps({"schema": "motrixsim.scene.v1", "source": str(input_path), "assets": []}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (package_dir / "viewer.html").write_text("<!doctype html><title>dry-run scene</title><main>dry-run scene</main>\n", encoding="utf-8")
    return path


def _write_scene_manifest(package_dir: Path, *, input_path: Path, renderer: str, simulators: list[str], dry_run: bool, external_assets: list[dict[str, Any]]) -> Path:
    package_dir.mkdir(parents=True, exist_ok=True)
    assets = enrich_scene_manifest_assets({"assets": external_assets}).get("assets", [])
    for path in _candidate_asset_files(package_dir):
        rel = path.relative_to(package_dir)
        assets.append(_viewer_asset(path, url=_run_static_url(path), role="package"))
    manifest = {
        "schema": "real2sim.scene_manifest.v1",
        "input_path": str(input_path),
        "renderer": renderer,
        "renderer_status": renderer_status(renderer, package_dir),
        "simulators": simulators,
        "dry_run": dry_run,
        "assets": assets,
        "simulator_artifacts": _simulator_artifact_plan(package_dir, simulators),
    }
    path = package_dir / "scene_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def enrich_scene_manifest_assets(manifest: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("assets"), list):
        return manifest
    enriched = dict(manifest)
    assets = []
    for item in manifest["assets"]:
        if not isinstance(item, dict):
            assets.append(item)
            continue
        asset = dict(item)
        path_text = str(asset.get("path") or "")
        path = Path(path_text) if path_text else None
        if path and path.exists():
            asset.setdefault("name", path.name)
            asset.setdefault("format", path.suffix.lower().lstrip("."))
            asset.setdefault("kind", _asset_kind(path))
            size_bytes = path.stat().st_size
            asset.setdefault("size_bytes", size_bytes)
            asset.setdefault("size_label", _format_bytes(size_bytes))
        else:
            size_bytes = int(asset.get("size_bytes") or 0)
            if size_bytes:
                asset.setdefault("size_label", _format_bytes(size_bytes))
        kind = str(asset.get("kind") or "")
        autoload_default = kind in {"splat_or_pointcloud", "mesh"} and size_bytes <= VIEWER_AUTOLOAD_MAX_BYTES
        asset.setdefault("autoload", autoload_default)
        if "viewer_hint" not in asset:
            if kind in {"splat_or_pointcloud", "mesh"} and size_bytes > VIEWER_AUTOLOAD_MAX_BYTES:
                asset["viewer_hint"] = f"manual_load_large_asset_over_{VIEWER_AUTOLOAD_MAX_BYTES}_bytes"
            elif autoload_default:
                asset["viewer_hint"] = "autoload"
            else:
                asset["viewer_hint"] = "manual_load"
        assets.append(asset)
    enriched["assets"] = assets
    return enriched


def renderer_status(renderer: str, package_dir: Path) -> dict[str, Any]:
    aholo_manifest = package_dir / "aholo" / "manifest.json"
    spark_available = SPARK_VENDOR_PATH.is_file()
    three_available = THREE_VENDOR_PATH.is_file()
    selected_backend = "aholo" if renderer == "aholo" and aholo_manifest.is_file() else "spark" if renderer == "spark" and spark_available else "three_fallback"
    return {
        "selected": renderer,
        "backend": selected_backend,
        "spark_available": spark_available,
        "spark_bundle": str(SPARK_VENDOR_PATH),
        "three_available": three_available,
        "three_bundle": str(THREE_VENDOR_PATH),
        "aholo_available": aholo_manifest.is_file(),
        "aholo_reason": "Aholo LOD package found" if aholo_manifest.is_file() else "Aholo unavailable: LOD package is not present.",
        "fallback_reason": "" if spark_available else "Three.js fallback: Spark bundle is not installed." if three_available else "Viewer unavailable: Three.js bundle is not installed.",
    }


def _normalize_reconstruction_quality(value: str) -> str:
    mapping = {
        "discoverse-high": "capture-high",
        "capture-high": "capture-high",
        "high": "capture-high",
        "capture-fast": "balanced",
        "fast": "balanced",
        "capture-debug": "balanced",
        "debug": "balanced",
        "motrix-high": "motrix-high",
        "balanced": "balanced",
    }
    return mapping.get(value.strip().lower(), "capture-high")


def _append_discoverse_reconstruction_flags(command: list[str], *, params: dict[str, Any], quality_preset: str) -> None:
    matcher = str(params.get("matcher") or params.get("sfm_matcher") or "exhaustive").strip().lower()
    if matcher not in {"exhaustive", "sequential"}:
        matcher = "exhaustive" if quality_preset == "capture-high" else "sequential"
    command.extend(["--sfm-matcher", matcher])
    camera_model = str(params.get("camera_model") or params.get("sfm_camera_model") or "auto").strip()
    if camera_model and camera_model.lower() != "auto":
        command.extend(["--sfm-camera-model", camera_model])
    for param_name, flag_name in [
        ("max_image_size", "--sfm-max-image-size"),
        ("sift_max_num_features", "--sfm-sift-max-num-features"),
        ("sfm_sequential_overlap", "--sfm-sequential-overlap"),
        ("sfm_feature_threads", "--sfm-feature-threads"),
        ("sfm_matcher_threads", "--sfm-matcher-threads"),
        ("min_sfm_registered_ratio", "--min-sfm-registered-ratio"),
        ("min_sfm_points", "--min-sfm-points"),
        ("anysplat_max_views", "--anysplat-max-views"),
        ("anysplat_mesh_grid", "--anysplat-mesh-grid"),
        ("pose_init_train_steps", "--pose-init-train-steps"),
        ("pose_init_train_max_frames", "--pose-init-train-max-frames"),
        ("pose_init_train_image_max_size", "--pose-init-train-image-max-size"),
        ("pose_init_train_target_gaussians", "--pose-init-train-target-gaussians"),
        ("pose_init_train_max_points", "--pose-init-train-max-points"),
        ("pose_init_train_dense_image_points_per_frame", "--pose-init-train-dense-image-points-per-frame"),
    ]:
        value = params.get(param_name)
        if value not in (None, ""):
            command.extend([flag_name, str(value)])
    if bool(params.get("sfm_guided_matching", quality_preset == "capture-high")):
        command.append("--sfm-guided-matching")
    sfm_frames = str(params.get("sfm_frames") or ("selected" if quality_preset == "capture-high" else "extracted"))
    if sfm_frames in {"extracted", "selected"}:
        command.extend(["--sfm-frames", sfm_frames])
    keyframe_strategy = str(params.get("anysplat_keyframe_strategy") or ("sfm-registered" if quality_preset == "capture-high" else "quality-diversity"))
    if keyframe_strategy in {"quality-diversity", "sfm-registered"}:
        command.extend(["--anysplat-keyframe-strategy", keyframe_strategy])
    if quality_preset == "capture-high":
        defaults = {
            "--anysplat-max-views": "80",
            "--pose-init-train-steps": "15000",
            "--pose-init-train-target-gaussians": "1500000",
            "--pose-init-train-max-points": "2000000",
            "--pose-init-train-image-max-size": "1600",
            "--pose-init-train-preview-frames": "16",
        }
        existing = set(command[1::2])
        for flag, value in defaults.items():
            if flag not in command:
                command.extend([flag, value])
        if "--pose-init-train-replace-scene" not in command and bool(params.get("pose_init_train_replace_scene", True)):
            command.append("--pose-init-train-replace-scene")
        if "--colmap-dense" not in command and bool(params.get("colmap_dense", False)):
            command.append("--colmap-dense")
        if "--colmap-texture-bake" not in command and bool(params.get("colmap_texture_bake", False)):
            command.append("--colmap-texture-bake")


def _postprocess_discoverse_scene_assets(package_dir: Path, work_dir: Path, emit: Callable[[str], None]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "real2sim.discoverse_scene_postprocess.v1",
        "status": "not_run",
        "exports": [],
        "warnings": [],
    }
    source = _best_full_3dgs_asset(package_dir)
    if source is None:
        report["warnings"].append("no full 3DGS PLY with f_dc/opactity fields found; keeping original package assets")
        _write_discoverse_postprocess_report(work_dir, report)
        return report
    target_dir = package_dir / "fallback" / "point_cloud"
    target = target_dir / f"{source.stem}_threejs_rgb.ply"
    export_report_path = target_dir / f"{source.stem}_threejs_rgb_report.json"
    try:
        export_report = _export_3dgs_rgb_fallback(source, target, export_report_path)
        report["status"] = "succeeded"
        report["source"] = str(source)
        report["exports"].append(export_report)
        emit(f"DISCOVERSE-style RGB fallback exported: {target.name} ({export_report.get('vertices', 0)} points)")
    except Exception as exc:
        report["status"] = "warning"
        report["source"] = str(source)
        report["warnings"].append(f"failed to export RGB fallback from 3DGS PLY: {type(exc).__name__}: {exc}")
    _write_discoverse_postprocess_report(work_dir, report)
    return report


def _write_discoverse_postprocess_report(work_dir: Path, report: dict[str, Any]) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "discoverse_postprocess_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _best_full_3dgs_asset(package_dir: Path) -> Path | None:
    candidates = sorted(package_dir.rglob("*.ply"), key=_scene_asset_sort_key)
    for path in candidates:
        try:
            header = _parse_ply_header(path)
        except Exception:
            continue
        names = {name for _type, name in header["properties"]}
        if {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"}.issubset(names):
            return path
    return None


def _parse_ply_header(path: Path) -> dict[str, Any]:
    properties: list[tuple[str, str]] = []
    vertex_count: int | None = None
    fmt = ""
    with path.open("rb") as handle:
        first = handle.readline().decode("ascii", errors="replace").strip()
        if first != "ply":
            raise ValueError("not a PLY file")
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError("PLY header ended unexpectedly")
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property "):
                parts = line.split()
                if len(parts) >= 3:
                    properties.append((parts[-2], parts[-1]))
            elif line == "end_header":
                if vertex_count is None:
                    raise ValueError("PLY header missing vertex count")
                return {"format": fmt, "vertex_count": vertex_count, "properties": properties, "data_offset": handle.tell()}


def _export_3dgs_rgb_fallback(source: Path, target: Path, report_path: Path) -> dict[str, Any]:
    header = _parse_ply_header(source)
    if header["format"] != "binary_little_endian":
        raise ValueError(f"unsupported PLY format for exact RGB fallback: {header['format']}")
    properties: list[tuple[str, str]] = header["properties"]
    type_map = {
        "char": "b",
        "uchar": "B",
        "int8": "b",
        "uint8": "B",
        "short": "h",
        "ushort": "H",
        "int16": "h",
        "uint16": "H",
        "int": "i",
        "uint": "I",
        "int32": "i",
        "uint32": "I",
        "float": "f",
        "float32": "f",
        "double": "d",
        "float64": "d",
    }
    try:
        input_struct = struct.Struct("<" + "".join(type_map[prop_type] for prop_type, _name in properties))
    except KeyError as exc:
        raise ValueError(f"unsupported PLY property type: {exc}") from exc
    output_struct = struct.Struct("<fffBBB")
    names = [name for _type, name in properties]
    indexes = {name: index for index, name in enumerate(names)}
    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"}
    missing = sorted(required.difference(indexes))
    if missing:
        raise ValueError(f"missing required 3DGS fields: {missing}")
    count = int(header["vertex_count"])
    bbox_min = [float("inf"), float("inf"), float("inf")]
    bbox_max = [float("-inf"), float("-inf"), float("-inf")]
    target.parent.mkdir(parents=True, exist_ok=True)
    sh_c0 = 0.28209479177387814
    with source.open("rb") as src, target.open("wb") as dst:
        src.seek(int(header["data_offset"]))
        dst.write((
            "ply\n"
            "format binary_little_endian 1.0\n"
            "comment exported_by real2sim DISCOVERSE-style exact 3DGS RGB fallback\n"
            f"comment source {source.as_posix()}\n"
            f"element vertex {count}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        ).encode("ascii"))
        for _ in range(count):
            raw = src.read(input_struct.size)
            if len(raw) != input_struct.size:
                raise ValueError("PLY data ended before declared vertex count")
            values = input_struct.unpack(raw)
            x, y, z = values[indexes["x"]], values[indexes["y"]], values[indexes["z"]]
            opacity = _sigmoid(float(values[indexes["opacity"]]))
            rgb = []
            for channel in ("f_dc_0", "f_dc_1", "f_dc_2"):
                linear = float(values[indexes[channel]]) * sh_c0 + 0.5
                linear = max(0.0, min(1.0, linear * min(1.0, opacity + 0.35)))
                rgb.append(int(round(linear * 255)))
            dst.write(output_struct.pack(float(x), float(y), float(z), rgb[0], rgb[1], rgb[2]))
            for axis, value in enumerate((float(x), float(y), float(z))):
                bbox_min[axis] = min(bbox_min[axis], value)
                bbox_max[axis] = max(bbox_max[axis], value)
    report = {
        "schema": "real2sim.discoverse_exact_3dgs_rgb_fallback.v1",
        "input": str(source),
        "output": str(target),
        "vertices": count,
        "source_bbox_min": bbox_min,
        "source_bbox_max": bbox_max,
        "target_bbox_min": bbox_min,
        "target_bbox_max": bbox_max,
        "scale_xyz": [1.0, 1.0, 1.0],
        "translate_xyz": [0.0, 0.0, 0.0],
        "note": "RGB fallback point cloud exported from the exact same 3DGS PLY DC color fields so visual layers stay aligned.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _add_preview_asset_artifacts(store: PlatformStore, task_id: str, input_path: Path) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for path in _candidate_asset_files(input_path)[:12]:
        artifact = store.add_artifact(task_id=task_id, name=path.name, path=path, kind=path.suffix.lower().lstrip(".") or "file")
        assets.append(_viewer_asset(path, url=f"/api/artifacts/{artifact['id']}/content", role="source"))
    return assets


def _viewer_asset(path: Path, *, url: str, role: str) -> dict[str, Any]:
    size_bytes = path.stat().st_size if path.exists() else 0
    suffix = path.suffix.lower().lstrip(".")
    kind = _asset_kind(path)
    autoload = kind in {"splat_or_pointcloud", "mesh"} and size_bytes <= VIEWER_AUTOLOAD_MAX_BYTES
    hint = "autoload" if autoload else "manual_load"
    if kind in {"splat_or_pointcloud", "mesh"} and size_bytes > VIEWER_AUTOLOAD_MAX_BYTES:
        hint = f"manual_load_large_asset_over_{VIEWER_AUTOLOAD_MAX_BYTES}_bytes"
    return {
        "name": path.name,
        "path": str(path),
        "url": url,
        "kind": kind,
        "role": role,
        "format": suffix,
        "size_bytes": size_bytes,
        "size_label": _format_bytes(size_bytes),
        "autoload": autoload,
        "viewer_hint": hint,
    }


def _candidate_asset_files(root: Path) -> list[Path]:
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = [path for path in root.rglob("*") if path.is_file()]
    else:
        return []
    return sorted([path for path in files if path.suffix.lower() in SCENE_ASSET_SUFFIXES], key=_scene_asset_sort_key)


def _scene_asset_sort_key(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    full = path.as_posix().lower()
    if name == "scene_cover.png":
        rank = 0
    elif name.endswith("_threejs_rgb.ply") or "fallback/point_cloud" in full:
        rank = 1
    elif name.endswith((".splat", ".spz", ".ksplat", ".sog")):
        rank = 2
    elif name == "pose_init_trained_scene.ply":
        rank = 3
    elif name == "scene.ply" or full.endswith("/3dgs/scene.ply"):
        rank = 4
    elif name in {"pointcloud_projection_preview.png", "source_gs_preview.png"}:
        rank = 5
    elif name.startswith("thumb_") and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        rank = 6
    elif name.endswith(".ply") and ("colmap_sparse" in name or "sparse_points" in name):
        rank = 10
    elif name.endswith(".ply"):
        rank = 7
    elif path.suffix.lower() in {".obj", ".glb", ".gltf"}:
        rank = 8
    elif path.suffix.lower() == ".xml":
        rank = 9
    elif path.suffix.lower() == ".json":
        rank = 11
    else:
        rank = 12
    size = path.stat().st_size if path.exists() else 0
    return (rank, -size, name)


def _asset_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in SPARK_ASSET_SUFFIXES:
        return "splat_or_pointcloud"
    if suffix in {".obj", ".glb", ".gltf"}:
        return "mesh"
    if suffix == ".xml":
        return "mujoco"
    if suffix == ".json":
        return "manifest"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "image"
    return "file"


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def _run_static_url(path: Path) -> str:
    parts = path.parts
    if "runs" in parts:
        index = parts.index("runs")
        return "/" + "/".join(parts[index:])
    return str(path)


def _scene_manifest_path_for_scene(scene: dict[str, Any]) -> Path:
    manifest_path = Path(str(scene.get("manifest_path") or ""))
    if manifest_path.is_file():
        return manifest_path
    scene_path = Path(str(scene.get("path") or ""))
    package_dir = scene_path if scene_path.is_dir() else scene_path.parent
    candidate = package_dir / "scene_manifest.json"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"scene manifest not found for scene path {scene_path}")


def _find_chromium_binary() -> Path | None:
    candidates = [
        os.environ.get("REAL2SIM_CHROME_BIN", ""),
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/snap/bin/chromium",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for item in candidates:
        if item and Path(item).is_file():
            return Path(item)
    for name in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def _render_scene_cover_candidates(config: PlatformConfig, chrome: Path, scene_id: str, artifacts_dir: Path, emit: Callable[[str], None]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for view in ["hero", "front", "side", "top"]:
        out_path = artifacts_dir / f"scene_cover_{view}.png"
        url = f"http://127.0.0.1:{config.port}/static/scene-viewer.html?scene_id={scene_id}&cover=1&cover_view={view}"
        command = [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            "--window-size=1400,820",
            "--virtual-time-budget=12000",
            f"--screenshot={out_path}",
            url,
        ]
        emit("cover command: " + " ".join(command))
        try:
            completed = subprocess.run(command, cwd=str(config.app_root), text=True, capture_output=True, timeout=45)
        except Exception as exc:
            emit(f"cover view {view} failed: {exc}")
            continue
        if completed.returncode != 0 or not out_path.is_file() or out_path.stat().st_size < 4096:
            reason = (completed.stderr or completed.stdout or f"chromium exited {completed.returncode}").strip()
            emit(f"cover view {view} failed: {reason}")
            continue
        score_info = _score_cover_image(out_path)
        candidates.append({
            "view": view,
            "path": str(out_path),
            "url": _run_static_url(out_path),
            "size_bytes": out_path.stat().st_size,
            **score_info,
        })
    return candidates


def _score_cover_image(path: Path) -> dict[str, Any]:
    size_score = min(path.stat().st_size / 180000.0, 1.2)
    try:
        from PIL import Image, ImageStat  # type: ignore

        image = Image.open(path).convert("RGB").resize((160, 94))
        pixels = list(image.getdata())
        non_dark = sum(1 for r, g, b in pixels if r + g + b > 48)
        non_dark_ratio = non_dark / max(len(pixels), 1)
        stat = ImageStat.Stat(image)
        variance = sum(stat.var) / 3.0
        variance_score = min(variance / 1800.0, 1.4)
        score = non_dark_ratio * 2.6 + variance_score + size_score
        return {
            "score": round(score, 4),
            "non_dark_ratio": round(non_dark_ratio, 4),
            "variance": round(variance, 2),
        }
    except Exception:
        return {"score": round(size_score, 4), "non_dark_ratio": None, "variance": None}


def _copy_existing_scene_preview(manifest: dict[str, Any], cover_path: Path) -> None:
    assets = [asset for asset in manifest.get("assets", []) if isinstance(asset, dict)]
    image_assets = [
        asset for asset in assets
        if str(asset.get("path") or "").lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        or str(asset.get("name") or "").lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    image_assets.sort(key=lambda asset: _scene_preview_rank(str(asset.get("name") or asset.get("path") or "")))
    for asset in image_assets:
        path = Path(str(asset.get("path") or ""))
        if path.is_file():
            shutil.copy2(path, cover_path)
            return


def _scene_preview_rank(name: str) -> tuple[int, str]:
    lowered = name.lower()
    if "scene_cover" in lowered:
        return (0, lowered)
    if "pointcloud_projection_preview" in lowered:
        return (1, lowered)
    if "source_gs_preview" in lowered:
        return (2, lowered)
    if "preview" in lowered:
        return (3, lowered)
    if "background_image" in lowered:
        return (4, lowered)
    if "thumb" in lowered:
        return (5, lowered)
    return (9, lowered)


def _simulator_artifact_plan(package_dir: Path, simulators: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "mujoco" in simulators:
        result["mujoco"] = {"expected": str(package_dir / "scene_mujoco.xml")}
    if "motrixsim" in simulators:
        result["motrixsim"] = {"expected": str(package_dir / "motrix_import.json")}
    return result


def _validate_sim_asset(candidates: list[Path], *, required_suffix: str) -> dict[str, Any]:
    existing = next((path for path in candidates if path.is_file() and path.suffix.lower() == required_suffix), None)
    if existing is None:
        return {"ok": False, "path": None, "error": f"no {required_suffix} simulator artifact found"}
    text = existing.read_text(encoding="utf-8", errors="ignore")[:20000]
    checks = {
        "exists": True,
        "non_empty": existing.stat().st_size > 0,
        "has_coordinate_or_schema": any(token in text.lower() for token in ["schema", "coordinate", "mujoco", "motrix", "worldbody"]),
    }
    return {"ok": all(checks.values()), "path": str(existing), "checks": checks}


def _validate_mujoco_asset(candidates: list[Path], *, level: str) -> dict[str, Any]:
    result = _validate_sim_asset(candidates, required_suffix=".xml")
    if not result["ok"] or level == "smoke":
        result["level_effective"] = "smoke"
        return result
    try:
        import mujoco  # type: ignore
    except Exception as exc:
        result["runtime"] = {"status": "unavailable", "error": str(exc)}
        result["level_effective"] = "smoke"
        result["ok"] = result["ok"] and level != "runtime_required"
        return result
    try:
        model = mujoco.MjModel.from_xml_path(str(result["path"]))
        result["runtime"] = {"status": "loaded", "nbody": int(model.nbody), "ngeom": int(model.ngeom)}
        result["level_effective"] = "runtime"
        result["ok"] = True
    except Exception as exc:
        result["runtime"] = {"status": "failed", "error": str(exc)}
        result["level_effective"] = "runtime"
        result["ok"] = False
    return result


def _validate_motrixsim_asset(candidates: list[Path], *, level: str) -> dict[str, Any]:
    descriptor = _validate_sim_asset(candidates, required_suffix=".json")
    if level == "smoke":
        descriptor["level_effective"] = "smoke"
        return descriptor
    runtime_scene = next((path for path in candidates if path.is_file() and path.suffix.lower() == ".xml"), None)
    if runtime_scene is None:
        return {
            "ok": False,
            "path": descriptor.get("path"),
            "descriptor": descriptor,
            "level_effective": "runtime",
            "runtime": {"status": "artifact_unavailable", "error": "no MJCF scene is available for MotrixSim runtime loading"},
        }
    try:
        from motrixsim import SceneData, load_model, step  # type: ignore
    except Exception as exc:
        return {
            "ok": level != "runtime_required" and bool(descriptor.get("ok")),
            "path": str(runtime_scene),
            "descriptor": descriptor,
            "level_effective": "runtime_if_available",
            "runtime": {"status": "unavailable", "error": str(exc), "tooling": _motrixsim_tooling()},
        }
    try:
        model = load_model(str(runtime_scene))
        data = SceneData(model)
        for _ in range(8):
            step(model, data)
        return {
            "ok": True,
            "path": str(runtime_scene),
            "descriptor": descriptor,
            "level_effective": "runtime",
            "runtime": {
                "status": "loaded_and_stepped",
                "steps": 8,
                "timestep": float(model.options.timestep),
                "nq": len(data.low.dof_positions),
                "nv": len(data.low.dof_velocities),
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "path": str(runtime_scene),
            "descriptor": descriptor,
            "level_effective": "runtime",
            "runtime": {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        }


def _motrixsim_tooling() -> dict[str, Any]:
    candidates = [
        Path("/opt/motrixsim-docs"),
        Path("/opt/Motphys/motrixsim-docs/examples/hello_motrixsim.py"),
        Path("/opt/Motphys/MotrixLab"),
        Path("/opt/Motphys/motrixsim_playground"),
    ]
    existing = [str(path) for path in candidates if path.exists()]
    return {"available": bool(existing), "paths": existing}


def _find_scene_json(package_dir: Path) -> Path | None:
    for name in ["scene.json", "motrix_import.json", "package_manifest.json"]:
        candidate = package_dir / name
        if candidate.is_file():
            return candidate
    matches = sorted(package_dir.rglob("scene.json"))
    return matches[0] if matches else None


def _index_artifacts(store: PlatformStore, task_id: str, root: Path) -> None:
    indexable_suffixes = {
        ".json", ".jsonl", ".html", ".svg", ".png", ".jpg", ".jpeg", ".mp4", ".zip", ".tar", ".gz",
        ".ply", ".spz", ".splat", ".ksplat", ".sog", ".rad", ".obj", ".glb", ".gltf", ".xml",
        ".hdf5", ".h5", ".ckpt", ".pkl", ".npy", ".npz", ".csv", ".yaml", ".yml", ".toml", ".py",
        ".log", ".txt",
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in indexable_suffixes:
            store.add_artifact(task_id=task_id, name=path.name, path=path, kind=path.suffix.lower().lstrip(".") or "file")


def index_artifacts_for_task(store: PlatformStore, task_id: str, root: Path) -> None:
    _index_artifacts(store, task_id, root)


def collect_task_reports(store: PlatformStore, task_id: str) -> dict[str, Any]:
    task = store.get_task(task_id)
    run_dir = Path(task["run_dir"])
    report_names = [
        "capture_preflight_report.json",
        "capture_acceptance_report.json",
        "reconstruction_input.json",
        "environment.json",
        "reconstruction_command.json",
        "reconstruction_report.json",
        "validation_report.json",
        "motrixsim_asset_import_report.json",
        "splat_optimization_report.json",
        "prepared_manifest.json",
        "dataset_manifest.json",
        "model_card.json",
    ]
    reports: list[dict[str, Any]] = []
    task_error = str(task.get("error") or "")
    report_errors: list[tuple[int, str]] = []
    candidates = list(sorted(run_dir.rglob("*.json")) if run_dir.exists() else [])
    for artifact in store.list_task_artifacts(task_id):
        artifact_path = Path(artifact["path"])
        if artifact_path.suffix.lower() == ".json" and artifact_path not in candidates:
            candidates.append(artifact_path)
    for path in candidates:
        if path.name not in report_names:
            continue
        payload = _read_optional_json(path)
        failure_summary = _report_failure_summary(path.name, payload)
        if failure_summary is not None:
            report_errors.append(failure_summary)
        try:
            relative_path = str(path.relative_to(run_dir))
        except ValueError:
            relative_path = path.name
        reports.append(
            {
                "name": path.name,
                "path": str(path),
                "relative_path": relative_path,
                "payload": payload,
            }
        )
    return {
        "task_id": task_id,
        "type": task["type"],
        "status": task["status"],
        "readable_error": _choose_readable_error(task_error, report_errors),
        "reports": reports,
    }


def _choose_readable_error(task_error: str, report_errors: list[tuple[int, str]]) -> str:
    clean = task_error.strip()
    if report_errors:
        report_error = max(report_errors, key=lambda item: item[0])[1]
        if not clean or _is_generic_task_error(clean):
            return report_error
    return clean


def _is_generic_task_error(value: str) -> bool:
    clean = value.strip().lower()
    return clean.startswith(("command failed with exit code", "subprocess exited with", "process failed with exit code"))


def _report_failure_summary(name: str, payload: dict[str, Any]) -> tuple[int, str] | None:
    if not isinstance(payload, dict) or not payload:
        return None
    schema = str(payload.get("schema") or "")
    status = str(payload.get("status") or "").lower()
    if name == "capture_acceptance_report.json" or "capture_acceptance" in schema:
        blockers = _string_items(payload.get("blockers") or payload.get("failed_gates"))
        decision = str(payload.get("decision") or payload.get("reason") or "").strip()
        accepted = payload.get("accepted")
        if accepted is False or blockers or status in {"failed", "not_run", "rejected"}:
            parts = ["capture acceptance failed"]
            if decision:
                parts[0] += f": {decision}"
            if blockers:
                parts.append("blockers=" + ", ".join(blockers))
            return (100, "; ".join(parts))
    if name == "capture_preflight_report.json" or "capture_preflight" in schema:
        failed_gates = _string_items(payload.get("failed_gates"))
        if failed_gates:
            return (90, "reconstruction preflight failed: " + ", ".join(failed_gates))
        message = str(payload.get("readable_error") or payload.get("error") or "").strip()
        if message:
            return (80, message)
    if name == "validation_report.json" or "simulator_validation" in schema:
        if payload.get("ok") is False:
            failures = []
            simulators = payload.get("simulators") or {}
            if isinstance(simulators, dict):
                for simulator, result in simulators.items():
                    if isinstance(result, dict) and not result.get("ok", False):
                        reason = str(result.get("reason") or result.get("status") or "failed")
                        failures.append(f"{simulator}:{reason}")
            return (80, "simulator validation failed: " + (", ".join(failures) if failures else "one or more simulators failed"))
    if name == "reconstruction_report.json" or "reconstruction_report" in schema:
        if status == "failed":
            message = str(payload.get("readable_error") or payload.get("error") or "").strip()
            if message:
                return (70, message)
    if payload.get("ok") is False or status in {"failed", "not_run", "rejected"}:
        message = str(payload.get("readable_error") or payload.get("error") or payload.get("reason") or "").strip()
        if message:
            return (50, message)
    return None


def _string_items(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            if isinstance(item, bool):
                if item:
                    items.append(str(key))
            elif item not in (None, ""):
                items.append(f"{key}={item}")
        return items
    if isinstance(value, list):
        items = []
        for item in value:
            if isinstance(item, dict):
                label = item.get("id") or item.get("name") or item.get("check") or item.get("label") or item.get("reason")
                if label:
                    items.append(str(label))
            elif item not in (None, ""):
                items.append(str(item))
        return items
    return [str(value)]


def _normalize_patterns(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]
    raise ValueError("allow_patterns must be a string or list")


def _safe_name(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value).strip(".-")
    return clean or "hf-dataset"


def _file_inventory(root: Path, *, limit: int) -> list[dict[str, Any]]:
    files = [path for path in root.rglob("*") if path.is_file()]
    items = []
    for path in sorted(files)[:limit]:
        items.append({"path": str(path.relative_to(root)), "size": path.stat().st_size, "kind": _asset_kind(path)})
    return items


def _env_with_paths(config: PlatformConfig, python_paths: list[Path] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    runtime_paths = [
        config.anysplat_root,
        config.runtime_root / "speedy-splat",
        config.runtime_root / "sam2",
        config.runtime_root / "sam-3d-objects",
        config.runtime_root / "sam-3d-objects" / "notebook",
    ]
    paths = [str(path) for path in [*(python_paths or []), *runtime_paths] if path.exists()]
    if paths:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(paths + ([existing] if existing else []))
    env.setdefault("REAL2SIM_COLMAP_BIN", str(config.colmap_binary))
    env.setdefault("REAL2SIM_ANYSPLAT_ROOT", str(config.anysplat_root))
    env.setdefault("REAL2SIM_ANY_SPLAT_PYTHON", str(config.anysplat_python))
    env.setdefault("REAL2SIM_POSE_INIT_TRAIN_PYTHON", str(config.gsplat_python))
    env.setdefault("CUDA_HOME", str(config.cuda_home))
    cuda_bin = config.cuda_home / "bin"
    cuda_lib = config.cuda_home / "lib64"
    if cuda_bin.is_dir():
        env["PATH"] = str(cuda_bin) + os.pathsep + env.get("PATH", "")
    if cuda_lib.is_dir():
        env["LD_LIBRARY_PATH"] = str(cuda_lib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any]:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {}


def _tail_text(path: Path, *, max_chars: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    return text[-max_chars:]


def _readable_failure_from_log(text: str) -> str:
    fallback = ""
    for line in reversed(text.splitlines()):
        clean = line.strip()
        if not clean or clean.startswith(("Traceback", "File ", "During handling of the above exception")):
            continue
        for prefix in ["ERROR:", "RuntimeError:", "ValueError:"]:
            if clean.startswith(prefix):
                clean = clean[len(prefix):].strip()
        if clean.startswith("$ "):
            continue
        if clean.startswith(("raise ", "subprocess.CalledProcessError", "CalledProcessError(")):
            continue
        if not fallback:
            fallback = clean
        lowered = clean.lower()
        if any(marker in lowered for marker in [" requires ", " failed", "error", "missing", "not found", "recapture", "blocker"]):
            return clean
    return fallback


def _append_log(path: Path, message: str) -> None:
    path.write_text(path.read_text(encoding="utf-8") + message.rstrip() + "\n", encoding="utf-8")


def _module_available(name: str) -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=1)
def _torch_cuda_ready() -> bool:
    if not _module_available("torch"):
        return False
    completed = subprocess.run(
        [sys.executable, "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 2)"],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return completed.returncode == 0


@lru_cache(maxsize=4)
def _act_runtime_import_ready(discoverse_root_value: str, discoverse_commit: str, act_commit: str) -> bool:
    discoverse_root = Path(discoverse_root_value)
    act_root = discoverse_root / "policies" / "act"
    if _git_head(discoverse_root) != discoverse_commit or _git_head(act_root) != act_commit or not _torch_cuda_ready():
        return False
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(act_root), str(discoverse_root), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-c", "from policies.act.act import ACTPolicy; assert ACTPolicy is not None"],
        cwd=act_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    return completed.returncode == 0


def _is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False
