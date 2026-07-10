#!/usr/bin/env bash
set -euo pipefail

KEY="${REAL2SIM_WUYING_KEY:-/Users/d-robotics/workSpace/document/80-private/orgs/drobotics/keys/wuying-39.106.101.95_ed25519}"
HOST="${REAL2SIM_WUYING_HOST:-root@39.106.101.95}"
BASE_URL="${REAL2SIM_ACCEPTANCE_BASE_URL:-http://127.0.0.1:18126}"
MODE="${REAL2SIM_ACCEPTANCE_MODE:-production}"
SCALE="${REAL2SIM_ACCEPTANCE_SCALE:-pressure}"
OUT="${REAL2SIM_ACCEPTANCE_OUT:-/opt/real2sim2model/real2sim-platform/workspace/acceptance_runs}"
HF_REPO="${REAL2SIM_ACCEPTANCE_HF_REPO:-shashuo0104/xarm7_pusht}"
HF_PATTERNS="${REAL2SIM_ACCEPTANCE_HF_PATTERNS:-videos/chunk-000/observation.images.front/*.mp4,data/chunk-000/*.parquet,meta/**,README.md}"
RECON_FIXTURE="${REAL2SIM_ACCEPTANCE_RECON_FIXTURE:-/opt/real2sim_paper_runtime/sam-3d-objects/notebook/images/shutterstock_modern_colorful_Interior_2620125197}"
RECON_FALLBACKS="${REAL2SIM_ACCEPTANCE_RECON_FALLBACKS:-/opt/real2sim_paper_runtime/sam2/notebooks/videos/bedroom}"
MIN_RECON_FRAMES="${REAL2SIM_ACCEPTANCE_MIN_RECON_FRAMES:-80}"
TARGET_HF_FRAMES="${REAL2SIM_ACCEPTANCE_TARGET_HF_FRAMES:-240}"
MIN_HF_FRAMES="${REAL2SIM_ACCEPTANCE_MIN_HF_FRAMES:-120}"
TIMEOUT_SECONDS="${REAL2SIM_ACCEPTANCE_TIMEOUT_SECONDS:-7200}"
REQUIRE_MOTRIXSIM="${REAL2SIM_REQUIRE_MOTRIXSIM:-0}"
REPEAT_RECONSTRUCTIONS="${REAL2SIM_ACCEPTANCE_REPEAT_RECONSTRUCTIONS:-2}"
PRESSURE_TASKS="${REAL2SIM_ACCEPTANCE_PRESSURE_TASKS:-12}"
CONTINUE_AFTER_RECON_QUALITY_FAIL="${REAL2SIM_ACCEPTANCE_CONTINUE_AFTER_RECON_QUALITY_FAIL:-1}"

ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$HOST" \
  env \
  REAL2SIM_ACCEPTANCE_BASE_URL="$BASE_URL" \
  REAL2SIM_ACCEPTANCE_MODE="$MODE" \
  REAL2SIM_ACCEPTANCE_SCALE="$SCALE" \
  REAL2SIM_ACCEPTANCE_OUT="$OUT" \
  REAL2SIM_ACCEPTANCE_HF_REPO="$HF_REPO" \
  REAL2SIM_ACCEPTANCE_HF_PATTERNS="$HF_PATTERNS" \
  REAL2SIM_ACCEPTANCE_RECON_FIXTURE="$RECON_FIXTURE" \
  REAL2SIM_ACCEPTANCE_RECON_FALLBACKS="$RECON_FALLBACKS" \
  REAL2SIM_ACCEPTANCE_MIN_RECON_FRAMES="$MIN_RECON_FRAMES" \
  REAL2SIM_ACCEPTANCE_TARGET_HF_FRAMES="$TARGET_HF_FRAMES" \
  REAL2SIM_ACCEPTANCE_MIN_HF_FRAMES="$MIN_HF_FRAMES" \
  REAL2SIM_ACCEPTANCE_TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
  REAL2SIM_REQUIRE_MOTRIXSIM="$REQUIRE_MOTRIXSIM" \
  REAL2SIM_ACCEPTANCE_REPEAT_RECONSTRUCTIONS="$REPEAT_RECONSTRUCTIONS" \
  REAL2SIM_ACCEPTANCE_PRESSURE_TASKS="$PRESSURE_TASKS" \
  REAL2SIM_ACCEPTANCE_CONTINUE_AFTER_RECON_QUALITY_FAIL="$CONTINUE_AFTER_RECON_QUALITY_FAIL" \
  python3 -s <<'PY'
from __future__ import annotations

import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE = os.environ["REAL2SIM_ACCEPTANCE_BASE_URL"].rstrip("/")
MODE = os.environ["REAL2SIM_ACCEPTANCE_MODE"]
SCALE = os.environ["REAL2SIM_ACCEPTANCE_SCALE"]
OUT_ROOT = Path(os.environ["REAL2SIM_ACCEPTANCE_OUT"])
HF_REPO = os.environ["REAL2SIM_ACCEPTANCE_HF_REPO"]
HF_PATTERNS = [item.strip() for item in os.environ["REAL2SIM_ACCEPTANCE_HF_PATTERNS"].split(",") if item.strip()]
RECON_FIXTURE = os.environ["REAL2SIM_ACCEPTANCE_RECON_FIXTURE"]
RECON_FALLBACKS = [item.strip() for item in os.environ["REAL2SIM_ACCEPTANCE_RECON_FALLBACKS"].split(":") if item.strip()]
MIN_RECON_FRAMES = int(os.environ["REAL2SIM_ACCEPTANCE_MIN_RECON_FRAMES"])
TARGET_HF_FRAMES = int(os.environ["REAL2SIM_ACCEPTANCE_TARGET_HF_FRAMES"])
MIN_HF_FRAMES = int(os.environ["REAL2SIM_ACCEPTANCE_MIN_HF_FRAMES"])
TIMEOUT = int(os.environ["REAL2SIM_ACCEPTANCE_TIMEOUT_SECONDS"])
REQUIRE_MOTRIXSIM = os.environ["REAL2SIM_REQUIRE_MOTRIXSIM"] == "1"
REPEAT_RECONSTRUCTIONS = int(os.environ["REAL2SIM_ACCEPTANCE_REPEAT_RECONSTRUCTIONS"])
PRESSURE_TASKS = int(os.environ["REAL2SIM_ACCEPTANCE_PRESSURE_TASKS"])
CONTINUE_AFTER_RECON_QUALITY_FAIL = os.environ["REAL2SIM_ACCEPTANCE_CONTINUE_AFTER_RECON_QUALITY_FAIL"] == "1"
TERMINAL_STATUSES = {"succeeded", "failed", "canceled", "failed_recovered"}

started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
run_dir = OUT_ROOT / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
run_dir.mkdir(parents=True, exist_ok=True)

report: dict[str, Any] = {
    "schema": "real2sim.acceptance_report.v1",
    "mode": MODE,
    "scale": SCALE,
    "started_at": started_at,
    "finished_at": "",
    "base_url": BASE,
    "inputs": {
        "hf_repo": HF_REPO,
        "hf_patterns": HF_PATTERNS,
        "reconstruction_fixture": RECON_FIXTURE,
        "reconstruction_fallbacks": RECON_FALLBACKS,
        "min_reconstruction_frames": MIN_RECON_FRAMES,
        "target_hf_frames": TARGET_HF_FRAMES,
        "min_hf_frames": MIN_HF_FRAMES,
        "repeat_reconstructions": REPEAT_RECONSTRUCTIONS,
        "pressure_tasks": PRESSURE_TASKS,
        "require_motrixsim": REQUIRE_MOTRIXSIM,
        "continue_after_reconstruction_quality_fail": CONTINUE_AFTER_RECON_QUALITY_FAIL,
    },
    "version": {},
    "doctor": {},
    "tasks": [],
    "datasets": [],
    "scenes": [],
    "models": [],
    "browser": {},
    "viewer_probe": {
        "status": "not_run",
        "reason": "Run scripts/scene_viewer_probe_local.mjs separately; production API acceptance does not launch local Playwright.",
    },
    "screenshots": [],
    "artifacts": [],
    "warnings": [],
    "failures": [],
    "pass": False,
}


def api(path: str, payload: Any | None = None, timeout: int = 60) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            if not body:
                return None
            ctype = response.headers.get("content-type", "")
            return json.loads(body) if "application/json" in ctype else body
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} failed with HTTP {exc.code}: {detail}") from exc


def record_failure(message: str, **context: Any) -> None:
    failure = {"message": message, **context}
    print("FAIL:", json.dumps(failure, ensure_ascii=False), flush=True)
    report["failures"].append(failure)


def save_report() -> None:
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (run_dir / "acceptance_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Real2Sim Production Acceptance",
        "",
        f"- pass: {str(report['pass']).lower()}",
        f"- started_at: {report['started_at']}",
        f"- finished_at: {report['finished_at']}",
        f"- build_id: {report.get('version', {}).get('build_id', '')}",
        f"- tasks: {len(report['tasks'])}",
        f"- failures: {len(report['failures'])}",
        f"- warnings: {len(report['warnings'])}",
        f"- viewer_probe: {report.get('viewer_probe', {}).get('status', 'not_run')}",
        "",
        "## Task IDs",
    ]
    for task in report["tasks"]:
        lines.append(f"- {task.get('type')} {task.get('id')} {task.get('status')}")
    if report["failures"]:
        lines.extend(["", "## Failures"])
        for failure in report["failures"]:
            lines.append(f"- {failure.get('message')} {json.dumps(failure, ensure_ascii=False)[:600]}")
    if report["warnings"]:
        lines.extend(["", "## Warnings"])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def task_snapshot(task_id: str) -> dict[str, Any]:
    task = api(f"/api/tasks/{task_id}")
    try:
        reports = api(f"/api/tasks/{task_id}/reports")
    except Exception as exc:
        reports = {"error": str(exc), "reports": []}
    try:
        artifacts = api(f"/api/tasks/{task_id}/artifacts")
    except Exception as exc:
        artifacts = [{"error": str(exc)}]
    try:
        logs = api(f"/api/tasks/{task_id}/logs")
    except Exception as exc:
        logs = f"log unavailable: {exc}"
    snapshot = {
        "id": task["id"],
        "type": task["type"],
        "status": task["status"],
        "params": task.get("params"),
        "result": task.get("result"),
        "error": task.get("error"),
        "run_dir": task.get("run_dir"),
        "reports": reports,
        "artifacts": artifacts,
        "log_tail": str(logs)[-5000:],
    }
    (run_dir / f"task_{task_id}.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return snapshot


def wait_task(task_id: str, *, required: bool = True) -> dict[str, Any]:
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        task = api(f"/api/tasks/{task_id}")
        if task["status"] in TERMINAL_STATUSES:
            snapshot = task_snapshot(task_id)
            report["tasks"].append(snapshot)
            if required and task["status"] != "succeeded":
                record_failure(f"required task {task['type']} did not succeed", task_id=task_id, status=task["status"], error=task.get("error"))
            print(json.dumps({"task_id": task_id, "type": task["type"], "status": task["status"]}, ensure_ascii=False), flush=True)
            return snapshot
        time.sleep(5)
    snapshot = task_snapshot(task_id)
    report["tasks"].append(snapshot)
    record_failure("task timed out", task_id=task_id)
    if required:
        raise TimeoutError(task_id)
    return snapshot


def create_task(task_type: str, params: dict[str, Any], *, required: bool = True) -> dict[str, Any]:
    task = api("/api/tasks", {"type": task_type, "params": params})
    print(f"{task_type}: {task['id']}", flush=True)
    return wait_task(task["id"], required=required)


def assert_artifacts(task: dict[str, Any], required_names: list[str] | None = None) -> None:
    artifacts = task.get("artifacts") or []
    names = {item.get("name") for item in artifacts}
    if not artifacts:
        record_failure("succeeded task has no artifacts", task_id=task.get("id"), task_type=task.get("type"))
    for name in required_names or []:
        if name not in names:
            record_failure("required artifact missing", task_id=task.get("id"), task_type=task.get("type"), artifact=name, available=sorted(n for n in names if n))
    for item in artifacts:
        if item.get("name"):
            report["artifacts"].append({"task_id": task.get("id"), "name": item.get("name"), "path": item.get("path"), "kind": item.get("kind")})


def refresh_entities() -> None:
    report["datasets"] = api("/api/datasets")
    report["scenes"] = api("/api/scenes")
    report["models"] = api("/api/models")


def check_doctor() -> None:
    report["version"] = api("/api/version")
    doctor = api("/api/doctor")
    report["doctor"] = doctor
    checks = doctor.get("checks", {})
    required = ["anysplat_root", "anysplat_python", "spark_bundle", "mujoco_python", "media_prepare_deps", "service_process"]
    for name in required:
        if not checks.get(name):
            record_failure("doctor required check failed", check=name, value=checks.get(name))
    for warning in doctor.get("warnings", []) or []:
        report["warnings"].append(f"doctor: {warning}")
    if not doctor.get("ok"):
        record_failure("doctor returned ok=false", doctor=doctor)


def latest_scene_id(task: dict[str, Any]) -> str:
    scene_id = (((task.get("result") or {}).get("scene_id")) or "")
    if not scene_id:
        record_failure("scene task did not return scene_id", task_id=task.get("id"))
    return scene_id


def run_pressure_tasks(scene_id: str, expert_dataset_id: str, model_id: str, reconstruction_input: str) -> None:
    queued = []
    templates = [
        ("simulator_validation", {"scene_id": scene_id, "simulators": ["mujoco", "motrixsim"], "level": "runtime_if_available"}),
        ("expert_collection", {"scene_id": scene_id, "episodes": 3, "steps": 24}),
        ("training", {"dataset_id": expert_dataset_id, "epochs": 1}),
        ("inference", {"scene_id": scene_id, "model_id": model_id, "steps": 40}),
        ("reconstruction_preflight", {"input_path": reconstruction_input, "min_frames": MIN_RECON_FRAMES}),
    ]
    for index in range(max(PRESSURE_TASKS, 12)):
        task_type, params = templates[index % len(templates)]
        task = api("/api/tasks", {"type": task_type, "params": params})
        queued.append(task)
        print(f"pressure {index + 1}: {task_type} {task['id']}", flush=True)
    for task in queued:
        wait_task(task["id"], required=True)
    active = [task for task in api("/api/tasks") if task.get("status") in {"queued", "running", "canceling"}]
    if active:
        record_failure("active tasks remain after pressure batch", active=active[:20])


def select_reconstruction_input() -> tuple[str, dict[str, Any]]:
    candidates = [RECON_FIXTURE, *RECON_FALLBACKS]
    snapshots = []
    for index, candidate in enumerate(candidates):
        preflight = create_task("reconstruction_preflight", {
            "input_path": candidate,
            "min_frames": MIN_RECON_FRAMES,
        })
        assert_artifacts(preflight, ["capture_preflight_report.json"])
        snapshots.append(preflight)
        result = preflight.get("result") or {}
        if result.get("ok"):
            if index > 0:
                report["warnings"].append(f"primary reconstruction fixture failed preflight; using fallback: {candidate}")
            return candidate, preflight
        report["warnings"].append(f"reconstruction fixture candidate failed preflight: {candidate}")
    record_failure("no reconstruction fixture passed preflight", candidates=candidates)
    return candidates[0], snapshots[-1] if snapshots else {}


def reconstruction_failed_on_quality_gate(task: dict[str, Any]) -> bool:
    reports = task.get("reports") or {}
    readable = str(reports.get("readable_error") or task.get("error") or "")
    for report_item in reports.get("reports", []) or []:
        payload = report_item.get("payload") or {}
        readable += " " + str(payload.get("readable_error") or payload.get("decision") or "")
        readable += " " + " ".join(str(item) for item in payload.get("blockers", []) or [])
    return any(marker in readable for marker in ["allow-low-quality", "capture acceptance failed", "recapture_required", "registered_ratio", "sparse_points"])


def run_reconstruction_with_optional_continuation(reconstruction_input: str, repeat_index: int) -> dict[str, Any]:
    params = {
        "input_path": reconstruction_input,
        "dry_run": False,
        "renderer": "spark",
        "simulators": ["mujoco", "motrixsim"],
        "min_frames": MIN_RECON_FRAMES,
        "quality_preset": "capture-high",
    }
    strict = create_task("scene_reconstruction", params)
    if strict.get("status") == "succeeded":
        assert_artifacts(strict, ["environment.json", "reconstruction_command.json", "capture_preflight_report.json", "reconstruction_report.json", "scene_manifest.json"])
        return strict
    assert_artifacts(strict, ["capture_preflight_report.json", "reconstruction_report.json"])
    if not (CONTINUE_AFTER_RECON_QUALITY_FAIL and reconstruction_failed_on_quality_gate(strict)):
        raise RuntimeError(f"scene_reconstruction failed: {strict.get('id')}")
    report["warnings"].append(f"strict reconstruction {strict.get('id')} failed quality gate; continuing with allow_low_quality for downstream coverage")
    continuation_params = {**params, "allow_low_quality": True, "name": f"acceptance-continuation-{repeat_index}"}
    continuation = create_task("scene_reconstruction", continuation_params)
    if continuation.get("status") != "succeeded":
        assert_artifacts(continuation, ["capture_preflight_report.json", "reconstruction_report.json"])
        raise RuntimeError(f"allow_low_quality scene_reconstruction failed: {continuation.get('id')}")
    assert_artifacts(continuation, ["environment.json", "reconstruction_command.json", "capture_preflight_report.json", "reconstruction_report.json", "scene_manifest.json"])
    return continuation


def main() -> int:
    try:
        check_doctor()
        hf_import = create_task("hf_import", {
            "repo_id": HF_REPO,
            "repo_type": "dataset",
            "allow_patterns": HF_PATTERNS,
            "local_name": "acceptance-xarm7-pusht",
        })
        assert_artifacts(hf_import)
        hf_dataset_id = (hf_import.get("result") or {}).get("dataset_id")
        if not hf_dataset_id:
            record_failure("hf_import did not return dataset_id", task_id=hf_import.get("id"))

        media = create_task("media_prepare", {
            "dataset_id": hf_dataset_id,
            "camera_label": "observation.images.front",
            "sample_fps": 2,
            "max_frames_per_video": 80,
            "max_videos": 16,
            "target_frame_count": TARGET_HF_FRAMES,
            "min_total_frames": MIN_HF_FRAMES,
            "output_name": "acceptance-xarm7-pusht-prepared",
        })
        assert_artifacts(media)
        prepared_id = (media.get("result") or {}).get("dataset_id")
        frame_count = int((media.get("result") or {}).get("frame_count") or 0)
        if frame_count < MIN_HF_FRAMES:
            record_failure("media_prepare produced too few frames", frame_count=frame_count, minimum=MIN_HF_FRAMES)

        hf_preflight = create_task("reconstruction_preflight", {
            "dataset_id": prepared_id,
            "min_frames": MIN_RECON_FRAMES,
            "allow_low_quality": True,
        })
        assert_artifacts(hf_preflight, ["capture_preflight_report.json"])

        reconstruction_input, fixture_preflight = select_reconstruction_input()
        if not ((fixture_preflight.get("result") or {}).get("ok")):
            raise RuntimeError("no reconstruction input passed preflight")

        reconstruction_tasks = []
        for index in range(max(REPEAT_RECONSTRUCTIONS, 2)):
            reconstruction_tasks.append(run_reconstruction_with_optional_continuation(reconstruction_input, index + 1))

        scene_id = latest_scene_id(reconstruction_tasks[-1])
        if not scene_id:
            raise RuntimeError("scene_reconstruction did not produce scene_id")
        validation = create_task("simulator_validation", {
            "scene_id": scene_id,
            "simulators": ["mujoco", "motrixsim"],
            "level": "runtime_required" if REQUIRE_MOTRIXSIM else "runtime_if_available",
        })
        assert_artifacts(validation, ["validation_report.json"])
        validation_report = (validation.get("result") or {}).get("report") or {}
        motrix = (validation_report.get("simulators") or {}).get("motrixsim") or {}
        if REQUIRE_MOTRIXSIM and not motrix.get("ok"):
            record_failure("MotrixSim runtime is required but validation failed", motrix=motrix)
        elif not motrix.get("ok"):
            report["warnings"].append(f"motrixsim validation not fully ok: {motrix.get('reason') or motrix.get('status')}")

        expert = create_task("expert_collection", {"scene_id": scene_id, "episodes": 12, "steps": 48, "name": "acceptance-expert-rollouts"})
        assert_artifacts(expert, ["manifest.json", "rollouts.jsonl"])
        expert_dataset_id = (expert.get("result") or {}).get("dataset_id")
        if not expert_dataset_id:
            record_failure("expert_collection did not return dataset_id", task_id=expert.get("id"))

        training = create_task("training", {"dataset_id": expert_dataset_id, "epochs": 3, "name": "acceptance-policy"})
        assert_artifacts(training, ["model_card.json", "policy_head.json"])
        model_id = (training.get("result") or {}).get("model_id")
        if not model_id:
            record_failure("training did not return model_id", task_id=training.get("id"))

        inference = create_task("inference", {"scene_id": scene_id, "model_id": model_id, "steps": 80})
        assert_artifacts(inference, ["prediction_rollout.json", "viewer.html"])
        if int((inference.get("result") or {}).get("rollout_steps") or 0) <= 0:
            record_failure("inference produced empty rollout", task_id=inference.get("id"), result=inference.get("result"))

        run_pressure_tasks(scene_id, expert_dataset_id, model_id, reconstruction_input)
        refresh_entities()
        report["pass"] = len(report["failures"]) == 0
        save_report()
        print(json.dumps({"pass": report["pass"], "run_dir": str(run_dir), "report": str(run_dir / "acceptance_report.json")}, ensure_ascii=False), flush=True)
        return 0 if report["pass"] else 1
    except Exception as exc:
        record_failure("acceptance runner crashed", error=str(exc), traceback=traceback.format_exc())
        try:
            refresh_entities()
        except Exception:
            pass
        report["pass"] = False
        save_report()
        print(json.dumps({"pass": False, "run_dir": str(run_dir), "report": str(run_dir / "acceptance_report.json"), "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1


sys.exit(main())
PY
