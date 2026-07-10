#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import json
import os
import secrets
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KUBECONFIG = Path("/Users/d-robotics/Downloads/poc.config")
DEFAULT_PLAYROOM = Path("/Users/d-robotics/Downloads/images")
NAMESPACE = "real2sim-poc"
TERMINAL = {"succeeded", "failed", "failed_recovered", "canceled"}


class AcceptanceError(RuntimeError):
    pass


class ApiError(AcceptanceError):
    def __init__(self, status: int, path: str, payload: Any) -> None:
        super().__init__(f"HTTP {status} {path}: {payload}")
        self.status = status
        self.path = path
        self.payload = payload


class ApiClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.cookies = http.cookiejar.CookieJar()
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=context),
            urllib.request.HTTPCookieProcessor(self.cookies),
        )

    def request(
        self,
        method: str,
        path: str,
        payload: Any | None = None,
        *,
        body: bytes | None = None,
        content_type: str = "application/json",
        timeout: int = 180,
    ) -> Any:
        data = body
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=timeout) as response:
                raw = response.read()
                response_type = response.headers.get("content-type", "")
                if "application/json" in response_type:
                    return json.loads(raw.decode("utf-8"))
                return raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                payload = raw.decode("utf-8", errors="replace")
            raise ApiError(exc.code, path, payload) from exc

    def get(self, path: str) -> Any:
        return self.request("GET", path)

    def post(self, path: str, payload: Any | None = None) -> Any:
        return self.request("POST", path, payload)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def kubectl(kubeconfig: Path, *args: str, timeout: int = 3600, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kubectl", "--kubeconfig", str(kubeconfig), *args],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=check,
    )


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_port_forward(kubeconfig: Path, port: int, out_dir: Path) -> subprocess.Popen[str]:
    log = (out_dir / "port-forward.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            "kubectl",
            "--kubeconfig",
            str(kubeconfig),
            "-n",
            NAMESPACE,
            "port-forward",
            "svc/real2sim-frontend",
            f"{port}:443",
            "--address",
            "127.0.0.1",
        ],
        text=True,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 90
    while time.time() < deadline:
        if process.poll() is not None:
            log.close()
            raise AcceptanceError((out_dir / "port-forward.log").read_text(encoding="utf-8", errors="replace"))
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return process
        except OSError:
            time.sleep(1)
    process.terminate()
    log.close()
    raise AcceptanceError("timed out waiting for the HTTPS port-forward")


class AcceptanceRun:
    def __init__(self, *, client: ApiClient, out_dir: Path, kubeconfig: Path, build_id: str, timeout_seconds: int) -> None:
        self.client = client
        self.out_dir = out_dir
        self.kubeconfig = kubeconfig
        self.build_id = build_id
        self.timeout_seconds = timeout_seconds
        self.report: dict[str, Any] = {
            "schema": "real2sim.poc_acceptance_report.v1",
            "started_at": utc_now(),
            "finished_at": "",
            "build_id": build_id,
            "namespace": NAMESPACE,
            "version": {},
            "doctor": {},
            "capabilities": {},
            "recipes": {},
            "identities": {},
            "tasks": [],
            "datasets": [],
            "scenes": [],
            "models": [],
            "storage": {},
            "browser": {"status": "pending", "note": "Run browser acceptance after the API gate."},
            "artifacts": [],
            "checks": [],
            "failures": [],
            "warnings": [],
            "pass": False,
        }

    def save(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "acceptance_report.json").write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        lines = [
            f"# Real2Sim POC Acceptance {self.build_id}",
            "",
            f"- Started: {self.report['started_at']}",
            f"- Finished: {self.report['finished_at'] or 'running'}",
            f"- Pass: **{self.report['pass']}**",
            f"- Tasks: {len(self.report['tasks'])}",
            f"- Failures: {len(self.report['failures'])}",
            "",
            "## Checks",
            *[f"- {'PASS' if item['ok'] else 'FAIL'}: {item['name']}" for item in self.report["checks"]],
            "",
            "## Failures",
            *([f"- {item['stage']}: {item['error']}" for item in self.report["failures"]] or ["- None"]),
            "",
            "## Warnings",
            *([f"- {item}" for item in self.report["warnings"]] or ["- None"]),
        ]
        (self.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def check(self, name: str, ok: bool, detail: Any = None) -> None:
        self.report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        self.save()
        if not ok:
            raise AcceptanceError(f"{name}: {detail}")

    def stage(self, name: str, action: Callable[[], Any]) -> Any | None:
        print(f"\n== {name} ==", flush=True)
        started = time.time()
        try:
            result = action()
            self.report["checks"].append({"name": name, "ok": True, "elapsed_seconds": round(time.time() - started, 2)})
            self.save()
            return result
        except Exception as exc:
            failure = {
                "stage": name,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.time() - started, 2),
                "next_action": "Inspect the task reports and logs recorded in this acceptance directory, fix the real runtime blocker, then rerun the gate.",
            }
            self.report["checks"].append({"name": name, "ok": False, "elapsed_seconds": failure["elapsed_seconds"]})
            self.report["failures"].append(failure)
            self.save()
            print(f"FAILED: {failure['error']}", file=sys.stderr, flush=True)
            return None

    def create_task(self, task_type: str, params: dict[str, Any], *, required_artifacts: tuple[str, ...] = ()) -> dict[str, Any]:
        created = self.client.post("/api/tasks", {"type": task_type, "params": params})
        print(f"{task_type}: {created['id']}", flush=True)
        return self.wait_task(created["id"], required_artifacts=required_artifacts)

    def wait_task(
        self,
        task_id: str,
        *,
        required_artifacts: tuple[str, ...] = (),
        expected_status: str = "succeeded",
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        deadline = time.time() + int(timeout_seconds or self.timeout_seconds)
        previous = None
        task: dict[str, Any] = {}
        while time.time() < deadline:
            task = self.client.get(f"/api/tasks/{task_id}")
            marker = (task.get("status"), task.get("progress"), task.get("stage"), task.get("message"))
            if marker != previous:
                print(f"  {task.get('status')} {task.get('progress')}% {task.get('stage')} {task.get('message')}", flush=True)
                previous = marker
            if task.get("status") in TERMINAL:
                break
            time.sleep(5)
        else:
            raise AcceptanceError(f"task {task_id} timed out after {timeout_seconds or self.timeout_seconds}s")

        artifacts = self.client.get(f"/api/tasks/{task_id}/artifacts")
        reports = self.client.get(f"/api/tasks/{task_id}/reports")
        task_record = {
            "id": task_id,
            "type": task.get("type"),
            "status": task.get("status"),
            "progress": task.get("progress"),
            "stage": task.get("stage"),
            "message": task.get("message"),
            "error": task.get("error"),
            "result": task.get("result"),
            "artifact_count": len(artifacts),
            "artifact_names": sorted({str(item.get("name") or "") for item in artifacts}),
            "reports": reports,
        }
        self.report["tasks"].append(task_record)
        self.save()
        if task.get("status") != expected_status:
            log = self.client.get(f"/api/tasks/{task_id}/logs")
            failure_dir = self.out_dir / "task_failures" / task_id
            failure_dir.mkdir(parents=True, exist_ok=True)
            (failure_dir / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (failure_dir / "reports.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (failure_dir / "artifacts.json").write_text(json.dumps(artifacts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (failure_dir / "task.log").write_text(str(log), encoding="utf-8")
            raise AcceptanceError(f"task {task_id} ended as {task.get('status')}: {reports.get('readable_error') or task.get('error')}")
        if expected_status == "succeeded" and not artifacts:
            raise AcceptanceError(f"succeeded task {task_id} has no indexed artifacts")
        names = {str(item.get("name") or "") for item in artifacts}
        missing = sorted(set(required_artifacts) - names)
        if missing:
            raise AcceptanceError(f"task {task_id} is missing required artifacts: {missing}")
        result = task.get("result") or {}
        if expected_status == "succeeded" and not str(result.get("run_storage_uri") or "").startswith("bos://testxyz/real2sim-poc/runs/"):
            raise AcceptanceError(f"task {task_id} has no owned BOS run URI")
        return task


def credential_candidates(root: Path, out_dir: Path) -> list[str]:
    candidates: list[str] = []
    explicit = os.environ.get("REAL2SIM_POC_ADMIN_PASSWORD", "").strip()
    if explicit:
        candidates.append(explicit)
    for path in [out_dir / "acceptance-admin-password.txt", out_dir / "bootstrap-admin-password.txt"]:
        if path.is_file():
            candidates.append(path.read_text(encoding="utf-8").strip())
    prior = sorted((root / "acceptance_runs" / "poc").glob("*/acceptance-admin-password.txt"), key=lambda path: path.stat().st_mtime, reverse=True)
    prior += sorted((root / "acceptance_runs" / "poc").glob("*/bootstrap-admin-password.txt"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in prior:
        candidates.append(path.read_text(encoding="utf-8").strip())
    return list(dict.fromkeys(value for value in candidates if value))


def authenticate_admin(run: AcceptanceRun) -> str:
    login = None
    current_password = ""
    errors: list[str] = []
    for candidate in credential_candidates(ROOT, run.out_dir):
        try:
            login = run.client.post("/api/auth/login", {"username": "admin", "password": candidate})
            current_password = candidate
            break
        except ApiError as exc:
            errors.append(str(exc.status))
    if login is None:
        raise AcceptanceError(f"admin login failed for all credential candidates: {errors}")
    if login["user"].get("must_change_password"):
        next_password = "Real2Sim!" + secrets.token_urlsafe(18) + "Aa1"
        run.client.post(
            "/api/auth/change-password",
            {"current_password": current_password, "new_password": next_password},
        )
        run.client.post("/api/auth/login", {"username": "admin", "password": next_password})
        current_password = next_password
    password_path = run.out_dir / "acceptance-admin-password.txt"
    password_path.write_text(current_password + "\n", encoding="utf-8")
    password_path.chmod(0o600)
    me = run.client.get("/api/auth/me")["user"]
    run.check("管理员登录且已完成首次改密", me.get("role") == "admin" and not me.get("must_change_password"), me)
    return current_password


def verify_rbac(run: AcceptanceRun) -> None:
    suffix = secrets.token_hex(4)
    records: dict[str, Any] = {}
    for role in ("engineer", "viewer"):
        username = f"acceptance.{role}.{suffix}"
        password = f"{role.title()}!Password123"
        user = run.client.post("/api/users", {"username": username, "password": password, "role": role})
        records[role] = {"id": user["id"], "username": username}
        role_client = ApiClient(run.client.base_url)
        login = role_client.post("/api/auth/login", {"username": username, "password": password})
        if login["user"].get("must_change_password"):
            changed = f"{role.title()}!Changed456"
            role_client.post("/api/auth/change-password", {"current_password": password, "new_password": changed})
            role_client.post("/api/auth/login", {"username": username, "password": changed})
        role_client.get("/api/datasets")
        try:
            payload = {"type": "motrixsim_asset_import" if role == "engineer" else "renderer_benchmark", "params": {}}
            role_client.post("/api/tasks", payload)
        except ApiError as exc:
            if exc.status != 403:
                raise
        else:
            raise AcceptanceError(f"{role} unexpectedly created a forbidden task")
    run.report["identities"] = records
    for record in records.values():
        run.client.request("DELETE", f"/api/users/{record['id']}")


def build_playroom_archive(source: Path, build_id: str) -> tuple[Path, int, str]:
    images = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if len(images) < 225:
        raise AcceptanceError(f"Playroom source has {len(images)} images; at least 225 are required")
    target = Path(tempfile.gettempdir()) / f"real2sim-playroom-{build_id}.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for image in images:
            archive.write(image, Path("playroom") / image.relative_to(source))
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return target, len(images), digest.hexdigest()


def upload_with_resume(run: AcceptanceRun, archive: Path, checksum: str) -> dict[str, Any]:
    chunk_size = 8 * 1024 * 1024
    created = run.client.post(
        "/api/uploads",
        {
            "filename": archive.name,
            "content_type": "application/zip",
            "total_size": archive.stat().st_size,
            "chunk_size": chunk_size,
            "checksum": checksum,
        },
    )
    upload_id = created["id"]
    total_parts = int(created["total_parts"])
    pause_after = max(1, total_parts // 2)
    run.report["storage"] = {
        **run.report.get("storage", {}),
        "upload_id": upload_id,
        "checksum": checksum,
        "size_bytes": archive.stat().st_size,
        "total_parts": total_parts,
        "resume_checkpoint_part": pause_after,
        "status": "uploading",
    }
    run.save()
    with archive.open("rb") as handle:
        for part in range(1, total_parts + 1):
            body = handle.read(chunk_size)
            run.client.request(
                "PUT",
                f"/api/uploads/{upload_id}/parts/{part}",
                body=body,
                content_type="application/octet-stream",
                timeout=600,
            )
            print(f"  uploaded part {part}/{total_parts}", flush=True)
            if part == pause_after:
                resumed = run.client.get(f"/api/uploads/{upload_id}")
                expected = list(range(1, pause_after + 1))
                run.check("BOS 上传分片状态可恢复", resumed.get("received_parts") == expected, resumed.get("received_parts"))
    completed = run.client.post(f"/api/uploads/{upload_id}/complete")
    dataset = completed["dataset"]
    run.check("上传数据写入受控 BOS 前缀", str(dataset.get("storage_uri") or "").startswith("bos://testxyz/real2sim-poc/uploads/"), dataset.get("storage_uri"))
    run.report["storage"] = {
        **run.report.get("storage", {}),
        "upload_id": upload_id,
        "storage_uri": dataset.get("storage_uri"),
        "checksum": checksum,
        "size_bytes": archive.stat().st_size,
        "total_parts": total_parts,
        "resume_checkpoint_part": pause_after,
        "status": "completed",
    }
    run.report["datasets"].append({"id": dataset["id"], "role": "playroom_upload", "storage_uri": dataset.get("storage_uri")})
    return dataset


def validate_infrastructure(run: AcceptanceRun) -> None:
    run.report["version"] = run.client.get("/api/version")
    run.report["doctor"] = run.client.get("/api/doctor")
    run.report["capabilities"] = run.client.get("/api/capabilities")
    run.report["recipes"] = run.client.get("/api/recipes")
    run.check("部署 build_id 与验收目标一致", run.report["version"].get("build_id") == run.build_id, run.report["version"].get("build_id"))
    run.check("生产 API doctor 通过", bool(run.report["doctor"].get("ok")), run.report["doctor"])
    run.check("A800 worker 心跳就绪", bool((run.report["doctor"].get("worker") or {}).get("ready")), run.report["doctor"].get("worker"))
    runtimes = run.report["capabilities"].get("runtimes") or {}
    for name in ("anysplat", "mujoco", "motrixsim", "discoverse_act", "rm65_controller"):
        run.check(f"真实运行时就绪: {name}", bool((runtimes.get(name) or {}).get("ready")), runtimes.get(name))
    bos = (run.report["capabilities"].get("storage") or {}).get("bos") or {}
    run.check("BOS 前缀隔离", bos.get("prefix") == "real2sim-poc", bos)


def exercise_task_operations(run: AcceptanceRun, scene_id: str | None) -> None:
    benchmark = run.create_task("renderer_benchmark", {})
    retry = run.client.post(f"/api/tasks/{benchmark['id']}/retry")
    retried = run.wait_task(retry["id"])
    bulk = run.client.post("/api/tasks/bulk", {"action": "retry", "ids": [benchmark["id"], retried["id"]], "params": {}})
    run.check("任务批量重试逐项返回", len(bulk.get("results") or []) == 2 and all(item.get("ok") for item in bulk["results"]), bulk)
    for item in bulk["results"]:
        run.wait_task(item["result"]["id"], timeout_seconds=300)

    if not scene_id:
        run.report["warnings"].append("RM65 scene unavailable; cancellation and restart-recovery probes were skipped")
        return
    cancel_probe = run.client.post(
        "/api/tasks",
        {
            "type": "expert_collection",
            "params": {
                "scene_id": scene_id,
                "policy_preset": "rm65_pick_place_6d_ik",
                "episodes": 50,
                "max_attempts": 150,
                "steps": 240,
                "name": "acceptance-cancel-probe",
            },
        },
    )
    wait_until_status(run.client, cancel_probe["id"], "running", 300)
    run.client.post(f"/api/tasks/{cancel_probe['id']}/cancel")
    run.wait_task(cancel_probe["id"], expected_status="canceled", timeout_seconds=300)

    recovery_probe = run.client.post(
        "/api/tasks",
        {
            "type": "expert_collection",
            "params": {
                "scene_id": scene_id,
                "policy_preset": "rm65_pick_place_6d_ik",
                "episodes": 50,
                "max_attempts": 150,
                "steps": 240,
                "name": "acceptance-recovery-probe",
            },
        },
    )
    wait_until_status(run.client, recovery_probe["id"], "running", 300)
    kubectl(run.kubeconfig, "-n", NAMESPACE, "rollout", "restart", "deployment/real2sim-worker")
    kubectl(run.kubeconfig, "-n", NAMESPACE, "rollout", "status", "deployment/real2sim-worker", "--timeout=3600s", timeout=3700)
    run.wait_task(recovery_probe["id"], expected_status="failed_recovered", timeout_seconds=600)
    run.check("worker 重启后历史任务仍可查询", run.client.get(f"/api/tasks/{benchmark['id']}").get("id") == benchmark["id"])


def wait_until_status(client: ApiClient, task_id: str, status: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        task = client.get(f"/api/tasks/{task_id}")
        if task.get("status") == status:
            return task
        if task.get("status") in TERMINAL:
            raise AcceptanceError(f"task {task_id} reached {task.get('status')} before {status}")
        time.sleep(2)
    raise AcceptanceError(f"task {task_id} did not reach {status} within {timeout_seconds}s")


def resume_reconstruction_evidence(run: AcceptanceRun, report_path: Path) -> tuple[str, list[str]]:
    if not report_path.is_file():
        raise AcceptanceError(f"resume report not found: {report_path}")
    previous = json.loads(report_path.read_text(encoding="utf-8"))
    if previous.get("schema") != "real2sim.poc_acceptance_report.v1":
        raise AcceptanceError(f"unsupported resume report schema: {previous.get('schema')}")
    source_build_id = str(previous.get("build_id") or "")
    if not source_build_id:
        raise AcceptanceError("resume report does not identify its source build")

    prepared = next((item for item in previous.get("datasets", []) if item.get("role") == "playroom_prepared"), None)
    scenes = [item for item in previous.get("scenes", []) if str(item.get("role") or "").startswith("playroom_reconstruction_")]
    tasks = list(previous.get("tasks", []))
    by_type: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        by_type.setdefault(str(task.get("type") or ""), []).append(task)
    if not prepared or len(scenes) != 2:
        raise AcceptanceError("resume report does not contain one prepared Playroom dataset and two reconstruction scenes")
    required_counts = {"media_prepare": 1, "reconstruction_preflight": 1, "scene_reconstruction": 2, "simulator_validation": 2}
    for task_type, count in required_counts.items():
        if len(by_type.get(task_type, [])) < count:
            raise AcceptanceError(f"resume report has only {len(by_type.get(task_type, []))} {task_type} task(s); expected {count}")

    artifact_requirements = {
        "media_prepare": ("prepared_manifest.json",),
        "reconstruction_preflight": ("capture_preflight_report.json",),
        "scene_reconstruction": (
            "environment.json",
            "reconstruction_command.json",
            "capture_preflight_report.json",
            "reconstruction_report.json",
            "scene_manifest.json",
        ),
        "simulator_validation": ("validation_report.json",),
    }
    for task_type, count in required_counts.items():
        for task in by_type[task_type][:count]:
            run.wait_task(str(task["id"]), required_artifacts=artifact_requirements[task_type])

    prepared_id = str(prepared["id"])
    prepared_manifest = run.client.get(f"/api/datasets/{prepared_id}/manifest")
    run.check("恢复数据集仍有至少 80 帧", int(prepared_manifest.get("frame_count") or 0) >= 80, prepared_manifest.get("frame_count"))
    scene_ids = [str(item["id"]) for item in scenes]
    for scene_id in scene_ids:
        manifest = run.client.get(f"/api/scenes/{scene_id}/manifest")
        names = {str(asset.get("name") or "") for asset in manifest.get("assets", []) if isinstance(asset, dict)}
        run.check(f"恢复场景 {scene_id[:8]} 仍有 3D 与仿真资产", bool({"scene.ply", "pose_init_trained_scene.ply"} & names) and "scene.xml" in names, sorted(names)[:20])
    run.check("恢复的两次重建场景相互独立", len(set(scene_ids)) == 2, scene_ids)
    run.report["datasets"] = [item for item in previous.get("datasets", []) if item.get("role") in {"playroom_upload", "playroom_prepared"}]
    run.report["scenes"] = scenes
    run.report["storage"] = dict(previous.get("storage") or {})
    run.report["resumed_evidence"] = {
        "report": str(report_path.resolve()),
        "source_build_id": source_build_id,
        "target_build_id": run.build_id,
        "cross_build": source_build_id != run.build_id,
        "reaudited_task_ids": [
            str(task["id"])
            for task_type, count in required_counts.items()
            for task in by_type[task_type][:count]
        ],
        "prepared_dataset_id": prepared_id,
        "scene_ids": scene_ids,
    }
    run.save()
    return prepared_id, scene_ids


def run_acceptance(args: argparse.Namespace) -> int:
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
    out_dir = args.out or ROOT / "acceptance_runs" / "poc" / build_id
    out_dir.mkdir(parents=True, exist_ok=True)
    port = args.port or free_port()
    forward = start_port_forward(args.kubeconfig, port, out_dir)
    archive: Path | None = None
    run = AcceptanceRun(
        client=ApiClient(f"https://127.0.0.1:{port}"),
        out_dir=out_dir,
        kubeconfig=args.kubeconfig,
        build_id=build_id,
        timeout_seconds=args.timeout_seconds,
    )
    try:
        if run.stage("管理员认证", lambda: authenticate_admin(run)) is None:
            return finalize(run)
        run.stage("运行时、存储与版本预检", lambda: validate_infrastructure(run))
        run.stage("用户 CRUD 与角色隔离", lambda: verify_rbac(run))

        reconstruction_scenes: list[str] = []
        prepared_dataset_id: str | None = None
        if args.resume_report:
            resumed = run.stage("恢复并复核已通过的 Playroom 重建证据", lambda: resume_reconstruction_evidence(run, args.resume_report))
            if resumed:
                prepared_dataset_id, reconstruction_scenes = resumed
        else:
            uploaded_dataset: dict[str, Any] | None = None
            playroom = run.stage("打包真实 Playroom 225 帧", lambda: build_playroom_archive(args.playroom, build_id))
            if playroom:
                archive, image_count, checksum = playroom
                run.report["storage"]["source_image_count"] = image_count
                uploaded_dataset = run.stage("BOS 分片上传与断点续传", lambda: upload_with_resume(run, archive, checksum))
            if uploaded_dataset:
                prepared = run.stage(
                    "媒体准备 225 帧",
                    lambda: run.create_task(
                        "media_prepare",
                        {
                            "dataset_id": uploaded_dataset["id"],
                            "camera_label": "playroom.rgb",
                            "target_frame_count": 225,
                            "min_total_frames": 80,
                            "max_frames_per_video": 225,
                            "max_videos": 1,
                            "sample_fps": 2,
                            "output_name": f"playroom-prepared-{build_id}",
                        },
                        required_artifacts=("prepared_manifest.json",),
                    ),
                )
                if prepared:
                    prepared_dataset_id = str(prepared["result"]["dataset_id"])
                    run.report["datasets"].append({"id": prepared_dataset_id, "role": "playroom_prepared", "frame_count": prepared["result"].get("frame_count")})
                    run.stage(
                        "Playroom 重建质量预检",
                        lambda: run.create_task(
                            "reconstruction_preflight",
                            {"dataset_id": prepared_dataset_id, "min_frames": 80, "target_frame_count": 225, "allow_low_quality": False, "quality_preset": "capture-high"},
                            required_artifacts=("capture_preflight_report.json",),
                        ),
                    )

            if prepared_dataset_id:
                for index in range(1, 3):
                    reconstructed = run.stage(
                        f"Playroom AnySplat 真实重建 #{index}",
                        lambda index=index: run.create_task(
                            "scene_reconstruction",
                            {
                                "dataset_id": prepared_dataset_id,
                                "scene_name": f"Playroom A800 Reconstruction {index}",
                                "renderer": "spark",
                                "simulators": ["mujoco", "motrixsim"],
                                "dry_run": False,
                                "preflight_required": True,
                                "min_frames": 80,
                                "allow_low_quality": False,
                                "quality_preset": "capture-high",
                                "pose_init_train": True,
                                "sfm_matcher": "exhaustive",
                                "sfm_guided_matching": True,
                                "sfm_frames": "selected",
                                "anysplat_keyframe_strategy": "sfm-registered",
                                "auto_cover": True,
                            },
                            required_artifacts=(
                                "environment.json",
                                "reconstruction_command.json",
                                "capture_preflight_report.json",
                                "reconstruction_report.json",
                                "scene_manifest.json",
                            ),
                        ),
                    )
                    if reconstructed:
                        scene_id = str(reconstructed["result"]["scene_id"])
                        reconstruction_scenes.append(scene_id)
                        run.report["scenes"].append({"id": scene_id, "role": f"playroom_reconstruction_{index}"})
                        run.stage(
                            f"Playroom 重建仿真双运行时校验 #{index}",
                            lambda scene_id=scene_id: run.create_task(
                                "simulator_validation",
                                {"scene_id": scene_id, "simulators": ["mujoco", "motrixsim"], "level": "runtime_required"},
                                required_artifacts=("validation_report.json",),
                            ),
                        )
        if len(reconstruction_scenes) == 2:
            run.stage(
                "两次重建产生独立场景包",
                lambda: run.check("重建 scene_id 独立", len(set(reconstruction_scenes)) == 2, reconstruction_scenes),
            )

        motrix_task = run.stage(
            "MotrixSim 官方客厅与 RM65 资产导入",
            lambda: run.create_task(
                "motrixsim_asset_import",
                {"source": "/opt/motrixsim-docs", "validation_steps": 120, "max_parlour_meshes": 240},
                required_artifacts=("motrixsim_asset_import_report.json", "rm65_table_pick_place_validation.json", "motrixsim_parlour_rm65_preview.glb"),
            ),
        )
        rm65_scene_id = str(motrix_task["result"]["scene_id"]) if motrix_task else None
        if rm65_scene_id:
            run.report["scenes"].append({"id": rm65_scene_id, "role": "rm65_pick_place"})
            run.stage(
                "RM65 场景 runtime_required 校验",
                lambda: run.create_task(
                    "simulator_validation",
                    {"scene_id": rm65_scene_id, "simulators": ["mujoco", "motrixsim"], "level": "runtime_required"},
                    required_artifacts=("validation_report.json",),
                ),
            )

        collection = None
        if rm65_scene_id:
            collection = run.stage(
                "RM65 真实物理专家采集 50 条成功 episode",
                lambda: run.create_task(
                    "expert_collection",
                    {
                        "scene_id": rm65_scene_id,
                        "policy_preset": "rm65_pick_place_6d_ik",
                        "episodes": 50,
                        "max_attempts": 150,
                        "steps": 240,
                        "workers": 1,
                        "seed": 7,
                        "image_width": 320,
                        "image_height": 240,
                        "name": f"RM65 Physical ACT Dataset {build_id}",
                    },
                    required_artifacts=("manifest.json", "episode_0.hdf5", "episode_0.mp4"),
                ),
            )
        collection_dataset_id = str(collection["result"]["dataset_id"]) if collection else None
        if collection_dataset_id:
            run.report["datasets"].append({"id": collection_dataset_id, "role": "rm65_physical_collection", "samples": collection["result"].get("samples")})
            run.stage("采集成功数门槛", lambda: run.check("成功 episode 至少 50", int(collection["result"].get("samples") or 0) >= 50, collection["result"]))

        trained = None
        if collection_dataset_id:
            trained = run.stage(
                "DISCOVERSE ACT 真实训练",
                lambda: run.create_task(
                    "training",
                    {
                        "dataset_id": collection_dataset_id,
                        "recipe": "discoverse_act",
                        "epochs": args.act_epochs,
                        "batch_size": 8,
                        "chunk_size": 32,
                        "seed": 1,
                        "name": f"RM65 ACT {build_id}",
                    },
                    required_artifacts=("policy_best.ckpt", "dataset_stats.pkl", "model_card.json", "task_config.py"),
                ),
            )
        model_id = str(trained["result"]["model_id"]) if trained else None
        if model_id:
            run.report["models"].append({"id": model_id, "role": "discoverse_act", "metrics": trained["result"].get("metrics")})

        if model_id and rm65_scene_id:
            inference = run.stage(
                "DISCOVERSE ACT 在线推理 10 回合",
                lambda: run.create_task(
                    "inference",
                    {
                        "model_id": model_id,
                        "scene_id": rm65_scene_id,
                        "recipe": "discoverse_act",
                        "episodes": 10,
                        "steps": 240,
                        "workers": 1,
                        "seed": 11,
                        "minimum_success_rate": 0.5,
                    },
                    required_artifacts=("inference_report.json", "episode_000.mp4", "episode_000_trajectory.json"),
                ),
            )
            if inference:
                run.stage("ACT 推理成功率门槛", lambda: run.check("推理成功率至少 50%", float(inference["result"].get("success_rate") or 0) >= 0.5, inference["result"]))

        run.stage("任务取消、重试、批量操作与重启恢复", lambda: exercise_task_operations(run, rm65_scene_id))
        return finalize(run)
    finally:
        if archive and archive.exists():
            archive.unlink()
        forward.terminate()
        try:
            forward.wait(timeout=10)
        except subprocess.TimeoutExpired:
            forward.kill()


def finalize(run: AcceptanceRun) -> int:
    run.report["finished_at"] = utc_now()
    run.report["pass"] = not run.report["failures"] and all(item.get("ok") for item in run.report["checks"])
    run.save()
    print(f"\nacceptance_report={run.out_dir / 'acceptance_report.json'}", flush=True)
    print(f"pass={run.report['pass']}", flush=True)
    return 0 if run.report["pass"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real Real2Sim production acceptance against the isolated POC namespace.")
    parser.add_argument("--kubeconfig", type=Path, default=Path(os.environ.get("REAL2SIM_POC_KUBECONFIG", DEFAULT_KUBECONFIG)))
    parser.add_argument("--playroom", type=Path, default=Path(os.environ.get("REAL2SIM_POC_PLAYROOM", DEFAULT_PLAYROOM)))
    parser.add_argument("--build-id", default=os.environ.get("REAL2SIM_POC_BUILD_ID", ""))
    parser.add_argument("--out", type=Path, default=Path(os.environ["REAL2SIM_POC_ACCEPTANCE_OUT"]) if os.environ.get("REAL2SIM_POC_ACCEPTANCE_OUT") else None)
    parser.add_argument("--resume-report", type=Path, default=Path(os.environ["REAL2SIM_POC_RESUME_REPORT"]) if os.environ.get("REAL2SIM_POC_RESUME_REPORT") else None)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=int(os.environ.get("REAL2SIM_POC_ACCEPTANCE_TIMEOUT_SECONDS", "28800")))
    parser.add_argument("--act-epochs", type=int, default=int(os.environ.get("REAL2SIM_POC_ACT_EPOCHS", "500")))
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run_acceptance(parse_args()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"acceptance bootstrap failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
