from __future__ import annotations

import concurrent.futures
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from real2sim_platform.config import PlatformConfig
from real2sim_platform.adapters import index_artifacts_for_task
from real2sim_platform.rm65_controller import _write_preview_video, collect
from real2sim_platform.runtime_gate import _motrixsim_assets_probe, runtime_key
from real2sim_platform.server import create_app
from real2sim_platform.storage import BosStorage, StorageBoundaryError, normalize_relative_path
from real2sim_platform.store import PlatformStore


ROOT = Path(__file__).resolve().parents[1]


def test_auth_first_login_rbac_and_user_crud(tmp_path: Path) -> None:
    config = PlatformConfig(
        app_root=ROOT,
        workspace_dir=tmp_path / "workspace",
        worker_mode="api",
        auth_enabled=True,
        bootstrap_admin_password="Admin!Password123",
    )
    with TestClient(create_app(config)) as client:
        assert client.get("/api/datasets").status_code == 401
        login = client.post("/api/auth/login", json={"username": "admin", "password": "Admin!Password123"})
        assert login.status_code == 200
        assert login.json()["user"]["must_change_password"] is True
        assert client.get("/api/datasets").status_code == 428
        changed = client.post(
            "/api/auth/change-password",
            json={"current_password": "Admin!Password123", "new_password": "Changed!Password456"},
        )
        assert changed.status_code == 200
        assert client.get("/api/datasets").status_code == 401
        assert client.post("/api/auth/login", json={"username": "admin", "password": "Changed!Password456"}).status_code == 200
        viewer = client.post(
            "/api/users",
            json={"username": "viewer.user", "password": "Viewer!Password123", "role": "viewer"},
        )
        assert viewer.status_code == 200
        assert viewer.json()["role"] == "viewer"
        assert client.patch(f"/api/users/{viewer.json()['id']}", json={"disabled": True}).json()["disabled"] is True

    viewer_config = PlatformConfig(
        app_root=ROOT,
        workspace_dir=config.workspace_dir,
        worker_mode="api",
        auth_enabled=True,
        bootstrap_admin_password="unused-Password!123",
    )
    store = PlatformStore(viewer_config.db_path)
    viewer_id = next(item["id"] for item in store.list_users() if item["username"] == "viewer.user")
    store.update_user(viewer_id, disabled=False)
    with TestClient(create_app(viewer_config)) as viewer_client:
        assert viewer_client.post("/api/auth/login", json={"username": "viewer.user", "password": "Viewer!Password123"}).status_code == 200
        assert viewer_client.get("/api/auth/me").status_code == 200
        assert viewer_client.post("/api/auth/change-password", json={"current_password": "Viewer!Password123", "new_password": "Viewer!Changed456"}).status_code == 200
        assert viewer_client.post("/api/auth/login", json={"username": "viewer.user", "password": "Viewer!Changed456"}).status_code == 200
        assert viewer_client.get("/api/datasets").status_code == 200
        assert viewer_client.post("/api/tasks", json={"type": "renderer_benchmark", "params": {}}).status_code == 403


def test_bos_prefix_boundary_is_enforced(tmp_path: Path) -> None:
    config = PlatformConfig(
        workspace_dir=tmp_path,
        bos_bucket="testxyz",
        bos_prefix="real2sim-poc",
        bos_access_key_id="configured-but-not-used",
        bos_secret_access_key="configured-but-not-used",
    )
    storage = BosStorage(config)
    assert storage.object_key("datasets", "team/a.json") == "real2sim-poc/datasets/team/a.json"
    assert storage.key_from_uri("bos://testxyz/real2sim-poc/models/m1/card.json") == "real2sim-poc/models/m1/card.json"
    with pytest.raises(StorageBoundaryError):
        storage.object_key("datasets", "../../droboticflow/secret")
    with pytest.raises(StorageBoundaryError):
        storage.assert_owned_key("droboticflow/workspaces/data")
    with pytest.raises(StorageBoundaryError):
        storage.key_from_uri("bos://another-bucket/real2sim-poc/datasets/a")
    with pytest.raises(StorageBoundaryError):
        normalize_relative_path("../outside")


def test_atomic_task_claim_has_no_duplicates(tmp_path: Path) -> None:
    store = PlatformStore(tmp_path / "state.db")
    for index in range(24):
        store.create_task(task_type="renderer_benchmark", params={"index": index}, run_dir=tmp_path / f"run-{index}")

    def claim(worker: int):
        task = store.claim_next_queued_task(f"worker-{worker}")
        return task["id"] if task else None

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        claimed = [item for item in executor.map(claim, range(24)) if item]
    assert len(claimed) == 24
    assert len(set(claimed)) == 24
    assert all(task["status"] == "running" and task["attempt"] == 1 for task in store.list_tasks())


def test_api_capabilities_follow_live_worker_heartbeat(tmp_path: Path) -> None:
    config = PlatformConfig(app_root=ROOT, workspace_dir=tmp_path / "workspace", worker_mode="api")
    store = PlatformStore(config.db_path)
    store.update_worker_heartbeat(
        "worker-a800",
        {
            "ready": True,
            "checks": {
                "ffmpeg": True,
                "colmap": True,
                "anysplat_root": True,
                "anysplat_python": True,
                "anysplat_pinned": True,
                "mujoco_python": True,
                "motrixsim_python": True,
                "discoverse_act": True,
                "torch_cuda": True,
                "act_import": True,
                "rm65_controller": True,
            },
        },
    )
    with TestClient(create_app(config)) as client:
        capabilities = client.get("/api/capabilities").json()
        assert capabilities["worker"]["connected"] is True
        assert capabilities["runtimes"]["anysplat"]["ready"] is True
        assert capabilities["runtimes"]["discoverse_act"]["ready"] is True
        assert capabilities["runtimes"]["rm65_controller"]["ready"] is True


def test_login_rate_limit_blocks_repeated_failures(tmp_path: Path) -> None:
    config = PlatformConfig(
        app_root=ROOT,
        workspace_dir=tmp_path / "workspace",
        worker_mode="api",
        auth_enabled=True,
        bootstrap_admin_password="Admin!Password123",
    )
    with TestClient(create_app(config)) as client:
        for _ in range(8):
            assert client.post("/api/auth/login", json={"username": "admin", "password": "wrong-password"}).status_code == 401
        assert client.post("/api/auth/login", json={"username": "admin", "password": "wrong-password"}).status_code == 429


def test_production_task_gates_reject_fake_routes(tmp_path: Path) -> None:
    config = PlatformConfig(app_root=ROOT, workspace_dir=tmp_path / "workspace", worker_mode="api", production_mode=True)
    client = TestClient(create_app(config))
    dry_run = client.post(
        "/api/tasks",
        json={"type": "scene_reconstruction", "params": {"input_path": str(tmp_path), "dry_run": True}},
    )
    assert dry_run.status_code == 400
    legacy = client.post(
        "/api/tasks",
        json={"type": "training", "params": {"dataset_path": str(tmp_path), "recipe": "bc_v2"}},
    )
    assert legacy.status_code == 400
    fake_expert = client.post(
        "/api/tasks",
        json={"type": "expert_collection", "params": {"scene_path": str(tmp_path), "policy_preset": "pick_and_place_v1"}},
    )
    assert fake_expert.status_code == 400


def test_motrixsim_asset_import_is_admin_only_and_validated(tmp_path: Path) -> None:
    config = PlatformConfig(
        app_root=ROOT,
        workspace_dir=tmp_path / "workspace",
        worker_mode="api",
        production_mode=True,
        auth_enabled=True,
        bootstrap_admin_password="Admin!Password123",
    )
    with TestClient(create_app(config)) as client:
        assert client.post("/api/auth/login", json={"username": "admin", "password": "Admin!Password123"}).status_code == 200
        assert client.post(
            "/api/auth/change-password",
            json={"current_password": "Admin!Password123", "new_password": "Changed!Password456"},
        ).status_code == 200
        assert client.post("/api/auth/login", json={"username": "admin", "password": "Changed!Password456"}).status_code == 200
        assert client.post(
            "/api/users",
            json={"username": "asset.engineer", "password": "Engineer!Password123", "role": "engineer"},
        ).status_code == 200
        invalid = client.post(
            "/api/tasks",
            json={"type": "motrixsim_asset_import", "params": {"validation_steps": 0}},
        )
        assert invalid.status_code == 400
        queued = client.post(
            "/api/tasks",
            json={"type": "motrixsim_asset_import", "params": {"validation_steps": 120, "max_parlour_meshes": 240}},
        )
        assert queued.status_code == 200
        assert queued.json()["type"] == "motrixsim_asset_import"
        assert client.post("/api/auth/logout").status_code == 200
        assert client.post("/api/auth/login", json={"username": "asset.engineer", "password": "Engineer!Password123"}).status_code == 200
        assert client.post(
            "/api/auth/change-password",
            json={"current_password": "Engineer!Password123", "new_password": "Engineer!Changed456"},
        ).status_code == 200
        assert client.post("/api/auth/login", json={"username": "asset.engineer", "password": "Engineer!Changed456"}).status_code == 200
        forbidden = client.post("/api/tasks", json={"type": "motrixsim_asset_import", "params": {}})
        assert forbidden.status_code == 403


@pytest.mark.integration
def test_runtime_required_validation_steps_mujoco_and_motrixsim(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("motrixsim")
    package = tmp_path / "runtime-scene"
    simulator_assets = package / "simulator_assets"
    simulator_assets.mkdir(parents=True)
    (package / "scene.json").write_text('{"schema":"motrixsim.scene.v1"}\n', encoding="utf-8")
    (simulator_assets / "scene.xml").write_text(
        '<mujoco model="runtime-validation"><option timestep="0.002"/><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .2"><freejoint/><geom type="sphere" size=".03" mass=".1"/></body></worldbody></mujoco>\n',
        encoding="utf-8",
    )
    config = PlatformConfig(app_root=ROOT, workspace_dir=tmp_path / "workspace", allowed_path_roots=(tmp_path,))
    with TestClient(create_app(config)) as client:
        created = client.post(
            "/api/tasks",
            json={
                "type": "simulator_validation",
                "params": {"scene_path": str(package), "simulators": ["mujoco", "motrixsim"], "level": "runtime_required"},
            },
        ).json()
        for _ in range(80):
            task = client.get(f"/api/tasks/{created['id']}").json()
            if task["status"] in {"succeeded", "failed"}:
                break
            __import__("time").sleep(0.1)
    assert task["status"] == "succeeded", task.get("error")
    simulators = task["result"]["report"]["simulators"]
    assert simulators["mujoco"]["runtime"]["status"] == "loaded"
    assert simulators["motrixsim"]["runtime"]["status"] == "loaded_and_stepped"
    assert simulators["motrixsim"]["runtime"]["steps"] == 8


def test_production_artifact_index_includes_viewer_dataset_and_model_outputs(tmp_path: Path) -> None:
    store = PlatformStore(tmp_path / "state.db")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    task = store.create_task(task_type="training", params={}, run_dir=run_dir)
    for filename in ("scene.glb", "episode_0.hdf5", "policy_best.ckpt", "dataset_stats.pkl"):
        (run_dir / filename).write_bytes(b"real-production-artifact")
    index_artifacts_for_task(store, task["id"], run_dir)
    names = {item["name"] for item in store.list_task_artifacts(task["id"])}
    assert names == {"scene.glb", "episode_0.hdf5", "policy_best.ckpt", "dataset_stats.pkl"}


def test_poc_frontend_proxies_authenticated_run_artifacts() -> None:
    nginx = (ROOT / "deploy/poc/nginx.conf").read_text(encoding="utf-8")
    runs_location = nginx.split("location /runs/ {", 1)[1].split("}", 1)[0]
    assert "proxy_pass http://real2sim_api;" in runs_location
    assert "proxy_buffering off;" in runs_location
    assert "proxy_read_timeout 7200s;" in runs_location
    deploy = (ROOT / "scripts/deploy_poc_cluster.sh").read_text(encoding="utf-8")
    assert "--frontend-only" in deploy
    assert "--worker-only" in deploy
    assert 'set image deployment/real2sim-frontend "frontend=$FRONTEND_IMAGE"' in deploy
    assert 'set image deployment/real2sim-worker "worker=$WORKER_IMAGE" "runtime-gate=$WORKER_IMAGE"' in deploy


def test_worker_image_materializes_motrixsim_lfs_assets() -> None:
    dockerfile = (ROOT / "docker/worker.Dockerfile").read_text(encoding="utf-8")
    assert "git-lfs" in dockerfile
    assert "git -C /opt/motrixsim-docs lfs pull --include='examples/assets/**'" in dockerfile
    assert "git-lfs.github.com/spec" in dockerfile
    manifest = (ROOT / "deploy/poc/app-template.yaml").read_text(encoding="utf-8")
    assert "REAL2SIM_MOTRIXSIM_PYTHON" in manifest
    assert "REAL2SIM_FFMPEG_BIN" in manifest
    assert "NVIDIA_DRIVER_CAPABILITIES" in manifest
    assert "compute,utility,graphics" in manifest
    assert "GIT_CONFIG_VALUE_3, value: /opt/motrixsim-docs" in manifest


def test_rollout_video_encoder_reports_the_selected_ffmpeg_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REAL2SIM_FFMPEG_BIN", "/usr/bin/false")
    with pytest.raises(RuntimeError, match="ffmpeg rollout encoding failed with /usr/bin/false"):
        _write_preview_video(tmp_path / "rollout.mp4", [np.zeros((16, 16, 3), dtype=np.uint8)])


def test_runtime_gate_rejects_unresolved_motrixsim_lfs_pointer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "motrixsim-docs"
    required = [
        root / "examples/assets/parlour/parlour.xml",
        root / "examples/assets/rm65_open_cabinet/assets/rm65/meshes/base_link.STL",
        root / "examples/assets/franka_emika_panda/scene_pick_cube.xml",
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<mujoco/>\n", encoding="utf-8")
    required[1].write_text("version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\nsize 10000\n", encoding="utf-8")
    monkeypatch.setenv("REAL2SIM_MOTRIXSIM_DOCS_ROOT", str(root))
    with pytest.raises(RuntimeError, match="Git LFS assets are unresolved"):
        _motrixsim_assets_probe()


def test_runtime_key_changes_with_deployed_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = PlatformConfig(app_root=ROOT, workspace_dir=tmp_path)
    monkeypatch.setenv("REAL2SIM_PLATFORM_BUILD_ID", "build-a")
    first = runtime_key(config)
    monkeypatch.setenv("REAL2SIM_PLATFORM_BUILD_ID", "build-b")
    assert runtime_key(config) != first


def test_motrixsim_runtime_scene_uses_current_python_when_module_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    script_path = ROOT / "scripts/generate_motrixsim_runtime_scene.py"
    spec = importlib.util.spec_from_file_location("real2sim_generate_motrixsim_runtime_scene", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("REAL2SIM_MOTRIXSIM_PYTHON", raising=False)
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object() if name == "motrixsim" else None)
    assert module.find_motrixsim_python("") == Path(sys.executable)


@pytest.mark.integration
def test_real_rm65_physical_collection_writes_act_hdf5(tmp_path: Path) -> None:
    scene = ROOT / "workspace/motrixsim_assets/parlour_rm65_pick_place/simulator_assets/rm65_table_pick_place.xml"
    if not scene.is_file():
        pytest.skip("real imported RM65 scene asset is not present")
    manifest = collect(scene, tmp_path / "collection", required_successes=1, max_attempts=2, max_steps=240, seed=7, width=160, height=120)
    assert manifest["successful_episodes"] == 1
    assert manifest["episodes"][0]["lift_height"] >= 0.10
    assert manifest["episodes"][0]["place_error"] <= 0.05
    episode = tmp_path / "collection/episode_0.hdf5"
    with h5py.File(episode, "r") as root:
        assert root["/observations/qpos"].shape[1] == 7
        assert root["/action"].shape[1] == 7
        assert root["/observations/images/overview"].shape[0] > 0
        assert root["/observations/contact"].shape[1] == 2
        assert bool(root.attrs["success"]) is True
    assert (tmp_path / "collection/episode_0.mp4").stat().st_size > 0
