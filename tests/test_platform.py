from __future__ import annotations

import shutil
import struct
import subprocess
import time
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from real2sim_platform.adapters import _export_3dgs_rgb_fallback, collect_task_reports
from real2sim_platform.config import PlatformConfig
from real2sim_platform import motrix_scene
from real2sim_platform.server import create_app
from real2sim_platform.store import PlatformStore


ROOT = Path(__file__).resolve().parents[2]


def test_health_and_doctor(tmp_path: Path) -> None:
    cfg = PlatformConfig(app_root=ROOT, workspace_dir=tmp_path / "workspace")
    client = TestClient(create_app(cfg))
    assert client.get("/api/health").json()["ok"] is True
    version = client.get("/api/version").json()
    assert version["features"]["media_prepare"] is True
    assert version["features"]["reconstruction_preflight"] is True
    assert "runtime_paths" in version
    assert "config_digest" in version
    doctor = client.get("/api/doctor").json()
    assert "checks" in doctor
    assert "media_prepare_deps" in doctor["checks"]
    assert "anysplat_root" in doctor["checks"]
    assert "three_bundle" in doctor["checks"]


def test_dry_run_scene_task(tmp_path: Path) -> None:
    cfg = PlatformConfig(app_root=ROOT, workspace_dir=tmp_path / "workspace")
    with TestClient(create_app(cfg)) as client:
        response = client.post(
            "/api/tasks",
            json={
                "type": "scene_reconstruction",
                "params": {
                    "input_path": str(ROOT / "real2sim-expert-policy/examples"),
                    "dry_run": True,
                    "renderer": "spark",
                    "simulators": ["mujoco", "motrixsim"],
                },
            },
        )
        task = response.json()
        for _ in range(40):
            current = client.get(f"/api/tasks/{task['id']}").json()
            if current["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.1)
        assert current["status"] == "succeeded"
        artifacts = client.get(f"/api/tasks/{task['id']}/artifacts").json()
        assert any(item["name"] == "scene.json" for item in artifacts)
        scene_id = current["result"]["scene_id"]
        manifest = client.get(f"/api/scenes/{scene_id}/manifest").json()
        assert manifest["schema"] == "real2sim.scene_manifest.v1"
        assert manifest["renderer"] == "spark"
        assert all("size_bytes" in item and "autoload" in item for item in manifest["assets"])
        scene_artifact = next(item for item in artifacts if item["name"] == "scene.json")
        assert client.get(scene_artifact["url"]).status_code == 200


def test_task_payload_validation(tmp_path: Path) -> None:
    cfg = PlatformConfig(app_root=ROOT, workspace_dir=tmp_path / "workspace")
    client = TestClient(create_app(cfg))
    response = client.post("/api/tasks", json={"type": "hf_import", "params": {"repo_id": "owner/name", "repo_type": "model"}})
    assert response.status_code == 400
    response = client.post("/api/tasks", json={"type": "media_prepare", "params": {}})
    assert response.status_code == 400
    response = client.post("/api/tasks", json={"type": "reconstruction_preflight", "params": {}})
    assert response.status_code == 400
    response = client.post("/api/tasks", json={"type": "scene_reconstruction", "params": {"input_path": str(ROOT), "renderer": "bad"}})
    assert response.status_code == 400
    response = client.post("/api/tasks", json={"type": "simulator_validation", "params": {"scene_path": str(ROOT), "level": "bad"}})
    assert response.status_code == 400
    response = client.post(
        "/api/tasks",
        json={
            "type": "expert_collection",
            "params": {"scene_path": str(ROOT), "policy_preset": "navigation_v1"},
        },
    )
    assert response.status_code == 400
    response = client.post(
        "/api/tasks",
        json={
            "type": "expert_collection",
            "params": {"scene_path": str(ROOT), "policy_preset": "pick_and_place_v1", "workers": 2},
        },
    )
    assert response.status_code == 400
    response = client.post(
        "/api/tasks",
        json={"type": "training", "params": {"dataset_id": "missing", "recipe": "diffusion_policy"}},
    )
    assert response.status_code == 400
    response = client.post(
        "/api/tasks",
        json={"type": "inference", "params": {"model_id": "missing", "scene_id": "missing", "workers": 2}},
    )
    assert response.status_code == 400


def test_crud_metadata_and_task_params(tmp_path: Path) -> None:
    cfg = PlatformConfig(app_root=ROOT, workspace_dir=tmp_path / "workspace", allowed_path_roots=(ROOT,))
    client = TestClient(create_app(cfg))
    dataset = client.post("/api/datasets/import-path", json={"path": str(ROOT / "real2sim-expert-policy/examples"), "name": "examples"}).json()
    updated = client.patch(f"/api/datasets/{dataset['id']}", json={"name": "examples edited", "tags": ["demo", "crud"], "version": "v2"}).json()
    assert updated["name"] == "examples edited"
    assert updated["tags"] == ["demo", "crud"]
    assert updated["version"] == "v2"
    task = client.post("/api/tasks", json={"type": "renderer_benchmark", "params": {}}).json()
    edited = client.patch(f"/api/tasks/{task['id']}", json={"params": {"note": "edited before run"}}).json()
    assert edited["params"]["note"] == "edited before run"
    assert client.delete(f"/api/tasks/{task['id']}").json()["ok"] is True
    assert client.delete(f"/api/datasets/{dataset['id']}").json()["ok"] is True


def test_task_progress_fields_and_bulk_actions(tmp_path: Path) -> None:
    cfg = PlatformConfig(app_root=ROOT, workspace_dir=tmp_path / "workspace")
    with TestClient(create_app(cfg)) as client:
        task = client.post("/api/tasks", json={"type": "renderer_benchmark", "params": {}}).json()
        current = _wait(client, task["id"])
        assert current["status"] == "succeeded"
        assert current["progress"] == 100
        assert current["stage"] == "已完成"
        retry = client.post("/api/tasks/bulk", json={"action": "retry", "ids": [task["id"]]}).json()
        assert retry["ok"] is True
        retry_id = retry["results"][0]["result"]["id"]
        cancel = client.post("/api/tasks/bulk", json={"action": "cancel", "ids": [retry_id]}).json()
        assert cancel["ok"] is True
        delete = client.post("/api/tasks/bulk", json={"action": "delete", "ids": [task["id"]]}).json()
        assert delete["ok"] is True


def test_dataset_scene_model_bulk_interfaces(tmp_path: Path) -> None:
    package = tmp_path / "scene_package"
    package.mkdir()
    scene_file = package / "scene.json"
    scene_file.write_text('{"schema":"motrixsim.scene.v1"}\n', encoding="utf-8")
    card_dir = tmp_path / "model"
    card_dir.mkdir()
    (card_dir / "model_card.json").write_text('{"metrics":{"success_rate":0.91}}\n', encoding="utf-8")
    cfg = PlatformConfig(app_root=ROOT, workspace_dir=tmp_path / "workspace", allowed_path_roots=(ROOT, tmp_path))
    client = TestClient(create_app(cfg))
    dataset = client.post("/api/datasets/import-path", json={"path": str(ROOT / "real2sim-expert-policy/examples"), "name": "examples"}).json()
    dataset_bulk = client.post("/api/datasets/bulk", json={"action": "preflight", "ids": [dataset["id"]], "params": {"min_frames": 1}}).json()
    assert dataset_bulk["ok"] is True
    assert dataset_bulk["results"][0]["result"]["type"] == "reconstruction_preflight"
    scene = client.post("/api/scenes/import", json={"path": str(scene_file), "name": "fixture scene"}).json()
    scene_bulk = client.post("/api/scenes/bulk", json={"action": "validate", "ids": [scene["id"]], "params": {"level": "smoke"}}).json()
    assert scene_bulk["ok"] is True
    assert scene_bulk["results"][0]["result"]["type"] == "simulator_validation"
    export_bulk = client.post("/api/scenes/bulk", json={"action": "export", "ids": [scene["id"]]}).json()
    assert export_bulk["results"][0]["result"]["scene_path"].endswith("scene.json")
    rejected_training = client.post(
        "/api/tasks",
        json={"type": "training", "params": {"dataset_id": dataset["id"], "epochs": 1}},
    )
    assert rejected_training.status_code == 400
    assert "专家采集" in rejected_training.json()["detail"]
    # Directly index a model fixture for bulk model API coverage without running a trainer.
    store = PlatformStore(cfg.db_path)
    expert_dir = tmp_path / "expert_collection"
    expert_dir.mkdir()
    (expert_dir / "rollouts.jsonl").write_text(
        '{"episode":0,"step":0,"observation":[0,0,0],"action":[0,0,0]}\n',
        encoding="utf-8",
    )
    expert_dataset = store.create_dataset(
        name="expert rollouts",
        path=expert_dir,
        scan={"input_kind": "expert_rollouts", "files": 1},
        source_type="expert_collection",
    )
    model = client.post(
        "/api/tasks",
        json={"type": "training", "params": {"dataset_id": expert_dataset["id"], "epochs": 1, "recipe": "bc_fast"}},
    )
    assert model.status_code == 200, model.text
    model_entity = store.create_model(name="fixture model", path=card_dir, metadata={"metrics": {"success_rate": 0.91}}, manifest_path=card_dir / "model_card.json", metrics={"success_rate": 0.91})
    model_bulk = client.post("/api/models/bulk", json={"action": "compare", "ids": [model_entity["id"]]}).json()
    assert model_bulk["ok"] is True
    assert model_bulk["results"][0]["result"]["metrics"]["success_rate"] == 0.91
    assert model.json()["type"] == "training"


def test_scene_asset_request_queues_real_revision_task_from_catalog(tmp_path: Path) -> None:
    package = tmp_path / "scene_package"
    package.mkdir()
    scene_file = package / "scene.json"
    scene_file.write_text('{"schema":"motrixsim.scene.v1"}\n', encoding="utf-8")
    asset_dataset = tmp_path / "asset_dataset"
    asset_dataset.mkdir()
    asset = asset_dataset / "fixture.obj"
    asset.write_text("v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nf 1 2 3\n", encoding="utf-8")
    cfg = PlatformConfig(app_root=ROOT, workspace_dir=tmp_path / "workspace", allowed_path_roots=(tmp_path,))
    client = TestClient(create_app(cfg))
    dataset = client.post("/api/datasets/import-path", json={"path": str(asset_dataset), "name": "asset library"}).json()
    scene = client.post("/api/scenes/import", json={"path": str(scene_file), "name": "editable scene"}).json()
    catalog = client.get("/api/assets").json()
    catalog_asset = next(item for item in catalog if item.get("relative_path") == "fixture.obj")
    assert catalog_asset["source"]["dataset_id"] == dataset["id"]
    response = client.post(
        f"/api/scenes/{scene['id']}/assets",
        json={
            "source": catalog_asset["source"],
            "name": "fixture mesh",
            "transform": {"translation": [1, 2, 3], "scale": [1.5, 1.5, 1.5]},
            "physics": {"dynamic": True, "mass": 0.2, "collision_mode": "visual_and_collision"},
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["scene_id"] == scene["id"]
    assert payload["task"]["type"] == "scene_asset_compose"
    assert payload["task"]["params"]["source"] == catalog_asset["source"]
    assert payload["task"]["params"]["physics"]["dynamic"] is True
    assert len(client.get("/api/scenes").json()) == 1


def test_scene_library_clone_and_asset_revision_requests_are_real_tasks(tmp_path: Path) -> None:
    package = tmp_path / "editable_package"
    runtime_dir = package / "simulator_assets"
    runtime_dir.mkdir(parents=True)
    runtime = runtime_dir / "scene.xml"
    runtime.write_text(
        '<mujoco model="fixture"><option timestep="0.002"/><worldbody><body name="added_box" pos="0 0 0"><geom type="box" size="0.05 0.05 0.05"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    manifest = package / "scene_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "real2sim.scene_manifest.v3",
                "assets": [
                    {
                        "name": "测试物体",
                        "role": "added_asset",
                        "kind": "primitive",
                        "runtime": {"body": "added_box"},
                        "transform": {"translation": [0, 0, 0], "rotation_euler_degrees": [0, 0, 0], "scale": [1, 1, 1]},
                        "physics": {"dynamic": False, "mass": 1, "collision_mode": "visual_and_collision", "dimensions": [0.1, 0.1, 0.1]},
                    }
                ],
                "simulator_artifacts": {"motrixsim": {"runtime_scene": str(runtime)}},
            }
        ),
        encoding="utf-8",
    )
    cfg = PlatformConfig(app_root=ROOT, workspace_dir=tmp_path / "workspace", allowed_path_roots=(tmp_path,))
    client = TestClient(create_app(cfg))
    scene = client.post(
        "/api/scenes/import",
        json={"path": str(runtime), "name": "可编辑场景", "metadata": {"package": str(package), "manifest_path": str(manifest)}},
    ).json()
    update = client.patch(
        f"/api/scenes/{scene['id']}/assets/added_box",
        json={"name": "新位置物体", "transform": {"translation": [1, 2, 3]}, "physics": {"dynamic": False}, "validation_steps": 1},
    )
    assert update.status_code == 200, update.text
    assert update.json()["task"]["type"] == "scene_asset_update"
    clone = client.post(f"/api/scenes/{scene['id']}/clone", json={"name": "可编辑场景副本", "validation_steps": 1})
    assert clone.status_code == 200, clone.text
    assert clone.json()["task"]["type"] == "scene_clone"
    delete = client.delete(f"/api/scenes/{scene['id']}/assets/added_box")
    assert delete.status_code == 200, delete.text
    assert delete.json()["task"]["type"] == "scene_asset_delete"


def test_scene_asset_delete_cleans_runtime_references(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "editable_package"
    runtime_dir = package / "simulator_assets"
    runtime_dir.mkdir(parents=True)
    runtime = runtime_dir / "scene.xml"
    runtime.write_text(
        """<mujoco model="fixture">
  <worldbody>
    <body name="base_link"><geom type="box" size="0.1 0.1 0.1" /></body>
    <body name="pick_cube"><freejoint name="pick_cube_free" /><geom type="box" size="0.03 0.03 0.03" /></body>
    <camera name="overview" mode="targetbody" target="pick_cube" pos="1 -1 1" />
  </worldbody>
  <equality><joint joint1="pick_cube_free" joint2="base_joint" /></equality>
  <keyframe><key name="home" qpos="0 0 0 0 0 0 0" qvel="0 0 0 0 0 0 0" /></keyframe>
</mujoco>""",
        encoding="utf-8",
    )
    (package / "scene_manifest.json").write_text(
        json.dumps(
            {
                "schema": "real2sim.scene_manifest.v3",
                "assets": [
                    {"name": "抓取方块", "role": "added_asset", "runtime": {"body": "pick_cube"}},
                    {"name": "机器人", "role": "robot", "runtime": {"body": "base_link"}},
                ],
                "simulator_artifacts": {"motrixsim": {"runtime_scene": str(runtime)}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        motrix_scene,
        "validate_runtime_scene",
        lambda _runtime, steps: {
            "steps": steps,
            "mujoco": {"ok": True},
            "motrixsim": {"ok": True},
            "ok": True,
        },
    )

    result = motrix_scene.revise_scene_asset(
        package,
        asset_id="pick_cube",
        delete=True,
        validation_steps=1,
    )

    assert result["operation"] == "deleted"
    root = __import__("xml.etree.ElementTree", fromlist=["parse"]).parse(runtime).getroot()
    assert root.find(".//body[@name='pick_cube']") is None
    camera = root.find(".//camera[@name='overview']")
    assert camera is not None
    assert camera.get("target") is None
    assert camera.get("mode") is None
    assert root.find("keyframe") is None
    assert root.find("equality") is not None and list(root.find("equality")) == []
    manifest = json.loads((package / "scene_manifest.json").read_text(encoding="utf-8"))
    assert all(item.get("runtime", {}).get("body") != "pick_cube" for item in manifest["assets"])


def test_scene_runtime_metadata_update(tmp_path: Path) -> None:
    store = PlatformStore(tmp_path / "state.db")
    scene_file = tmp_path / "scene.xml"
    scene_file.write_text("<mujoco/>", encoding="utf-8")
    scene = store.create_scene(
        name="runtime scene",
        path=scene_file,
        metadata={"status": "adapter_pending", "adapter_pending": True},
        tags=["motrixsim", "adapter-pending"],
    )
    updated = store.update_scene(
        scene["id"],
        metadata={
            "status": "runtime_validated",
            "adapter_pending": False,
            "runtime_validation": {"mujoco": True, "motrixsim": True},
        },
        metrics={"runtime_count": 2},
        tags=["motrixsim", "runtime-validated", "motrixsim-ready"],
    )
    assert updated["metadata"]["runtime_validation"] == {"mujoco": True, "motrixsim": True}
    assert updated["metadata"]["adapter_pending"] is False
    assert updated["metrics"]["runtime_count"] == 2
    assert "adapter-pending" not in updated["tags"]


def test_media_prepare_video_task(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for media_prepare video fixture")
    video = tmp_path / "input.mp4"
    _write_fixture_video(video, tmp_path / "frames")
    cfg = PlatformConfig(app_root=ROOT, workspace_dir=tmp_path / "workspace")
    with TestClient(create_app(cfg)) as client:
        task = client.post(
            "/api/tasks",
            json={
                "type": "media_prepare",
                "params": {"input_path": str(video), "sample_fps": 2, "max_frames_per_video": 5, "max_videos": 1},
            },
        ).json()
        current = _wait(client, task["id"])
        assert current["status"] == "succeeded", current
        manifest = client.get(f"/api/datasets/{current['result']['dataset_id']}/manifest").json()
        assert manifest["schema"] == "real2sim.prepared_manifest.v1"
        assert manifest["frame_count"] > 0
        assert manifest["video_frame_counts"][0]["frame_count"] > 0
        assert Path(manifest["frames_dir"]).is_dir()


def test_scene_reconstruction_uses_prepared_dataset(tmp_path: Path) -> None:
    gs_root = tmp_path / "gs"
    script = gs_root / "scripts" / "real2sim_video_motrix_scene.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser(); p.add_argument('--output'); p.add_argument('--work-dir'); p.add_argument('--frames-dir'); p.add_argument('--quality-preset'); p.add_argument('--sfm-report', action='store_true'); p.add_argument('--capture-acceptance', action='store_true'); p.add_argument('--pose-init-train', action='store_true'); p.add_argument('--anysplat-root'); p.add_argument('--anysplat-python'); args,_=p.parse_known_args()\n"
        "out=Path(args.output); out.mkdir(parents=True, exist_ok=True)\n"
        "(out/'scene.json').write_text(json.dumps({'schema':'fake.scene','coordinate_frame':'gs_metric_zup'}), encoding='utf-8')\n"
        "(out/'scene_mujoco.xml').write_text('<mujoco model=\"fake\"><worldbody /></mujoco>\\n', encoding='utf-8')\n"
        "(out/'motrix_import.json').write_text(json.dumps({'schema':'motrixsim.scene.v1'}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    frames = tmp_path / "prepared" / "frames"
    frames.mkdir(parents=True)
    for index in range(30):
        Image.new("RGB", (640, 480), (index * 3, 48, 120)).save(frames / f"frame_{index:06d}.jpg")
    manifest = {
        "schema": "real2sim.prepared_manifest.v1",
        "frames_dir": str(frames),
        "frame_count": 30,
        "source_uri": str(tmp_path),
    }
    prepared_root = frames.parent
    (prepared_root / "prepared_manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")
    anysplat_root = tmp_path / "AnySplat"
    anysplat_root.mkdir()
    cfg = PlatformConfig(
        app_root=ROOT,
        workspace_dir=tmp_path / "workspace",
        gs_playground_root=gs_root,
        gsplat_python=Path(__import__("sys").executable),
        anysplat_root=anysplat_root,
        anysplat_python=Path(__import__("sys").executable),
        allowed_path_roots=(tmp_path,),
    )
    with TestClient(create_app(cfg)) as client:
        dataset = client.post("/api/datasets/import-path", json={"path": str(prepared_root), "name": "prepared"}).json()
        task = client.post(
            "/api/tasks",
            json={"type": "scene_reconstruction", "params": {"dataset_id": dataset["id"], "dry_run": False, "renderer": "spark", "simulators": ["mujoco", "motrixsim"], "preflight_required": False}},
        ).json()
        current = _wait(client, task["id"])
        assert current["status"] == "succeeded", current
        artifacts = client.get(f"/api/tasks/{task['id']}/artifacts").json()
        assert any(item["name"] == "reconstruction_report.json" for item in artifacts)
        reports = client.get(f"/api/tasks/{task['id']}/reports").json()
        assert any(item["name"] == "capture_preflight_report.json" for item in reports["reports"])


def test_reconstruction_preflight_blocks_short_dataset(tmp_path: Path) -> None:
    frames = tmp_path / "prepared" / "frames"
    frames.mkdir(parents=True)
    Image.new("RGB", (320, 240), "red").save(frames / "frame_000001.jpg")
    prepared_root = frames.parent
    (prepared_root / "prepared_manifest.json").write_text(
        __import__("json").dumps({"schema": "real2sim.prepared_manifest.v1", "frames_dir": str(frames), "frame_count": 1}),
        encoding="utf-8",
    )
    cfg = PlatformConfig(app_root=ROOT, workspace_dir=tmp_path / "workspace", anysplat_root=tmp_path / "missing-anysplat", allowed_path_roots=(tmp_path,))
    with TestClient(create_app(cfg)) as client:
        dataset = client.post("/api/datasets/import-path", json={"path": str(prepared_root), "name": "short"}).json()
        task = client.post("/api/tasks", json={"type": "reconstruction_preflight", "params": {"dataset_id": dataset["id"]}}).json()
        current = _wait(client, task["id"])
        assert current["status"] == "succeeded"
        assert current["result"]["ok"] is False
        assert "enough_frames" in current["result"]["report"]["failed_gates"]
        assert current["result"]["report"]["readiness_score"] < 100
        assert current["result"]["report"]["next_action"] == "run_media_prepare_or_recapture_more_frames"


def test_scene_reconstruction_unprepared_media_error(tmp_path: Path) -> None:
    source = tmp_path / "hf_video_manifest"
    source.mkdir()
    (source / "0000.parquet").write_bytes(b"not a real parquet")
    cfg = PlatformConfig(app_root=ROOT, workspace_dir=tmp_path / "workspace")
    with TestClient(create_app(cfg)) as client:
        task = client.post("/api/tasks", json={"type": "scene_reconstruction", "params": {"input_path": str(source), "dry_run": False}}).json()
        current = _wait(client, task["id"])
        assert current["status"] == "failed"
        assert "media_prepare" in current["error"]


def test_simulator_validation_task(tmp_path: Path) -> None:
    cfg = PlatformConfig(app_root=ROOT, workspace_dir=tmp_path / "workspace")
    with TestClient(create_app(cfg)) as client:
        scene_task = client.post(
            "/api/tasks",
            json={
                "type": "scene_reconstruction",
                "params": {
                    "input_path": str(ROOT / "real2sim-expert-policy/examples"),
                    "dry_run": True,
                    "renderer": "auto",
                    "simulators": ["mujoco", "motrixsim"],
                },
            },
        ).json()
        scene_task = _wait(client, scene_task["id"])
        assert scene_task["status"] == "succeeded"
        validation = client.post(
            "/api/tasks",
            json={"type": "simulator_validation", "params": {"scene_id": scene_task["result"]["scene_id"], "simulators": ["mujoco", "motrixsim"], "level": "smoke"}},
        ).json()
        validation = _wait(client, validation["id"])
        assert validation["status"] == "succeeded"
        assert validation["result"]["ok"] is True
        assert validation["result"]["report"]["level"] == "smoke"


def test_simulator_validation_accepts_scene_package_layout(tmp_path: Path) -> None:
    package = tmp_path / "scene_package"
    (package / "mjcf").mkdir(parents=True)
    (package / "mjcf" / "scene.xml").write_text('<mujoco model="fixture"><worldbody /></mujoco>\n', encoding="utf-8")
    (package / "motrix_import.json").write_text('{"schema":"motrixsim.scene.v1"}\n', encoding="utf-8")
    cfg = PlatformConfig(app_root=ROOT, workspace_dir=tmp_path / "workspace", allowed_path_roots=(tmp_path,))
    with TestClient(create_app(cfg)) as client:
        task = client.post(
            "/api/tasks",
            json={
                "type": "simulator_validation",
                "params": {"scene_path": str(package), "simulators": ["mujoco", "motrixsim"], "level": "smoke"},
            },
        ).json()
        current = _wait(client, task["id"])
        assert current["status"] == "succeeded"
        report = current["result"]["report"]
        assert report["ok"] is True
        assert report["simulators"]["mujoco"]["path"].endswith("mjcf/scene.xml")
        assert report["simulators"]["motrixsim"]["path"].endswith("motrix_import.json")


def test_task_reports_prefer_capture_acceptance_failure(tmp_path: Path) -> None:
    store = PlatformStore(tmp_path / "state.db")
    run_dir = tmp_path / "run"
    work_dir = run_dir / "work"
    work_dir.mkdir(parents=True)
    task = store.create_task(task_type="scene_reconstruction", params={}, run_dir=run_dir)
    store.update_task(task["id"], status="failed", error="command failed with exit code 1; see logs")
    (work_dir / "capture_acceptance_report.json").write_text(
        __import__("json").dumps(
            {
                "schema": "real2sim.capture_acceptance.v1",
                "accepted": False,
                "decision": "recapture_required_before_high_fidelity_reconstruction",
                "blockers": ["enough_registered_images", "registered_ratio"],
            }
        ),
        encoding="utf-8",
    )

    reports = collect_task_reports(store, task["id"])

    assert reports["readable_error"].startswith("capture acceptance failed")
    assert "registered_ratio" in reports["readable_error"]


def test_recover_interrupted_tasks(tmp_path: Path) -> None:
    store = PlatformStore(tmp_path / "state.db")
    task = store.create_task(task_type="renderer_benchmark", params={}, run_dir=tmp_path / "run")
    store.update_task(task["id"], status="running")
    assert store.recover_interrupted_tasks() == 1
    recovered = store.get_task(task["id"])
    assert recovered["status"] == "failed_recovered"


def test_discoverse_3dgs_rgb_fallback_export(tmp_path: Path) -> None:
    source = tmp_path / "scene.ply"
    target = tmp_path / "fallback" / "scene_threejs_rgb.ply"
    report_path = tmp_path / "fallback" / "scene_threejs_rgb_report.json"
    fields = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    source.write_bytes(
        (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 2\n"
            + "".join(f"property float {name}\n" for name in fields)
            + "end_header\n"
        ).encode("ascii")
        + struct.pack("<" + "f" * len(fields), 0, 0, 0, 0.5, 0.0, -0.5, 4.0, 0.01, 0.01, 0.01, 1, 0, 0, 0)
        + struct.pack("<" + "f" * len(fields), 1, 2, 3, -0.5, 0.2, 0.5, 4.0, 0.01, 0.01, 0.01, 1, 0, 0, 0)
    )

    report = _export_3dgs_rgb_fallback(source, target, report_path)

    assert report["schema"] == "real2sim.discoverse_exact_3dgs_rgb_fallback.v1"
    assert report["vertices"] == 2
    assert target.is_file()
    assert report_path.is_file()
    header = target.read_bytes().split(b"end_header\n", 1)[0].decode("ascii")
    assert "property uchar red" in header


def _wait(client: TestClient, task_id: str) -> dict:
    current = {}
    for _ in range(60):
        current = client.get(f"/api/tasks/{task_id}").json()
        if current["status"] in {"succeeded", "failed", "canceled", "failed_recovered"}:
            break
        time.sleep(0.1)
    return current


def _write_fixture_video(video_path: Path, frame_dir: Path) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, 9):
        Image.new("RGB", (96, 64), (index * 20, 80, 120)).save(frame_dir / f"frame_{index:03d}.jpg")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-framerate",
            "8",
            "-i",
            str(frame_dir / "frame_%03d.jpg"),
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
