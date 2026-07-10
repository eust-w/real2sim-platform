from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from .adapters import run_doctor
from .config import PlatformConfig, ensure_workspace, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the production Real2Sim GPU runtime")
    parser.add_argument("--mode", choices=["bootstrap", "readiness"], default="readiness")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config()
    ensure_workspace(config)
    marker = marker_path(config)
    if args.mode == "readiness":
        return readiness(config, marker)
    if marker.is_file() and not args.force:
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
            if existing.get("runtime_key") == runtime_key(config) and existing.get("ok") is True:
                print(json.dumps(existing, ensure_ascii=False))
                return 0
        except (OSError, json.JSONDecodeError):
            pass
    report = bootstrap(config)
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(marker)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ok"] else 1


def marker_path(config: PlatformConfig) -> Path:
    root = Path(os.environ.get("REAL2SIM_RUNTIME_GATE_DIR", config.workspace_dir / "runtime_gate"))
    return root / f"{runtime_key(config)}.json"


def runtime_key(config: PlatformConfig) -> str:
    raw = ":".join(
        [
            config.anysplat_commit,
            config.anysplat_model_revision,
            config.vggt_model_revision,
            config.discoverse_commit,
            config.act_commit,
            os.environ.get("REAL2SIM_MOTRIXSIM_DOCS_COMMIT", ""),
            os.environ.get("REAL2SIM_PLATFORM_BUILD_ID", ""),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def readiness(config: PlatformConfig, marker: Path) -> int:
    doctor = run_doctor(config)
    if not marker.is_file():
        print("runtime bootstrap report is missing", file=sys.stderr)
        return 1
    try:
        report = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"runtime bootstrap report is unreadable: {exc}", file=sys.stderr)
        return 1
    ready = bool(report.get("ok") and report.get("runtime_key") == runtime_key(config) and doctor.get("ok"))
    if not ready:
        print(json.dumps({"bootstrap": report, "doctor": doctor}, ensure_ascii=False), file=sys.stderr)
    return 0 if ready else 1


def bootstrap(config: PlatformConfig) -> dict[str, Any]:
    started = time.time()
    checks: dict[str, Any] = {}
    doctor = run_doctor(config)
    checks["doctor"] = {"ok": doctor.get("ok"), "core_checks": doctor.get("core_checks"), "checks": doctor.get("checks")}
    try:
        checks["nvidia"] = _nvidia_probe()
        checks["simulators"] = _simulator_probe()
        checks["motrixsim_assets"] = _motrixsim_assets_probe()
        checks["act"] = _act_probe(config)
        checks["anysplat"] = _anysplat_probe(config)
    except Exception as exc:
        checks["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    ok = bool(doctor.get("ok")) and all(bool(value.get("ok")) for key, value in checks.items() if key != "failure") and "failure" not in checks
    return {
        "schema": "real2sim.runtime_gate.v1",
        "runtime_key": runtime_key(config),
        "ok": ok,
        "started_at": started,
        "finished_at": time.time(),
        "checks": checks,
        "revisions": {
            "anysplat_code": config.anysplat_commit,
            "anysplat_model": config.anysplat_model_revision,
            "vggt_model": config.vggt_model_revision,
            "discoverse": config.discoverse_commit,
            "act": config.act_commit,
            "motrixsim_docs": os.environ.get("REAL2SIM_MOTRIXSIM_DOCS_COMMIT", ""),
        },
    }


def _nvidia_probe() -> dict[str, Any]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        timeout=20,
        check=True,
    )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")
    value = torch.ones((1024, 1024), device="cuda")
    checksum = float((value @ value).mean().item())
    del value
    torch.cuda.empty_cache()
    return {"ok": checksum == 1024.0, "device": torch.cuda.get_device_name(0), "nvidia_smi": completed.stdout.strip().splitlines()[0]}


def _simulator_probe() -> dict[str, Any]:
    xml = """<mujoco model="runtime-gate"><option timestep="0.002"/><worldbody><body pos="0 0 0.2"><freejoint/><geom type="sphere" size="0.03" mass="0.1"/></body><geom type="plane" size="1 1 .1"/></worldbody></mujoco>"""
    import mujoco

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(8):
        mujoco.mj_step(model, data)
    with tempfile.TemporaryDirectory(prefix="real2sim-runtime-gate-") as temporary:
        scene_path = Path(temporary) / "scene.xml"
        scene_path.write_text(xml, encoding="utf-8")
        from motrixsim import SceneData, load_model, step

        motrix_model = load_model(str(scene_path))
        motrix_data = SceneData(motrix_model)
        for _ in range(8):
            step(motrix_model, motrix_data)
    return {"ok": True, "mujoco_time": float(data.time), "motrix_timestep": float(motrix_model.options.timestep)}


def _motrixsim_assets_probe() -> dict[str, Any]:
    root = Path(os.environ.get("REAL2SIM_MOTRIXSIM_DOCS_ROOT", "/opt/motrixsim-docs"))
    required = {
        "parlour": root / "examples/assets/parlour/parlour.xml",
        "rm65_mesh": root / "examples/assets/rm65_open_cabinet/assets/rm65/meshes/base_link.STL",
        "pick_place_template": root / "examples/assets/franka_emika_panda/scene_pick_cube.xml",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError("MotrixSim assets are missing: " + ", ".join(missing))
    pointers = []
    for name, path in required.items():
        if path.stat().st_size <= 1024 and path.read_bytes()[:160].startswith(b"version https://git-lfs.github.com/spec/"):
            pointers.append(name)
    if pointers:
        raise RuntimeError("MotrixSim Git LFS assets are unresolved: " + ", ".join(pointers))
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        timeout=20,
        check=True,
    )
    commit = completed.stdout.strip()
    expected = os.environ.get("REAL2SIM_MOTRIXSIM_DOCS_COMMIT", "")
    if expected and commit != expected:
        raise RuntimeError(f"MotrixSim docs revision mismatch: expected {expected}, got {commit}")
    return {"ok": True, "commit": commit, "asset_sizes": {name: path.stat().st_size for name, path in required.items()}}


def _act_probe(config: PlatformConfig) -> dict[str, Any]:
    act_root = config.discoverse_root / "policies" / "act"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(act_root), str(config.discoverse_root), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-c", "import torch; from policies.act.act import ACTPolicy; print(torch.__version__); assert ACTPolicy is not None and torch.cuda.is_available()"],
        cwd=act_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    )
    return {"ok": True, "torch": completed.stdout.strip().splitlines()[-1]}


def _anysplat_probe(config: PlatformConfig) -> dict[str, Any]:
    cache_dir = Path(os.environ.get("HF_HOME", config.workspace_dir / "cache" / "huggingface"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    anysplat_snapshot = Path(
        snapshot_download(
            repo_id="lhjiang/anysplat",
            revision=config.anysplat_model_revision,
            cache_dir=cache_dir,
        )
    )
    snapshot_download(repo_id="facebook/VGGT-1B", revision=config.vggt_model_revision, cache_dir=cache_dir)
    if str(config.anysplat_root) not in sys.path:
        sys.path.insert(0, str(config.anysplat_root))
    from src.model.model.anysplat import AnySplat

    model = AnySplat.from_pretrained(str(anysplat_snapshot))
    model = model.to("cuda").eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters <= 0:
        raise RuntimeError("AnySplat model loaded with zero parameters")
    del model
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    return {"ok": True, "parameter_count": parameters, "snapshot": anysplat_snapshot.name}


if __name__ == "__main__":
    raise SystemExit(main())
