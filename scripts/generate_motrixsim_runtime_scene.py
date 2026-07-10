#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
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


SCENE_ID = "1f2618bdfcac4e45456569980716405a"
PACKAGE_SLUG = "parlour_rm65_pick_place"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a runnable RM65 + coffee-table pick-place smoke MJCF for the MotrixSim asset package.")
    parser.add_argument("--scene-id", default=SCENE_ID)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--motrixsim-python", default="")
    args = parser.parse_args()

    config = load_config()
    ensure_workspace(config)
    store = PlatformStore(config.db_path)
    scene = store.get_scene(args.scene_id)
    package_dir = Path(str((scene.get("metadata") or {}).get("package") or config.workspace_dir / "motrixsim_assets" / PACKAGE_SLUG))
    manifest_path = Path(str(scene.get("manifest_path") or package_dir / "motrixsim_scene_manifest.json"))
    sim_dir = package_dir / "simulator_assets"
    sim_dir.mkdir(parents=True, exist_ok=True)

    runtime_rm65 = sim_dir / "rmjx_rm65_runtime.xml"
    source_rm65 = package_dir / "assets/rm65_open_cabinet/mjx_rm65.xml"
    runtime_xml = source_rm65.read_text(encoding="utf-8").replace(
        'assetdir="assets"',
        'assetdir="../assets/rm65_open_cabinet/assets"',
    )
    # The upstream pad capsules are visual-only diagnostics. Enable collision on
    # the runtime copy so pick-place validation measures real bilateral contact.
    runtime_xml = runtime_xml.replace(
        'contype="0" conaffinity="0" density="0" group="2"/>',
        'contype="1" conaffinity="1" density="0" friction="2.0 0.02 0.002" group="3"/>',
    )
    runtime_xml = runtime_xml.replace(
        'name="left_finger_pad_z_axis" type="capsule" fromto="-0.021275 0.04375 -0.06 -0.021275 0.04375 0.06"\n                      size="0.003"',
        'name="left_finger_pad_z_axis" type="capsule" fromto="-0.021275 0.04375 -0.06 -0.021275 0.04375 0.06"\n                      size="0.010"',
    ).replace(
        'name="right_finger_pad_z_axis" type="capsule" fromto="-0.021275 0.04375 -0.06 -0.021275 0.04375 0.06"\n                      size="0.003"',
        'name="right_finger_pad_z_axis" type="capsule" fromto="-0.021275 0.04375 -0.06 -0.021275 0.04375 0.06"\n                      size="0.010"',
    )
    runtime_rm65.write_text(runtime_xml, encoding="utf-8")

    combined = sim_dir / "rm65_table_pick_place.xml"
    combined.write_text(runtime_scene_xml(), encoding="utf-8")

    mujoco_report = validate_mujoco(combined, steps=args.steps)
    motrixsim_python = find_motrixsim_python(args.motrixsim_python)
    motrixsim_report = validate_motrixsim(combined, steps=args.steps, python=motrixsim_python)
    report = {
        "schema": "real2sim.motrixsim_runtime_scene_validation.v2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "combined_mjcf": str(combined),
        "steps_requested": args.steps,
        "ok": bool(mujoco_report.get("ok") and motrixsim_report.get("ok")),
        "runtimes": {"mujoco": mujoco_report, "motrixsim": motrixsim_report},
    }
    report_path = sim_dir / "rm65_table_pick_place_validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    scene_artifact = store.add_artifact(task_id=None, name=combined.name, path=combined, kind="simulator")
    rm65_artifact = store.add_artifact(task_id=None, name=runtime_rm65.name, path=runtime_rm65, kind="simulator")
    report_artifact = store.add_artifact(task_id=None, name=report_path.name, path=report_path, kind="json")
    update_manifest(manifest_path, combined, scene_artifact["id"], runtime_rm65, rm65_artifact["id"], report_path, report_artifact["id"], report)
    runtime_ready = bool(report["ok"])
    metadata = dict(scene.get("metadata") or {})
    metadata.update(
        {
            "status": "runtime_validated" if runtime_ready else "runtime_validation_failed",
            "adapter_pending": not runtime_ready,
            "runtime_validation": {"mujoco": bool(mujoco_report.get("ok")), "motrixsim": bool(motrixsim_report.get("ok"))},
        }
    )
    tags = [tag for tag in scene.get("tags", []) if tag not in {"adapter-pending", "runtime-validated", "motrixsim-ready"}]
    if runtime_ready:
        tags.extend(["runtime-validated", "motrixsim-ready"])
    else:
        tags.append("adapter-pending")
    store.update_scene(args.scene_id, metadata=metadata, tags=tags)

    print(json.dumps({"ok": report["ok"], "scene_id": args.scene_id, "combined_mjcf": str(combined), "report": str(report_path), "validation": report}, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def runtime_scene_xml() -> str:
    return """<mujoco model="rm65 table pick place smoke">
  <include file="rmjx_rm65_runtime.xml"/>
  <visual>
    <global offwidth="960" offheight="540"/>
  </visual>
  <worldbody>
    <body name="coffee_table" pos="0.65 0 0.25">
      <geom name="coffee_table_top" type="box" size="0.45 0.32 0.03" rgba="0.45 0.30 0.18 1" friction="0.8 0.01 0.001"/>
      <geom name="coffee_table_leg_1" type="box" pos="0.36 0.25 -0.14" size="0.025 0.025 0.14" rgba="0.20 0.13 0.08 1"/>
      <geom name="coffee_table_leg_2" type="box" pos="-0.36 0.25 -0.14" size="0.025 0.025 0.14" rgba="0.20 0.13 0.08 1"/>
      <geom name="coffee_table_leg_3" type="box" pos="0.36 -0.25 -0.14" size="0.025 0.025 0.14" rgba="0.20 0.13 0.08 1"/>
      <geom name="coffee_table_leg_4" type="box" pos="-0.36 -0.25 -0.14" size="0.025 0.025 0.14" rgba="0.20 0.13 0.08 1"/>
    </body>
    <body name="pick_cube" pos="0.65 0 0.33">
      <freejoint name="pick_cube_free"/>
      <geom name="pick_cube_geom" type="box" size="0.03 0.03 0.03" rgba="0.9 0.2 0.12 1" mass="0.05" friction="0.9 0.01 0.001"/>
    </body>
    <body name="place_target" pos="0.43 0.18 0.335">
      <geom name="place_target_marker" type="cylinder" size="0.055 0.004" rgba="0.1 0.8 0.6 0.35" contype="0" conaffinity="0"/>
    </body>
    <camera name="overview" pos="1.7 -1.6 1.25" xyaxes="0.68 0.73 0 -0.37 0.34 0.86"/>
  </worldbody>
  <keyframe>
    <key name="home" qpos="0 0 0 -1.57079 0 1.57079 -0.7853 0 0 0 0 0 0.65 0 0.33 1 0 0 0" ctrl="0 0 0 -1.57079 0 1.57079 -0.7853"/>
  </keyframe>
</mujoco>
"""


def validate_mujoco(path: Path, *, steps: int) -> dict[str, Any]:
    started = time.time()
    report: dict[str, Any] = {
        "schema": "real2sim.motrixsim_runtime_scene_validation.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "combined_mjcf": str(path),
        "steps_requested": steps,
        "ok": False,
    }
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        for _ in range(max(0, steps)):
            mujoco.mj_step(model, data)
        report.update(
            {
                "ok": True,
                "runtime": "mujoco",
                "nq": int(model.nq),
                "nv": int(model.nv),
                "nbody": int(model.nbody),
                "ngeom": int(model.ngeom),
                "nkey": int(model.nkey),
                "sim_time": float(data.time),
                "elapsed_ms": round((time.time() - started) * 1000),
            }
        )
    except Exception as exc:
        report.update({"ok": False, "error": f"{type(exc).__name__}: {exc}", "elapsed_ms": round((time.time() - started) * 1000)})
    return report


def find_motrixsim_python(explicit: str) -> Path | None:
    current_python = sys.executable if importlib.util.find_spec("motrixsim") is not None else ""
    candidates = [
        explicit,
        os.environ.get("REAL2SIM_MOTRIXSIM_PYTHON", ""),
        current_python,
        "/Users/d-robotics/workSpace/motrixsim-docs/.venv/bin/python",
        "/opt/Motphys/motrixsim-docs/.venv/bin/python",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def validate_motrixsim(path: Path, *, steps: int, python: Path | None) -> dict[str, Any]:
    started = time.time()
    report: dict[str, Any] = {"ok": False, "runtime": "motrixsim", "python": str(python or "")}
    if python is None:
        report.update({"status": "runtime_unavailable", "error": "motrixsim python environment was not found"})
        return report
    probe = """
import json, sys
from motrixsim import SceneData, load_model, step
model = load_model(sys.argv[1])
data = SceneData(model)
for _ in range(int(sys.argv[2])):
    step(model, data)
print("REAL2SIM_RESULT=" + json.dumps({
    "timestep": float(model.options.timestep),
    "nq": len(data.low.dof_positions),
    "nv": len(data.low.dof_velocities),
}))
"""
    try:
        completed = subprocess.run(
            [str(python), "-c", probe, str(path), str(steps)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        result_line = next(line for line in completed.stdout.splitlines() if line.startswith("REAL2SIM_RESULT="))
        result = json.loads(result_line.split("=", 1)[1])
        report.update(
            {
                "ok": True,
                "status": "loaded_and_stepped",
                "steps": steps,
                **result,
                "elapsed_ms": round((time.time() - started) * 1000),
            }
        )
    except Exception as exc:
        stderr = getattr(exc, "stderr", "") or ""
        report.update(
            {
                "ok": False,
                "status": "runtime_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "stderr": stderr[-2000:],
                "elapsed_ms": round((time.time() - started) * 1000),
            }
        )
    return report


def update_manifest(
    manifest_path: Path,
    combined: Path,
    combined_artifact_id: str,
    runtime_rm65: Path,
    runtime_rm65_artifact_id: str,
    report_path: Path,
    report_artifact_id: str,
    report: dict[str, Any],
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generated_names = {combined.name, runtime_rm65.name, report_path.name}
    assets = [item for item in manifest.get("assets", []) if not (isinstance(item, dict) and item.get("name") in generated_names)]
    generated = [
        asset_entry(combined, combined_artifact_id, "runtime_scene", "simulator", "可运行 RM65 茶几 pick-place smoke MJCF", manifest_path),
        asset_entry(runtime_rm65, runtime_rm65_artifact_id, "runtime_robot", "simulator", "RM65 runtime assetdir 修正版", manifest_path),
        asset_entry(report_path, report_artifact_id, "runtime_validation_report", "metadata", "运行时加载校验报告", manifest_path),
    ]
    manifest["assets"] = [*generated, *assets]
    manifest["runtime_scene"] = {
        "combined_mjcf": generated[0],
        "runtime_rm65": generated[1],
        "validation_report": generated[2],
        "validation": report,
        "status": "mujoco_and_motrixsim_smoke_passed" if report.get("ok") else "failed",
        "scope": "combined_rm65_table_scene_runtime_load_and_step",
    }
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def asset_entry(path: Path, artifact_id: str, role: str, kind: str, name: str, manifest_path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    return {
        "name": path.name,
        "display_name": name,
        "role": role,
        "kind": kind,
        "path": str(path),
        "relative_path": str(path.relative_to(manifest_path.parent)),
        "url": f"/api/artifacts/{artifact_id}/content",
        "exists": True,
        "size_bytes": size,
        "size_label": format_bytes(size),
        "generated": True,
    }


def format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{value} B"


if __name__ == "__main__":
    raise SystemExit(main())
