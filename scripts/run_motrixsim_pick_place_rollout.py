#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


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
    parser = argparse.ArgumentParser(description="Run a scripted RM65 pick-place rollout on the generated smoke MJCF.")
    parser.add_argument("--scene-id", default=SCENE_ID)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    config = load_config()
    ensure_workspace(config)
    store = PlatformStore(config.db_path)
    scene = store.get_scene(args.scene_id)
    package_dir = Path(str((scene.get("metadata") or {}).get("package") or config.workspace_dir / "motrixsim_assets" / PACKAGE_SLUG))
    manifest_path = Path(str(scene.get("manifest_path") or package_dir / "motrixsim_scene_manifest.json"))
    sim_path = package_dir / "simulator_assets/rm65_table_pick_place.xml"
    if not sim_path.is_file():
        raise SystemExit(f"runtime scene not found: {sim_path}")

    rollout_dir = package_dir / "rollouts" / time.strftime("%Y%m%d_%H%M%S")
    rollout_dir.mkdir(parents=True, exist_ok=True)
    result = run_rollout(
        sim_path,
        rollout_dir,
        fps=args.fps,
        width=args.width,
        height=args.height,
        render=not args.no_render,
    )

    artifacts = {}
    for key, path in result["artifact_paths"].items():
        if Path(path).is_file():
            artifacts[key] = store.add_artifact(task_id=None, name=Path(path).name, path=Path(path), kind=artifact_kind(Path(path)))
    update_manifest(manifest_path, rollout_dir, result, artifacts)
    print(json.dumps({"ok": result["ok"], "scene_id": args.scene_id, "rollout_dir": str(rollout_dir), "summary": result["summary"], "artifacts": result["artifact_paths"]}, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


def run_rollout(
    sim_path: Path,
    out_dir: Path,
    *,
    fps: int,
    width: int,
    height: int,
    render: bool,
) -> dict[str, Any]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(sim_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    cube_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pick_cube_free")
    cube_qadr = int(model.jnt_qposadr[cube_joint])
    cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pick_cube_geom")
    finger_geoms = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad_z_axis"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad_z_axis"),
    }
    ctrl_ranges = model.actuator_ctrlrange.copy()
    gripper_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_Left_1_Joint")
    gripper_qadr = int(model.jnt_qposadr[gripper_joint])
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i}") for i in range(1, 7)]
    qpos_adrs = [int(model.jnt_qposadr[jid]) for jid in joint_ids]
    dof_adrs = [int(model.jnt_dofadr[jid]) for jid in joint_ids]

    duration_scale = 8
    phases = [
        ("home", np.array([0.42, 0.00, 0.72]), 0.0, 35 * duration_scale, False),
        ("approach", np.array([0.71, 0.015, 0.42]), 0.0, 55 * duration_scale, False),
        ("descend", np.array([0.71, 0.015, 0.24]), 0.0, 40 * duration_scale, False),
        ("grasp", np.array([0.71, 0.015, 0.24]), -0.91, 45 * duration_scale, True),
        ("lift", np.array([0.71, 0.015, 0.60]), -0.91, 200 * duration_scale, True),
        ("place", np.array([0.43, 0.18, 0.40]), -0.91, 65 * duration_scale, True),
        ("release", np.array([0.43, 0.18, 0.40]), 0.0, 35 * duration_scale, False),
        ("retreat", np.array([0.42, 0.00, 0.72]), 0.0, 45 * duration_scale, False),
    ]

    trajectory = []
    frames = []
    renderer = None
    if render:
        renderer = mujoco.Renderer(model, height=height, width=width)

    contact_steps = 0
    bilateral_contact_steps = 0
    phase_contacts = {phase: {"contact_steps": 0, "bilateral_contact_steps": 0} for phase, *_ in phases}
    contact_force_metrics = {"max_normal": 0.0, "max_tangent": 0.0, "max_normal_z": 0.0, "samples": 0}
    gripper_qpos_range = [float(data.qpos[gripper_qadr]), float(data.qpos[gripper_qadr])]
    max_cube_z = float(data.qpos[cube_qadr + 2])
    initial_cube = data.qpos[cube_qadr : cube_qadr + 3].copy()
    settled_cube_z = float(initial_cube[2])
    started = time.time()
    frame_stride = max(1, round((1.0 / model.opt.timestep) / fps))
    step_index = 0
    for phase, target, gripper_ctrl, count, _contact_expected in phases:
        if phase == "grasp":
            settled_cube_z = float(data.qpos[cube_qadr + 2])
            max_cube_z = settled_cube_z
        q_target = solve_ik(model, data, site_id, target, qpos_adrs, dof_adrs, ctrl_ranges)
        for local in range(count):
            alpha = (local + 1) / count
            current = data.ctrl[:6].copy()
            data.ctrl[:6] = current * (1 - alpha) + q_target * alpha
            data.ctrl[6] = gripper_ctrl
            mujoco.mj_step(model, data)
            touching_fingers = cube_finger_contacts(data, cube_geom, finger_geoms)
            accumulate_contact_metrics(model, data, cube_geom, finger_geoms, contact_force_metrics)
            if touching_fingers:
                contact_steps += 1
                phase_contacts[phase]["contact_steps"] += 1
            if len(touching_fingers) == 2:
                bilateral_contact_steps += 1
                phase_contacts[phase]["bilateral_contact_steps"] += 1
            gripper_qpos_range[0] = min(gripper_qpos_range[0], float(data.qpos[gripper_qadr]))
            gripper_qpos_range[1] = max(gripper_qpos_range[1], float(data.qpos[gripper_qadr]))
            max_cube_z = max(max_cube_z, float(data.qpos[cube_qadr + 2]))
            if step_index % frame_stride == 0:
                trajectory.append(sample_state(model, data, phase, step_index, site_id, cube_qadr))
                if renderer is not None:
                    renderer.update_scene(data, camera="overview")
                    frames.append(renderer.render().copy())
            step_index += 1

    if renderer is not None:
        renderer.close()

    trajectory_path = out_dir / "rm65_pick_place_trajectory.json"
    report_path = out_dir / "rm65_pick_place_rollout_report.json"
    gif_path = out_dir / "rm65_pick_place_rollout.gif"
    keyframe_path = out_dir / "rm65_pick_place_final.png"
    trajectory_path.write_text(json.dumps({"schema": "real2sim.rm65_pick_place_trajectory.v1", "samples": trajectory}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    artifact_paths = {"trajectory": str(trajectory_path)}
    if frames:
        imageio.mimsave(gif_path, frames, fps=fps)
        imageio.imwrite(keyframe_path, frames[-1])
        artifact_paths["gif"] = str(gif_path)
        artifact_paths["final_frame"] = str(keyframe_path)

    final = trajectory[-1] if trajectory else {}
    final_cube = np.asarray(final.get("cube_pos") or initial_cube, dtype=float)
    target_cube = np.array([0.43, 0.18, 0.31])
    lift_height = max_cube_z - settled_cube_z
    place_error = float(np.linalg.norm(final_cube - target_cube))
    physical_success = bilateral_contact_steps > 0 and lift_height >= 0.10 and place_error <= 0.12
    summary = {
        "mode": "scripted_ik_physical_contact",
        "steps": step_index,
        "samples": len(trajectory),
        "sim_time": float(data.time),
        "final_gripper": final.get("gripper_pos"),
        "final_cube": final.get("cube_pos"),
        "initial_cube": [float(x) for x in initial_cube],
        "settled_cube_z": settled_cube_z,
        "max_cube_z": max_cube_z,
        "lift_height": lift_height,
        "place_target": [float(x) for x in target_cube],
        "place_error": place_error,
        "contact_steps": contact_steps,
        "bilateral_contact_steps": bilateral_contact_steps,
        "phase_contacts": phase_contacts,
        "gripper_qpos_range": gripper_qpos_range,
        "contact_force_metrics": contact_force_metrics,
        "physical_success": physical_success,
        "elapsed_ms": round((time.time() - started) * 1000),
    }
    report = {
        "schema": "real2sim.rm65_pick_place_rollout_report.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ok": physical_success,
        "runtime": "mujoco",
        "scene": str(sim_path),
        "summary": summary,
        "phases": [phase for phase, *_ in phases],
        "note": "Scripted IK with contact-physics grasp; the cube pose is never overwritten. This is not a learned policy.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    artifact_paths["report"] = str(report_path)
    return {"ok": bool(report["ok"]), "summary": summary, "artifact_paths": artifact_paths}


def cube_finger_contacts(data: Any, cube_geom: int, finger_geoms: set[int]) -> set[int]:
    touching: set[int] = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if cube_geom not in pair:
            continue
        touching.update(pair & finger_geoms)
    return touching


def accumulate_contact_metrics(model: Any, data: Any, cube_geom: int, finger_geoms: set[int], metrics: dict[str, Any]) -> None:
    import mujoco

    for index in range(data.ncon):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if cube_geom not in pair or not (pair & finger_geoms):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        metrics["max_normal"] = max(float(metrics["max_normal"]), abs(float(force[0])))
        metrics["max_tangent"] = max(float(metrics["max_tangent"]), float(np.linalg.norm(force[1:3])))
        metrics["max_normal_z"] = max(float(metrics["max_normal_z"]), abs(float(contact.frame[2])))
        metrics["samples"] = int(metrics["samples"]) + 1


def solve_ik(model: Any, data: Any, site_id: int, target: np.ndarray, qpos_adrs: list[int], dof_adrs: list[int], ctrl_ranges: np.ndarray) -> np.ndarray:
    import mujoco

    q_backup = data.qpos.copy()
    q = data.qpos[qpos_adrs].copy()
    damping = 1e-3
    for _ in range(80):
        data.qpos[qpos_adrs] = q
        mujoco.mj_forward(model, data)
        err = target - data.site_xpos[site_id]
        if np.linalg.norm(err) < 0.012:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        j = jacp[:, dof_adrs]
        dq = j.T @ np.linalg.solve(j @ j.T + damping * np.eye(3), err)
        q = q + np.clip(dq, -0.08, 0.08)
        for i in range(6):
            q[i] = float(np.clip(q[i], ctrl_ranges[i, 0], ctrl_ranges[i, 1]))
    data.qpos[:] = q_backup
    mujoco.mj_forward(model, data)
    return q


def sample_state(model: Any, data: Any, phase: str, step: int, site_id: int, cube_qadr: int) -> dict[str, Any]:
    return {
        "step": step,
        "time": float(data.time),
        "phase": phase,
        "qpos_arm": [float(x) for x in data.qpos[:6]],
        "gripper_ctrl": float(data.ctrl[6]),
        "gripper_pos": [float(x) for x in data.site_xpos[site_id]],
        "cube_pos": [float(x) for x in data.qpos[cube_qadr : cube_qadr + 3]],
    }


def update_manifest(manifest_path: Path, rollout_dir: Path, result: dict[str, Any], artifacts: dict[str, dict[str, Any]]) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    new_assets = []
    for key, artifact in artifacts.items():
        path = Path(artifact["path"])
        new_assets.append(
            {
                "name": path.name,
                "role": f"rollout_{key}",
                "kind": artifact_kind(path),
                "path": str(path),
                "relative_path": str(path.relative_to(manifest_path.parent)),
                "url": f"/api/artifacts/{artifact['id']}/content",
                "exists": True,
                "size_bytes": path.stat().st_size,
                "size_label": format_bytes(path.stat().st_size),
                "generated": True,
            }
        )
    old_assets = [item for item in manifest.get("assets", []) if not (isinstance(item, dict) and str(item.get("role", "")).startswith("rollout_"))]
    manifest["assets"] = [*new_assets, *old_assets]
    manifest["latest_rollout"] = {
        "dir": str(rollout_dir),
        "summary": result["summary"],
        "artifacts": new_assets,
        "status": "scripted_rollout_succeeded",
    }
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        return "image"
    if suffix == ".json":
        return "metadata"
    return suffix.lstrip(".") or "file"


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
