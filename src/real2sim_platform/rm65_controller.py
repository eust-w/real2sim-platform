from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EpisodeResult:
    success: bool
    lift_height: float
    place_error: float
    bilateral_contact_steps: int
    observations: list[np.ndarray]
    actions: list[np.ndarray]
    contacts: list[np.ndarray]
    images: list[np.ndarray]
    trajectory: list[dict[str, Any]]
    initial_cube: list[float]
    target: list[float]


class RM65PickPlaceRuntime:
    def __init__(self, scene_path: Path, *, width: int, height: int, seed: int) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.rng = np.random.default_rng(seed)
        self.site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "gripper")
        self.cube_joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, "pick_cube_free")
        self.cube_qadr = int(self.model.jnt_qposadr[self.cube_joint])
        self.cube_geom = self._id(mujoco.mjtObj.mjOBJ_GEOM, "pick_cube_geom")
        self.finger_geoms = {
            self._id(mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad_z_axis"),
            self._id(mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad_z_axis"),
        }
        self.target_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "place_target")
        self.gripper_joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, "gripper_Left_1_Joint")
        self.gripper_qadr = int(self.model.jnt_qposadr[self.gripper_joint])
        self.joint_ids = [self._id(mujoco.mjtObj.mjOBJ_JOINT, f"joint_{index}") for index in range(1, 7)]
        self.qpos_adrs = [int(self.model.jnt_qposadr[joint]) for joint in self.joint_ids]
        self.dof_adrs = [int(self.model.jnt_dofadr[joint]) for joint in self.joint_ids]
        self.ctrl_ranges = self.model.actuator_ctrlrange.copy()

    def _id(self, kind: Any, name: str) -> int:
        value = self.mujoco.mj_name2id(self.model, kind, name)
        if value < 0:
            raise ValueError(f"required RM65 scene object is missing: {name}")
        return int(value)

    def close(self) -> None:
        self.renderer.close()

    def run_episode(self, *, max_steps: int) -> EpisodeResult:
        mujoco = self.mujoco
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        cube = np.array([self.rng.uniform(0.645, 0.655), self.rng.uniform(-0.010, 0.010), 0.33], dtype=float)
        target = np.array([self.rng.uniform(0.42, 0.51), self.rng.uniform(0.13, 0.22), 0.31], dtype=float)
        self.data.qpos[self.cube_qadr : self.cube_qadr + 3] = cube
        self.data.qpos[self.cube_qadr + 3 : self.cube_qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[self.model.jnt_dofadr[self.cube_joint] : self.model.jnt_dofadr[self.cube_joint] + 6] = 0
        self.model.body_pos[self.target_body] = np.array([target[0], target[1], 0.335])
        mujoco.mj_forward(self.model, self.data)
        for _ in range(120):
            mujoco.mj_step(self.model, self.data)
        settled_z = float(self.data.qpos[self.cube_qadr + 2])

        # The named site is about 3 cm above the cube center at bilateral pad contact.
        grasp_site = np.array([cube[0] - 0.014, cube[1] - 0.020, settled_z + 0.020])
        # Contact-calibrated negative-wrist branch for the packaged RM65 gripper.
        # This seed is a robot calibration pose, not an object-pose override.
        seed_q = np.array([-1.2468, -0.1011, 1.8666, -1.9230, -1.5982, 1.5636], dtype=float)
        q_backup = self.data.qpos.copy()
        self.data.qpos[self.qpos_adrs] = seed_q
        mujoco.mj_forward(self.model, self.data)
        grasp_orientation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        self.data.qpos[:] = q_backup
        mujoco.mj_forward(self.model, self.data)

        phases = [
            ("approach", grasp_site + np.array([0.0, 0.0, 0.20]), 0.0, 24),
            ("descend", grasp_site, 0.0, 24),
            ("grasp", grasp_site, -0.28, 30),
            ("lift", grasp_site + np.array([0.0, 0.0, 0.20]), -0.28, 60),
            ("transfer", target + np.array([-0.040, 0.0, 0.21]), -0.28, 34),
            ("place", target + np.array([-0.040, 0.0, 0.02]), -0.28, 26),
            ("release", target + np.array([-0.040, 0.0, 0.02]), 0.0, 18),
            ("retreat", target + np.array([-0.040, 0.0, 0.20]), 0.0, 20),
        ]
        scale = max(0.25, max_steps / sum(phase[3] for phase in phases))
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        contacts: list[np.ndarray] = []
        images: list[np.ndarray] = []
        trajectory: list[dict[str, Any]] = []
        bilateral_contact_steps = 0
        max_cube_z = settled_z
        step_index = 0
        for phase, position, gripper, nominal_steps in phases:
            target_q = self.solve_ik_6d(position, grasp_orientation, seed_q=seed_q)
            phase_steps = max(2, int(round(nominal_steps * scale)))
            start_ctrl = self.data.ctrl[:6].copy()
            for local_step in range(phase_steps):
                alpha = _smoothstep((local_step + 1) / phase_steps)
                action = np.empty(7, dtype=np.float32)
                action[:6] = start_ctrl * (1 - alpha) + target_q * alpha
                action[6] = gripper
                self.data.ctrl[:7] = action
                # The RM65 model has conservative actuator force limits; each policy
                # command is held long enough for the physical joints to settle.
                for _ in range(35):
                    mujoco.mj_step(self.model, self.data)
                touching = self.finger_contacts()
                bilateral_contact_steps += int(len(touching) == 2)
                max_cube_z = max(max_cube_z, float(self.data.qpos[self.cube_qadr + 2]))
                self.renderer.update_scene(self.data, camera="overview")
                frame = self.renderer.render().copy()
                observation = np.concatenate([self.data.qpos[self.qpos_adrs], [self.data.qpos[self.gripper_qadr]]]).astype(np.float32)
                observations.append(observation)
                actions.append(action.copy())
                contacts.append(np.array([int(geom in touching) for geom in sorted(self.finger_geoms)], dtype=np.uint8))
                images.append(frame)
                trajectory.append(
                    {
                        "step": step_index,
                        "time": float(self.data.time),
                        "phase": phase,
                        "qpos": observation.tolist(),
                        "action": action.tolist(),
                        "cube_pos": self.data.qpos[self.cube_qadr : self.cube_qadr + 3].astype(float).tolist(),
                        "gripper_pos": self.data.site_xpos[self.site_id].astype(float).tolist(),
                        "bilateral_contact": len(touching) == 2,
                        "touching_fingers": sorted(int(item) for item in touching),
                        "finger_positions": [self.data.geom_xpos[item].astype(float).tolist() for item in sorted(self.finger_geoms)],
                    }
                )
                step_index += 1

        final_cube = self.data.qpos[self.cube_qadr : self.cube_qadr + 3].copy()
        lift_height = float(max_cube_z - settled_z)
        place_error = float(np.linalg.norm(final_cube - target))
        success = bilateral_contact_steps > 0 and lift_height >= 0.10 and place_error <= 0.05
        return EpisodeResult(
            success=success,
            lift_height=lift_height,
            place_error=place_error,
            bilateral_contact_steps=bilateral_contact_steps,
            observations=observations,
            actions=actions,
            contacts=contacts,
            images=images,
            trajectory=trajectory,
            initial_cube=cube.tolist(),
            target=target.tolist(),
        )

    def finger_contacts(self) -> set[int]:
        touching: set[int] = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.cube_geom in pair:
                touching.update(pair & self.finger_geoms)
        return touching

    def solve_position_ik(self, target_position: np.ndarray) -> np.ndarray:
        q_backup = self.data.qpos.copy()
        q = self.data.qpos[self.qpos_adrs].copy()
        for _ in range(120):
            self.data.qpos[self.qpos_adrs] = q
            self.mujoco.mj_forward(self.model, self.data)
            error = target_position - self.data.site_xpos[self.site_id]
            if np.linalg.norm(error) < 0.004:
                break
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            self.mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            jacobian = jacp[:, self.dof_adrs]
            q += np.clip(_damped_least_squares(jacobian, error, 2e-3), -0.06, 0.06)
            q = self._clip_arm(q)
        self.data.qpos[:] = q_backup
        self.mujoco.mj_forward(self.model, self.data)
        return q

    def solve_ik_6d(self, target_position: np.ndarray, target_rotation: np.ndarray, *, seed_q: np.ndarray | None = None) -> np.ndarray:
        q_backup = self.data.qpos.copy()
        q = seed_q.copy() if seed_q is not None else self.data.qpos[self.qpos_adrs].copy()
        for _ in range(160):
            self.data.qpos[self.qpos_adrs] = q
            self.mujoco.mj_forward(self.model, self.data)
            position_error = target_position - self.data.site_xpos[self.site_id]
            current_rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
            rotation_error = _rotation_vector(target_rotation @ current_rotation.T)
            error = np.concatenate([position_error, 0.45 * rotation_error])
            if np.linalg.norm(position_error) < 0.005 and np.linalg.norm(rotation_error) < 0.04:
                break
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            self.mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            jacobian = np.vstack([jacp[:, self.dof_adrs], 0.45 * jacr[:, self.dof_adrs]])
            q += np.clip(_damped_least_squares(jacobian, error, 4e-3), -0.05, 0.05)
            q = self._clip_arm(q)
        self.data.qpos[:] = q_backup
        self.mujoco.mj_forward(self.model, self.data)
        return q

    def _clip_arm(self, q: np.ndarray) -> np.ndarray:
        return np.array([np.clip(q[index], self.ctrl_ranges[index, 0], self.ctrl_ranges[index, 1]) for index in range(6)])


def collect(scene_path: Path, out_dir: Path, *, required_successes: int, max_attempts: int, max_steps: int, seed: int, width: int, height: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime = RM65PickPlaceRuntime(scene_path, width=width, height=height, seed=seed)
    reports: list[dict[str, Any]] = []
    success_count = 0
    try:
        for attempt in range(max_attempts):
            result = runtime.run_episode(max_steps=max_steps)
            episode_report = {
                "attempt": attempt,
                "success": result.success,
                "lift_height": result.lift_height,
                "place_error": result.place_error,
                "bilateral_contact_steps": result.bilateral_contact_steps,
                "initial_cube": result.initial_cube,
                "target": result.target,
            }
            reports.append(episode_report)
            trajectory_path = out_dir / f"attempt_{attempt:04d}_trajectory.json"
            trajectory_path.write_text(json.dumps({"schema": "real2sim.rm65.trajectory.v2", "report": episode_report, "samples": result.trajectory}, ensure_ascii=False) + "\n", encoding="utf-8")
            if result.success:
                _write_hdf5_episode(out_dir / f"episode_{success_count}.hdf5", result)
                if success_count < 3:
                    _write_preview_video(out_dir / f"episode_{success_count}.mp4", result.images)
                success_count += 1
                if success_count >= required_successes:
                    break
    finally:
        runtime.close()
    manifest = {
        "schema": "real2sim.discoverse_act_dataset.v1",
        "controller": "rm65_pick_place_6d_ik",
        "runtime": "mujoco",
        "scene": str(scene_path),
        "camera_names": ["overview"],
        "state_dim": 7,
        "action_dim": 7,
        "required_successes": required_successes,
        "successful_episodes": success_count,
        "attempts": len(reports),
        "success_rate": success_count / max(1, len(reports)),
        "success_definition": {"minimum_lift_m": 0.10, "maximum_place_error_m": 0.05, "bilateral_contact_required": True},
        "episodes": reports,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _write_hdf5_episode(path: Path, result: EpisodeResult) -> None:
    import cv2
    import h5py

    observations = np.asarray(result.observations, dtype=np.float32)
    actions = np.asarray(result.actions, dtype=np.float32)
    contacts = np.asarray(result.contacts, dtype=np.uint8)
    encoded = []
    for image in result.images:
        ok, buffer = cv2.imencode(".jpg", image[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            raise RuntimeError("failed to encode collection frame")
        encoded.append(buffer.astype(np.uint8))
    with h5py.File(path, "w") as root:
        root.attrs["sim"] = True
        root.attrs["compress"] = True
        root.attrs["success"] = bool(result.success)
        root.create_dataset("/observations/qpos", data=observations, dtype="f4")
        root.create_dataset("/action", data=actions, dtype="f4")
        root.create_dataset("/observations/contact", data=contacts, dtype="u1")
        root.create_dataset("/success", data=np.full(len(actions), result.success, dtype=np.uint8))
        variable = h5py.vlen_dtype(np.dtype("uint8"))
        dataset = root.create_dataset("/observations/images/overview", (len(encoded),), dtype=variable)
        for index, value in enumerate(encoded):
            dataset[index] = value


def _write_preview_video(path: Path, images: list[np.ndarray]) -> None:
    if not images:
        raise ValueError("cannot encode an empty rollout")
    height, width = images[0].shape[:2]
    ffmpeg = os.environ.get("REAL2SIM_FFMPEG_BIN", "").strip() or shutil.which("ffmpeg") or "ffmpeg"
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        "20",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except OSError as exc:
        raise RuntimeError(f"ffmpeg rollout encoder could not start {ffmpeg}: {exc}") from exc
    assert process.stdin is not None
    write_error: BrokenPipeError | None = None
    try:
        for image in images:
            process.stdin.write(np.ascontiguousarray(image, dtype=np.uint8).tobytes())
    except BrokenPipeError as exc:
        write_error = exc
    try:
        process.stdin.close()
    except BrokenPipeError as exc:
        write_error = write_error or exc
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if write_error or return_code != 0 or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg rollout encoding failed with {ffmpeg} (exit {return_code}): {stderr[-2000:]}") from write_error


def _damped_least_squares(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    identity = np.eye(jacobian.shape[0])
    return jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * identity, error)


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-7:
        return np.zeros(3)
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    denominator = 2.0 * math.sin(angle)
    return axis * (angle / denominator)


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect physical RM65 pick-place episodes for DISCOVERSE ACT")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--required-successes", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=150)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    args = parser.parse_args()
    manifest = collect(
        args.scene,
        args.out,
        required_successes=args.required_successes,
        max_attempts=args.max_attempts,
        max_steps=args.steps,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["successful_episodes"] >= manifest["required_successes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
