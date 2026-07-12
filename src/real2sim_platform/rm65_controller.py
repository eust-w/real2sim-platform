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


CONTROL_HOLD_STEPS = 35
ACTION_ALIGNMENT = "observation_t_to_action_t"


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
        import motrixsim
        from motrixsim import ik
        from motrixsim.render import RenderApp

        self.motrixsim = motrixsim
        self.model = motrixsim.load_model(str(scene_path))
        self.data = motrixsim.SceneData(self.model)
        self.rng = np.random.default_rng(seed)
        self.gripper_site = self._require(self.model.get_site("gripper"), "site gripper")
        self.finger_sites = [
            self._require(self.model.get_site("left_finger_pad"), "site left_finger_pad"),
            self._require(self.model.get_site("right_finger_pad"), "site right_finger_pad"),
        ]
        self.cube_body = self._require(self.model.get_body("pick_cube"), "body pick_cube")
        self.cube_floating_base = self._require(self.cube_body.floatingbase, "pick_cube floating base")
        self.cube_geom = self._require_index(self.model.get_geom_index("pick_cube_geom"), "geom pick_cube_geom")
        self.finger_geoms = [
            self._finger_collision_geom("gripper_Left_Support_Link"),
            self._finger_collision_geom("gripper_Right_Support_Link"),
        ]
        self.target_body = self._require(self.model.get_body("place_target"), "body place_target")
        self.joints = [self._require(self.model.get_joint(f"joint_{index}"), f"joint_{index}") for index in range(1, 7)]
        self.gripper_joint = self._require(self.model.get_joint("gripper_Left_1_Joint"), "joint gripper_Left_1_Joint")
        self.actuators = [self._require(self.model.get_actuator(index), f"actuator {index}") for index in range(7)]
        ctrl_ranges = np.asarray(self.model.actuator_ctrl_limits, dtype=float)
        if ctrl_ranges.shape == (2, self.model.num_actuators):
            ctrl_ranges = ctrl_ranges.T
        if ctrl_ranges.shape != (self.model.num_actuators, 2):
            raise ValueError(f"MotrixSim actuator control limits have an unexpected shape: {ctrl_ranges.shape}")
        self.ctrl_ranges = ctrl_ranges.copy()
        self.home_keyframe = next((item for item in self.model.keyframes if item.name == "home"), self.model.keyframes[0] if self.model.keyframes else None)
        self._reset_home()
        self.target_home = np.asarray(self.target_body.get_position(self.data), dtype=float).copy()
        offset = [*np.asarray(self.gripper_site.local_pos, dtype=float), *np.asarray(self.gripper_site.local_quat, dtype=float)]
        self.ik_chain = ik.IkChain(
            self.model,
            end_link=self.gripper_site.parent_link.name,
            start_link="base_link",
            end_effector_offset=offset,
        )
        self.ik_solver = ik.DlsSolver(max_iter=200, step_size=0.4, tolerance=1e-3, damping=4e-3)
        camera_index = self.model.cameras.get_index("overview")
        self.model.cameras[camera_index].set_render_target("image", width, height)
        self._render_context = RenderApp(headless=True)
        self.renderer = self._render_context.__enter__()
        self.renderer.launch(self.model)
        self.render_camera = self.renderer.get_camera(camera_index)
        self.renderer.sync(self.data, wait=True)
        self.simulation_time = 0.0

    @staticmethod
    def _require(value: Any, description: str) -> Any:
        if value is None:
            raise ValueError(f"required RM65 scene object is missing: {description}")
        return value

    @staticmethod
    def _require_index(value: Any, description: str) -> int:
        if value is None:
            raise ValueError(f"required RM65 scene object is missing: {description}")
        return int(value)

    def _finger_collision_geom(self, link_name: str) -> int:
        candidates = [
            geom
            for geom in self.model.geoms
            if getattr(geom.link, "name", None) == link_name and int(geom.collision_group) != 0 and int(geom.collision_affinity) != 0
        ]
        if len(candidates) != 1:
            raise ValueError(f"expected one collidable finger geom on {link_name}, received {len(candidates)}")
        return int(candidates[0].index)

    def _reset_home(self) -> None:
        self.data.reset(self.model)
        if self.home_keyframe is not None:
            self.home_keyframe.apply(self.data)
        self.model.forward_kinematic(self.data)

    def close(self) -> None:
        self._render_context.__exit__(None, None, None)

    def reset_randomized(self) -> tuple[np.ndarray, np.ndarray, float]:
        self._reset_home()
        cube_home = np.asarray(self.cube_floating_base.get_translation(self.data), dtype=float).copy()
        cube = cube_home + np.array([self.rng.uniform(-0.010, 0.010), self.rng.uniform(-0.010, 0.010), 0.0], dtype=float)
        self.cube_floating_base.set_translation(self.data, cube)
        self.cube_floating_base.set_rotation(self.data, np.array([0.0, 0.0, 0.0, 1.0], dtype=float))
        self.cube_floating_base.set_global_linear_velocity(self.data, np.zeros(3, dtype=float))
        self.cube_floating_base.set_global_angular_velocity(self.data, np.zeros(3, dtype=float))
        target = np.array(
            [
                self.target_home[0] + self.rng.uniform(-0.025, 0.025),
                self.target_home[1] + self.rng.uniform(-0.025, 0.025),
                self.target_home[2] - 0.025,
            ],
            dtype=float,
        )
        if self.target_body.mocap is not None:
            self.target_body.mocap.set_pose(
                self.data,
                np.array([target[0], target[1], self.target_home[2], 0.0, 0.0, 0.0, 1.0], dtype=float),
            )
        else:
            target[:2] = self.target_home[:2]
        self.model.forward_kinematic(self.data)
        self.model.step_n(self.data, 120)
        self.simulation_time = 120 * float(self.model.options.timestep)
        self.renderer.sync(self.data, wait=True)
        return cube, target, float(self.cube_position()[2])

    def capture_frame(self) -> np.ndarray:
        capture = self.render_camera.capture()
        self.renderer.sync(self.data, wait=True)
        image = capture.take_image()
        if image is None:
            raise RuntimeError("MotrixSim overview camera did not return an RGB frame")
        pixels = np.asarray(image.pixels, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] < 3:
            raise RuntimeError(f"MotrixSim overview camera returned invalid pixels: {pixels.shape}")
        return np.ascontiguousarray(pixels[:, :, :3]).copy()

    def joint_state(self) -> np.ndarray:
        values = [float(joint.get_dof_pos(self.data)[0]) for joint in self.joints]
        values.append(float(self.gripper_joint.get_dof_pos(self.data)[0]))
        return np.asarray(values, dtype=np.float32)

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).copy()
        for index in range(7):
            action[index] = np.clip(action[index], self.ctrl_ranges[index, 0], self.ctrl_ranges[index, 1])
        return action

    def apply_action(self, action: np.ndarray, *, hold_steps: int = CONTROL_HOLD_STEPS) -> None:
        action = self.clip_action(action)
        for actuator, value in zip(self.actuators, action):
            actuator.set_ctrl(self.data, float(value))
        self.model.step_n(self.data, hold_steps)
        self.simulation_time += hold_steps * float(self.model.options.timestep)

    def cube_position(self) -> np.ndarray:
        return np.asarray(self.cube_body.get_position(self.data), dtype=float).copy()

    def gripper_position(self) -> np.ndarray:
        return np.asarray(self.gripper_site.get_position(self.data), dtype=float).copy()

    def run_episode(self, *, max_steps: int) -> EpisodeResult:
        cube, target, settled_z = self.reset_randomized()

        # The named site is about 3 cm above the cube center at bilateral pad contact.
        grasp_site = np.array([cube[0] - 0.014, cube[1] - 0.020, settled_z + 0.020])
        # Contact-calibrated negative-wrist branch for the packaged RM65 gripper.
        # This seed is a robot calibration pose, not an object-pose override.
        seed_q = np.array([-1.2468, -0.1011, 1.8666, -1.9230, -1.5982, 1.5636], dtype=float)
        q_backup = np.asarray(self.data.dof_pos, dtype=float).copy()
        for joint, value in zip(self.joints, seed_q):
            joint.set_dof_pos(self.data, np.array([value], dtype=float))
        self.model.forward_kinematic(self.data)
        grasp_orientation = np.asarray(self.gripper_site.get_pose(self.data), dtype=float)[3:7].copy()
        self.data.set_dof_pos(q_backup, self.model)
        self.model.forward_kinematic(self.data)

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
            start_ctrl = np.asarray(self.data.actuator_ctrls[:6], dtype=float).copy()
            for local_step in range(phase_steps):
                alpha = _smoothstep((local_step + 1) / phase_steps)
                action = np.empty(7, dtype=np.float32)
                action[:6] = start_ctrl * (1 - alpha) + target_q * alpha
                action[6] = gripper
                frame = self.capture_frame()
                observation = self.joint_state()
                cube_before = self.cube_position()
                self.apply_action(action)
                touching = self.finger_contacts()
                bilateral_contact_steps += int(len(touching) == 2)
                cube_after = self.cube_position()
                max_cube_z = max(max_cube_z, float(cube_after[2]))
                observations.append(observation)
                actions.append(action.copy())
                contacts.append(np.array([int(geom in touching) for geom in sorted(self.finger_geoms)], dtype=np.uint8))
                images.append(frame)
                trajectory.append(
                    {
                        "step": step_index,
                        "time": self.simulation_time,
                        "phase": phase,
                        "qpos": observation.tolist(),
                        "action": action.tolist(),
                        "action_alignment": ACTION_ALIGNMENT,
                        "cube_pos_before": cube_before.tolist(),
                        "cube_pos": cube_after.tolist(),
                        "gripper_pos": self.gripper_position().tolist(),
                        "bilateral_contact": len(touching) == 2,
                        "touching_fingers": sorted(int(item) for item in touching),
                        "finger_positions": [np.asarray(site.get_position(self.data), dtype=float).tolist() for site in self.finger_sites],
                    }
                )
                step_index += 1

        final_cube = self.cube_position()
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
        pairs = np.asarray([[self.cube_geom, geom] for geom in self.finger_geoms], dtype=np.uint32)
        active = np.asarray(self.model.get_contact_query(self.data).is_colliding(pairs), dtype=bool)
        return {geom for geom, colliding in zip(self.finger_geoms, active) if bool(colliding)}

    def solve_ik_6d(self, target_position: np.ndarray, target_quaternion: np.ndarray, *, seed_q: np.ndarray | None = None) -> np.ndarray:
        q_backup = np.asarray(self.data.dof_pos, dtype=float).copy()
        if seed_q is not None:
            for joint, value in zip(self.joints, seed_q):
                joint.set_dof_pos(self.data, np.array([value], dtype=float))
            self.model.forward_kinematic(self.data)
        target_pose = np.concatenate([np.asarray(target_position, dtype=float), np.asarray(target_quaternion, dtype=float)])
        solution = np.asarray(self.ik_solver.solve(self.ik_chain, self.data, target_pose), dtype=float)
        q = self._clip_arm(solution[2:8])
        self.data.set_dof_pos(q_backup, self.model)
        self.model.forward_kinematic(self.data)
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
        "runtime": "motrixsim",
        "scene": str(scene_path),
        "camera_names": ["overview"],
        "image_encoding": {"codec": "jpeg", "semantic_color_order": "RGB", "opencv_decode_color_order": "BGR"},
        "action_alignment": ACTION_ALIGNMENT,
        "control_hold_steps": CONTROL_HOLD_STEPS,
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
        root.attrs["image_color_order"] = "RGB"
        root.attrs["opencv_decode_color_order"] = "BGR"
        root.attrs["action_alignment"] = ACTION_ALIGNMENT
        root.attrs["control_hold_steps"] = CONTROL_HOLD_STEPS
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
    parser = argparse.ArgumentParser(description="Collect physical RM65 pick-place episodes in MotrixSim for DISCOVERSE ACT")
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
