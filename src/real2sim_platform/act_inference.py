from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .rm65_controller import ACTION_ALIGNMENT, CONTROL_HOLD_STEPS, RM65PickPlaceRuntime, _write_preview_video


def load_policy(model_dir: Path, act_root: Path, max_timesteps: int):
    import torch

    sys.path.insert(0, str(act_root))
    from configurations.task_configs.config_tools.basic_configer import config_policy

    config_path = model_dir / "task_config.py"
    checkpoint = model_dir / "core" / "policy_best.ckpt"
    stats_path = model_dir / "core" / "dataset_stats.pkl"
    for path in (config_path, checkpoint, stats_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("real2sim_act_task_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load ACT task config")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    task_config = module.TASK_CONFIG
    merged = {**task_config["common"], **task_config["train"]}
    merged["learning_rate"] = task_config["train"]["learning_rate"]
    policy_config = config_policy(merged)
    policy_config["ckpt_path"] = str(checkpoint)
    policy_config["max_timesteps"] = max_timesteps
    policy = module.policy_maker(policy_config, "eval")
    with stats_path.open("rb") as handle:
        stats = pickle.load(handle)
    if not all(key in stats for key in ("qpos_mean", "qpos_std", "action_mean", "action_std")):
        raise ValueError("ACT dataset_stats.pkl is missing normalization arrays")
    return policy, stats, task_config["common"]["camera_names"], torch


def infer(
    *,
    scene_path: Path,
    model_dir: Path,
    act_root: Path,
    out_dir: Path,
    episodes: int,
    steps: int,
    seed: int,
    minimum_success_rate: float,
    width: int,
    height: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    policy, stats, camera_names, torch = load_policy(model_dir, act_root, steps)
    if camera_names != ["overview"]:
        raise ValueError(f"RM65 runtime currently exposes camera ['overview']; model expects {camera_names}")
    runtime = RM65PickPlaceRuntime(scene_path, width=width, height=height, seed=seed)
    results: list[dict[str, Any]] = []
    all_actions: list[np.ndarray] = []
    try:
        for episode in range(episodes):
            policy.reset()
            cube, target, settled_z = _reset_randomized(runtime)
            trajectory: list[dict[str, Any]] = []
            frames: list[np.ndarray] = []
            max_cube_z = settled_z
            bilateral_steps = 0
            for step in range(steps):
                frame = runtime.capture_frame()
                qpos = runtime.joint_state()
                normalized_qpos = (qpos - np.asarray(stats["qpos_mean"], dtype=np.float32)) / np.asarray(stats["qpos_std"], dtype=np.float32)
                image = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
                qpos_tensor = torch.from_numpy(normalized_qpos).unsqueeze(0).float().cuda()
                with torch.inference_mode():
                    predicted = policy(qpos_tensor, image)
                normalized_action = predicted[0, 0].detach().cpu().numpy()
                action = normalized_action * np.asarray(stats["action_std"], dtype=np.float32) + np.asarray(stats["action_mean"], dtype=np.float32)
                action = np.asarray(action, dtype=np.float32)
                if action.shape != (7,):
                    raise ValueError(f"ACT action shape must be (7,), received {action.shape}")
                action = runtime.clip_action(action)
                runtime.apply_action(action)
                touching = runtime.finger_contacts()
                bilateral_steps += int(len(touching) == 2)
                cube_position = runtime.cube_position()
                max_cube_z = max(max_cube_z, float(cube_position[2]))
                all_actions.append(action.copy())
                frames.append(frame)
                trajectory.append(
                    {
                        "step": step,
                        "time": runtime.simulation_time,
                        "qpos": qpos.tolist(),
                        "action": action.tolist(),
                        "cube_pos": cube_position.tolist(),
                        "target": target.tolist(),
                        "bilateral_contact": len(touching) == 2,
                    }
                )
            final_cube = runtime.cube_position()
            lift_height = float(max_cube_z - settled_z)
            place_error = float(np.linalg.norm(final_cube - target))
            success = bilateral_steps > 0 and lift_height >= 0.10 and place_error <= 0.05
            result = {
                "episode": episode,
                "success": success,
                "lift_height": lift_height,
                "place_error": place_error,
                "bilateral_contact_steps": bilateral_steps,
                "initial_cube": cube.tolist(),
                "target": target.tolist(),
            }
            results.append(result)
            (out_dir / f"episode_{episode:03d}_trajectory.json").write_text(json.dumps({"schema": "real2sim.act_rollout.v1", "result": result, "samples": trajectory}, ensure_ascii=False) + "\n", encoding="utf-8")
            _write_preview_video(out_dir / f"episode_{episode:03d}.mp4", frames)
    finally:
        runtime.close()
    success_rate = sum(int(item["success"]) for item in results) / max(1, len(results))
    action_matrix = np.asarray(all_actions, dtype=np.float32)
    action_std = action_matrix.std(axis=0).tolist() if len(action_matrix) else []
    nonconstant = bool(len(action_matrix) > 1 and np.max(np.std(action_matrix, axis=0)) > 1e-5)
    report = {
        "schema": "real2sim.discoverse_act_inference.v1",
        "runtime": "motrixsim",
        "scene": str(scene_path),
        "model": str(model_dir),
        "episodes": results,
        "success_rate": success_rate,
        "minimum_success_rate": minimum_success_rate,
        "action_alignment": ACTION_ALIGNMENT,
        "control_hold_steps": CONTROL_HOLD_STEPS,
        "action_std": action_std,
        "actions_nonconstant": nonconstant,
        "pass": success_rate >= minimum_success_rate and nonconstant,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "inference_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _reset_randomized(runtime: RM65PickPlaceRuntime) -> tuple[np.ndarray, np.ndarray, float]:
    return runtime.reset_randomized()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DISCOVERSE ACT in the physical RM65 MotrixSim scene")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--act-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--minimum-success-rate", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    args = parser.parse_args()
    report = infer(
        scene_path=args.scene,
        model_dir=args.model,
        act_root=args.act_root,
        out_dir=args.out,
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        minimum_success_rate=args.minimum_success_rate,
        width=args.width,
        height=args.height,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
