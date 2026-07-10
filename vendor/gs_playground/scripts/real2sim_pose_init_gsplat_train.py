#!/usr/bin/env python3
"""Train a small 3DGS scene from Real2Sim pose-init artifacts.

The input is the dataset exported by real2sim_export_pose_init_dataset.py:

- transforms.json with registered COLMAP camera-to-world matrices
- images/ with the registered frames
- colmap_sparse_points.ply with sparse SfM points and RGB colors

This is intentionally dependency-light: it only needs torch, gsplat, numpy and
Pillow at runtime. It is used as a real end-to-end training backend when the
full upstream post-optimization stack is not installed on the target machine.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PIL_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
SH_C0 = 0.28209479177387814


@dataclass
class FrameSample:
    image_path: Path
    pixels: np.ndarray
    c2w: np.ndarray
    K: np.ndarray
    width: int
    height: int


def main() -> int:
    parser = argparse.ArgumentParser(description="Train pose-init 3DGS from Real2Sim COLMAP artifacts")
    parser.add_argument("--pose-init-dir", type=Path, required=True, help="Directory containing transforms.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for trained outputs")
    parser.add_argument("--output-ply", type=Path, default=None, help="Output trained 3DGS PLY")
    parser.add_argument("--preview-output", type=Path, default=None, help="Output side-by-side preview PNG")
    parser.add_argument("--multi-preview-output", type=Path, default=None, help="Output multi-frame source/render contact sheet")
    parser.add_argument("--metrics-output", type=Path, default=None, help="Output metrics JSON")
    parser.add_argument("--camera-matrix-key", default="colmap_transform_matrix")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--image-max-size", type=int, default=448)
    parser.add_argument("--max-points", type=int, default=80_000)
    parser.add_argument("--target-gaussians", type=int, default=50_000)
    parser.add_argument("--dense-image-points-per-frame", type=int, default=1600)
    parser.add_argument("--dense-depth-neighbors", type=int, default=4)
    parser.add_argument("--dense-depth-max-distance", type=float, default=48.0)
    parser.add_argument("--adaptive-densify-every", type=int, default=0, help="Add residual-driven Gaussians every N steps; 0 disables")
    parser.add_argument("--adaptive-densify-start", type=int, default=50)
    parser.add_argument("--adaptive-densify-stop", type=int, default=0, help="Last step for adaptive densify; 0 means max-steps")
    parser.add_argument("--adaptive-densify-points", type=int, default=2500)
    parser.add_argument("--adaptive-densify-error-percentile", type=float, default=82.0)
    parser.add_argument("--adaptive-densify-depth-neighbors", type=int, default=4)
    parser.add_argument("--adaptive-densify-max-distance", type=float, default=64.0)
    parser.add_argument("--adaptive-densify-max-projection-points", type=int, default=25_000)
    parser.add_argument("--preview-frames", type=int, default=4)
    parser.add_argument("--trim-outlier-percentile", type=float, default=99.5)
    parser.add_argument("--initial-scale", type=float, default=0.0, help="World-space Gaussian scale; <=0 auto-estimates")
    parser.add_argument("--means-lr", type=float, default=0.0015)
    parser.add_argument("--scale-lr", type=float, default=0.004)
    parser.add_argument("--opacity-lr", type=float, default=0.02)
    parser.add_argument("--color-lr", type=float, default=0.03)
    parser.add_argument("--scale-reg", type=float, default=0.0005)
    parser.add_argument("--opacity-reg", type=float, default=0.0002)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    if args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")
    if args.image_max_size <= 0:
        raise SystemExit("--image-max-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_ply = args.output_ply or (args.output_dir / "pose_init_trained_scene.ply")
    preview_output = args.preview_output or (args.output_dir / "pose_init_training_preview.png")
    multi_preview_output = args.multi_preview_output or (args.output_dir / "pose_init_training_contact_sheet.png")
    metrics_output = args.metrics_output or (args.output_dir / "pose_init_training_metrics.json")

    set_seed(args.seed)
    start = time.time()
    result = train_pose_init_scene(
        args.pose_init_dir,
        output_ply=output_ply,
        preview_output=preview_output,
        multi_preview_output=multi_preview_output,
        metrics_output=metrics_output,
        camera_matrix_key=args.camera_matrix_key,
        max_steps=args.max_steps,
        max_frames=args.max_frames,
        image_max_size=args.image_max_size,
        max_points=args.max_points,
        target_gaussians=args.target_gaussians,
        dense_image_points_per_frame=args.dense_image_points_per_frame,
        dense_depth_neighbors=args.dense_depth_neighbors,
        dense_depth_max_distance=args.dense_depth_max_distance,
        adaptive_densify_every=args.adaptive_densify_every,
        adaptive_densify_start=args.adaptive_densify_start,
        adaptive_densify_stop=args.adaptive_densify_stop,
        adaptive_densify_points=args.adaptive_densify_points,
        adaptive_densify_error_percentile=args.adaptive_densify_error_percentile,
        adaptive_densify_depth_neighbors=args.adaptive_densify_depth_neighbors,
        adaptive_densify_max_distance=args.adaptive_densify_max_distance,
        adaptive_densify_max_projection_points=args.adaptive_densify_max_projection_points,
        preview_frames=args.preview_frames,
        trim_outlier_percentile=args.trim_outlier_percentile,
        initial_scale=args.initial_scale,
        means_lr=args.means_lr,
        scale_lr=args.scale_lr,
        opacity_lr=args.opacity_lr,
        color_lr=args.color_lr,
        scale_reg=args.scale_reg,
        opacity_reg=args.opacity_reg,
        device=args.device,
        seed=args.seed,
    )
    result["elapsed_seconds"] = round(time.time() - start, 3)
    metrics_output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def train_pose_init_scene(
    pose_init_dir: Path,
    *,
    output_ply: Path,
    preview_output: Path,
    multi_preview_output: Path,
    metrics_output: Path,
    camera_matrix_key: str,
    max_steps: int,
    max_frames: int,
    image_max_size: int,
    max_points: int,
    target_gaussians: int,
    dense_image_points_per_frame: int,
    dense_depth_neighbors: int,
    dense_depth_max_distance: float,
    adaptive_densify_every: int,
    adaptive_densify_start: int,
    adaptive_densify_stop: int,
    adaptive_densify_points: int,
    adaptive_densify_error_percentile: float,
    adaptive_densify_depth_neighbors: int,
    adaptive_densify_max_distance: float,
    adaptive_densify_max_projection_points: int,
    preview_frames: int,
    trim_outlier_percentile: float,
    initial_scale: float,
    means_lr: float,
    scale_lr: float,
    opacity_lr: float,
    color_lr: float,
    scale_reg: float,
    opacity_reg: float,
    device: str,
    seed: int,
) -> dict[str, Any]:
    import torch
    from gsplat import rasterization

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "cpu":
        raise SystemExit("pose-init gsplat training requires CUDA for practical end-to-end validation")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    frames = load_frames(
        pose_init_dir,
        camera_matrix_key=camera_matrix_key,
        max_frames=max_frames,
        image_max_size=image_max_size,
    )
    points, colors = load_sparse_points(pose_init_dir / "colmap_sparse_points.ply")
    points, colors, trim_report = trim_sparse_outliers(points, colors, percentile=trim_outlier_percentile)
    dense_points, dense_colors, dense_report = augment_points_from_registered_images(
        points,
        colors,
        frames,
        points_per_frame=dense_image_points_per_frame,
        neighbors=dense_depth_neighbors,
        max_distance_px=dense_depth_max_distance,
        seed=seed,
    )
    if len(dense_points):
        points = np.concatenate([points, dense_points], axis=0).astype(np.float32, copy=False)
        colors = np.concatenate([colors, dense_colors], axis=0).astype(np.float32, copy=False)
    points, colors, densify_report = densify_sparse_points(
        points,
        colors,
        target_gaussians=max(target_gaussians, len(points)),
        max_points=max_points,
        seed=seed,
    )
    if len(points) == 0:
        raise SystemExit("pose-init sparse point cloud is empty after filtering")

    scale_value = float(initial_scale) if initial_scale > 0 else estimate_initial_scale(points)
    means = torch.tensor(points, dtype=torch.float32, device=device).requires_grad_(True)
    log_scales = torch.full((len(points), 3), math.log(scale_value), dtype=torch.float32, device=device, requires_grad=True)
    opacity_logits = torch.full((len(points),), 0.0, dtype=torch.float32, device=device, requires_grad=True)
    color_logits = torch.logit(torch.tensor(colors, dtype=torch.float32, device=device).clamp(1e-4, 1.0 - 1e-4))
    color_logits.requires_grad_(True)
    quats = torch.zeros((len(points), 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0

    frame_tensors = [
        {
            "image_path": frame.image_path,
            "pixels": torch.tensor(frame.pixels, dtype=torch.float32, device=device).unsqueeze(0),
            "c2w": torch.tensor(frame.c2w, dtype=torch.float32, device=device).unsqueeze(0),
            "K": torch.tensor(frame.K, dtype=torch.float32, device=device).unsqueeze(0),
            "width": frame.width,
            "height": frame.height,
        }
        for frame in frames
    ]

    optimizer = build_optimizer(
        torch,
        means=means,
        log_scales=log_scales,
        opacity_logits=opacity_logits,
        color_logits=color_logits,
        means_lr=means_lr,
        scale_lr=scale_lr,
        opacity_lr=opacity_lr,
        color_lr=color_lr,
    )
    losses: list[float] = []
    psnrs: list[float] = []
    maes: list[float] = []
    adaptive_events: list[dict[str, Any]] = []
    densify_stop = max_steps if adaptive_densify_stop <= 0 else min(max_steps, adaptive_densify_stop)
    adaptive_rng = np.random.default_rng(seed + 17)
    for step in range(max_steps):
        frame_index = step % len(frame_tensors)
        frame = frame_tensors[frame_index]
        optimizer.zero_grad(set_to_none=True)
        render, _alpha, _info = rasterization(
            means=means,
            quats=quats,
            scales=torch.exp(log_scales).clamp(1e-5, 1.0),
            opacities=torch.sigmoid(opacity_logits),
            colors=torch.sigmoid(color_logits),
            viewmats=torch.linalg.inv(frame["c2w"]),
            Ks=frame["K"],
            width=int(frame["width"]),
            height=int(frame["height"]),
            packed=False,
            backgrounds=torch.zeros((1, 3), dtype=torch.float32, device=device),
            render_mode="RGB",
            rasterize_mode="antialiased",
            camera_model="pinhole",
        )
        target = frame["pixels"]
        l1 = torch.mean(torch.abs(render - target))
        scale_loss = torch.mean(torch.exp(log_scales))
        opacity_loss = torch.mean(torch.sigmoid(opacity_logits))
        loss = l1 + scale_reg * scale_loss + opacity_reg * opacity_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            metrics = image_metrics(render.detach(), target)
            losses.append(float(loss.detach().cpu()))
            psnrs.append(metrics["psnr"])
            maes.append(metrics["mae"])
            if should_adaptive_densify(
                step=step + 1,
                every=adaptive_densify_every,
                start=adaptive_densify_start,
                stop=densify_stop,
                current_count=int(means.shape[0]),
                max_points=max_points,
            ):
                event, updated = adaptive_densify_from_residual(
                    torch,
                    frame_sample=frames[frame_index],
                    render=render.detach(),
                    target=target.detach(),
                    means=means,
                    log_scales=log_scales,
                    opacity_logits=opacity_logits,
                    color_logits=color_logits,
                    quats=quats,
                    scale_value=scale_value,
                    max_points=max_points,
                    add_points=adaptive_densify_points,
                    error_percentile=adaptive_densify_error_percentile,
                    depth_neighbors=adaptive_densify_depth_neighbors,
                    max_distance_px=adaptive_densify_max_distance,
                    max_projection_points=adaptive_densify_max_projection_points,
                    rng=adaptive_rng,
                    device=device,
                )
                adaptive_events.append(event)
                if updated is not None:
                    means, log_scales, opacity_logits, color_logits, quats = updated
                    optimizer = build_optimizer(
                        torch,
                        means=means,
                        log_scales=log_scales,
                        opacity_logits=opacity_logits,
                        color_logits=color_logits,
                        means_lr=means_lr,
                        scale_lr=scale_lr,
                        opacity_lr=opacity_lr,
                        color_lr=color_lr,
                    )
        if step % 25 == 24 or step == max_steps - 1:
            print(
                f"step={step + 1}/{max_steps} loss={losses[-1]:.5f} "
                f"psnr={psnrs[-1]:.3f} mae={maes[-1]:.4f} gaussians={int(means.shape[0])}",
                flush=True,
            )

    with torch.no_grad():
        preview_frame = frame_tensors[0]
        preview_render, _alpha, _info = rasterization(
            means=means,
            quats=quats,
            scales=torch.exp(log_scales).clamp(1e-5, 1.0),
            opacities=torch.sigmoid(opacity_logits),
            colors=torch.sigmoid(color_logits),
            viewmats=torch.linalg.inv(preview_frame["c2w"]),
            Ks=preview_frame["K"],
            width=int(preview_frame["width"]),
            height=int(preview_frame["height"]),
            packed=False,
            backgrounds=torch.zeros((1, 3), dtype=torch.float32, device=device),
            render_mode="RGB",
            rasterize_mode="antialiased",
            camera_model="pinhole",
        )
        final_metrics = image_metrics(preview_render, preview_frame["pixels"])
        render_np = tensor_image_to_numpy(preview_render[0])
        source_np = tensor_image_to_numpy(preview_frame["pixels"][0])
        write_side_by_side_preview(source_np, render_np, preview_output)
        multi_preview_metrics = write_multi_frame_preview(
            rasterization,
            frame_tensors,
            means=means,
            quats=quats,
            log_scales=log_scales,
            opacity_logits=opacity_logits,
            color_logits=color_logits,
            output=multi_preview_output,
            preview_frames=preview_frames,
            device=device,
        )

    write_3dgs_ply(
        output_ply,
        means.detach().cpu().numpy(),
        torch.sigmoid(color_logits).detach().cpu().numpy(),
        log_scales.detach().cpu().numpy(),
        opacity_logits.detach().cpu().numpy(),
    )

    return {
        "format": "real2sim_pose_init_gsplat_training_v1",
        "pose_init_dir": pose_init_dir.as_posix(),
        "output_ply": output_ply.as_posix(),
        "preview_output": preview_output.as_posix(),
        "multi_preview_output": multi_preview_output.as_posix(),
        "metrics_output": metrics_output.as_posix(),
        "device": device,
        "max_steps": max_steps,
        "frame_count": len(frames),
        "source_frame_size": [frames[0].width, frames[0].height],
        "input_sparse_point_count": int(trim_report["input_point_count"]),
        "trimmed_sparse_point_count": int(trim_report["kept_point_count"]),
        "dense_image_point_count": int(dense_report["output_count"]),
        "trained_gaussian_count": int(means.shape[0]),
        "initial_scale": scale_value,
        "final_loss": losses[-1],
        "mean_loss_last_10": float(np.mean(losses[-10:])),
        "final_preview_psnr": final_metrics["psnr"],
        "final_preview_mae": final_metrics["mae"],
        "mean_step_psnr_last_10": float(np.mean(psnrs[-10:])),
        "mean_step_mae_last_10": float(np.mean(maes[-10:])),
        "multi_preview_mean_psnr": float(np.mean([item["psnr"] for item in multi_preview_metrics])),
        "multi_preview_mean_mae": float(np.mean([item["mae"] for item in multi_preview_metrics])),
        "multi_preview_metrics": multi_preview_metrics,
        "adaptive_densify": {
            "enabled": adaptive_densify_every > 0,
            "every": adaptive_densify_every,
            "start": adaptive_densify_start,
            "stop": densify_stop,
            "points_per_event": adaptive_densify_points,
            "event_count": len(adaptive_events),
            "added_point_count": int(sum(event.get("added_count", 0) for event in adaptive_events)),
            "events": adaptive_events,
        },
        "trim_report": trim_report,
        "dense_image_report": dense_report,
        "densify_report": densify_report,
    }


def build_optimizer(
    torch: Any,
    *,
    means: Any,
    log_scales: Any,
    opacity_logits: Any,
    color_logits: Any,
    means_lr: float,
    scale_lr: float,
    opacity_lr: float,
    color_lr: float,
) -> Any:
    return torch.optim.Adam(
        [
            {"params": [means], "lr": means_lr},
            {"params": [log_scales], "lr": scale_lr},
            {"params": [opacity_logits], "lr": opacity_lr},
            {"params": [color_logits], "lr": color_lr},
        ]
    )


def should_adaptive_densify(
    *,
    step: int,
    every: int,
    start: int,
    stop: int,
    current_count: int,
    max_points: int,
) -> bool:
    return every > 0 and step >= start and step <= stop and step % every == 0 and current_count < max_points


def adaptive_densify_from_residual(
    torch: Any,
    *,
    frame_sample: FrameSample,
    render: Any,
    target: Any,
    means: Any,
    log_scales: Any,
    opacity_logits: Any,
    color_logits: Any,
    quats: Any,
    scale_value: float,
    max_points: int,
    add_points: int,
    error_percentile: float,
    depth_neighbors: int,
    max_distance_px: float,
    max_projection_points: int,
    rng: np.random.Generator,
    device: str,
) -> tuple[dict[str, Any], tuple[Any, Any, Any, Any, Any] | None]:
    remaining = max(0, int(max_points) - int(means.shape[0]))
    requested = min(max(int(add_points), 0), remaining)
    event: dict[str, Any] = {
        "frame": frame_sample.image_path.name,
        "requested_count": int(add_points),
        "current_count": int(means.shape[0]),
        "max_points": int(max_points),
        "added_count": 0,
    }
    if requested <= 0:
        event["reason"] = "max_points_reached"
        return event, None

    error = torch.mean(torch.abs(torch.clamp(render[0], 0.0, 1.0) - torch.clamp(target[0], 0.0, 1.0)), dim=-1)
    error_np = error.detach().cpu().numpy()
    pixels = sample_high_error_pixels(
        error_np,
        count=requested,
        percentile=error_percentile,
        rng=rng,
    )
    if len(pixels) == 0:
        event["reason"] = "no_high_error_pixels"
        return event, None

    projection_points = means.detach().cpu().numpy().astype(np.float32, copy=False)
    if len(projection_points) > max_projection_points > 0:
        selected = rng.choice(len(projection_points), size=max_projection_points, replace=False)
        projection_points = projection_points[np.sort(selected)]
    projected = project_world_points(projection_points, frame_sample)
    if len(projected["uv"]) < max(32, depth_neighbors):
        event["reason"] = "too_few_projected_points"
        event["projected_count"] = int(len(projected["uv"]))
        return event, None

    depths, nearest_distances = estimate_depths_from_sparse_projection(
        pixels,
        projected["uv"],
        projected["depth"],
        neighbors=max(1, depth_neighbors),
    )
    valid = np.isfinite(depths) & (depths > 0.0) & (nearest_distances <= max_distance_px)
    if int(valid.sum()) < max(32, int(requested * 0.1)):
        valid = np.isfinite(depths) & (depths > 0.0)
    pixels = pixels[valid]
    depths = depths[valid]
    nearest_distances = nearest_distances[valid]
    if len(pixels) == 0:
        event["reason"] = "no_valid_depths"
        event["projected_count"] = int(len(projected["uv"]))
        return event, None

    if len(pixels) > requested:
        chosen = rng.choice(len(pixels), size=requested, replace=False)
        pixels = pixels[np.sort(chosen)]
        depths = depths[np.sort(chosen)]
        nearest_distances = nearest_distances[np.sort(chosen)]

    world = unproject_pixels_to_world(pixels, depths, frame_sample).astype(np.float32, copy=False)
    rgb = frame_sample.pixels[
        np.clip(np.rint(pixels[:, 1]).astype(np.int32), 0, frame_sample.height - 1),
        np.clip(np.rint(pixels[:, 0]).astype(np.int32), 0, frame_sample.width - 1),
    ].astype(np.float32, copy=False)

    new_means = torch.tensor(world, dtype=torch.float32, device=device)
    new_log_scales = torch.full((len(world), 3), math.log(scale_value), dtype=torch.float32, device=device)
    new_opacity_logits = torch.full((len(world),), 0.0, dtype=torch.float32, device=device)
    new_color_logits = torch.logit(torch.tensor(rgb, dtype=torch.float32, device=device).clamp(1e-4, 1.0 - 1e-4))
    new_quats = torch.zeros((len(world), 4), dtype=torch.float32, device=device)
    new_quats[:, 0] = 1.0

    updated_means = torch.cat([means.detach(), new_means], dim=0).requires_grad_(True)
    updated_scales = torch.cat([log_scales.detach(), new_log_scales], dim=0).requires_grad_(True)
    updated_opacity = torch.cat([opacity_logits.detach(), new_opacity_logits], dim=0).requires_grad_(True)
    updated_colors = torch.cat([color_logits.detach(), new_color_logits], dim=0).requires_grad_(True)
    updated_quats = torch.cat([quats.detach(), new_quats], dim=0)
    event.update(
        {
            "added_count": int(len(world)),
            "projected_count": int(len(projected["uv"])),
            "new_count": int(updated_means.shape[0]),
            "error_percentile": float(error_percentile),
            "mean_selected_error": float(np.mean(error_np[np.clip(np.rint(pixels[:, 1]).astype(np.int32), 0, frame_sample.height - 1), np.clip(np.rint(pixels[:, 0]).astype(np.int32), 0, frame_sample.width - 1)])),
            "nearest_distance_mean": float(np.mean(nearest_distances)),
            "depth_median": float(np.median(depths)),
        }
    )
    return event, (updated_means, updated_scales, updated_opacity, updated_colors, updated_quats)


def sample_high_error_pixels(
    error: np.ndarray,
    *,
    count: int,
    percentile: float,
    rng: np.random.Generator,
) -> np.ndarray:
    threshold = float(np.percentile(error, max(0.0, min(100.0, percentile))))
    ys, xs = np.nonzero(error >= threshold)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(xs) > count:
        chosen = rng.choice(len(xs), size=count, replace=False)
        xs = xs[chosen]
        ys = ys[chosen]
    return np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)


def write_multi_frame_preview(
    rasterization: Any,
    frame_tensors: list[dict[str, Any]],
    *,
    means: Any,
    quats: Any,
    log_scales: Any,
    opacity_logits: Any,
    color_logits: Any,
    output: Path,
    preview_frames: int,
    device: str,
) -> list[dict[str, float]]:
    if preview_frames <= 0:
        return []
    indices = np.linspace(0, len(frame_tensors) - 1, min(preview_frames, len(frame_tensors)), dtype=np.int64)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    metrics: list[dict[str, float]] = []
    for index in indices:
        frame = frame_tensors[int(index)]
        render, _alpha, _info = rasterization(
            means=means,
            quats=quats,
            scales=torch_exp_clamped(log_scales),
            opacities=torch_sigmoid(opacity_logits),
            colors=torch_sigmoid(color_logits),
            viewmats=frame["c2w"].inverse(),
            Ks=frame["K"],
            width=int(frame["width"]),
            height=int(frame["height"]),
            packed=False,
            backgrounds=zeros_background_like(means, device=device),
            render_mode="RGB",
            rasterize_mode="antialiased",
            camera_model="pinhole",
        )
        image_metric = image_metrics(render, frame["pixels"])
        metrics.append(
            {
                "frame_index": int(index),
                "psnr": image_metric["psnr"],
                "mae": image_metric["mae"],
            }
        )
        pairs.append((tensor_image_to_numpy(frame["pixels"][0]), tensor_image_to_numpy(render[0])))
    write_preview_contact_sheet(pairs, output)
    return metrics


def torch_exp_clamped(value: Any) -> Any:
    return value.exp().clamp(1e-5, 1.0)


def torch_sigmoid(value: Any) -> Any:
    return value.sigmoid()


def zeros_background_like(value: Any, *, device: str) -> Any:
    import torch

    return torch.zeros((1, 3), dtype=torch.float32, device=device)


def write_preview_contact_sheet(pairs: list[tuple[np.ndarray, np.ndarray]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not pairs:
        Image.new("RGB", (1, 1), "black").save(output)
        return
    tile_w = max(source.shape[1] + render.shape[1] for source, render in pairs)
    tile_h = max(max(source.shape[0], render.shape[0]) for source, render in pairs)
    canvas = np.zeros((tile_h * len(pairs), tile_w, 3), dtype=np.uint8)
    for row, (source, render) in enumerate(pairs):
        y = row * tile_h
        canvas[y : y + source.shape[0], : source.shape[1]] = source
        canvas[y : y + render.shape[0], source.shape[1] : source.shape[1] + render.shape[1]] = render
    Image.fromarray(canvas).save(output)


def load_frames(
    pose_init_dir: Path,
    *,
    camera_matrix_key: str,
    max_frames: int,
    image_max_size: int,
) -> list[FrameSample]:
    transforms_path = pose_init_dir / "transforms.json"
    if not transforms_path.is_file():
        raise SystemExit(f"pose-init transforms.json not found: {transforms_path}")
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    raw_frames = transforms.get("frames") or []
    if not raw_frames:
        raise SystemExit(f"pose-init transforms.json has no frames: {transforms_path}")

    selected_frames = select_evenly_spaced(raw_frames, max_frames)
    samples: list[FrameSample] = []
    for frame in selected_frames:
        image_path = pose_init_dir / str(frame["file_path"])
        if not image_path.is_file():
            raise SystemExit(f"pose-init image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        scale = min(1.0, float(image_max_size) / max(float(original_width), float(original_height)))
        width = max(1, int(round(original_width * scale)))
        height = max(1, int(round(original_height * scale)))
        if (width, height) != image.size:
            image = image.resize((width, height), PIL_BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0

        matrix = frame.get(camera_matrix_key)
        if matrix is None:
            matrix = frame.get("colmap_transform_matrix") or frame.get("transform_matrix")
        if matrix is None:
            raise SystemExit(f"pose-init frame has no usable camera matrix: {image_path}")
        c2w = np.asarray(matrix, dtype=np.float32)
        if c2w.shape != (4, 4):
            raise SystemExit(f"camera matrix must be 4x4 for {image_path}")

        fl_x = float(frame.get("fl_x", transforms.get("fl_x", 0.0))) * scale
        fl_y = float(frame.get("fl_y", transforms.get("fl_y", fl_x))) * scale
        cx = float(frame.get("cx", transforms.get("cx", original_width * 0.5))) * scale
        cy = float(frame.get("cy", transforms.get("cy", original_height * 0.5))) * scale
        K = np.asarray([[fl_x, 0.0, cx], [0.0, fl_y, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        samples.append(FrameSample(image_path=image_path, pixels=pixels, c2w=c2w, K=K, width=width, height=height))

    if not samples:
        raise SystemExit("no pose-init frames were loaded")
    return samples


def select_evenly_spaced(items: list[Any], max_count: int) -> list[Any]:
    if len(items) <= max_count:
        return list(items)
    indices = np.linspace(0, len(items) - 1, max_count, dtype=np.int64)
    return [items[int(index)] for index in indices]


def load_sparse_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise SystemExit(f"pose-init sparse point PLY not found: {path}")
    with path.open("r", encoding="ascii") as handle:
        first = handle.readline().strip()
        if first != "ply":
            raise SystemExit(f"unsupported sparse point file, expected PLY: {path}")
        vertex_count = None
        properties: list[str] = []
        in_vertex = False
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("format ") and stripped != "format ascii 1.0":
                raise SystemExit(f"only ascii PLY sparse points are supported: {path}")
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
                in_vertex = True
                continue
            if stripped.startswith("element ") and not stripped.startswith("element vertex "):
                in_vertex = False
            if in_vertex and stripped.startswith("property "):
                properties.append(stripped.split()[-1])
            if stripped == "end_header":
                break
        if vertex_count is None:
            raise SystemExit(f"PLY has no vertex count: {path}")
        required = {"x", "y", "z"}
        if not required.issubset(set(properties)):
            raise SystemExit(f"PLY missing xyz properties: {path}")
        names = {name: index for index, name in enumerate(properties)}
        points = np.zeros((vertex_count, 3), dtype=np.float32)
        colors = np.ones((vertex_count, 3), dtype=np.float32) * 0.7
        for row in range(vertex_count):
            values = handle.readline().split()
            if len(values) < len(properties):
                raise SystemExit(f"PLY vertex row is truncated at row {row}: {path}")
            points[row] = [float(values[names["x"]]), float(values[names["y"]]), float(values[names["z"]])]
            if {"red", "green", "blue"}.issubset(names):
                colors[row] = [
                    float(values[names["red"]]) / 255.0,
                    float(values[names["green"]]) / 255.0,
                    float(values[names["blue"]]) / 255.0,
                ]
    finite = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    points = points[finite]
    colors = np.clip(colors[finite], 0.0, 1.0)
    return points.astype(np.float32, copy=False), colors.astype(np.float32, copy=False)


def trim_sparse_outliers(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if len(points) == 0:
        return points, colors, {"input_point_count": 0, "kept_point_count": 0, "percentile": percentile}
    if percentile <= 0.0 or percentile >= 100.0:
        return points, colors, {
            "input_point_count": int(len(points)),
            "kept_point_count": int(len(points)),
            "percentile": percentile,
            "enabled": False,
        }
    center = np.median(points, axis=0)
    radius = np.linalg.norm(points - center[None, :], axis=1)
    threshold = float(np.percentile(radius, percentile))
    keep = radius <= threshold
    if int(keep.sum()) < max(256, int(len(points) * 0.5)):
        keep = np.ones(len(points), dtype=bool)
    return points[keep], colors[keep], {
        "input_point_count": int(len(points)),
        "kept_point_count": int(keep.sum()),
        "percentile": percentile,
        "radius_threshold": threshold,
        "center": [float(value) for value in center],
    }


def densify_sparse_points(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    target_gaussians: int,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if len(points) == 0:
        return points, colors, {"input_count": 0, "output_count": 0}
    target = min(max(int(target_gaussians), len(points)), max(int(max_points), len(points)))
    if target <= len(points):
        return points, colors, {"input_count": int(len(points)), "output_count": int(len(points)), "jittered_count": 0}

    rng = np.random.default_rng(seed)
    extra_count = target - len(points)
    base_indices = rng.integers(0, len(points), size=extra_count)
    span = np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0)
    jitter_scale = max(float(np.max(span)) / 900.0, 1e-4)
    jitter = rng.normal(0.0, jitter_scale, size=(extra_count, 3)).astype(np.float32)
    extra_points = points[base_indices] + jitter
    extra_colors = np.clip(colors[base_indices] + rng.normal(0.0, 0.015, size=(extra_count, 3)), 0.0, 1.0).astype(np.float32)
    return (
        np.concatenate([points, extra_points], axis=0).astype(np.float32, copy=False),
        np.concatenate([colors, extra_colors], axis=0).astype(np.float32, copy=False),
        {
            "input_count": int(len(points)),
            "output_count": int(target),
            "jittered_count": int(extra_count),
            "jitter_scale": jitter_scale,
        },
    )


def augment_points_from_registered_images(
    points: np.ndarray,
    colors: np.ndarray,
    frames: list[FrameSample],
    *,
    points_per_frame: int,
    neighbors: int,
    max_distance_px: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if points_per_frame <= 0 or len(points) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            {"enabled": False, "output_count": 0},
        )

    rng = np.random.default_rng(seed)
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    frame_reports: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        projected = project_world_points(points, frame)
        if projected["uv"].shape[0] < max(32, neighbors):
            frame_reports.append(
                {
                    "frame": frame.image_path.name,
                    "projected_sparse_count": int(projected["uv"].shape[0]),
                    "sampled_count": 0,
                }
            )
            continue
        pixels = sample_candidate_pixels(frame.width, frame.height, points_per_frame, rng=rng)
        depths, nearest_distances = estimate_depths_from_sparse_projection(
            pixels,
            projected["uv"],
            projected["depth"],
            neighbors=max(1, neighbors),
        )
        valid = np.isfinite(depths) & (depths > 0.0) & (nearest_distances <= max_distance_px)
        if int(valid.sum()) < max(64, int(points_per_frame * 0.15)):
            valid = np.isfinite(depths) & (depths > 0.0)
        pixels = pixels[valid]
        depths = depths[valid]
        if len(pixels) == 0:
            frame_reports.append(
                {
                    "frame": frame.image_path.name,
                    "projected_sparse_count": int(projected["uv"].shape[0]),
                    "sampled_count": 0,
                }
            )
            continue
        world = unproject_pixels_to_world(pixels, depths, frame)
        rgb = frame.pixels[
            np.clip(np.rint(pixels[:, 1]).astype(np.int32), 0, frame.height - 1),
            np.clip(np.rint(pixels[:, 0]).astype(np.int32), 0, frame.width - 1),
        ]
        all_points.append(world.astype(np.float32, copy=False))
        all_colors.append(rgb.astype(np.float32, copy=False))
        frame_reports.append(
            {
                "frame": frame.image_path.name,
                "projected_sparse_count": int(projected["uv"].shape[0]),
                "sampled_count": int(len(world)),
                "nearest_distance_mean": float(np.mean(nearest_distances[valid])),
                "depth_median": float(np.median(depths)),
            }
        )

    if not all_points:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            {
                "enabled": True,
                "points_per_frame": points_per_frame,
                "output_count": 0,
                "frames": frame_reports,
            },
        )
    dense_points = np.concatenate(all_points, axis=0)
    dense_colors = np.concatenate(all_colors, axis=0)
    return dense_points, dense_colors, {
        "enabled": True,
        "points_per_frame": points_per_frame,
        "neighbors": neighbors,
        "max_distance_px": max_distance_px,
        "output_count": int(len(dense_points)),
        "frames": frame_reports,
    }


def project_world_points(points: np.ndarray, frame: FrameSample) -> dict[str, np.ndarray]:
    w2c = np.linalg.inv(frame.c2w)
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    camera = homogeneous @ w2c.T
    depth = camera[:, 2]
    valid_depth = depth > 1e-5
    camera = camera[valid_depth]
    depth = depth[valid_depth]
    if len(camera) == 0:
        return {"uv": np.zeros((0, 2), dtype=np.float32), "depth": np.zeros((0,), dtype=np.float32)}
    uv = np.empty((len(camera), 2), dtype=np.float32)
    uv[:, 0] = frame.K[0, 0] * (camera[:, 0] / depth) + frame.K[0, 2]
    uv[:, 1] = frame.K[1, 1] * (camera[:, 1] / depth) + frame.K[1, 2]
    in_frame = (uv[:, 0] >= 0.0) & (uv[:, 0] < frame.width) & (uv[:, 1] >= 0.0) & (uv[:, 1] < frame.height)
    return {"uv": uv[in_frame], "depth": depth[in_frame].astype(np.float32, copy=False)}


def sample_candidate_pixels(width: int, height: int, count: int, *, rng: np.random.Generator) -> np.ndarray:
    count = max(1, int(count))
    stride = max(1, int(math.sqrt((width * height) / float(count))))
    xs = np.arange(stride // 2, width, stride, dtype=np.float32)
    ys = np.arange(stride // 2, height, stride, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pixels = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    if len(pixels) > count:
        selected = rng.choice(len(pixels), size=count, replace=False)
        pixels = pixels[np.sort(selected)]
    return pixels.astype(np.float32, copy=False)


def estimate_depths_from_sparse_projection(
    pixels: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    *,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    estimated = np.zeros((len(pixels),), dtype=np.float32)
    nearest = np.zeros((len(pixels),), dtype=np.float32)
    chunk = 512
    k = min(max(1, neighbors), len(uv))
    for start in range(0, len(pixels), chunk):
        stop = min(start + chunk, len(pixels))
        delta = pixels[start:stop, None, :] - uv[None, :, :]
        dist2 = np.sum(delta * delta, axis=2)
        if k == 1:
            idx = np.argmin(dist2, axis=1)[:, None]
        else:
            idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
        chosen_dist2 = np.take_along_axis(dist2, idx, axis=1)
        chosen_depth = depth[idx]
        weights = 1.0 / np.maximum(chosen_dist2, 1e-4)
        estimated[start:stop] = np.sum(weights * chosen_depth, axis=1) / np.sum(weights, axis=1)
        nearest[start:stop] = np.sqrt(np.min(chosen_dist2, axis=1))
    return estimated, nearest


def unproject_pixels_to_world(pixels: np.ndarray, depths: np.ndarray, frame: FrameSample) -> np.ndarray:
    x = (pixels[:, 0] - frame.K[0, 2]) / frame.K[0, 0] * depths
    y = (pixels[:, 1] - frame.K[1, 2]) / frame.K[1, 1] * depths
    camera = np.stack([x, y, depths, np.ones_like(depths)], axis=1).astype(np.float32, copy=False)
    world = camera @ frame.c2w.T
    return world[:, :3]


def estimate_initial_scale(points: np.ndarray) -> float:
    span = np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0)
    extent = max(float(np.max(span)), 1e-3)
    return max(min(extent / 220.0, 0.08), 0.004)


def image_metrics(render: Any, target: Any) -> dict[str, float]:
    import torch

    diff = torch.clamp(render, 0.0, 1.0) - torch.clamp(target, 0.0, 1.0)
    mae = float(torch.mean(torch.abs(diff)).detach().cpu())
    mse = float(torch.mean(diff * diff).detach().cpu())
    psnr = 99.0 if mse <= 1e-12 else -10.0 * math.log10(mse)
    return {"psnr": psnr, "mae": mae}


def tensor_image_to_numpy(image: Any) -> np.ndarray:
    array = image.detach().clamp(0.0, 1.0).cpu().numpy()
    return (array * 255.0 + 0.5).astype(np.uint8)


def write_side_by_side_preview(source: np.ndarray, render: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    h = max(source.shape[0], render.shape[0])
    w = source.shape[1] + render.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: source.shape[0], : source.shape[1]] = source
    canvas[: render.shape[0], source.shape[1] :] = render
    Image.fromarray(canvas).save(output)


def write_3dgs_ply(
    path: Path,
    means: np.ndarray,
    colors: np.ndarray,
    log_scales: np.ndarray,
    opacity_logits: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sh = (np.clip(colors, 0.0, 1.0) - 0.5) / SH_C0
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(means)}\n")
        for name in ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"):
            handle.write(f"property float {name}\n")
        for index in range(45):
            handle.write(f"property float f_rest_{index}\n")
        for name in ("opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"):
            handle.write(f"property float {name}\n")
        handle.write("end_header\n")
        zeros_rest = " ".join(["0"] * 45)
        for point, color, scale, opacity in zip(means, sh, log_scales, opacity_logits):
            handle.write(
                f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} 0 0 0 "
                f"{color[0]:.7g} {color[1]:.7g} {color[2]:.7g} {zeros_rest} "
                f"{opacity:.7g} {scale[0]:.7g} {scale[1]:.7g} {scale[2]:.7g} 1 0 0 0\n"
            )


if __name__ == "__main__":
    raise SystemExit(main())
