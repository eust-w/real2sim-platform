#!/usr/bin/env python3
"""Desktop viewer for a generated Real2Sim MotrixSim scene package.

The viewer can inspect only the MJCF scene, or load a Real2Sim package config
and render the corresponding 3D Gaussian assets without attaching a robot.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("ALSOFT_DRIVERS", "null")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
if "XDG_RUNTIME_DIR" not in os.environ:
    runtime_dir = Path(f"/run/user/{os.getuid()}")
    if runtime_dir.is_dir():
        os.environ["XDG_RUNTIME_DIR"] = runtime_dir.as_posix()

sys.path[:] = [
    entry
    for entry in sys.path
    if not entry
    or (
        Path(entry).name != "speedy-splat"
        and "real2sim_paper_runtime/speedy-splat" not in Path(entry).as_posix()
    )
]

import numpy as np
from motrixsim import SceneData, msd, run
from motrixsim.render import Layout, RenderApp, RenderSettings

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from real2sim.camera import load_anysplat_source_camera


def main() -> int:
    parser = argparse.ArgumentParser(description="Open a Real2Sim scene package without attaching a robot")
    parser.add_argument("--config", type=Path, default=None, help="Path to a generated configs/*_real2sim.json")
    parser.add_argument("--scene", type=Path, default=None, help="Path to the generated mjcf/scene.xml")
    parser.add_argument(
        "--camera",
        default="overview",
        help="Camera name from scene.xml to use as the main view; use 'system' for the interactive free camera",
    )
    parser.add_argument("--no-gs", action="store_true", help="Disable 3DGS preview widgets and load only the Motrix/MJCF scene")
    parser.add_argument("--gs-width", type=int, default=640, help="3DGS camera widget width")
    parser.add_argument("--gs-height", type=int, default=480, help="3DGS camera widget height")
    parser.add_argument("--gs-widget-left", type=int, default=10, help="Left offset for 3DGS preview widgets")
    parser.add_argument("--gs-widget-top", type=int, default=10, help="Top offset for 3DGS preview widgets")
    parser.add_argument("--gs-render-fps", type=float, default=12.0, help="Maximum 3DGS widget render refresh rate; use 0 for every frame")
    parser.add_argument("--static-gs-preview", action="store_true", help="Render 3DGS preview widgets once, then leave interaction to MotrixRender")
    parser.add_argument("--viewer-fps", type=float, default=60.0, help="MotrixRender window refresh rate")
    parser.add_argument("--frame-sleep-ms", type=float, default=0.0, help="Extra sleep after each render frame to keep desktop interaction responsive")
    parser.add_argument("--show-source-gs", action="store_true", help="Render the AnySplat source camera from camera_artifacts.cameras_json")
    parser.add_argument("--source-gs-only", action="store_true", help="Use the source-camera 3DGS render as the only 3DGS preview widget")
    parser.add_argument("--source-camera-json", type=Path, default=None, help="Override source-view cameras.json artifact")
    parser.add_argument("--save-source-gs", type=Path, default=None, help="Optional PNG path for the rendered source-camera 3DGS view")
    parser.add_argument("--save-main-gs", type=Path, default=None, help="Optional PNG path for the rendered main-camera 3DGS view")
    parser.add_argument("--show-system-gs", action="store_true", help="Also render a 3DGS view from the interactive camera")
    parser.add_argument("--exit-after-static-preview", action="store_true", help="Exit after static 3DGS preview images have been rendered and saved")
    parser.add_argument("--no-sync", action="store_true", help="Disable real-time sync")
    args = parser.parse_args()

    config = load_config(args.config)
    scene_path = resolve_scene_path(args.scene, config, args.config)
    scene_gaussians = {} if args.no_gs else collect_scene_gaussians(config, args.config)
    camera_artifacts = collect_camera_artifacts(config, args.config)
    source_camera_json = args.source_camera_json or camera_artifacts.get("cameras_json")
    gaussian_links = {
        str(gs_name): str(link_name)
        for gs_name, link_name in config.get("gaussian_links", {}).items()
    }

    world = msd.from_file(scene_path.as_posix())
    model = world.build()
    data = SceneData(model)

    render_target_fps = max(int(round(float(args.viewer_fps))), 1)
    with RenderApp(fps=render_target_fps) as render:
        print(f"Launching Real2Sim scene viewer: {scene_path}", flush=True)
        render.launch(model, render_settings=RenderSettings.performance())
        for group_id in range(6):
            try:
                render.opt.set_group_vis(group_id, True)
            except Exception:
                pass

        use_system_camera = args.camera.lower() in {"system", "free", "interactive"}
        camera = None if use_system_camera else find_camera(model, args.camera)
        camera_id = None if use_system_camera else find_camera_id(model, args.camera)
        if use_system_camera:
            render.set_main_camera(None)
            seed_camera = find_camera(model, "overview") or (model.cameras[0] if len(model.cameras) else None)
            if seed_camera is not None:
                try:
                    render.system_camera.pose = seed_camera.get_pose(data).astype(np.float32)
                except Exception:
                    pass
            print("Using system camera", flush=True)
        elif camera is not None:
            render.set_main_camera(camera)
            print(f"Using camera: {args.camera}", flush=True)

        gs_renderer = None
        gs_image = None
        gs_source_image = None
        gs_system_image = None
        gs_debug_printed = False
        source_rendered = False
        source_render_failed = False
        source_camera = None
        if scene_gaussians:
            print(f"Loading 3DGS assets: {', '.join(sorted(scene_gaussians))}", flush=True)
            from gaussian_renderer import GSRendererMotrixSim

            gs_renderer = GSRendererMotrixSim(scene_gaussians, model)
            apply_gaussian_links(gs_renderer, model, gaussian_links)
            widget_top = int(args.gs_widget_top)
            widget_left = int(args.gs_widget_left)
            if args.source_gs_only:
                args.show_source_gs = True
            if args.show_source_gs or args.save_source_gs is not None:
                if source_camera_json is None:
                    print("Source 3DGS view requested but no camera_artifacts.cameras_json was found.", flush=True)
                elif source_camera_json.is_file():
                    source_camera = load_anysplat_source_camera(source_camera_json)
                    gs_source_image = render.create_image(np.zeros((args.gs_height, args.gs_width, 3), dtype=np.uint8))
                    if args.show_source_gs:
                        render.widgets.create_image_widget(
                            gs_source_image,
                            layout=Layout(left=widget_left, top=widget_top, width=args.gs_width, height=args.gs_height),
                        )
                        widget_top += args.gs_height + 10
                    print(f"Source 3DGS camera ready: {source_camera_json}", flush=True)
                else:
                    print(f"Source camera JSON is missing: {source_camera_json}", flush=True)
            if not args.source_gs_only:
                gs_image = render.create_image(np.zeros((args.gs_height, args.gs_width, 3), dtype=np.uint8))
                render.widgets.create_image_widget(
                    gs_image,
                    layout=Layout(left=widget_left, top=widget_top, width=args.gs_width, height=args.gs_height),
                )
                widget_top += args.gs_height + 10
            if args.show_system_gs:
                gs_system_image = render.create_image(np.zeros((args.gs_height, args.gs_width, 3), dtype=np.uint8))
                render.widgets.create_image_widget(
                    gs_system_image,
                    layout=Layout(left=widget_left, top=widget_top, width=args.gs_width, height=args.gs_height),
                )
            print("3DGS renderer ready.", flush=True)

        print(f"Scene viewer ready. Bodies: {len(model.body_names)} Links: {len(model.link_names)}", flush=True)

        def phys_step() -> None:
            return None

        last_gs_render_at = 0.0
        gs_preview_rendered = False
        gs_render_interval = 0.0 if args.gs_render_fps <= 0 else 1.0 / max(float(args.gs_render_fps), 1e-6)

        def render_step() -> bool:
            nonlocal gs_debug_printed, source_rendered, source_render_failed, last_gs_render_at, gs_preview_rendered
            if render.input.is_key_just_pressed("esc") or render.input.is_key_just_pressed("escape"):
                return False
            if gs_renderer is not None:
                now = time.monotonic()
                should_render_gs = (
                    not (args.static_gs_preview and gs_preview_rendered)
                    and (gs_render_interval <= 0.0 or (now - last_gs_render_at) >= gs_render_interval)
                )
                if source_camera is not None and gs_source_image is not None and not source_rendered and not source_render_failed:
                    try:
                        gs_renderer.update_gaussians(data)
                        source_rgb, _source_depth = gs_renderer.render_batch(
                            source_camera.position.reshape(1, 3),
                            source_camera.xmat.reshape(1, 9),
                            args.gs_height,
                            args.gs_width,
                            np.array([source_camera.fovy_degrees], dtype=np.float32),
                            y_up=source_camera.y_up,
                        )
                        gs_source_image.pixels = rgb_to_uint8(source_rgb[0])
                        if args.save_source_gs is not None:
                            args.save_source_gs.parent.mkdir(parents=True, exist_ok=True)
                            gs_source_image.save_to_disk(args.save_source_gs.as_posix())
                            print(f"Saved source 3DGS render: {args.save_source_gs}", flush=True)
                        source_rendered = True
                        print("3DGS rendered source camera.", flush=True)
                        if args.source_gs_only and args.static_gs_preview and not gs_preview_rendered:
                            gs_preview_rendered = True
                            print("3DGS static preview rendered.", flush=True)
                    except Exception as exc:
                        source_render_failed = True
                        print(f"Source 3DGS render failed: {exc}", flush=True)
                if should_render_gs and not args.source_gs_only:
                    last_gs_render_at = now
                    gs_renderer.update_gaussians(data)
                    render_ids = []
                    if camera_id is not None:
                        render_ids.append(camera_id)
                    if args.show_system_gs:
                        render_ids.append(-1)
                    if render_ids:
                        results = gs_renderer.render(
                            model,
                            data,
                            render_ids,
                            args.gs_width,
                            args.gs_height,
                            system_camera=render.system_camera,
                        )
                        if not gs_debug_printed:
                            print(f"3DGS rendered camera ids: {sorted(results.keys())}", flush=True)
                            gs_debug_printed = True
                        if camera_id in results and gs_image is not None:
                            gs_image.pixels = rgb_to_uint8(results[camera_id][0])
                            if args.save_main_gs is not None:
                                args.save_main_gs.parent.mkdir(parents=True, exist_ok=True)
                                gs_image.save_to_disk(args.save_main_gs.as_posix())
                                print(f"Saved main 3DGS render: {args.save_main_gs}", flush=True)
                        if -1 in results and gs_system_image is not None:
                            gs_system_image.pixels = rgb_to_uint8(results[-1][0])
                        if args.static_gs_preview:
                            gs_preview_rendered = True
                            print("3DGS static preview rendered.", flush=True)
            should_exit_after_preview = args.exit_after_static_preview and args.static_gs_preview and gs_preview_rendered
            render.sync(data)
            if args.frame_sleep_ms > 0:
                time.sleep(float(args.frame_sleep_ms) / 1000.0)
            if should_exit_after_preview:
                # Motrix's render loop can ignore a false-y callback return on
                # the Wuying headless display. This flag is only for automated
                # static screenshot capture, so exit explicitly after flushing.
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0)
            return True

        render_fps = 1000000.0 if args.no_sync else max(float(args.viewer_fps), 1.0)
        run.render_loop(model.options.timestep, render_fps, phys_step, render_step)
    return 0


def find_camera(model, name: str):
    try:
        return model.cameras[name]
    except Exception:
        pass
    for camera in model.cameras:
        if getattr(camera, "name", None) == name:
            return camera
    return model.cameras[0] if len(model.cameras) else None


def find_camera_id(model, name: str) -> int | None:
    for idx, camera in enumerate(model.cameras):
        if getattr(camera, "name", None) == name:
            return idx
    return 0 if len(model.cameras) else None


def load_config(config_path: Path | None) -> dict:
    if config_path is None:
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Real2Sim viewer config must be a JSON object: {config_path}")
    return config


def resolve_config_path(path: str | Path, config_path: Path | None) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    if config_path is not None:
        return (config_path.resolve().parent / resolved).resolve()
    return resolved.resolve()


def resolve_scene_path(scene_arg: Path | None, config: dict, config_path: Path | None) -> Path:
    scene = scene_arg if scene_arg is not None else config.get("scene")
    if not scene:
        raise ValueError("Provide --scene or a --config containing a scene path")
    return resolve_config_path(scene, config_path)


def collect_scene_gaussians(config: dict, config_path: Path | None) -> dict[str, str]:
    scene_gaussians = config.get("scene_gaussians", {})
    if not isinstance(scene_gaussians, dict):
        raise ValueError("scene_gaussians must be an object mapping Gaussian names to PLY paths")

    gaussians: dict[str, str] = {}
    missing: list[Path] = []
    for name, path in scene_gaussians.items():
        ply = resolve_config_path(path, config_path)
        if ply.exists():
            gaussians[str(name)] = ply.as_posix()
        else:
            missing.append(ply)
    if missing:
        print("Skipping missing 3DGS assets: " + ", ".join(path.as_posix() for path in missing), flush=True)
    return gaussians


def collect_camera_artifacts(config: dict, config_path: Path | None) -> dict[str, Path]:
    raw_artifacts = config.get("camera_artifacts", {})
    if not isinstance(raw_artifacts, dict):
        raise ValueError("camera_artifacts must be an object mapping artifact names to paths")

    artifacts: dict[str, Path] = {}
    missing: list[Path] = []
    for name, path in raw_artifacts.items():
        artifact = resolve_config_path(path, config_path)
        if artifact.exists():
            artifacts[str(name)] = artifact
        else:
            missing.append(artifact)
    if missing:
        print("Skipping missing camera artifacts: " + ", ".join(path.as_posix() for path in missing), flush=True)
    return artifacts


def apply_gaussian_links(gs_renderer, model, gaussian_links: dict[str, str]) -> None:
    if not gaussian_links:
        return

    link_ids = {name: idx for idx, name in enumerate(model.link_names)}
    objects_info = []
    body_ids = []
    skipped = []
    for gs_name, link_name in gaussian_links.items():
        if gs_name not in gs_renderer.gaussian_start_indices:
            skipped.append(f"{gs_name}:missing-gaussian")
            continue
        if link_name not in link_ids:
            skipped.append(f"{gs_name}:{link_name}:missing-link")
            continue
        objects_info.append((gs_name, gs_renderer.gaussian_start_indices[gs_name], gs_renderer.gaussian_end_indices[gs_name]))
        body_ids.append(link_ids[link_name])

    if skipped:
        print("Skipping Gaussian links: " + ", ".join(skipped), flush=True)
    if not objects_info:
        return

    gs_renderer.gs_idx_start = np.array([start for _, start, _ in objects_info])
    gs_renderer.gs_idx_end = np.array([end for _, _, end in objects_info])
    gs_renderer.gs_body_ids = np.array(body_ids)
    gs_renderer.set_objects_mapping(objects_info)
    print(f"Bound 3DGS assets to MJCF links: {', '.join(name for name, _, _ in objects_info)}", flush=True)


def rgb_to_uint8(rgb_tensor) -> np.ndarray:
    rgb = rgb_tensor.detach().cpu().numpy()
    if rgb.dtype == np.uint8:
        return rgb
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    raise SystemExit(main())
