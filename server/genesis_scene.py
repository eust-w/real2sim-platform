"""Run an exported Real2Sim scene. All geometry and parameters come from the package."""

import argparse
import json
from pathlib import Path

import genesis as gs
import numpy as np
from physics_metrics import shape_deformation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--capture-every", type=int, default=5)
    args = parser.parse_args()
    if args.steps < 1 or args.capture_every < 1:
        parser.error("steps and capture-every must be positive")

    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    config = manifest["genesis"]
    params = config["parameters"]
    gravity = -float(config["gravity"])
    is_cloth = config["solver"] == "pbd_cloth"
    dt = 0.005 if is_cloth else 0.002
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.gpu, seed=0, logging_level="warning")
    options = dict(
        sim_options=gs.options.SimOptions(dt=dt, substeps=4, gravity=(0, 0, gravity)),
        show_viewer=args.viewer,
    )
    if is_cloth:
        options["pbd_options"] = gs.options.PBDOptions(max_stretch_solver_iterations=10)
    else:
        options["fem_options"] = gs.options.FEMOptions(
            use_implicit_solver=True, n_newton_iterations=4, n_pcg_iterations=100,
            damping=params["damping"], damping_alpha=params["damping"],
        )
    scene = gs.Scene(**options)
    scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.84, 0.87, 0.88)))
    mesh_options = dict(file=str(root / manifest["assets"]["model"]), pos=(0, 0, 0.35), scale=1.0)
    if not is_cloth:
        mesh_options.update(force_retet=True, maxvolume=params["maxVolume"])
    material = (
        gs.materials.PBD.Cloth(
            stretch_compliance=params["stretchCompliance"], bending_compliance=params["bendCompliance"],
            air_resistance=params["damping"],
        )
        if is_cloth else gs.materials.FEM.Elastic(E=params["E"], nu=params["nu"], rho=params["rho"])
    )
    body = scene.add_entity(
        gs.morphs.Mesh(**mesh_options), material=material,
        surface=gs.surfaces.Default(color=(0.12, 0.72, 0.62)),
    )
    camera = None
    if args.record:
        camera = scene.add_camera(res=(1280, 720), pos=(0.65, -0.85, 0.55), lookat=(0, 0, 0.16), fov=40, GUI=False)
    print("Building Genesis scene...", flush=True)
    scene.build()
    print("Running solver...", flush=True)
    def positions():
        return body.get_state().pos.detach().cpu().numpy().reshape(-1, 3).copy()
    initial = positions()
    frames = [initial]
    max_deformation = 0.0
    min_z = float(initial[:, 2].min())
    if camera:
        camera.start_recording(save_to_filename=str(root / "simulation.mp4"), fps=50)
    for step in range(args.steps):
        scene.step()
        if (step + 1) % args.capture_every == 0 or step == args.steps - 1:
            pos = positions()
            if not np.isfinite(pos).all() or np.abs(pos).max() > 100:
                raise RuntimeError(f"Unstable solver state at step {step + 1}")
            max_deformation = max(max_deformation, shape_deformation(initial, pos))
            min_z = min(min_z, float(pos[:, 2].min()))
            frames.append(pos)
    if camera:
        camera.stop_recording()
    final = positions()
    np.savez_compressed(root / "trajectory.npz", positions=np.asarray(frames), dt=dt * args.capture_every)
    metrics = {
        "genesis_version": gs.__version__, "backend": args.backend, "solver": config["solver"],
        "steps": args.steps, "dt": dt, "gravity": config["gravity"], "vertex_count": len(initial),
        "initial_center_z": float(initial[:, 2].mean()), "final_center_z": float(final[:, 2].mean()),
        "min_vertex_z": min_z, "max_shape_deformation_m": max_deformation,
        "all_finite": bool(np.isfinite(final).all()), "recorded": bool(camera),
    }
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
