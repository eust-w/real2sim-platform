# Reproduce the procedural example

This sample is an original, watertight mesh generated from an icosphere. The model and generator are MIT-licensed with the repository. It is not a scanned object or a paid generation result. GLB uses metres and Y-up coordinates.

## Inspect and export

```bash
npm ci
npm run demo
```

Open the local Vite URL, select **Load example**, then **Export Genesis Package**. The archive includes the current sliders, mesh, Python entry point, requirements and manifest. Unpack it before running. To export the default example without the UI:

```bash
npm run example:export
```

The default export goes to `.data/genesis-packages/example-soft-puck/`. Re-exporting replaces this generated folder, including previous run results. Keep any results you need elsewhere first.

## Run Genesis

The reference environment is Python 3.11, Genesis World 1.4.5 and PyTorch 2.14.1. Use a virtual environment; the Python dependencies are separate from the Node application.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r examples/requirements.txt
python .data/genesis-packages/example-soft-puck/genesis_scene.py --steps 800
```

The default CPU run writes `metrics.json` and `trajectory.npz` next to the script. The first run may take several minutes to compile kernels. `--backend gpu` requests the Genesis GPU backend but is not covered by the reference CPU verification.

For a native Genesis camera recording:

```bash
python .data/genesis-packages/example-soft-puck/genesis_scene.py --steps 800 --record
```

This writes `simulation.mp4`. Rendering needs a supported graphics environment; a headless numerical success does not imply camera support. `--viewer` additionally opens the interactive Genesis viewer.

## Interpret the result

- FEM elastic material presets use an implicit solver with a 0.002 s step and four substeps. The sample is dropped onto a plane.
- The browser displays geometry only. Deformation footage comes from the exported Genesis scene.
- Gravity comes from the exported manifest, not a fixed value in the Python script.
- Presets are illustrative, not measured silicone properties. No physical calibration or sim-to-real accuracy is claimed.
- Cloth uses a separate experimental PBD path and requires a suitable surface mesh. This solid sample validates the FEM path only.
- Arbitrary generated models may have wrong units, disconnected surfaces or invalid topology. Inspect and repair them before simulation.

To regenerate the source geometry after installing the Python requirements, run `python scripts/generate-example.py`. This overwrites only `examples/soft-puck/model.glb`.
