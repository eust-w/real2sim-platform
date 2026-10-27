# Real2Sim Studio

**A local-first workbench for 3D assets and Genesis simulation exports.**

[中文](docs/README.zh-CN.md) · [Offline example](examples/README.md) · [Roadmap](docs/ROADMAP.md) · [Contributing](CONTRIBUTING.md) · [MIT license](LICENSE)

![Real2Sim Studio workbench](docs/images/workbench.png)

[Download v0.1.0 Alpha](https://github.com/eust-w/real2sim-platform/releases/tag/v0.1.0-alpha.1) · [90-second introduction](https://github.com/eust-w/real2sim-platform/releases/download/v0.1.0-alpha.1/real2sim-intro-en.mp4) · [中文视频](https://github.com/eust-w/real2sim-platform/releases/download/v0.1.0-alpha.1/real2sim-intro-zh.mp4) · [Gitea mirror](https://gitea.d-robotics.cc/longtao.wu/real2sim-platform)

Inspect a GLB, configure a material preset and export a self-contained Genesis scene. Start with the included procedural example without an API key, or connect your own DashScope account for Tripo text/image-to-3D generation.

## Try it

Requires Node.js **22.12 or later**.

```bash
git clone https://github.com/eust-w/real2sim-platform.git
cd real2sim-platform
npm ci
npm run demo
```

Open the Vite URL printed in the terminal (normally `http://127.0.0.1:5173`). Select **Load example**, rotate the model, adjust the material controls, then **Export Genesis Package**. Demo mode cannot make paid generation calls and uses `.data/demo/`, separate from your usual tasks.

## What is included

- Text, single-image and multi-image generation through DashScope / Tripo using your own account.
- Local task history and asset caching, GLB preview and downloads.
- A self-contained, MIT-licensed procedural sample.
- Material preset controls and Genesis export using the current settings, including gravity.
- FEM elastic export and an experimental PBD cloth export path.
- English / Chinese UI, unit tests, backend type checking and browser smoke tests.

The package contains the mesh, `flex-profile.json`, `manifest.json`, `genesis_scene.py`, pinned runtime requirements and instructions. Physics runs in **Genesis**, not in the browser preview.

## Run the exported scene

See the [complete example guide](examples/README.md) for Python environment setup.

```bash
npm run example:export
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r examples/requirements.txt
python .data/genesis-packages/example-soft-puck/genesis_scene.py --steps 800 --record
```

The first run compiles solver kernels. Without `--record`, the scene runs numerical checks and saves metrics and a trajectory without requesting a camera. The recorded result uses the actual Genesis solver, not an animated preview.

**Alpha security limitation:** the optional Python runtime has known Pillow advisories constrained by an upstream MoviePy dependency. Use only the trusted example in an isolated environment; see [the audit note](SECURITY.md#known-optional-runtime-risk-2026-10-28). The Node audit and Python audit are separate.

## Optional AI generation

```bash
cp .env.example .env.local
# Set your DashScope API key in .env.local.
npm run dev
```

Generation uses an external paid service. Provider models, generated assets and service terms are not covered by this repository's MIT license. A flexible variant is a new prompt-based generation; it is not deformation or reconstruction of the original GLB.

The local API defaults to `127.0.0.1:8789`. Set `PORT` before starting to change it; the development proxy uses the same value. This is a single-user local application, not a publicly hosted service. See [Security](SECURITY.md).

## Limits

- Material values are illustrative presets, not calibrated estimates from real objects.
- The included mesh is procedural. Arbitrary generated meshes can fail tetrahedralization and may need repair or unit correction.
- Cloth requires an appropriate surface mesh. Inflatable/soft-robot presets do not simulate pressure or actuators.
- No robot training, scene reconstruction, measured sim-to-real accuracy or universal model compatibility is claimed.
- The current exporter targets Genesis World 1.4.5. Newer releases may change APIs.

## Development

```bash
npm run test:run
npm run lint
npm run build
npx playwright install chromium
npm run test:e2e
npm audit
```

The browser test uses isolated data, exports a real GLB package, verifies current slider values, checks canvas pixels and rotation, and captures desktop/mobile screenshots. Physics execution is a separate Python check, not implied by passing the TypeScript tests.

## License and credits

MIT for this workbench and its procedural sample. See [third-party notices](THIRD_PARTY_NOTICES.md). Built with React, Three.js, Express and Genesis World; optional generation is provided by Tripo through DashScope. This project is independent of those upstream projects and providers.
