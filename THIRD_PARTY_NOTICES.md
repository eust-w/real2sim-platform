# Third-party software and assets

The workbench and the procedural sample in `examples/soft-puck/` are distributed under the root MIT license. `scripts/generate-example.py` constructs that mesh from an icosphere; it contains no scanned, purchased, provider-generated, or company-internal asset.

- React, Express, Vite, Three.js, React Three Fiber, Drei and Zod retain their respective upstream licenses. See their installed package license files and the exact dependency versions in `package-lock.json`.
- Lucide icons use the ISC license.
- Genesis World is a separate Apache-2.0 project, installed by the user. Its transitive dependencies retain their own licenses. This repository does not redistribute Genesis, PyTorch or their model assets.
- Tripo through Alibaba Cloud DashScope is an optional external generation service. The root MIT license does not license that service, its models or generated outputs. Users supply their own credentials and must check their service terms and input/output rights.
- Video recordings supplied with a release show this project's procedural sample and workbench. They are not evidence of real-object material identification or physical calibration.

Upstream references: [Genesis World](https://github.com/Genesis-Embodied-AI/genesis-world), [Three.js](https://github.com/mrdoob/three.js), [Lucide](https://github.com/lucide-icons/lucide).
