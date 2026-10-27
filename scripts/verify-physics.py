"""Check the bundled drop-test evidence, not arbitrary materials or meshes."""
import json
from pathlib import Path
import sys

import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".data/genesis-packages/example-soft-puck")
metrics = json.loads((root / "metrics.json").read_text())
trajectory = np.load(root / "trajectory.npz")["positions"]
assert metrics["solver"] == "fem_elastic"
assert metrics["gravity"] == 9.8
assert metrics["steps"] >= 500
assert np.isfinite(trajectory).all()
assert np.abs(trajectory).max() < 1
assert 0.30 < metrics["initial_center_z"] < 0.40
assert 0.0 < metrics["final_center_z"] < 0.15
assert metrics["min_vertex_z"] > -0.01
assert 0.001 < metrics["max_shape_deformation_m"] < 0.2
print("PASS: finite trajectory, falling body, ground contact, nonzero deformation")
