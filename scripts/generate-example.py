"""Regenerate the MIT-licensed sample from primitives, without external assets."""
from pathlib import Path

import trimesh

root = Path(__file__).resolve().parents[1] / "examples" / "soft-puck"
root.mkdir(parents=True, exist_ok=True)
mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.08)
mesh.apply_scale([1, 0.65, 1])
mesh.visual = trimesh.visual.TextureVisuals(
    material=trimesh.visual.material.PBRMaterial(
        baseColorFactor=[31, 184, 158, 255], metallicFactor=0.0, roughnessFactor=0.38
    )
)
assert mesh.is_watertight and mesh.volume > 0
mesh.export(root / "model.glb")
print(f"Generated watertight sample: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
