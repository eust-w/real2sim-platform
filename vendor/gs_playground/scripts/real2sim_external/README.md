# Real2Sim External Model Adapters

These scripts connect the GS-Playground Real2Sim stage contract to public model
implementations. They are optional: the built-in backend still runs without any
of these dependencies.

## Quality Upgrade Path

1. `groundingdino_hf_detect.py`
   - Purpose: replace heuristic connected components with open-vocabulary boxes.
   - Source: official GroundingDINO / Grounded-SAM-2 Hugging Face path.
   - Main dependency: `transformers`, `Pillow`, `torch`.
   - Default model: `IDEA-Research/grounding-dino-tiny`.

2. `sam2_segment.py`
   - Purpose: replace coarse masks with SAM2 box-prompted masks.
   - Source: official `facebookresearch/sam2`.
   - Required env:
     - `REAL2SIM_SAM2_CHECKPOINT=/path/to/sam2.1_hiera_large.pt`
     - `REAL2SIM_SAM2_MODEL_CFG=configs/sam2.1/sam2.1_hiera_l.yaml`

3. `instance_refine.py`
   - Purpose: match the paper's visual-only instance de-duplication stage after
     SAM/SAM2.
   - Method: composite detection/segmentation confidence, mask IoU, mask
     inclusion, and boundary overlap.

4. `lama_inpaint.py`
   - Purpose: replace median-fill background removal with sequential LaMa
     inpainting.
   - Runtime used here: `simple-lama-inpainting`, a lightweight LaMa wrapper
     listed by the official LaMa repository.
   - Method: one-object-at-a-time mask expansion and sequential background
     update. Re-detection of newly exposed regions must be supplied by the
     configured detection/segmentation loop for fully strict paper parity.
     The strict config uses `REAL2SIM_GROUNDING_DINO_COMMAND` and
     `REAL2SIM_SAM2_COMMAND` for this nested loop.

5. `depth_anything_depth.py`
   - Purpose: replace flat pseudo-depth detail with a monocular Depth Anything
     V2 depth prior before building the fallback 3DGS assets.
   - Main dependency: `transformers`, `Pillow`, `torch`.
   - Default model: `depth-anything/Depth-Anything-V2-Small-hf`.

6. `anysplat_scene.py`
   - Purpose: replace pseudo-depth scene Gaussians with AnySplat scene Gaussians.
   - Source: official AnySplat repo and Hugging Face model `lhjiang/anysplat`.
   - Output: official `export_ply` scene Gaussians, decoder depth, camera
     metadata, and a depth-derived scene mesh.
   - Required env:
     - `REAL2SIM_ANYSPLAT_ROOT=/path/to/AnySplat`
   - Best input: multiple nearby views or a short video, not a single repeated
     frame.

7. `sam3d_objects_reconstruct.py`
   - Purpose: replace object box meshes and object pseudo-Gaussians with SAM 3D
     Objects outputs.
   - Source: official Meta `facebookresearch/sam-3d-objects`.
   - Output: object Gaussian PLY from the SAM-3D `save_ply` output, an
     exportable object mesh, and pose/scale metadata. The script fails if no
     mesh is returned unless `--allow-fallback-mesh` is passed for debugging.
   - Required env:
     - `REAL2SIM_SAM3D_ROOT=/path/to/sam-3d-objects`
     - `REAL2SIM_SAM3D_CONFIG=/path/to/checkpoints/hf/pipeline.yaml`
   - Checkpoints are gated on Hugging Face and require accepted access. The
     upstream setup currently recommends an NVIDIA GPU with at least 32 GB VRAM.
   - Use `--cameras {cameras_json} --match-camera-preprocess` when the scene
     was reconstructed by AnySplat, so SAM-3D receives the same
     aspect-preserving center crop and mask geometry as the scene camera.
   - Use `--canonicalize-gaussian-to-mesh` to rotate the object Gaussian PLY
     into the SAM-3D mesh canonical frame using only geometric extent axes. It
     updates positions, normals, and splat quaternions without source-image
     photometric scoring.
   - Use `--remove-video-overlay` only for screenshots/video thumbnails with a
     central play button; it inpaints detected bright, low-saturation UI pixels
     near the image center before SAM-3D inference.

8. `depth_mask_pose_align.py`
   - Purpose: align SAM-3D objects back into the AnySplat scene using rendered
     object Gaussian PLY depth/mask, scene depth, and camera intrinsics.
     Mesh rendering is used only as a fallback when the object Gaussian asset is
     unavailable or invalid.
   - Output: an updated `{aligned_assets_json}` manifest with refined
     `pose.position`, `pose.quat_wxyz`, and `pose.scale` values plus alignment
     metadata including the selected `render_source`, candidate axis search
     label, mask IoU, centroid error, depth error, and quality summary.
     If the object Gaussian directory contains SAM-3D metadata showing
     `gaussian_canonicalization.status=ok`, the aligner keeps the canonical
     rotation and uses only a mask/depth position-scale seed instead of
     source-view axis search.

9. Speedy-Splat pruning
   - Purpose: match the paper's 3DGS pruning stage.
   - Configure `pruning.mode=speedy_splat` and set
     `pruning.external_command` or `REAL2SIM_SPEEDY_SPLAT_COMMAND`.
   - The pipeline fails if the command is missing in Speedy-Splat mode.
   - The adapter keeps Speedy-Splat's official score computation, but applies
     an exact-count tie-safe prune mask so all-zero or heavily tied object
     scores cannot delete every Gaussian in a canonical local object asset.

## Example

```bash
python -m real2sim run \
  --backend hybrid \
  --config configs/real2sim.external.example.json \
  --input media/teaser.png \
  --output /tmp/gs_real2sim_hybrid \
  --prompt "robot. person. table. chair. object."
```

Use `--backend external` only after AnySplat and SAM 3D Objects are installed,
their checkpoints are available, and the machine meets the SAM-3D memory
requirements. Use `python -m real2sim run ... --dry-run --backend hybrid` to
check which stage commands are configured before launching a long model run.
