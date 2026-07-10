#!/usr/bin/env bash
# Runtime wiring for the 48 GiB Wuying Real2Sim paper-alignment machine.
#
# Usage:
#   source scripts/real2sim_wuying_paper_env.sh
#   python -m real2sim check --paper-strict --config configs/real2sim.external.example.json --backend external

set -euo pipefail

export REAL2SIM_PROJECT_ROOT="${REAL2SIM_PROJECT_ROOT:-/opt/gs_playground_real2sim_48q}"
export REAL2SIM_RUNTIME_ROOT="${REAL2SIM_RUNTIME_ROOT:-/opt/real2sim_paper_runtime}"

export REAL2SIM_ANYSPLAT_ROOT="${REAL2SIM_ANYSPLAT_ROOT:-${REAL2SIM_RUNTIME_ROOT}/AnySplat}"
export REAL2SIM_SAM2_ROOT="${REAL2SIM_SAM2_ROOT:-${REAL2SIM_RUNTIME_ROOT}/sam2}"
export REAL2SIM_SAM3D_ROOT="${REAL2SIM_SAM3D_ROOT:-${REAL2SIM_RUNTIME_ROOT}/sam-3d-objects}"
export REAL2SIM_SPEEDY_SPLAT_ROOT="${REAL2SIM_SPEEDY_SPLAT_ROOT:-${REAL2SIM_RUNTIME_ROOT}/speedy-splat}"

export REAL2SIM_ANY_SPLAT_PYTHON="${REAL2SIM_ANY_SPLAT_PYTHON:-${REAL2SIM_RUNTIME_ROOT}/envs/anysplat/bin/python}"
export REAL2SIM_SAM3D_PYTHON="${REAL2SIM_SAM3D_PYTHON:-${REAL2SIM_RUNTIME_ROOT}/envs/sam3d-objects/bin/python}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
export CONDA_PREFIX="${CONDA_PREFIX:-${CUDA_HOME}}"
export LIDRA_SKIP_INIT="${LIDRA_SKIP_INIT:-1}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export REAL2SIM_SAM2_CHECKPOINT="${REAL2SIM_SAM2_CHECKPOINT:-${REAL2SIM_SAM2_ROOT}/sam2.1_hiera_large.pt}"
export REAL2SIM_SAM2_MODEL_CFG="${REAL2SIM_SAM2_MODEL_CFG:-configs/sam2.1/sam2.1_hiera_l.yaml}"
export REAL2SIM_SAM3D_CONFIG="${REAL2SIM_SAM3D_CONFIG:-${REAL2SIM_SAM3D_ROOT}/checkpoints/hf/pipeline.yaml}"

_real2sim_pythonpath_parts=(
  "${REAL2SIM_SAM2_ROOT}"
  "${REAL2SIM_SAM3D_ROOT}/notebook"
  "${REAL2SIM_SAM3D_ROOT}"
  "${REAL2SIM_ANYSPLAT_ROOT}"
  "${REAL2SIM_SPEEDY_SPLAT_ROOT}"
)
if [[ -n "${PYTHONPATH:-}" ]]; then
  _real2sim_pythonpath_parts+=("${PYTHONPATH}")
fi
export PYTHONPATH="$(IFS=:; echo "${_real2sim_pythonpath_parts[*]}")"
unset _real2sim_pythonpath_parts

export REAL2SIM_SYNTHETIC_MULTIVIEW_COMMAND="${REAL2SIM_SAM3D_PYTHON} scripts/real2sim_external/synthetic_multiview_expand.py --image {image} --output-dir {synthetic_multiview_dir} --frames-dir {synthetic_frames_dir} --prompt {synthetic_multiview_prompt} --mode {synthetic_multiview_mode} --size 1536x1024"
export REAL2SIM_GROUNDING_DINO_COMMAND="${REAL2SIM_SAM3D_PYTHON} scripts/real2sim_external/groundingdino_hf_detect.py --image {image} --prompt {prompt} --output {detections_json} --max-detections 12"
export REAL2SIM_SAM2_COMMAND="${REAL2SIM_SAM3D_PYTHON} scripts/real2sim_external/sam2_segment.py --image {image} --detections {detections_json} --masks-dir {masks_dir} --output {instances_json}"
export REAL2SIM_INSTANCE_REFINEMENT_COMMAND="${REAL2SIM_SAM3D_PYTHON} scripts/real2sim_external/instance_refine.py --instances {instances_json} --output {instances_json} --image {image} --merge-saturated-accessories --accessory-search-margin-ratio 0.25 --accessory-contact-passes 4 --accessory-saturation-threshold 0.45 --accessory-max-area-growth 0.15 --accessory-min-component-area-ratio 0.001 --accessory-max-components 8"
export REAL2SIM_LAMA_COMMAND="${REAL2SIM_SAM3D_PYTHON} scripts/real2sim_external/lama_inpaint.py --image {image} --instances {instances_json} --output {background_image} --prompt {prompt} --work-dir {work_dir} --redetect-command-env REAL2SIM_GROUNDING_DINO_COMMAND --segment-command-env REAL2SIM_SAM2_COMMAND --require-redetect-loop"
export REAL2SIM_DEPTH_COMMAND="${REAL2SIM_SAM3D_PYTHON} scripts/real2sim_external/depth_anything_depth.py --image {background_image} --output {depth_map} --model depth-anything/Depth-Anything-V2-Small-hf"
export REAL2SIM_ANYSPLAT_COMMAND="${REAL2SIM_ANY_SPLAT_PYTHON} scripts/real2sim_external/anysplat_scene.py --image {background_image} --frames-dir {frames_dir} --output-dir {scene_reconstruction_dir} --ply {scene_ply} --mesh {scene_mesh} --cameras {cameras_json} --depth {depth_map} --max-views 8"
export REAL2SIM_SAM3D_COMMAND="${REAL2SIM_SAM3D_PYTHON} scripts/real2sim_external/sam3d_objects_reconstruct.py --image {image} --mask {mask} --instance {instance_json} --output-dir {object_dir} --ply {object_ply} --mesh {object_mesh} --pose {object_pose} --cameras {cameras_json} --match-camera-preprocess --object-centric-preprocess --object-preprocess-size 896 --object-crop-margin 0.12 --canonicalize-gaussian-to-mesh --remove-video-overlay --texture-baking --texture-render-backend gsplat --texture-baking-engine pytorch3d"
export REAL2SIM_POSE_ALIGNMENT_COMMAND="${REAL2SIM_SAM3D_PYTHON} scripts/real2sim_external/depth_mask_pose_align.py --assets {aligned_assets_json} --instances {instances_json} --depth {depth_map} --cameras {cameras_json} --work-dir {work_dir} --output {aligned_assets_json}"
export REAL2SIM_SPEEDY_SPLAT_COMMAND="${REAL2SIM_ANY_SPLAT_PYTHON} scripts/real2sim_external/speedy_splat_prune.py --input {input} --output {output} --image {source_image} --cameras {cameras_json} --keep-ratio {keep_ratio}"
