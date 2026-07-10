#!/usr/bin/env bash
# Download SAM-3D Objects checkpoints into the layout expected by Real2Sim.
#
# Official paper-equivalent source:
#   REAL2SIM_SAM3D_CHECKPOINT_REPO=facebook/sam-3d-objects
#
# Public mirror fallback used only when the official gated repo is unavailable:
#   REAL2SIM_SAM3D_CHECKPOINT_REPO=licesma/sam-3d-objects-weights

set -euo pipefail

if [[ -z "${REAL2SIM_SAM3D_ROOT:-}" ]]; then
  echo "REAL2SIM_SAM3D_ROOT is required. Source scripts/real2sim_wuying_paper_env.sh first." >&2
  exit 2
fi

repo="${REAL2SIM_SAM3D_CHECKPOINT_REPO:-facebook/sam-3d-objects}"
max_workers="${REAL2SIM_SAM3D_DOWNLOAD_WORKERS:-1}"
dest="${REAL2SIM_SAM3D_CHECKPOINT_DIR:-${REAL2SIM_SAM3D_ROOT}/checkpoints/hf}"
hf_bin="${REAL2SIM_HF_BIN:-hf}"
force="${REAL2SIM_SAM3D_CHECKPOINT_FORCE:-0}"

if [[ -e "${dest}/pipeline.yaml" && "${force}" != "1" ]]; then
  echo "SAM-3D checkpoints already exist at ${dest}. Set REAL2SIM_SAM3D_CHECKPOINT_FORCE=1 to replace." >&2
  exit 0
fi

mkdir -p "${REAL2SIM_SAM3D_ROOT}/checkpoints"
download_dir="$(mktemp -d "${REAL2SIM_SAM3D_ROOT}/checkpoints/.sam3d-download.XXXXXX")"
cleanup() {
  rm -rf "${download_dir}"
}
trap cleanup EXIT

"${hf_bin}" download \
  "${repo}" \
  --repo-type model \
  --local-dir "${download_dir}" \
  --max-workers "${max_workers}"

source_dir=""
if [[ -f "${download_dir}/checkpoints/pipeline.yaml" ]]; then
  source_dir="${download_dir}/checkpoints"
elif [[ -f "${download_dir}/hf/pipeline.yaml" ]]; then
  source_dir="${download_dir}/hf"
else
  echo "Downloaded ${repo}, but no checkpoints/pipeline.yaml or hf/pipeline.yaml was found." >&2
  find "${download_dir}" -maxdepth 3 -type f | sort >&2
  exit 1
fi

tmp_dest="${dest}.tmp.$$"
rm -rf "${tmp_dest}"
mkdir -p "$(dirname "${dest}")"
cp -a "${source_dir}" "${tmp_dest}"
if [[ -e "${dest}" || -L "${dest}" ]]; then
  rm -rf "${dest}"
fi
mv "${tmp_dest}" "${dest}"

cat > "${dest}.SOURCE.txt" <<EOF
repo=${repo}
downloaded_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
destination=${dest}
official_repo=facebook/sam-3d-objects
EOF

echo "SAM-3D checkpoints are ready at ${dest}"
