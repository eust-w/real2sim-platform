#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  scripts/real2sim_wuying_launch_package.sh --archive /path/to/real2sim_scene.tar.gz [--dest /tmp/real2sim_import] [--with-gs]

Extract a generated one-image Real2Sim Motrix package on the Wuying desktop and
launch the visible MotrixSim viewer.
USAGE
}

archive=""
dest="/tmp/real2sim_import"
with_gs=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive)
      archive="${2:-}"
      shift 2
      ;;
    --dest)
      dest="${2:-}"
      shift 2
      ;;
    --with-gs)
      with_gs=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${archive}" || ! -f "${archive}" ]]; then
  echo "--archive must point to an existing .tar.gz package" >&2
  exit 2
fi

mkdir -p "${dest}"
tar -xzf "${archive}" -C "${dest}"
package_dir="$(find "${dest}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
if [[ -z "${package_dir}" || ! -f "${package_dir}/configs/go2_real2sim.json" ]]; then
  echo "Extracted package missing configs/go2_real2sim.json under ${dest}" >&2
  exit 1
fi

recommended_camera=""
if [[ -f "${package_dir}/motrix_import.json" ]]; then
  recommended_camera="$(python3 - "${package_dir}/motrix_import.json" <<'PY' || true
import json
import sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    data = {}
print(data.get("recommended_camera", ""))
PY
)"
fi

viewer_camera="overview"
if [[ "${recommended_camera}" == "source_3dgs_preview" ]]; then
  viewer_camera="closeup"
fi

args=(--config "${package_dir}/configs/go2_real2sim.json" --camera "${viewer_camera}" --gs-width 900 --gs-height 700 --static-gs-preview)
if [[ "${with_gs}" -eq 1 ]]; then
  args+=(--show-source-gs)
  if [[ "${recommended_camera}" == "source_3dgs_preview" ]]; then
    args+=(--source-gs-only)
  fi
  args=(--with-gs "${args[@]}")
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime_repo="${REAL2SIM_WUYING_RUNTIME_REPO:-$(cd "${script_dir}/.." && pwd)}"
if [[ ! -x "${runtime_repo}/scripts/real2sim_wuying_viewer.sh" || ! -x "${runtime_repo}/.venv/bin/python" ]]; then
  for candidate in \
    /opt/gs_playground_real2sim_48q \
    /opt/gs_playground-real2sim_48q \
    /opt/gs_playground \
    /workspace/gs_playground \
    /root/gs_playground; do
    if [[ -x "${candidate}/scripts/real2sim_wuying_viewer.sh" && -x "${candidate}/.venv/bin/python" ]]; then
      runtime_repo="${candidate}"
      break
    fi
  done
fi
if [[ ! -x "${runtime_repo}/scripts/real2sim_wuying_viewer.sh" || ! -x "${runtime_repo}/.venv/bin/python" ]]; then
  echo "Could not find a Real2Sim runtime repo with scripts/real2sim_wuying_viewer.sh and .venv/bin/python." >&2
  echo "Set REAL2SIM_WUYING_RUNTIME_REPO=/path/to/gs_playground on the Wuying desktop." >&2
  exit 1
fi

echo "Launching MotrixSim Real2Sim package: ${package_dir}"
echo "Runtime repo: ${runtime_repo}"
cd "${runtime_repo}"
exec scripts/real2sim_wuying_viewer.sh "${args[@]}"
