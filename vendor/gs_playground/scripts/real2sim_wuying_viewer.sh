#!/usr/bin/env bash
# Launch the generated Real2Sim viewer on the visible Wuying desktop.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

foreground=0
with_gs=0
force_render_app=0
package_dir="${REAL2SIM_WUYING_PACKAGE:-}"
config_path="${REAL2SIM_WUYING_CONFIG:-}"
scene_path="${REAL2SIM_WUYING_SCENE:-}"
while [[ "$#" -gt 0 ]]; do
  case "${1}" in
    --foreground)
      foreground=1
      shift
      ;;
    --with-gs)
      with_gs=1
      force_render_app=1
      shift
      ;;
    --render-app)
      force_render_app=1
      shift
      ;;
    --native)
      force_render_app=0
      shift
      ;;
    --package)
      package_dir="${2:?--package requires a package directory}"
      shift 2
      ;;
    --config)
      config_path="${2:?--config requires a config JSON path}"
      force_render_app=1
      shift 2
      ;;
    --scene)
      scene_path="${2:?--scene requires an MJCF scene path}"
      shift 2
      ;;
    --stop)
      pkill -f "real2sim_scene_viewer[.]py" || true
      pkill -f "motrixsim[.]interactive_viewer" || true
      echo "Stopped Real2Sim viewer processes."
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

unset PYTHONPATH
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
export ALSOFT_DRIVERS="${ALSOFT_DRIVERS:-null}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
if [[ -z "${XDG_RUNTIME_DIR:-}" && -d "/run/user/$(id -u)" ]]; then
  export XDG_RUNTIME_DIR="/run/user/$(id -u)"
fi

choose_display() {
  local first_ok=""
  local candidate
  local seen=" "
  for candidate in "${DISPLAY:-}" :1 :92; do
    [[ -n "${candidate}" ]] || continue
    [[ "${seen}" == *" ${candidate} "* ]] && continue
    seen+="${candidate} "
    DISPLAY="${candidate}" timeout 2s xdpyinfo >/dev/null 2>&1 || continue
    if [[ -z "${first_ok}" ]]; then
      first_ok="${candidate}"
    fi
    if DISPLAY="${candidate}" timeout 2s xwininfo -root -tree 2>/dev/null | grep -Eq 'gnome-shell|gnome-terminal|MotrixRender'; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  if [[ -n "${first_ok}" ]]; then
    printf '%s\n' "${first_ok}"
    return 0
  fi
  printf '%s\n' "${DISPLAY:-:1}"
}

export DISPLAY="$(choose_display)"
echo "Launching Real2Sim viewer on DISPLAY=${DISPLAY}"

resolve_latest_package() {
  find outputs -path "*/mjcf/scene.xml" -type f -printf '%T@ %h\n' 2>/dev/null \
    | sort -nr \
    | head -1 \
    | awk '{print $2}' \
    | sed 's#/mjcf$##'
}

if [[ -z "${package_dir}" && ( -z "${scene_path}" || -z "${config_path}" ) ]]; then
  package_dir="$(resolve_latest_package || true)"
fi
if [[ -n "${package_dir}" ]]; then
  if [[ -z "${scene_path}" ]]; then
    scene_path="${package_dir%/}/mjcf/scene.xml"
  fi
  if [[ -z "${config_path}" ]]; then
    config_path="$(find "${package_dir%/}/configs" -maxdepth 1 -name '*_real2sim.json' -type f 2>/dev/null | sort | head -1 || true)"
  fi
fi

if [[ "$#" -eq 0 ]]; then
  if [[ "${force_render_app}" -eq 0 ]]; then
    if [[ -z "${scene_path}" ]]; then
      echo "No scene path found. Pass --scene or --package." >&2
      exit 1
    fi
    set -- \
      --file="${scene_path}"
  elif [[ "${with_gs}" -eq 1 ]]; then
    if [[ -z "${config_path}" ]]; then
      echo "No Real2Sim config found. Pass --config or --package." >&2
      exit 1
    fi
    set -- \
      --config "${config_path}" \
      --camera overview \
      --gs-width 320 \
      --gs-height 210 \
      --gs-widget-left 930 \
      --gs-widget-top 10 \
      --gs-render-fps 8 \
      --static-gs-preview \
      --viewer-fps 20 \
      --frame-sleep-ms 30 \
      --show-source-gs \
      --show-system-gs
  else
    if [[ -n "${config_path}" ]]; then
      set -- \
        --config "${config_path}" \
        --camera system \
        --no-gs \
        --viewer-fps 12 \
        --frame-sleep-ms 55
    elif [[ -n "${scene_path}" ]]; then
      set -- \
        --scene "${scene_path}" \
        --camera system \
        --no-gs \
        --viewer-fps 12 \
        --frame-sleep-ms 55
    else
      echo "No Real2Sim config or scene found. Pass --config, --scene, or --package." >&2
      exit 1
    fi
  fi
else
  force_render_app=1
  parsed_args=()
  if [[ -n "${config_path}" ]]; then
    parsed_args+=(--config "${config_path}")
  elif [[ -n "${scene_path}" ]]; then
    parsed_args+=(--scene "${scene_path}")
  fi
  set -- "${parsed_args[@]}" "$@"
fi

pkill -f "real2sim_scene_viewer[.]py" || true
pkill -f "motrixsim[.]interactive_viewer" || true

if [[ "${foreground}" -eq 1 ]]; then
  if [[ "${force_render_app}" -eq 0 ]]; then
    exec .venv/bin/python -m motrixsim.interactive_viewer "$@"
  fi
  exec .venv/bin/python scripts/real2sim_scene_viewer.py "$@"
fi

log_path="${REAL2SIM_WUYING_VIEWER_LOG:-/tmp/real2sim_wuying_viewer.log}"
rm -f "${log_path}"
if [[ "${force_render_app}" -eq 0 ]]; then
  cmd=(.venv/bin/python -m motrixsim.interactive_viewer "$@")
else
  cmd=(.venv/bin/python scripts/real2sim_scene_viewer.py "$@")
fi
printf 'Command:' >"${log_path}"
printf ' %q' "${cmd[@]}" >>"${log_path}"
printf '\n' >>"${log_path}"
setsid "${cmd[@]}" >>"${log_path}" 2>&1 < /dev/null &
pid=$!
echo "Started Real2Sim viewer PID=${pid}"
echo "Log: ${log_path}"
sleep 2
if ! ps -p "${pid}" >/dev/null 2>&1; then
  echo "Viewer exited early; last log lines:" >&2
  tail -80 "${log_path}" >&2 || true
  exit 1
fi
