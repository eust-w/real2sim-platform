#!/usr/bin/env bash
set -euo pipefail

KEY="${REAL2SIM_WUYING_KEY:-/Users/d-robotics/workSpace/document/80-private/orgs/drobotics/keys/wuying-39.106.101.95_ed25519}"
HOST="${REAL2SIM_WUYING_HOST:-root@39.106.101.95}"
PROJECT_ROOT="${REAL2SIM_REMOTE_PLATFORM_ROOT:-/opt/real2sim2model/real2sim-platform}"
SERVICE="${REAL2SIM_PLATFORM_SERVICE:-real2sim-platform.service}"
PORT="${REAL2SIM_PLATFORM_PORT:-18126}"
DESKTOP_NAME="${REAL2SIM_DESKTOP_NAME:-Real2Sim 平台}"
DESKTOP_FILE_NAME="${REAL2SIM_DESKTOP_FILE_NAME:-Real2Sim Platform.desktop}"
MODE="local"
CLEAN_DESKTOP=1
CREATE_LAUNCHER=1

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_wuying_desktop_launcher.sh --remote
  scripts/setup_wuying_desktop_launcher.sh --local

Options:
  --remote             Run the setup on the Wuying host over SSH.
  --local              Run on the current Linux desktop host.
  --no-clean           Do not move existing desktop icons into backup.
  --no-launcher        Do not create/update the Real2Sim launcher.
  --project-root PATH  Remote Real2Sim platform root. Default: /opt/real2sim2model/real2sim-platform
  --service NAME       systemd service name. Default: real2sim-platform.service
  --port PORT          Local service port. Default: 18126

The cleanup is reversible: existing desktop entries are moved into a hidden
.real2sim_desktop_backup_<timestamp> directory on the Desktop, not deleted.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) MODE="remote"; shift ;;
    --local) MODE="local"; shift ;;
    --no-clean) CLEAN_DESKTOP=0; shift ;;
    --no-launcher) CREATE_LAUNCHER=0; shift ;;
    --project-root) PROJECT_ROOT="$2"; shift 2 ;;
    --service) SERVICE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$MODE" == "remote" ]]; then
  ssh -i "$KEY" \
    -o BatchMode=yes \
    -o ConnectTimeout=20 \
    -o StrictHostKeyChecking=accept-new \
    "$HOST" \
    "REAL2SIM_REMOTE_PLATFORM_ROOT='$PROJECT_ROOT' REAL2SIM_PLATFORM_SERVICE='$SERVICE' REAL2SIM_PLATFORM_PORT='$PORT' REAL2SIM_DESKTOP_NAME='$DESKTOP_NAME' REAL2SIM_DESKTOP_FILE_NAME='$DESKTOP_FILE_NAME' bash -s -- --local $([[ "$CLEAN_DESKTOP" == "0" ]] && printf '%s' '--no-clean') $([[ "$CREATE_LAUNCHER" == "0" ]] && printf '%s' '--no-launcher')" < "$0"
  exit 0
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This setup must run on the Wuying Linux desktop host." >&2
  exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"

add_desktop_dir() {
  local dir="$1"
  [[ -n "$dir" ]] || return 0
  if [[ ! -d "$dir" ]]; then
    mkdir -p "$dir"
  fi
  printf '%s\n' "$dir"
}

discover_desktop_dirs() {
  local dirs=()
  if [[ -n "${XDG_DESKTOP_DIR:-}" ]]; then
    dirs+=("$XDG_DESKTOP_DIR")
  fi
  dirs+=("$HOME/Desktop" "$HOME/桌面")
  if [[ "$(id -u)" == "0" ]]; then
    dirs+=("/root/Desktop" "/root/桌面")
    local home_dir
    for home_dir in /home/*; do
      [[ -d "$home_dir" ]] || continue
      dirs+=("$home_dir/Desktop" "$home_dir/桌面")
    done
  fi

  local seen="" dir
  for dir in "${dirs[@]}"; do
    [[ -n "$dir" ]] || continue
    case ":$seen:" in
      *":$dir:"*) continue ;;
    esac
    seen="$seen:$dir"
    add_desktop_dir "$dir"
  done
}

owner_for_path() {
  local path="$1"
  stat -c '%U:%G' "$path" 2>/dev/null || printf 'root:root\n'
}

write_launcher_script() {
  local home_dir="$1"
  local owner_group="$2"
  local bin_dir="$home_dir/.local/bin"
  local state_dir="$home_dir/.local/state"
  local launcher="$bin_dir/real2sim-platform-launch"

  mkdir -p "$bin_dir" "$state_dir"
  cat > "$launcher" <<EOF
#!/usr/bin/env bash
set -u

PROJECT_ROOT="$PROJECT_ROOT"
SERVICE="$SERVICE"
PORT="$PORT"
URL="http://127.0.0.1:$PORT/"
LOG="\${XDG_STATE_HOME:-\$HOME/.local/state}/real2sim-launcher.log"
mkdir -p "\$(dirname "\$LOG")"
{
  echo "==== \$(date '+%Y-%m-%d %H:%M:%S') Real2Sim launcher ===="

  run_with_timeout() {
    if command -v timeout >/dev/null 2>&1; then
      timeout 25 "\$@"
    else
      "\$@"
    fi
  }

  health_ok() {
    if command -v curl >/dev/null 2>&1; then
      curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1
    else
      python3 - <<'PY' >/dev/null 2>&1
import urllib.request
urllib.request.urlopen('http://127.0.0.1:$PORT/api/health', timeout=2).read()
PY
    fi
  }

  wait_for_health() {
    local i
    for i in \$(seq 1 40); do
      health_ok && return 0
      sleep 1
    done
    return 1
  }

  if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files "$SERVICE" >/dev/null 2>&1; then
    if systemctl is-failed --quiet "$SERVICE"; then
      run_with_timeout systemctl restart "$SERVICE" || true
    else
      run_with_timeout systemctl start "$SERVICE" || true
    fi
  fi

  if ! wait_for_health; then
    if [[ -x "\$PROJECT_ROOT/.venv/bin/real2sim-platform" ]]; then
      if ! pgrep -f "\$PROJECT_ROOT/.venv/bin/real2sim-platform" >/dev/null 2>&1; then
        cd "\$PROJECT_ROOT" || exit 1
        REAL2SIM_PLATFORM_CONFIG="\$PROJECT_ROOT/config.toml" nohup "\$PROJECT_ROOT/.venv/bin/real2sim-platform" >> "\$LOG" 2>&1 &
      fi
      wait_for_health || true
    fi
  fi

  launch_chrome_app() {
    local browser="\$1"
    mkdir -p "\$HOME/.config/real2sim-platform-chrome"
    nohup "\$browser" \\
      --app="\$URL" \\
      --new-window \\
      --user-data-dir="\$HOME/.config/real2sim-platform-chrome" \\
      --no-first-run \\
      --disable-sync \\
      --disable-translate \\
      --disable-background-networking \\
      --disable-component-update \\
      --disable-domain-reliability \\
      --disable-client-side-phishing-detection \\
      --no-default-browser-check \\
      --no-service-autorun \\
      --disable-features=Translate,TranslateUI,AutofillServerCommunication \\
      --window-size=1440,900 \\
      --force-device-scale-factor=1 \\
      >/dev/null 2>&1 &
  }

  if command -v google-chrome >/dev/null 2>&1; then
    launch_chrome_app google-chrome
  elif command -v chromium-browser >/dev/null 2>&1; then
    launch_chrome_app chromium-browser
  elif command -v xdg-open >/dev/null 2>&1; then
    nohup xdg-open "\$URL" >/dev/null 2>&1 &
  elif command -v firefox >/dev/null 2>&1; then
    nohup firefox "\$URL" >/dev/null 2>&1 &
  else
    echo "No browser opener found. Open \$URL manually."
  fi

  if ! health_ok; then
    echo "Real2Sim did not become healthy. Check: \$LOG"
    if command -v x-terminal-emulator >/dev/null 2>&1; then
      x-terminal-emulator -e "bash -lc 'tail -n 120 \"\$LOG\"; echo; read -p \"Press Enter to close\"'" &
    elif command -v gnome-terminal >/dev/null 2>&1; then
      gnome-terminal -- bash -lc "tail -n 120 \"\$LOG\"; echo; read -p 'Press Enter to close'" &
    fi
  fi
} >> "\$LOG" 2>&1
EOF
  chmod 0755 "$launcher"
  chown "$owner_group" "$launcher" "$bin_dir" "$state_dir" 2>/dev/null || true
  printf '%s\n' "$launcher"
}

write_icon() {
  local home_dir="$1"
  local owner_group="$2"
  local icon_dir="$home_dir/.local/share/icons"
  local icon_file="$icon_dir/real2sim-platform.svg"
  mkdir -p "$icon_dir"
  cat > "$icon_file" <<'EOF'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128">
  <defs>
    <linearGradient id="g" x1="16" y1="16" x2="112" y2="112" gradientUnits="userSpaceOnUse">
      <stop stop-color="#66ead4"/>
      <stop offset="0.55" stop-color="#23bca4"/>
      <stop offset="1" stop-color="#6aa8ff"/>
    </linearGradient>
  </defs>
  <rect width="128" height="128" rx="28" fill="#071119"/>
  <path d="M64 16 104 39v50L64 112 24 89V39L64 16Z" fill="none" stroke="url(#g)" stroke-width="8" stroke-linejoin="round"/>
  <path d="M64 36 84 47.5v25L64 84 44 72.5v-25L64 36Z" fill="none" stroke="#9ff7ea" stroke-width="6" stroke-linejoin="round"/>
  <path d="M64 36v48M44 48l40 24M84 48 44 72" stroke="#2fdbc4" stroke-width="4" stroke-linecap="round"/>
</svg>
EOF
  chmod 0644 "$icon_file"
  chown "$owner_group" "$icon_file" "$icon_dir" 2>/dev/null || true
  printf '%s\n' "$icon_file"
}

clean_desktop_dir() {
  local desktop_dir="$1"
  local keep_file="$desktop_dir/$DESKTOP_FILE_NAME"
  local backup_dir="$desktop_dir/.real2sim_desktop_backup_$timestamp"
  local moved=0
  shopt -s nullglob dotglob
  local item base
  for item in "$desktop_dir"/* "$desktop_dir"/.[!.]* "$desktop_dir"/..?*; do
    [[ -e "$item" ]] || continue
    base="$(basename "$item")"
    case "$base" in
      "$DESKTOP_FILE_NAME"|".directory"|".real2sim_desktop_backup_"*) continue ;;
    esac
    if [[ "$item" == "$keep_file" ]]; then
      continue
    fi
    mkdir -p "$backup_dir"
    mv "$item" "$backup_dir/"
    moved=$((moved + 1))
  done
  shopt -u nullglob dotglob
  if [[ "$moved" == "0" && -d "$backup_dir" ]]; then
    rmdir "$backup_dir" 2>/dev/null || true
  fi
  printf '%s\n' "$moved"
}

write_desktop_file() {
  local desktop_dir="$1"
  local owner_group="$2"
  local launcher="$3"
  local icon_file="$4"
  local desktop_file="$desktop_dir/$DESKTOP_FILE_NAME"

  cat > "$desktop_file" <<EOF
[Desktop Entry]
Type=Application
Name=$DESKTOP_NAME
Comment=启动 Real2Sim 服务并打开平台
Exec=$launcher
Icon=$icon_file
Terminal=false
Categories=Development;Science;
StartupNotify=true
EOF
  chmod 0755 "$desktop_file"
  chown "$owner_group" "$desktop_file" 2>/dev/null || true
  if command -v gio >/dev/null 2>&1; then
    gio set "$desktop_file" metadata::trusted true >/dev/null 2>&1 || true
  fi
  printf '%s\n' "$desktop_file"
}

main() {
  if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo "warning: project root does not exist yet: $PROJECT_ROOT" >&2
  fi

  local desktop_dir owner_group home_dir launcher icon_file moved desktop_file
  local count=0
  while IFS= read -r desktop_dir; do
    [[ -d "$desktop_dir" ]] || continue
    owner_group="$(owner_for_path "$desktop_dir")"
    home_dir="$(dirname "$desktop_dir")"
    if [[ "$CREATE_LAUNCHER" == "1" ]]; then
      launcher="$(write_launcher_script "$home_dir" "$owner_group")"
      icon_file="$(write_icon "$home_dir" "$owner_group")"
      desktop_file="$(write_desktop_file "$desktop_dir" "$owner_group" "$launcher" "$icon_file")"
    else
      desktop_file="$desktop_dir/$DESKTOP_FILE_NAME"
    fi
    if [[ "$CLEAN_DESKTOP" == "1" ]]; then
      moved="$(clean_desktop_dir "$desktop_dir")"
    else
      moved=0
    fi
    echo "desktop=$desktop_dir launcher=$desktop_file moved_to_backup=$moved"
    count=$((count + 1))
  done < <(discover_desktop_dirs)

  echo "Real2Sim desktop launcher setup complete. desktop_dirs=$count"
}

main
