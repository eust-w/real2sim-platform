#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="$(cd "$ROOT/.." && pwd)"
KEY="${REAL2SIM_WUYING_KEY:-/Users/d-robotics/workSpace/document/80-private/orgs/drobotics/keys/wuying-39.106.101.95_ed25519}"
HOST="${REAL2SIM_WUYING_HOST:-root@39.106.101.95}"
SSH_CONNECT_TIMEOUT="${REAL2SIM_SSH_CONNECT_TIMEOUT:-20}"
REMOTE_BASE="${REAL2SIM_REMOTE_BASE:-/opt/real2sim2model}"
REMOTE="$REMOTE_BASE/real2sim-platform"
SERVICE="real2sim-platform.service"
BUILD_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
BUILD_ID="${REAL2SIM_PLATFORM_BUILD_ID:-$BUILD_TIME}"
RUNTIME_ROOT="${REAL2SIM_RUNTIME_ROOT:-/opt/real2sim_paper_runtime}"
ANYSPLAT_ROOT="${REAL2SIM_ANYSPLAT_ROOT:-$RUNTIME_ROOT/AnySplat}"
ANYSPLAT_PYTHON="${REAL2SIM_ANY_SPLAT_PYTHON:-$RUNTIME_ROOT/envs/anysplat/bin/python}"
CUDA_HOME_REMOTE="${CUDA_HOME_REMOTE:-/usr/local/cuda-12.1}"
RESTART=0
STATUS_ONLY=0
SETUP_DESKTOP=0

for arg in "$@"; do
  case "$arg" in
    --restart) RESTART=1 ;;
    --status) STATUS_ONLY=1 ;;
    --desktop-launcher) SETUP_DESKTOP=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

ssh_opts=(-i "$KEY" -o BatchMode=yes -o ConnectTimeout="$SSH_CONNECT_TIMEOUT" -o ServerAliveInterval=10 -o ServerAliveCountMax=2 -o StrictHostKeyChecking=accept-new)
ssh_cmd=(ssh "${ssh_opts[@]}" "$HOST")

if [[ "$STATUS_ONLY" == "1" ]]; then
  "${ssh_cmd[@]}" "systemctl status '$SERVICE' --no-pager || true; curl -fsS http://127.0.0.1:18126/api/health || true; echo; curl -fsS http://127.0.0.1:18126/api/doctor || true"
  exit 0
fi

if [[ ! -f "$ROOT/src/real2sim_platform/static/vendor/spark.module.min.js" ]]; then
  bash "$ROOT/scripts/vendor_spark.sh"
fi
if [[ ! -f "$ROOT/src/real2sim_platform/static/vendor/three/three.module.js" ]]; then
  bash "$ROOT/scripts/vendor_three.sh"
fi

"${ssh_cmd[@]}" "mkdir -p '$REMOTE_BASE'"
rsync -az --delete \
  --exclude '.git/' \
  --exclude 'real2sim-platform/workspace/' \
  --exclude 'real2sim-platform/.venv/' \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  -e "ssh -i $KEY -o BatchMode=yes -o ConnectTimeout=$SSH_CONNECT_TIMEOUT -o ServerAliveInterval=10 -o ServerAliveCountMax=2 -o StrictHostKeyChecking=accept-new" \
  "$PROJECT_ROOT/" "$HOST:$REMOTE_BASE/"

"${ssh_cmd[@]}" "cd '$REMOTE' && python3 -m venv .venv && .venv/bin/python -m pip install -q --upgrade pip && .venv/bin/python -m pip install -q -e ."
"${ssh_cmd[@]}" "cd '$REMOTE' && .venv/bin/python - <<'PY'
import importlib.util
required = ['huggingface_hub', 'pyarrow', 'PIL', 'imageio', 'cv2', 'mujoco']
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit('missing runtime deps: ' + ', '.join(missing))
PY
command -v ffmpeg >/dev/null
test -f '$REMOTE/src/real2sim_platform/static/vendor/spark.module.min.js'
test -f '$REMOTE/src/real2sim_platform/static/vendor/three/three.module.js'
test -d '$ANYSPLAT_ROOT'
test -x '$ANYSPLAT_PYTHON'"

"${ssh_cmd[@]}" "cat > '$REMOTE/config.toml' <<'EOF'
app_root = \"$REMOTE\"
workspace_dir = \"$REMOTE/workspace\"
host = \"127.0.0.1\"
port = 18126
default_renderer = \"spark\"
real2sim2model_root = \"$REMOTE_BASE\"
expert_policy_root = \"$REMOTE_BASE/real2sim-expert-policy\"
training_root = \"$REMOTE_BASE/real2sim-policy-training\"
inference_root = \"$REMOTE_BASE/real2sim-policy-inference\"
discoverse_root = \"/opt/gs_playground_real2sim_48q/DISCOVERSE\"
gs_playground_root = \"/opt/gs_playground_real2sim_48q\"
images23dgs_root = \"/opt/images23dgs_app/src\"
colmap_binary = \"/usr/local/bin/colmap\"
gsplat_python = \"$ANYSPLAT_PYTHON\"
runtime_root = \"$RUNTIME_ROOT\"
anysplat_root = \"$ANYSPLAT_ROOT\"
anysplat_python = \"$ANYSPLAT_PYTHON\"
cuda_home = \"$CUDA_HOME_REMOTE\"
runtime_env_script = \"/opt/gs_playground_real2sim_48q/scripts/real2sim_wuying_paper_env.sh\"
allowed_path_roots = \"$REMOTE_BASE:$REMOTE/workspace:/opt/gs_playground_real2sim_48q:$RUNTIME_ROOT:$ANYSPLAT_ROOT:/tmp\"
EOF"

"${ssh_cmd[@]}" "cat > /etc/systemd/system/$SERVICE <<'EOF'
[Unit]
Description=Real2Sim Platform
After=network.target

[Service]
Type=simple
WorkingDirectory=$REMOTE
Environment=REAL2SIM_PLATFORM_CONFIG=$REMOTE/config.toml
Environment=REAL2SIM_PLATFORM_BUILD_TIME=$BUILD_TIME
Environment=REAL2SIM_PLATFORM_BUILD_ID=$BUILD_ID
Environment=MUJOCO_GL=egl
ExecStart=$REMOTE/.venv/bin/real2sim-platform
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable '$SERVICE' >/dev/null"

if [[ "$RESTART" == "1" ]]; then
  "${ssh_cmd[@]}" "systemctl stop '$SERVICE' || true; port_pid=\$(ss -ltnp 2>/dev/null | sed -n 's/.*127\\.0\\.0\\.1:18126.*pid=\\([0-9][0-9]*\\).*/\\1/p' | head -1); if [ -n \"\$port_pid\" ]; then kill \"\$port_pid\" || true; fi; sleep 2; systemctl restart '$SERVICE'; sleep 2; systemctl status '$SERVICE' --no-pager --lines=20"
fi

if [[ "$SETUP_DESKTOP" == "1" ]]; then
  "${ssh_cmd[@]}" "bash '$REMOTE/scripts/setup_wuying_desktop_launcher.sh' --local --project-root '$REMOTE' --service '$SERVICE' --port 18126"
fi

echo "deployed to $HOST:$REMOTE"
echo "service: $SERVICE"
echo "health: ssh -i $KEY -L 18126:127.0.0.1:18126 $HOST"
echo "restart: bash scripts/deploy_wuying.sh --restart"
echo "desktop launcher: bash scripts/deploy_wuying.sh --restart --desktop-launcher"
