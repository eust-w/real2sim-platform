#!/usr/bin/env bash
set -euo pipefail

KEY="${REAL2SIM_WUYING_KEY:-/Users/d-robotics/workSpace/document/80-private/orgs/drobotics/keys/wuying-39.106.101.95_ed25519}"
HOST="${REAL2SIM_WUYING_HOST:-root@39.106.101.95}"
FIXTURE="${REAL2SIM_ACCEPTANCE_FIXTURE:-/opt/real2sim_paper_runtime/sam-3d-objects/notebook/images/shutterstock_modern_colorful_Interior_2620125197}"
MIN_FRAMES="${REAL2SIM_ACCEPTANCE_MIN_FRAMES:-30}"
TIMEOUT_SECONDS="${REAL2SIM_ACCEPTANCE_TIMEOUT_SECONDS:-3600}"

ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$HOST" \
  "REAL2SIM_ACCEPTANCE_FIXTURE='$FIXTURE' REAL2SIM_ACCEPTANCE_MIN_FRAMES='$MIN_FRAMES' REAL2SIM_ACCEPTANCE_TIMEOUT_SECONDS='$TIMEOUT_SECONDS' python3 - <<'PY'
import json
import os
import time
import urllib.error
import urllib.request

BASE = 'http://127.0.0.1:18126'
FIXTURE = os.environ['REAL2SIM_ACCEPTANCE_FIXTURE']
MIN_FRAMES = int(os.environ['REAL2SIM_ACCEPTANCE_MIN_FRAMES'])
TIMEOUT = int(os.environ['REAL2SIM_ACCEPTANCE_TIMEOUT_SECONDS'])


def api(path, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode('utf-8'))


def create_task(task_type, params):
    task = api('/api/tasks', {'type': task_type, 'params': params})
    print(f'{task_type}: {task[\"id\"]}', flush=True)
    return wait_task(task['id'])


def wait_task(task_id):
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        task = api(f'/api/tasks/{task_id}')
        if task['status'] in {'succeeded', 'failed', 'canceled', 'failed_recovered'}:
            print(json.dumps({'task_id': task_id, 'type': task['type'], 'status': task['status'], 'result': task.get('result'), 'error': task.get('error')}, ensure_ascii=False), flush=True)
            return task
        time.sleep(5)
    raise SystemExit(f'timeout waiting for {task_id}')


doctor = api('/api/doctor')
version = api('/api/version')
print(json.dumps({'doctor_ok': doctor['ok'], 'warnings': doctor.get('warnings'), 'version': version.get('version'), 'build_id': version.get('build_id')}, ensure_ascii=False), flush=True)

preflight = create_task('reconstruction_preflight', {'input_path': FIXTURE, 'min_frames': MIN_FRAMES})
if not preflight.get('result', {}).get('ok'):
    raise SystemExit('preflight failed; inspect reports for task ' + preflight['id'])

reconstruction = create_task('scene_reconstruction', {
    'input_path': FIXTURE,
    'dry_run': False,
    'renderer': 'spark',
    'simulators': ['mujoco', 'motrixsim'],
    'min_frames': MIN_FRAMES,
    'quality_preset': 'capture-high',
})
if reconstruction['status'] != 'succeeded':
    raise SystemExit('scene_reconstruction failed; inspect reports for task ' + reconstruction['id'])

scene_id = reconstruction['result']['scene_id']
validation = create_task('simulator_validation', {'scene_id': scene_id, 'simulators': ['mujoco', 'motrixsim'], 'level': 'runtime_if_available'})
print(json.dumps({'scene_id': scene_id, 'validation_task': validation['id'], 'validation_ok': validation.get('result', {}).get('ok')}, ensure_ascii=False), flush=True)
PY"
