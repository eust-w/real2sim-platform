#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m real2sim_platform.server --host 127.0.0.1 --port 18126
