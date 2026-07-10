#!/usr/bin/env bash
set -euo pipefail
KUBECONFIG_PATH="${REAL2SIM_POC_KUBECONFIG:-/Users/d-robotics/Downloads/poc.config}"
K=(kubectl --kubeconfig "$KUBECONFIG_PATH" -n real2sim-poc)
for deployment in real2sim-frontend real2sim-api real2sim-worker; do
  "${K[@]}" rollout undo "deployment/$deployment"
  "${K[@]}" rollout status "deployment/$deployment" --timeout=3600s
done
echo "rollback completed; PV and BOS objects were preserved"
