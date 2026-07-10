#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KUBECONFIG_PATH="${REAL2SIM_POC_KUBECONFIG:-/Users/d-robotics/Downloads/poc.config}"
NAMESPACE="real2sim-poc"
GPU_NODE="${REAL2SIM_POC_GPU_NODE:-192.168.80.136}"
REGISTRY="${REAL2SIM_POC_REGISTRY:-ccr-29eug8s3-pub.cnc.bj.baidubce.com/dataflow-poc}"
SOURCE_COMMIT="$(git -C "$ROOT" rev-parse --short=12 HEAD)"
BUILD_ID="${REAL2SIM_POC_BUILD_ID:-${SOURCE_COMMIT}-$(date -u +%Y%m%d%H%M%S)}"
OUT="${REAL2SIM_POC_ACCEPTANCE_OUT:-$ROOT/acceptance_runs/poc/$BUILD_ID}"
BUILD_IMAGES=1
DEPLOY_APP=1
ONLY_COMPONENT="all"
tls_dir=""
context_archive=""
rendered_manifest=""
trap 'rm -rf "${tls_dir:-}" "${context_archive:-}" "${rendered_manifest:-}"' EXIT

for arg in "$@"; do
  case "$arg" in
    --skip-build) BUILD_IMAGES=0 ;;
    --build-only) DEPLOY_APP=0 ;;
    --frontend-only) ONLY_COMPONENT="frontend" ;;
    --api-only) ONLY_COMPONENT="api" ;;
    --worker-only) ONLY_COMPONENT="worker" ;;
    --help)
      echo "usage: $0 [--skip-build] [--build-only] [--frontend-only|--api-only|--worker-only]"
      exit 0
      ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

K=(kubectl --kubeconfig "$KUBECONFIG_PATH")
mkdir -p "$OUT"

require() { command -v "$1" >/dev/null || { echo "missing required command: $1" >&2; exit 1; }; }
for command in kubectl jq openssl tar git sed; do require "$command"; done
test -f "$KUBECONFIG_PATH" || { echo "kubeconfig not found: $KUBECONFIG_PATH" >&2; exit 1; }
test "$(git -C "$ROOT" branch --show-current)" = "feat/poc-cluster-productization" || {
  echo "deployment is only allowed from feat/poc-cluster-productization" >&2
  exit 1
}

gpu_capacity="$("${K[@]}" get node "$GPU_NODE" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')"
gpu_in_use="$("${K[@]}" get pods -A --field-selector "spec.nodeName=$GPU_NODE" -o json | jq '[.items[] | select(.status.phase=="Running" or .status.phase=="Pending") | ([.spec.containers[]?.resources.requests["nvidia.com/gpu"] // "0" | tonumber] | add // 0)] | add // 0')"
if (( gpu_capacity - gpu_in_use < 1 )); then
  echo "A800 node has no free GPU: capacity=$gpu_capacity in_use=$gpu_in_use" >&2
  exit 1
fi
printf 'gpu_capacity=%s\ngpu_in_use=%s\nbuild_id=%s\n' "$gpu_capacity" "$gpu_in_use" "$BUILD_ID" > "$OUT/deploy_preflight.txt"

"${K[@]}" apply -f "$ROOT/deploy/poc/bootstrap.yaml"

ensure_host_directory() {
  local pod="real2sim-hostpath-bootstrap"
  "${K[@]}" -n "$NAMESPACE" delete pod "$pod" --ignore-not-found --wait=true >/dev/null
  cat <<YAML | "${K[@]}" apply -f - >/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: $pod
  namespace: $NAMESPACE
  labels:
    app.kubernetes.io/part-of: real2sim-poc
    app.kubernetes.io/component: storage-bootstrap
spec:
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: "$GPU_NODE"
  containers:
    - name: bootstrap
      image: busybox:1.36
      command: [sh, -eu, -c]
      args:
        - mkdir -p /host-var-lib/real2sim-poc && chmod 0770 /host-var-lib/real2sim-poc && chown 10001:10001 /host-var-lib/real2sim-poc
      resources:
        requests: {cpu: 20m, memory: 32Mi}
        limits: {cpu: 100m, memory: 64Mi}
      securityContext:
        runAsUser: 0
        allowPrivilegeEscalation: false
        capabilities: {drop: [ALL], add: [CHOWN, FOWNER]}
      volumeMounts:
        - {name: host-var-lib, mountPath: /host-var-lib}
  volumes:
    - name: host-var-lib
      hostPath: {path: /var/lib, type: Directory}
YAML
  if ! "${K[@]}" -n "$NAMESPACE" wait --for=jsonpath='{.status.phase}'=Succeeded "pod/$pod" --timeout=180s; then
    "${K[@]}" -n "$NAMESPACE" describe "pod/$pod" >&2 || true
    exit 1
  fi
  "${K[@]}" -n "$NAMESPACE" delete pod "$pod" --wait=true >/dev/null
}

ensure_host_directory

clone_secret() {
  local source_namespace="$1" source_name="$2" target_name="$3"
  "${K[@]}" -n "$source_namespace" get secret "$source_name" -o json |
    jq --arg ns "$NAMESPACE" --arg name "$target_name" '
      del(.metadata.uid,.metadata.resourceVersion,.metadata.creationTimestamp,.metadata.managedFields,.metadata.ownerReferences) |
      .metadata.namespace=$ns | .metadata.name=$name |
      .metadata.labels={"app.kubernetes.io/part-of":"real2sim-poc"}' |
    "${K[@]}" apply -f - >/dev/null
}

clone_secret droboticflow droboticflow-registry real2sim-registry
clone_secret droboticflow droboticflow-bos-workspace real2sim-bos

if ! "${K[@]}" -n "$NAMESPACE" get secret real2sim-app-secrets >/dev/null 2>&1; then
  admin_password="$(openssl rand -base64 30 | tr -d '/+=' | cut -c1-28)Aa1!"
  session_secret="$(openssl rand -hex 32)"
  "${K[@]}" -n "$NAMESPACE" create secret generic real2sim-app-secrets \
    --from-literal=bootstrap-admin-password="$admin_password" \
    --from-literal=session-secret="$session_secret" \
    --dry-run=client -o yaml |
    "${K[@]}" label --local -f - app.kubernetes.io/part-of=real2sim-poc -o yaml |
    "${K[@]}" apply -f - >/dev/null
fi

if ! "${K[@]}" -n "$NAMESPACE" get secret real2sim-tls >/dev/null 2>&1; then
  tls_dir="$(mktemp -d)"
  openssl req -x509 -newkey rsa:3072 -sha256 -nodes -days 365 \
    -subj "/CN=real2sim-poc.local" \
    -addext "subjectAltName=DNS:real2sim-poc.local,DNS:localhost,IP:127.0.0.1" \
    -keyout "$tls_dir/tls.key" -out "$tls_dir/tls.crt" >/dev/null 2>&1
  "${K[@]}" -n "$NAMESPACE" create secret tls real2sim-tls \
    --key "$tls_dir/tls.key" --cert "$tls_dir/tls.crt" --dry-run=client -o yaml |
    "${K[@]}" label --local -f - app.kubernetes.io/part-of=real2sim-poc -o yaml |
    "${K[@]}" apply -f - >/dev/null
fi

"${K[@]}" -n "$NAMESPACE" get secret real2sim-app-secrets -o jsonpath='{.data.bootstrap-admin-password}' |
  base64 --decode > "$OUT/bootstrap-admin-password.txt"
chmod 600 "$OUT/bootstrap-admin-password.txt"
printf 'admin\n' > "$OUT/bootstrap-admin-username.txt"
chmod 600 "$OUT/bootstrap-admin-username.txt"

prepare_context() {
  context_archive="$(mktemp -t real2sim-poc-context.XXXXXX.tgz)"
  tar -C "$ROOT" -czf "$context_archive" \
    --exclude='.git' --exclude='.venv' --exclude='.pytest_cache' --exclude='workspace' \
    --exclude='acceptance_runs' --exclude='qa' --exclude='docs' --exclude='tests' .
  local loader="real2sim-context-${BUILD_ID//[^a-zA-Z0-9-]/-}"
  loader="${loader:0:60}"
  "${K[@]}" -n "$NAMESPACE" delete pod "$loader" --ignore-not-found --wait=true >/dev/null
  "${K[@]}" -n "$NAMESPACE" run "$loader" \
    --image=busybox:1.36 --restart=Never \
    --labels="app.kubernetes.io/part-of=real2sim-poc,app.kubernetes.io/component=build-context" \
    --overrides="$(jq -cn --arg node "$GPU_NODE" '{spec:{nodeSelector:{"kubernetes.io/hostname":$node},containers:[{name:"context-loader",image:"busybox:1.36",command:["sh","-c","sleep 3600"],resources:{requests:{cpu:"100m",memory:"64Mi"},limits:{cpu:"500m",memory:"256Mi"}},volumeMounts:[{name:"data",mountPath:"/data"}]}],volumes:[{name:"data",persistentVolumeClaim:{claimName:"real2sim-poc-data"}}]}}')" >/dev/null
  "${K[@]}" -n "$NAMESPACE" wait --for=condition=Ready "pod/$loader" --timeout=180s
  "${K[@]}" -n "$NAMESPACE" cp "$context_archive" "$loader:/tmp/context.tgz"
  "${K[@]}" -n "$NAMESPACE" exec "$loader" -- sh -eu -c \
    "mkdir -p '/data/build-context/$BUILD_ID' /data/workspace /data/cache /data/scratch; rm -rf '/data/build-context/$BUILD_ID/'*; tar -xzf /tmp/context.tgz -C '/data/build-context/$BUILD_ID'; chown -R 10001:10001 /data/workspace /data/cache /data/scratch; chmod 0770 /data/workspace /data/cache /data/scratch"
  "${K[@]}" -n "$NAMESPACE" delete pod "$loader" --wait=true >/dev/null
}

build_image() {
  local component="$1" dockerfile="$2" destination="$3"
  local job="real2sim-build-${component}-${BUILD_ID//[^a-zA-Z0-9-]/-}"
  job="${job:0:60}"
  "${K[@]}" -n "$NAMESPACE" delete job "$job" --ignore-not-found --wait=true >/dev/null
  cat <<YAML | "${K[@]}" apply -f - >/dev/null
apiVersion: batch/v1
kind: Job
metadata:
  name: $job
  namespace: $NAMESPACE
  labels:
    app.kubernetes.io/part-of: real2sim-poc
    app.kubernetes.io/component: image-build
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app.kubernetes.io/part-of: real2sim-poc
        app.kubernetes.io/component: image-build
    spec:
      restartPolicy: Never
      nodeSelector:
        kubernetes.io/hostname: "$GPU_NODE"
      containers:
        - name: kaniko
          image: gcr.io/kaniko-project/executor:v1.23.2-debug
          args:
            - --context=dir:///data/build-context/$BUILD_ID
            - --dockerfile=/data/build-context/$BUILD_ID/$dockerfile
            - --destination=$destination
            - --build-arg=BUILD_ID=$BUILD_ID
            - --build-arg=SOURCE_COMMIT=$SOURCE_COMMIT
            - --snapshot-mode=redo
            - --use-new-run
            - --cache=true
            - --cache-repo=$REGISTRY/real2sim-build-cache
            - --verbosity=info
          resources:
            requests: {cpu: "2", memory: 4Gi}
            limits: {cpu: "16", memory: 32Gi}
          volumeMounts:
            - {name: data, mountPath: /data}
            - {name: docker-config, mountPath: /kaniko/.docker, readOnly: true}
      volumes:
        - name: data
          persistentVolumeClaim: {claimName: real2sim-poc-data}
        - name: docker-config
          secret:
            secretName: real2sim-registry
            items:
              - {key: .dockerconfigjson, path: config.json}
YAML
  local deadline=$((SECONDS + 7200))
  local build_state="running"
  while (( SECONDS < deadline )); do
    local succeeded failed
    succeeded="$("${K[@]}" -n "$NAMESPACE" get "job/$job" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
    failed="$("${K[@]}" -n "$NAMESPACE" get "job/$job" -o jsonpath='{.status.failed}' 2>/dev/null || true)"
    if [[ "$succeeded" == "1" ]]; then
      build_state="succeeded"
      break
    fi
    if [[ -n "$failed" && "$failed" != "0" ]]; then
      build_state="failed"
      break
    fi
    sleep 5
  done
  if [[ "$build_state" != "succeeded" ]]; then
    "${K[@]}" -n "$NAMESPACE" logs "job/$job" --all-containers=true > "$OUT/build-${component}.log" 2>&1 || true
    "${K[@]}" -n "$NAMESPACE" describe "job/$job" > "$OUT/build-${component}-describe.txt" 2>&1 || true
    echo "image build $component ended as $build_state; see $OUT/build-${component}.log" >&2
    return 1
  fi
  "${K[@]}" -n "$NAMESPACE" logs "job/$job" --all-containers=true > "$OUT/build-${component}.log"
  "${K[@]}" -n "$NAMESPACE" delete job "$job" --wait=true >/dev/null
}

FRONTEND_IMAGE="$REGISTRY/real2sim-frontend:$BUILD_ID"
API_IMAGE="$REGISTRY/real2sim-api:$BUILD_ID"
WORKER_IMAGE="$REGISTRY/real2sim-worker:$BUILD_ID"
printf '%s\n' "$FRONTEND_IMAGE" "$API_IMAGE" "$WORKER_IMAGE" > "$OUT/images.txt"

if [[ "$BUILD_IMAGES" == "1" ]]; then
  prepare_context
  [[ "$ONLY_COMPONENT" == "all" || "$ONLY_COMPONENT" == "frontend" ]] && build_image frontend docker/frontend.Dockerfile "$FRONTEND_IMAGE"
  [[ "$ONLY_COMPONENT" == "all" || "$ONLY_COMPONENT" == "api" ]] && build_image api docker/api.Dockerfile "$API_IMAGE"
  [[ "$ONLY_COMPONENT" == "all" || "$ONLY_COMPONENT" == "worker" ]] && build_image worker docker/worker.Dockerfile "$WORKER_IMAGE"
fi

if [[ "$DEPLOY_APP" == "1" ]]; then
  if [[ "$ONLY_COMPONENT" == "all" ]]; then
    rendered_manifest="$(mktemp -t real2sim-poc-manifest.XXXXXX.yaml)"
    sed \
      -e "s|__BUILD_ID__|$BUILD_ID|g" \
      -e "s|__FRONTEND_IMAGE__|$FRONTEND_IMAGE|g" \
      -e "s|__API_IMAGE__|$API_IMAGE|g" \
      -e "s|__WORKER_IMAGE__|$WORKER_IMAGE|g" \
      "$ROOT/deploy/poc/app-template.yaml" > "$rendered_manifest"
    "${K[@]}" apply -f "$rendered_manifest"
    "${K[@]}" -n "$NAMESPACE" rollout status deployment/real2sim-api --timeout=600s
    "${K[@]}" -n "$NAMESPACE" rollout status deployment/real2sim-frontend --timeout=600s
    "${K[@]}" -n "$NAMESPACE" rollout status deployment/real2sim-worker --timeout=3600s
  elif [[ "$ONLY_COMPONENT" == "frontend" ]]; then
    "${K[@]}" -n "$NAMESPACE" set image deployment/real2sim-frontend "frontend=$FRONTEND_IMAGE"
    "${K[@]}" -n "$NAMESPACE" rollout status deployment/real2sim-frontend --timeout=600s
  elif [[ "$ONLY_COMPONENT" == "api" ]]; then
    "${K[@]}" -n "$NAMESPACE" set image deployment/real2sim-api "api=$API_IMAGE"
    "${K[@]}" -n "$NAMESPACE" rollout status deployment/real2sim-api --timeout=600s
  elif [[ "$ONLY_COMPONENT" == "worker" ]]; then
    "${K[@]}" -n "$NAMESPACE" set image deployment/real2sim-worker "worker=$WORKER_IMAGE" "runtime-gate=$WORKER_IMAGE"
    "${K[@]}" -n "$NAMESPACE" rollout status deployment/real2sim-worker --timeout=3600s
  fi
  "${K[@]}" -n "$NAMESPACE" get pods,svc,pvc -l app.kubernetes.io/part-of=real2sim-poc -o wide > "$OUT/resources.txt"
  "${K[@]}" -n "$NAMESPACE" get svc real2sim-frontend -o json > "$OUT/frontend-service.json"
fi

printf 'build_id=%s\nnamespace=%s\ncredentials=%s\n' "$BUILD_ID" "$NAMESPACE" "$OUT/bootstrap-admin-password.txt"
