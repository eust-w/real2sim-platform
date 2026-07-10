#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${THREE_VERSION:-0.185.1}"
VENDOR="$ROOT/src/real2sim_platform/static/vendor/three"
BASE="https://unpkg.com/three@$VERSION"

mkdir -p "$VENDOR/addons/controls" "$VENDOR/addons/loaders" "$VENDOR/addons/postprocessing" "$VENDOR/addons/utils"
curl -fsSL "$BASE/build/three.module.js" -o "$VENDOR/three.module.js"
curl -fsSL "$BASE/build/three.core.js" -o "$VENDOR/three.core.js"
curl -fsSL "$BASE/examples/jsm/controls/OrbitControls.js" -o "$VENDOR/addons/controls/OrbitControls.js"
curl -fsSL "$BASE/examples/jsm/loaders/OBJLoader.js" -o "$VENDOR/addons/loaders/OBJLoader.js"
curl -fsSL "$BASE/examples/jsm/loaders/PLYLoader.js" -o "$VENDOR/addons/loaders/PLYLoader.js"
curl -fsSL "$BASE/examples/jsm/loaders/GLTFLoader.js" -o "$VENDOR/addons/loaders/GLTFLoader.js"
curl -fsSL "$BASE/examples/jsm/postprocessing/Pass.js" -o "$VENDOR/addons/postprocessing/Pass.js"
curl -fsSL "$BASE/examples/jsm/utils/BufferGeometryUtils.js" -o "$VENDOR/addons/utils/BufferGeometryUtils.js"
curl -fsSL "$BASE/examples/jsm/utils/SkeletonUtils.js" -o "$VENDOR/addons/utils/SkeletonUtils.js"
printf '%s\n' "$VERSION" > "$ROOT/src/real2sim_platform/static/vendor/three.VERSION"
echo "vendored three@$VERSION"
