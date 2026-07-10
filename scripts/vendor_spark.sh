#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${SPARK_VERSION:-2.1.0}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cd "$TMP"
npm pack "@sparkjsdev/spark@$VERSION" >/dev/null
tar -xzf "sparkjsdev-spark-$VERSION.tgz"
mkdir -p "$ROOT/src/real2sim_platform/static/vendor"
cp package/dist/spark.module.min.js "$ROOT/src/real2sim_platform/static/vendor/spark.module.min.js"
cp package/LICENSE "$ROOT/src/real2sim_platform/static/vendor/spark.LICENSE"
printf '%s\n' "$VERSION" > "$ROOT/src/real2sim_platform/static/vendor/spark.VERSION"
echo "vendored @sparkjsdev/spark@$VERSION"
