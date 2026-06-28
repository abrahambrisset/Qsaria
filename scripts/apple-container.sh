#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v container >/dev/null 2>&1; then
  echo "Apple container CLI not found. Install it first: https://github.com/apple/container"
  exit 1
fi

set -a
[ -f .env ] && . ./.env
set +a

IMAGE="${IMAGE:-cs-copilot-apple:latest}"
NAME="${NAME:-cs-copilot-apple}"
PORT="${CHAINLIT_PORT:-8000}"
PUBLISH_HOST="${APPLE_CONTAINER_PUBLISH_HOST:-127.0.0.1}"
CPUS="${APPLE_CONTAINER_CPUS:-8}"
MEMORY="${APPLE_CONTAINER_MEMORY:-12G}"
MIN_FREE_GB="${APPLE_CONTAINER_MIN_FREE_GB:-10}"
BUILD_HEADROOM_GB="${APPLE_CONTAINER_BUILD_HEADROOM_GB:-25}"
MIN_FREE_KB=$((MIN_FREE_GB * 1024 * 1024))

free_kb() {
  df -Pk / | awk 'NR == 2 { print $4 }'
}

require_free_gb() {
  local min_gb="$1"
  local label="$2"
  local min_kb=$((min_gb * 1024 * 1024))
  local available_kb
  available_kb="$(free_kb)"
  if [ "$available_kb" -le "$min_kb" ]; then
    echo "Not enough free disk for $label: need >${min_gb}G, have ~$((available_kb / 1024 / 1024))G."
    exit 1
  fi
}

cleanup_builder() {
  container builder delete --force >/dev/null 2>&1 || true
}

mkdir -p data .files models

BUILD_CONTEXT="$(mktemp -d "/tmp/cs-copilot-apple-context.XXXXXX")"
trap 'rm -rf "$BUILD_CONTEXT"' EXIT

for path in \
  Dockerfile \
  docker-entrypoint.sh \
  pyproject.toml \
  uv.lock \
  README.md \
  package.json \
  chainlit.md \
  chainlit.toml \
  chainlit_app.py \
  .modelconf
do
  [ -e "$path" ] && rsync -a "$path" "$BUILD_CONTEXT/"
done

for path in public src examples; do
  if [ -d "$path" ]; then
    mkdir -p "$BUILD_CONTEXT/$path"
    rsync -a "$path/" "$BUILD_CONTEXT/$path/"
  fi
done

if [ -d prisma ]; then
  mkdir -p "$BUILD_CONTEXT/prisma"
  rsync -a prisma/ "$BUILD_CONTEXT/prisma/"
fi

container system start >/dev/null 2>&1 || true
if [ "${APPLE_CONTAINER_SKIP_BUILD:-false}" != "1" ]; then
  build_args=(-t "$IMAGE")
  if [ "${APPLE_CONTAINER_NO_CACHE:-false}" = "1" ]; then
    build_args+=(--no-cache)
  fi
  require_free_gb "$((MIN_FREE_GB + BUILD_HEADROOM_GB))" "Apple container build"
  container build "${build_args[@]}" "$BUILD_CONTEXT" &
  build_pid=$!
  while kill -0 "$build_pid" 2>/dev/null; do
    if [ "$(free_kb)" -le "$MIN_FREE_KB" ]; then
      echo "Free disk dropped below ${MIN_FREE_GB}G; stopping Apple container build."
      kill "$build_pid" 2>/dev/null || true
      wait "$build_pid" 2>/dev/null || true
      cleanup_builder
      exit 1
    fi
    sleep 2
  done
  if ! wait "$build_pid"; then
    cleanup_builder
    exit 1
  fi
  cleanup_builder
fi
require_free_gb "$MIN_FREE_GB" "Apple container run"

env_args=(
  -e "PYTHONPATH=/app/src"
  -e "CS_COPILOT_STORAGE_ROOT=/app/.files"
  -e "USE_S3=${USE_S3:-false}"
  -e "CS_COPILOT_AGENT_TEAM=${CS_COPILOT_AGENT_TEAM:-qsar}"
  -e "AGNO_TELEMETRY=${AGNO_TELEMETRY:-false}"
)

for key in MODEL_PROVIDER MODEL_ID OLLAMA_HOST DEEPSEEK_API_KEY CHAINLIT_AUTH_SECRET AWS_REGION; do
  if [ -n "${!key:-}" ]; then
    env_args+=(-e "$key=${!key}")
  fi
done

exec container run --rm --name "$NAME" \
  --cpus "$CPUS" \
  --memory "$MEMORY" \
  -p "$PUBLISH_HOST:$PORT:8000" \
  -v "$ROOT/data:/app/data" \
  -v "$ROOT/.files:/app/.files" \
  -v "$ROOT/models:/app/models" \
  -v "$ROOT/src:/app/src" \
  -v "$ROOT/public:/app/public" \
  -v "$ROOT/examples:/app/examples" \
  -v "$ROOT/chainlit_app.py:/app/chainlit_app.py" \
  "${env_args[@]}" \
  "$IMAGE"
