#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_PATH="${IMAGE_PATH:-$REPO_DIR/chemspacecopilot.sif}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
AGENT_TEAM="${AGENT_TEAM:-main}"
CHAINLIT_AUTH_SECRET="${CHAINLIT_AUTH_SECRET:-18074e1b85d9f6dde64ae2faffaa85f4bcdb7435b17d1de20addbde9d7e3e63f}"
MODEL_PROVIDER="${MODEL_PROVIDER:-deepseek}"
MODEL_ID="${MODEL_ID:-deepseek-chat}"
MODEL_MAX_TOKENS="${MODEL_MAX_TOKENS:-8192}"

if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "Missing image: $IMAGE_PATH"
  echo "Build it with: apptainer build chemspacecopilot.sif scripts/chemspacecopilot.def"
  exit 1
fi

env_args=(
  --env PYTHONPATH=/app/src
  --env CS_COPILOT_AGENT_TEAM="$AGENT_TEAM"
  --env CHAINLIT_AUTH_SECRET="$CHAINLIT_AUTH_SECRET"
  --env MODEL_PROVIDER="$MODEL_PROVIDER"
  --env MODEL_ID="$MODEL_ID"
  --env MODEL_MAX_TOKENS="$MODEL_MAX_TOKENS"
)

for key in DEEPSEEK_API_KEY OPENROUTER_API_KEY OLLAMA_HOST AWS_REGION; do
  if [[ -n "${!key:-}" ]]; then
    env_args+=(--env "$key=${!key}")
  fi
done

apptainer exec --nv \
  --bind "$REPO_DIR:/app" \
  "${env_args[@]}" \
  "$IMAGE_PATH" \
  bash -lc "source /opt/venv/bin/activate && cd /app && chainlit run chainlit_app.py --host '$HOST' --port '$PORT'"
