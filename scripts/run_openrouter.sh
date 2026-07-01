#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "Missing OPENROUTER_API_KEY"
  echo "Run: export OPENROUTER_API_KEY='your-key'"
  exit 1
fi

MODEL_PROVIDER=openrouter \
MODEL_ID="${MODEL_ID:-deepseek/deepseek-v4-flash}" \
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_apptainer.sh"
