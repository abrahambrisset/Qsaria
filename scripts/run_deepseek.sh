#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "Missing DEEPSEEK_API_KEY"
  echo "Run: export DEEPSEEK_API_KEY='your-key'"
  exit 1
fi

MODEL_PROVIDER=deepseek \
MODEL_ID="${MODEL_ID:-deepseek-chat}" \
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_apptainer.sh"
