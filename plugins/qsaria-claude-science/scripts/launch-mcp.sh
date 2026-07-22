#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
REPO_ROOT=${SCRIPT_DIR:h:h:h}
PYTHON=${REPO_ROOT}/.venv/bin/python

if [[ ! -x ${PYTHON} ]]; then
  print -u2 "Qsaria Claude Science: missing Python environment at ${PYTHON}"
  exit 2
fi

export CS_COPILOT_STORAGE_ROOT=${REPO_ROOT}/.files
export USE_S3=false
export QSARIA_MODEL_CATALOG_PATH=${REPO_ROOT}/data/model_assets/catalog/qsaria_model_catalog.json

if [[ -z ${QSARIA_SCIENCE_ARTIFACT_ROOT:-} ]]; then
  print -u2 "Qsaria Claude Science: QSARIA_SCIENCE_ARTIFACT_ROOT is required"
  exit 2
fi
if [[ ! -d ${QSARIA_SCIENCE_ARTIFACT_ROOT} ]]; then
  print -u2 "Qsaria Claude Science: artifact root does not exist"
  exit 2
fi
export QSARIA_MCP_ALLOWED_INPUT_ROOTS=${QSARIA_SCIENCE_ARTIFACT_ROOT}

cd ${REPO_ROOT}
exec ${PYTHON} -m cs_copilot.mcp \
  --profile qsaria \
  --llm-policy disabled \
  --log-level info \
  --no-chatgpt-compat \
  --no-prompts \
  --no-resources
