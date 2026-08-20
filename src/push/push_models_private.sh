#!/usr/bin/env bash
set -euo pipefail

# Wrapper to publish model artifacts to Hugging Face as PRIVATE repos only.
# Requires: HF_TOKEN env var or prior `huggingface-cli login`.

BASE_DIR="${1:-./models}"
USERNAME="${2:-romaricnadjire}"
MODE="${3:-live}" # live | dry-run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${MODE}" == "dry-run" ]]; then
  exec "${PYTHON_BIN}" "${SCRIPT_DIR}/push_models_private.py" \
    --base-dir "${BASE_DIR}" \
    --username "${USERNAME}" \
    --dry-run
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/push_models_private.py" \
  --base-dir "${BASE_DIR}" \
  --username "${USERNAME}"
