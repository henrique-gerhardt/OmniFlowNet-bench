#!/usr/bin/env bash
set -euo pipefail

SCENARIO="${1:-official_reproduction}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
else
  echo "No Python interpreter found in PATH." >&2
  exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}:${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ -z "${OMNIFLOWNET_CHECKPOINT:-}" ]]; then
  while IFS= read -r _ckpt; do
    export OMNIFLOWNET_CHECKPOINT="${_ckpt}"
    break
  done < <(find /opt/checkpoints -maxdepth 1 -type f -name '*.caffemodel' 2>/dev/null | sort)
fi

mkdir -p "${SCRIPT_DIR}/results/raw_logs"
mkdir -p "${SCRIPT_DIR}/results/optional_predictions"
mkdir -p "${SCRIPT_DIR}/outputs"

"${PYTHON_BIN}" "${SCRIPT_DIR}/export_results.py" --phase metadata --scenario "${SCENARIO}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_inference.py" --scenario "${SCENARIO}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate.py" --scenario "${SCENARIO}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/profile.py" --scenario "${SCENARIO}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/export_results.py" --phase finalize --scenario "${SCENARIO}"

echo "Benchmark contract finished for scenario: ${SCENARIO}"
