#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/porelm_flow_matching.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

torchrun --standalone --nproc-per-node="${NUM_PROCESSES}" \
  "${ROOT_DIR}/scripts/train_porelm.py" "${CONFIG_PATH}" "${@:2}"
