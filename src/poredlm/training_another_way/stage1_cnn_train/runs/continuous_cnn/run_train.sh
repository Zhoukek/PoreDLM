#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT="$(cd "$(dirname "$0")" && pwd)"
STAGE_ROOT="$(cd "$RUN_ROOT/../.." && pwd)"
export PYTHONPATH="$STAGE_ROOT/../../..:${PYTHONPATH:-}"
python "$STAGE_ROOT/train.py" --config "$RUN_ROOT/config.yaml"
