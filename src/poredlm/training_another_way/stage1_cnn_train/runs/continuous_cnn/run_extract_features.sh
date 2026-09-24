#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT="$(cd "$(dirname "$0")" && pwd)"
STAGE_ROOT="$(cd "$RUN_ROOT/../.." && pwd)"
export PYTHONPATH="$STAGE_ROOT/../../..:${PYTHONPATH:-}"
python "$STAGE_ROOT/extract_features.py" --config "$RUN_ROOT/config.yaml" --checkpoint "${1:?checkpoint required}" --split "${2:?train or valid required}" --output-dir "${3:?output directory required}"
