#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ROOT="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/eval-data/LB06/embedding-space-base-flow-matching"
OUTPUT_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/eval-data/LB06/embedding-space-base-flow-matching"

python "${SCRIPT_DIR}/plot_ode_feature_metrics_png.py" \
  --root "${ROOT}" \
  --output-dir "${OUTPUT_DIR}"

echo "PNG charts: ${OUTPUT_DIR}"
