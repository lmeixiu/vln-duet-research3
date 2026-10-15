#!/usr/bin/env bash
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${DUET_CKPT:?Set DUET_CKPT}"
: "${PILOT_ROOT:?Set PILOT_ROOT}"
cd "$(dirname "$0")/.."
python tools/evaluate_residual_ablation.py \
  --base "$DUET_CKPT" \
  --adapter "$PILOT_ROOT/semantic_chunks/ckpts/latest_dict.pilot.pt" \
  --data-root "$DATA_ROOT" \
  --output "$PILOT_ROOT/residual_ablation_$(date +%Y%m%d_%H%M%S)" \
  --limit-val-examples "${VAL_EXAMPLES:-64}"
