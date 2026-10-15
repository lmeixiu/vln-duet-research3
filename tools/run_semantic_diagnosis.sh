#!/usr/bin/env bash
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${DUET_CKPT:?Set DUET_CKPT}"
: "${PILOT_ROOT:?Set PILOT_ROOT}"
cd "$(dirname "$0")/../map_nav_src"
python ../tools/diagnose_semantic_pilot.py \
  --base "$DUET_CKPT" \
  --adapter "$PILOT_ROOT/semantic_chunks/ckpts/latest_dict.pilot.pt" \
  --output "$PILOT_ROOT/diagnosis_semantic_$(date +%Y%m%d_%H%M%S)" \
  --evaluate --data-root "$DATA_ROOT" --limit-val-examples 64
