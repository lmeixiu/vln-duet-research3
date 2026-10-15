#!/usr/bin/env bash
set -euo pipefail

# Set these three paths for the server before launching this script.
DATA_ROOT="${DATA_ROOT:-/path/to/datasets}"
DUET_CKPT="${DUET_CKPT:-/path/to/R2R/trained_models/best_val_unseen}"
PILOT_ROOT="${PILOT_ROOT:-/path/to/duet-semantic-pilot}"
SEED="${SEED:-0}"
IMAGE_FEAT_SIZE="${IMAGE_FEAT_SIZE:-768}"

if [[ ! -f "$DUET_CKPT" ]]; then
  echo "Fine-tuned DUET checkpoint not found: $DUET_CKPT" >&2
  exit 2
fi
if [[ ! -d "$DATA_ROOT/R2R/annotations" ]]; then
  echo "R2R data not found under DATA_ROOT: $DATA_ROOT" >&2
  exit 2
fi

cd "$(dirname "$0")/../map_nav_src"
python ../tools/test_semantic_pilot.py --require-torch

common=(
  --root_dir "$DATA_ROOT" --resume_file "$DUET_CKPT"
  --dataset r2r --tokenizer bert --fusion dynamic
  --enc_full_graph --graph_sprels --train_alg imitation
  --features vitbase --image_feat_size "$IMAGE_FEAT_SIZE" --angle_feat_size 4
  --batch_size 4 --max_instr_len 200 --max_action_len 15 --seed "$SEED"
  --optim adamW --lr 1e-4
)

python ../tools/run_semantic_pilot.py --mode off \
  "${common[@]}" --output_dir "$PILOT_ROOT/checkpoint" \
  --iters 0 --log_every 1 --eval_first

for mode in off mean token_attention semantic_chunks equal_chunks; do
  python ../tools/run_semantic_pilot.py --mode "$mode" \
    "${common[@]}" --output_dir "$PILOT_ROOT/$mode" \
    --iters 100 --log_every 100 --eval_first
done

echo "Pilot logs are under: $PILOT_ROOT"
