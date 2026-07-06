#!/usr/bin/env bash
# Multi-seed replication of the Stage 7 v3 (content-memory + visible-glimpse) pilot.
#
# For each seed it trains a fresh checkpoint under an otherwise-identical v3 recipe, then runs the
# permuted-label noise floor (scripts/stage7_latent_noise_floor.py) on the two claimed-positive
# interface widths. The aggregate answers whether the floor-cleared content advantages hold across
# independent training seeds, or were a single-seed draw.
#
# Verdict recorded in docs/NEXT_STEPS.md: the `memory` (remembered-content) leg clears the floor on
# all seeds (robust); the strict `content_only` leg is significant-when-present but seed-fragile.
#
# Usage: bash scripts/stage7_multiseed_v3.sh
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

for S in 107 207 307; do
  CFG="configs/stage7_content_memory_v3_seed${S}.yaml"
  CKPT="outputs/stage7_content_memory_v3_seed${S}/experiment.pt"
  echo "=== [$(date +%H:%M:%S)] seed ${S}: train ==="
  if [ ! -f "$CKPT" ]; then
    "$PY" scripts/train_minimal.py --config "$CFG"
  else
    echo "checkpoint exists, skipping train"
  fi
  echo "=== [$(date +%H:%M:%S)] seed ${S}: noise floor ==="
  "$PY" scripts/stage7_latent_noise_floor.py \
    --config "$CFG" \
    --checkpoint "$CKPT" \
    --state-key content_memory_state_seq \
    --interfaces 32x8,48x8 \
    --permutations 80 \
    --out "audits/stage7_latent_noise_floor_v3_seed${S}.json"
  echo "=== [$(date +%H:%M:%S)] seed ${S}: DONE ==="
done
echo "ALL SEEDS COMPLETE"
