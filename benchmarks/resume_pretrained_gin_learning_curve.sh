#!/bin/bash
#
# Resume the Pretrained GIN learning curve sweep (interrupted 2026-07-24).
#
# Status as of interruption:
#   DONE   (19/24): frac 0.01, 0.02, 0.05, 0.1, 0.2, 0.4  x  seeds 0,1,2
#                   + frac 0.7 seed 0 (finished right before the kill signal
#                   landed, at 176 epochs — turned out not to be interrupted)
#   MISSING (5/24): frac 0.7 seeds 1,2   +   frac 1.0 seeds 0,1,2
#
# Actual per-fraction timing observed on this laptop (RTX 4090 Laptop, 16GB):
#   frac 0.01-0.05: ~3-4 min/seed
#   frac 0.1-0.2:   ~6 min/seed
#   frac 0.4:       ~16 min/seed   (48m9s for all 3 seeds)
#   frac 0.7, 1.0:  not yet measured — extrapolating ~26-37 min/seed,
#                   so the remaining 6 runs are estimated at ~3+ hours total.
#
# Usage (from repo root, laptop with GPU):
#   ./benchmarks/resume_pretrained_gin_learning_curve.sh
#
# After it finishes, aggregate everything (all 24 points) with:
#   python benchmarks/aggregate_learning_curve.py

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .DeepRT/bin/activate

mkdir -p /tmp/lc_sweep_logs
# Only the 5 missing (frac, seed) points — frac=0.7 seed=0 is already done.
REMAINING="0.7:1 0.7:2 1.0:0 1.0:1 1.0:2"
for PAIR in $REMAINING; do
  FRAC="${PAIR%%:*}"
  SEED="${PAIR##*:}"
  echo "=== frac=$FRAC seed=$SEED ==="
  python3 benchmarks/pretrained_gin_learning_curve.py --frac "$FRAC" --seed "$SEED" --epochs 1000 \
    2>&1 | tee "/tmp/lc_sweep_logs/frac${FRAC}_seed${SEED}.log" | tail -8
done
echo "LEARNING_CURVE_SWEEP_COMPLETE (all 24/24 points)"
