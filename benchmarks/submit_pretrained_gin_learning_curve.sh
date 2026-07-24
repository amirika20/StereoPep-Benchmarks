#!/bin/bash
#
# Submit the Pretrained GIN learning curve as a grid of SLURM jobs: one
# ./submit.sh call per training-set fraction, each sweeping --seeds as a
# SLURM array job (reuses the existing deeprt-env submission path — no new
# conda env or sbatch template needed, unlike PepLand).
#
# Usage (from repo root):
#   ./benchmarks/submit_pretrained_gin_learning_curve.sh
#   ./benchmarks/submit_pretrained_gin_learning_curve.sh --fracs "0.05 0.1 1.0" --seeds 0-4
#   ./benchmarks/submit_pretrained_gin_learning_curve.sh --dry-run
#
set -euo pipefail

FRACS="0.01 0.02 0.05 0.1 0.2 0.4 0.7 1.0"
SEEDS="0-2"
EPOCHS=1000
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fracs)   FRACS="$2"; shift 2 ;;
        --seeds)   SEEDS="$2"; shift 2 ;;
        --epochs)  EPOCHS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

for FRAC in $FRACS; do
    NAME="pretrained_gin_lc_frac${FRAC}"
    CMD=(./submit.sh --benchmark --seeds "$SEEDS" --time 0-06:00 --mem 32G --gpus 1 --name "$NAME")
    $DRY_RUN && CMD+=(--dry-run)
    CMD+=(-- benchmarks/pretrained_gin_learning_curve.py --frac "$FRAC" --epochs "$EPOCHS")

    echo "── frac=$FRAC ──"
    "${CMD[@]}"
done

echo
echo "Submitted learning curve grid: fracs=[$FRACS]  seeds=$SEEDS"
echo "Results land in benchmarks/output/learning_curve/"
echo "Once done, aggregate with: python benchmarks/aggregate_learning_curve.py"
