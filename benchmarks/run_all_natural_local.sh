#!/bin/bash
#
# Run all 13 model configs already in the stereopep summary table against
# the StereoPep 'natural' config (canonical-amino-acid-only), locally and
# serially (no SLURM/cluster — matches how the learning curve sweep was run
# on this laptop). 3 seeds each, --epochs 1000 to match the same
# early-stopping convention (patience = 10% of max) used for every existing
# stereopep result.
#
# Usage (from repo root):
#   ./benchmarks/run_all_natural_local.sh
#
# Results land in benchmarks/output/results_<model>_natural_seed<N>.json
# (or _embedding_natural_ for the ESM family). Logs go to /tmp/natural_sweep_logs/.

set -uo pipefail   # not -e: one failed run shouldn't abort the whole sweep
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .DeepRT/bin/activate

mkdir -p /tmp/natural_sweep_logs
EPOCHS=1000
SEEDS="0 1 2"

run() {
    local name="$1"; shift
    for SEED in $SEEDS; do
        echo "=== $name seed=$SEED ==="
        LOG="/tmp/natural_sweep_logs/${name}_seed${SEED}.log"
        "$@" --dataset natural --seed "$SEED" --epochs "$EPOCHS" > "$LOG" 2>&1
        RC=$?
        echo "  -> exit=$RC  $(grep -E 'RMSE=|Ordering accuracy' "$LOG" | tr '\n' ' ')"
    done
}

run morgan_fp_mlp        python3 benchmarks/morgan_fp_mlp.py
run transformer_scratch  python3 benchmarks/transformer_scratch.py
run deeplc               python3 benchmarks/deeplc.py
run deeprt_capsnet       python3 benchmarks/deeprt_capsnet.py
run gin_scratch          python3 benchmarks/gin_scratch.py
run pretrained_gin       python3 benchmarks/pretrained_gin.py
run pepland              python3 benchmarks/pepland.py
run peptideclm2_hybrid   python3 benchmarks/peptideclm2_embedding.py --model hybrid_base
run chemberta2_mlm_77m   python3 benchmarks/chemberta2_embedding.py --model mlm_77m
run chemberta2_mtr_77m   python3 benchmarks/chemberta2_embedding.py --model mtr_77m
run esm3_sm              python3 benchmarks/esm3_embedding.py --model esm3_sm
run esmc_300m            python3 benchmarks/esm3_embedding.py --model esmc_300m
run esmc_600m            python3 benchmarks/esm3_embedding.py --model esmc_600m

echo "NATURAL_SWEEP_COMPLETE"
