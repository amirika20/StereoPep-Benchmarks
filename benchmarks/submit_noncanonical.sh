#!/bin/bash
#
# SLURM array job: the non-canonical (D-Phe-only) ablation promised during review.
#
# Complement of the natural (canonical-only) sweep. The two subsets partition the
# full dataset exactly, so together they answer the complementary questions:
#   natural       — does regression performance survive with every
#                   stereochemically ambiguous example removed?
#   noncanonical  — does it also hold on the non-canonical peptides alone?
#
# Submit from the repo root:
#   sbatch benchmarks/submit_noncanonical.sh
#
# One array task per (model, seed) pair, so everything runs in parallel:
# 10 models x 3 seeds = 30 tasks. Seeds match the natural sweep so the two
# ablations are directly comparable — if you change SEEDS below, update
# --array to (n_models * n_seeds - 1) to match; the script aborts if they
# disagree rather than silently skipping work.
#
# Results:  benchmarks/output/results_<model>_noncanonical_seed<N>.json
# Logs:     logs/noncanonical_<jobid>_<taskid>.{out,err}
# Then:     python benchmarks/summarize_results.py
#           -> metrics/latex_overall_performance_noncanonical.tex
#
#SBATCH --job-name=noncanonical
#SBATCH --array=0-29
#SBATCH --partition=kempner
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --time=0-08:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module load python
module load cuda/12.9.1-fasrc01

# The `deeprt` env, as used by submit.sh for every benchmark. PepLand only needs
# its separate `pepland_gpu` env for the precompute step (dgl's GPU wheels pin an
# incompatible torch); pepland.py itself just reads the cached embeddings, so it
# runs here like the rest.
CONDA_ENV="$HOME/.conda/envs/deeprt"
EPOCHS=1000

# "label|script and any model flags" — the label becomes part of the log line only;
# each script derives its own output stem.
#
# The ESM family is deliberately absent. Its benchmark patches in a new 'f'
# (D-Phe) token embedding initialised from L-Phe, and has a separate no-patch
# code path for --dataset natural (where 'f' never occurs). The noncanonical
# subset needs the patched path on subset data with pair evals skipped — a
# combination esm3_embedding.py is not structured for, and wiring it in would
# change what is measured rather than just the split.
MODELS=(
    "morgan_fp_mlp|benchmarks/morgan_fp_mlp.py"
    "transformer_scratch|benchmarks/transformer_scratch.py"
    "deeplc|benchmarks/deeplc.py"
    "deeprt_capsnet|benchmarks/deeprt_capsnet.py"
    "gin_scratch|benchmarks/gin_scratch.py"
    "pretrained_gin|benchmarks/pretrained_gin.py"
    "pepland|benchmarks/pepland.py"
    "peptideclm2_hybrid|benchmarks/peptideclm2_embedding.py --model hybrid_base"
    "chemberta2_mlm_77m|benchmarks/chemberta2_embedding.py --model mlm_77m"
    "chemberta2_mtr_77m|benchmarks/chemberta2_embedding.py --model mtr_77m"
)
SEEDS=(0 1 2)

N_MODELS=${#MODELS[@]}
N_SEEDS=${#SEEDS[@]}
N_TASKS=$((N_MODELS * N_SEEDS))

# Fail loudly if --array above does not cover exactly the model x seed grid.
# Silently running a subset of the grid is the failure mode worth preventing:
# summarize_results.py would happily average whatever seeds happen to exist.
EXPECTED_MAX=$((N_TASKS - 1))
if [[ "${SLURM_ARRAY_TASK_MAX:-$EXPECTED_MAX}" -ne "$EXPECTED_MAX" ]]; then
    echo "ERROR: --array=0-${SLURM_ARRAY_TASK_MAX} does not match the grid" >&2
    echo "       ${N_MODELS} models x ${N_SEEDS} seeds = ${N_TASKS} tasks" >&2
    echo "       Set '#SBATCH --array=0-${EXPECTED_MAX}' and resubmit." >&2
    exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?must be run as a SLURM array job}"
ENTRY="${MODELS[$((TASK_ID / N_SEEDS))]}"
SEED="${SEEDS[$((TASK_ID % N_SEEDS))]}"
LABEL="${ENTRY%%|*}"
SCRIPT_AND_ARGS="${ENTRY#*|}"

# Local /tmp for the HF cache — the shared filesystem hits file-locking errors
# when many array tasks read the dataset at once (same reason as submit.sh).
export HF_DATASETS_CACHE="/tmp/${USER}/hf_cache"
export HF_HOME="/tmp/${USER}/hf_home"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Task $TASK_ID/$EXPECTED_MAX | model=$LABEL | seed=$SEED | node=$SLURM_NODELIST"
echo "Command: $SCRIPT_AND_ARGS --dataset noncanonical --seed $SEED --epochs $EPOCHS"

# Frozen-embedding models read a precomputed cache that is gitignored, so it must
# exist on the cluster filesystem before these tasks run. Check first and say
# exactly what to run, rather than failing several minutes into the job.
case "$LABEL" in
    pepland)            CACHE="benchmarks/pretrained_weights/pepland_embeddings.pt" ;;
    peptideclm2_hybrid) CACHE="benchmarks/pretrained_weights/peptideclm2_hybrid_base_embeddings.pt" ;;
    chemberta2_mlm_77m) CACHE="benchmarks/pretrained_weights/chemberta2_mlm_77m_embeddings.pt" ;;
    chemberta2_mtr_77m) CACHE="benchmarks/pretrained_weights/chemberta2_mtr_77m_embeddings.pt" ;;
    *)                  CACHE="" ;;
esac
if [[ -n "$CACHE" && ! -f "$CACHE" ]]; then
    echo "ERROR: $LABEL needs $CACHE, which is not present." >&2
    echo "       Precompute it on the cluster first, e.g.:" >&2
    echo "         python benchmarks/chemberta2_precompute_embeddings.py --model mlm_77m,mtr_77m" >&2
    echo "         python benchmarks/peptideclm2_precompute_embeddings.py --model hybrid_base" >&2
    exit 1
fi

# shellcheck disable=SC2086  # SCRIPT_AND_ARGS intentionally word-splits
"$CONDA_ENV/bin/python" $SCRIPT_AND_ARGS \
    --dataset noncanonical --seed "$SEED" --epochs "$EPOCHS"

echo "Task $TASK_ID done: $LABEL seed=$SEED"
