#!/bin/bash
#
# SLURM submission for the one-off PepLand embedding precompute step.
#
# Unlike the other benchmarks (submitted via ./submit.sh, which always uses
# the `deeprt` conda env), this job needs the separate `pepland_gpu` env —
# see benchmarks/pepland_precompute_embeddings.py's docstring for why (dgl's
# GPU wheels are pinned to a torch version incompatible with `deeprt`).
#
# Usage (from repo root):
#   sbatch benchmarks/submit_pepland_precompute.sh
#
#SBATCH --job-name=pepland_precompute
#SBATCH --partition=kempner
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --time=0-02:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load python
module load cuda/12.9.1-fasrc01

CONDA_ENV="$HOME/.conda/envs/pepland_gpu"

# Use local /tmp for HF cache to avoid NFS file-locking errors
export HF_DATASETS_CACHE=/tmp/${USER}/hf_cache
export HF_HOME=/tmp/${USER}/hf_home
mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME"

echo "Node: $SLURM_NODELIST | GPU: 1 | Env: $CONDA_ENV"

"$CONDA_ENV/bin/python" benchmarks/pepland_precompute_embeddings.py
