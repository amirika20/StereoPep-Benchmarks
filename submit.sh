#!/bin/bash
#
# SLURM submission script for Kempner cluster.
#
# Usage:
#   ./submit.sh [OPTIONS] -- SCRIPT [ARGS...]
#
# Options:
#   -s, --seeds SEEDS   Seed list, e.g. "1 2 3 4 5" or "0-9" (default: single run)
#   -t, --time TIME     Wall time (default: 0-24:00)
#   -g, --gpus GPUS     Number of GPUs (default: 1)
#   -m, --mem MEM       Memory (default: 64G)
#   -n, --name NAME     Job name (default: derived from script name)
#   -b, --benchmark     Pass seed as --seed N (default: experiment.train_seed=N)
#   -d, --dry-run       Print the sbatch script without submitting
#
set -euo pipefail

CONDA_ENV="/n/home04/akazeminia/.conda/envs/deeprt"

SEEDS=""
TIME="0-24:00"
GPUS="1"
MEM="64G"
JOB_NAME=""
BENCHMARK=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--seeds)     SEEDS="$2"; shift 2 ;;
        -t|--time)      TIME="$2"; shift 2 ;;
        -g|--gpus)      GPUS="$2"; shift 2 ;;
        -m|--mem)       MEM="$2"; shift 2 ;;
        -n|--name)      JOB_NAME="$2"; shift 2 ;;
        -b|--benchmark) BENCHMARK=true; shift ;;
        -d|--dry-run)   DRY_RUN=true; shift ;;
        --)             shift; break ;;
        -*)             echo "Unknown option: $1" >&2; exit 1 ;;
        *)              break ;;
    esac
done

if [[ $# -lt 1 ]]; then
    echo "Error: no script specified. Usage: ./submit.sh [OPTIONS] -- SCRIPT [ARGS...]" >&2
    exit 1
fi

SCRIPT="$1"; shift
EXTRA_ARGS="$*"

[[ -z "$JOB_NAME" ]] && JOB_NAME=$(basename "$SCRIPT" .py)

# Parse seeds
SEED_ARRAY=()
if [[ -n "$SEEDS" ]]; then
    if [[ "$SEEDS" == *-* ]]; then
        START="${SEEDS%-*}"; END="${SEEDS#*-}"
        for ((i=START; i<=END; i++)); do SEED_ARRAY+=("$i"); done
    else
        read -ra SEED_ARRAY <<< "$SEEDS"
    fi
fi
N_SEEDS=${#SEED_ARRAY[@]}

ARRAY_LINE=""
SEED_LOGIC='SEED_OVERRIDE=""'
if [[ $N_SEEDS -gt 0 ]]; then
    ARRAY_LINE="#SBATCH --array=0-$((N_SEEDS - 1))"
    $BENCHMARK && SEED_KEY="--seed" || SEED_KEY="experiment.train_seed"
    SEED_LOGIC="SEEDS=(${SEED_ARRAY[*]})
SEED=\${SEEDS[\$SLURM_ARRAY_TASK_ID]}
echo \"Seed: \$SEED\"
SEED_OVERRIDE=\"$SEED_KEY \$SEED\""
fi

mkdir -p logs
SBATCH_SCRIPT=$(mktemp /tmp/deeprt_submit.XXXXXX.sh)

cat > "$SBATCH_SCRIPT" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --time=$TIME
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:$GPUS
#SBATCH --mem=$MEM
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
$ARRAY_LINE
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module load cuda/12.9.1-fasrc01

# Use local /tmp for HF cache to avoid NFS file-locking errors
export HF_DATASETS_CACHE=/tmp/\${USER}/hf_cache
export HF_HOME=/tmp/\${USER}/hf_home
mkdir -p "\$HF_DATASETS_CACHE" "\$HF_HOME"

echo "Node: \$SLURM_NODELIST | GPUs: $GPUS | Time: $TIME"

$SEED_LOGIC

$CONDA_ENV/bin/python $SCRIPT $EXTRA_ARGS \$SEED_OVERRIDE
SBATCH_EOF

if $DRY_RUN; then
    cat "$SBATCH_SCRIPT"
else
    echo "Submitting: $SCRIPT"
    [[ $N_SEEDS -gt 0 ]] && echo "  Seeds: ${SEED_ARRAY[*]} ($N_SEEDS jobs)"
    sbatch "$SBATCH_SCRIPT"
fi

rm -f "$SBATCH_SCRIPT"
