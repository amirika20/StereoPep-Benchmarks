#!/bin/bash
#
# Unified SLURM submission script for NGSFM experiments.
#
# Usage:
#   ./submit.sh [OPTIONS] -- SCRIPT [ARGS...]
#
# Options:
#   -c, --cluster CLUSTER    Cluster preset: kempner, o2, fas (default: kempner)
#   -s, --seeds SEEDS        Seed list, e.g. "1 2 3 4 5" or "0-9" (default: single run, no array)
#   -t, --time TIME          Wall time, e.g. "24:00:00" (default: per-cluster)
#   -g, --gpus GPUS          Number of GPUs (default: 1)
#   -m, --mem MEM            Memory, e.g. "64G" (default: per-cluster)
#   -n, --name NAME          Job name (default: derived from script name)
#   -p, --partition PART     Override partition
#   -b, --benchmark          Benchmark mode: pass seed as --seed N (default: Hydra style)
#   -d, --dry-run            Print the sbatch command without submitting
#
# Examples:
#   # Single GMM training run on Kempner
#   ./submit.sh -- scripts/main_gmm.py
#
#   # 10-seed MNIST sweep on Kempner
#   ./submit.sh -s "1-10" -- scripts/main.py --config-name=config_mnist \
#       distributions@source=gaussian_mnist distributions@target=mnist \
#       architecture.n_source_classes=0
#
#   # Single CIFAR run on O2 with 2 GPUs
#   ./submit.sh -c o2 -g 2 -- scripts/main.py --config-name=config_cifar \
#       distributions@source=gaussian_cifar distributions@target=cifar \
#       architecture=cifar training=cifar
#
#   # Dry run to inspect the generated sbatch script
#   ./submit.sh -d -s "1-5" -- scripts/main_gmm.py
#
# Environment:
#   Set NGSFM_CONDA_ENV to your conda/mamba environment path, or
#   NGSFM_VENV to your venv path. The script auto-detects if neither is set.
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CLUSTER="kempner"
SEEDS=""
TIME=""
GPUS=""
MEM=""
PARTITION=""
JOB_NAME=""
BENCHMARK=false
DRY_RUN=false

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--cluster)    CLUSTER="$2"; shift 2 ;;
        -s|--seeds)      SEEDS="$2"; shift 2 ;;
        -t|--time)       TIME="$2"; shift 2 ;;
        -g|--gpus)       GPUS="$2"; shift 2 ;;
        -m|--mem)        MEM="$2"; shift 2 ;;
        -n|--name)       JOB_NAME="$2"; shift 2 ;;
        -p|--partition)  PARTITION="$2"; shift 2 ;;
        -b|--benchmark)  BENCHMARK=true; shift ;;
        -d|--dry-run)    DRY_RUN=true; shift ;;
        --)              shift; break ;;
        -*)              echo "Unknown option: $1" >&2; exit 1 ;;
        *)               break ;;  # first positional arg = script
    esac
done

if [[ $# -lt 1 ]]; then
    echo "Error: no script specified. Usage: ./submit.sh [OPTIONS] -- SCRIPT [HYDRA_OVERRIDES...]" >&2
    exit 1
fi

SCRIPT="$1"; shift
HYDRA_ARGS="$*"

# ── Cluster presets ───────────────────────────────────────────────────────────
case "$CLUSTER" in
    kempner)
        PARTITION="${PARTITION:-kempner_h100}"
        TIME="${TIME:-0-24:00}"
        MEM="${MEM:-64G}"
        GPUS="${GPUS:-1}"
        GRES="gpu:nvidia_h100_80gb_hbm3:${GPUS}"
        ACCOUNT="kempner_mzitnik_lab"
        MODULE_CMDS="module load python
module load cuda/12.9.1-fasrc01"
        ;;
    o2)
        PARTITION="${PARTITION:-gpu}"
        TIME="${TIME:-0-12:00}"
        MEM="${MEM:-32G}"
        GPUS="${GPUS:-2}"
        GRES="gpu:${GPUS}"
        ACCOUNT=""
        MODULE_CMDS="module load gcc/14.2.0
module load python/3.13.1
module load cuda/12.8"
        ;;
    fas)
        PARTITION="${PARTITION:-mit_normal_gpu}"
        TIME="${TIME:-0-05:59}"
        MEM="${MEM:-32G}"
        GPUS="${GPUS:-1}"
        GRES="gpu:${GPUS}"
        ACCOUNT=""
        MODULE_CMDS="module load cuda/13.0.1
module load miniforge/25.11.0-0"
        ;;
    *)
        echo "Unknown cluster: $CLUSTER (choose: kempner, o2, fas)" >&2
        exit 1
        ;;
esac

# ── Derive job name from script if not set ────────────────────────────────────
if [[ -z "$JOB_NAME" ]]; then
    JOB_NAME=$(basename "$SCRIPT" .py | sed 's/^main_//')
fi

# ── Parse seeds into array ────────────────────────────────────────────────────
SEED_ARRAY=()
if [[ -n "$SEEDS" ]]; then
    if [[ "$SEEDS" == *-* ]]; then
        # Range notation: "1-10"
        START="${SEEDS%-*}"
        END="${SEEDS#*-}"
        for ((i=START; i<=END; i++)); do SEED_ARRAY+=("$i"); done
    else
        # Space-separated list: "1 2 3 4 5"
        read -ra SEED_ARRAY <<< "$SEEDS"
    fi
fi

N_SEEDS=${#SEED_ARRAY[@]}

# ── Build SLURM array directive ───────────────────────────────────────────────
ARRAY_LINE=""
SEED_LOGIC=""
if [[ $N_SEEDS -gt 0 ]]; then
    ARRAY_LINE="#SBATCH --array=0-$((N_SEEDS - 1))"
    SEED_LIST="${SEED_ARRAY[*]}"
    if $BENCHMARK; then
        SEED_LOGIC="SEEDS=($SEED_LIST)
SEED=\${SEEDS[\$SLURM_ARRAY_TASK_ID]}
echo \"  Seed: \$SEED\"
SEED_OVERRIDE=\"--seed \$SEED\""
    else
        SEED_LOGIC="SEEDS=($SEED_LIST)
SEED=\${SEEDS[\$SLURM_ARRAY_TASK_ID]}
echo \"  Seed: \$SEED\"
SEED_OVERRIDE=\"experiment.train_seed=\$SEED\""
    fi
else
    SEED_LOGIC='SEED_OVERRIDE=""'
fi

# ── Build environment activation ──────────────────────────────────────────────
if [[ -n "${NGSFM_CONDA_ENV:-}" ]]; then
    ENV_ACTIVATE='eval "$(mamba shell hook --shell bash)"
mamba activate '"$NGSFM_CONDA_ENV"
elif [[ -n "${NGSFM_VENV:-}" ]]; then
    ENV_ACTIVATE="source $NGSFM_VENV/bin/activate"
else
    # Auto-detect: prefer conda env if mamba is available, else look for .venv
    ENV_ACTIVATE='if command -v mamba &>/dev/null; then
    eval "$(mamba shell hook --shell bash)"
    mamba activate "${CONDA_DEFAULT_ENV:-base}"
elif [[ -d "$HOME/.venv" ]]; then
    source "$HOME/.venv/bin/activate"
elif [[ -d ".venv" ]]; then
    source .venv/bin/activate
else
    echo "Warning: no conda/venv environment detected" >&2
fi'
fi

# ── Account line (only for kempner) ───────────────────────────────────────────
ACCOUNT_LINE=""
if [[ -n "$ACCOUNT" ]]; then
    ACCOUNT_LINE="#SBATCH --account=$ACCOUNT"
fi

# ── Log directory ─────────────────────────────────────────────────────────────
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ── Generate sbatch script ────────────────────────────────────────────────────
SBATCH_SCRIPT=$(mktemp /tmp/ngsfm_submit.XXXXXX.sh)

cat > "$SBATCH_SCRIPT" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
$ACCOUNT_LINE
#SBATCH --time=$TIME
#SBATCH --gres=$GRES
#SBATCH --mem=$MEM
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
$ARRAY_LINE
#SBATCH --output=$LOG_DIR/%x_%A_%a.out
#SBATCH --error=$LOG_DIR/%x_%A_%a.err

set -euo pipefail

# ── Modules ───────────────────────────────────────────────────────────────────
$MODULE_CMDS

# ── Environment ───────────────────────────────────────────────────────────────
$ENV_ACTIVATE

# ── Job info ──────────────────────────────────────────────────────────────────
echo "========================================"
echo "Node:      \$SLURM_NODELIST"
echo "GPUs:      \${SLURM_GPUS:-N/A}"
echo "Partition: \$SLURM_JOB_PARTITION"

# ── Seed handling ─────────────────────────────────────────────────────────────
$SEED_LOGIC

echo "========================================"

# ── Run ───────────────────────────────────────────────────────────────────────
python $SCRIPT $HYDRA_ARGS \$SEED_OVERRIDE
SBATCH_EOF

# ── Submit or print ───────────────────────────────────────────────────────────
if $DRY_RUN; then
    echo "=== Generated sbatch script ==="
    cat "$SBATCH_SCRIPT"
    echo "=== End ==="
    echo "(dry run — not submitted)"
else
    echo "Submitting: $SCRIPT"
    [[ $N_SEEDS -gt 0 ]] && echo "  Seeds: ${SEED_ARRAY[*]} ($N_SEEDS jobs)"
    echo "  Cluster: $CLUSTER ($PARTITION)"
    echo "  GPUs: $GPUS, Mem: $MEM, Time: $TIME"
    sbatch "$SBATCH_SCRIPT"
fi

rm -f "$SBATCH_SCRIPT"
