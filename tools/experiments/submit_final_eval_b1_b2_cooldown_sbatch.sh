#!/usr/bin/env bash
# Submit the B1/B2 cooldown5ep clean evaluation as a single Slurm job.
#
# Generates and submits an sbatch that calls
#   run_final_eval_b1_b2_cooldown.sh --launcher slurm-step ...
# Outputs land under:
#   work_dirs/experiments/b1_b2/{b1,b2}_stage3_cooldown5ep/{b1,b2}/eval_deferred/cooldown5ep_latest/clean/
#
# Usage:
#   bash tools/experiments/submit_final_eval_b1_b2_cooldown_sbatch.sh [options]
#
# Required:
#   --mail-user EMAIL
#
# Optional:
#   --work-root DIR          (default: /scratch/gpfs/FHEIDE/yk3904/maptracker_v1/work_dirs)
#   --partition NAME
#   --qos NAME               (default: gpu-short)
#   --constraint EXPR        (default: a100&nomig)
#   --gpus N                 GPUs per eval step (default: 4)
#   --cpus-per-task N        (default: 6)
#   --mem VALUE              (default: 96G)
#   --time LIMIT             (default: 08:00:00)
#   --seed N                 (default: 0)
#   --skip-cmap              Skip C-mAP post-processing
#   --extra-wrap-args "..."  Extra args appended to the eval wrapper
#   --dry-run                Write sbatch file but do not submit
#   -h, --help
set -euo pipefail

usage() {
  grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \?//'
  exit 0
}

PROJECT_ROOT="/scratch/gpfs/FHEIDE/yk3904/maptracker_v1"
WORK_ROOT="$PROJECT_ROOT/work_dirs"
MAIL_USER=""
PARTITION=""
QOS="gpu-short"
CONSTRAINT="a100&nomig"
GPUS=4
CPUS_PER_TASK=6
MEMORY="96G"
TIME_LIMIT="08:00:00"
SEED=0
SKIP_CMAP=0
EXTRA_WRAP_ARGS_STR=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mail-user) MAIL_USER="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --constraint) CONSTRAINT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --skip-cmap) SKIP_CMAP=1; shift ;;
    --extra-wrap-args) EXTRA_WRAP_ARGS_STR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MAIL_USER" ]]; then
  echo "Missing required argument: --mail-user EMAIL" >&2
  exit 2
fi

if (( GPUS <= 0 || CPUS_PER_TASK <= 0 )); then
  echo "--gpus and --cpus-per-task must be positive integers." >&2
  exit 2
fi

TS="$(date -u +%Y%m%d_%H%M%S)"
SBATCH_ROOT="$WORK_ROOT/sbatch/final_eval_b1_b2_cooldown/${TS}"
SBATCH_LOG_DIR="$SBATCH_ROOT/logs"
SBATCH_SCRIPT="$SBATCH_ROOT/submit.sbatch"
mkdir -p "$SBATCH_LOG_DIR"

# Build the run_final_eval_b1_b2_cooldown.sh invocation.
WRAP_CMD=(bash tools/experiments/run_final_eval_b1_b2_cooldown.sh
  --launcher slurm-step
  --gpus "$GPUS"
  --seed "$SEED"
  --work-root "$WORK_ROOT"
  --rerun)
if [[ $SKIP_CMAP -eq 1 ]]; then WRAP_CMD+=(--skip-cmap); fi
if [[ -n "$EXTRA_WRAP_ARGS_STR" ]]; then
  read -r -a EXTRA_WRAP_ARGS_ARR <<< "$EXTRA_WRAP_ARGS_STR"
  WRAP_CMD+=("${EXTRA_WRAP_ARGS_ARR[@]}")
fi
printf -v WRAP_CMD_STR '%q ' "${WRAP_CMD[@]}"

# Optional sbatch directives.
OPTIONAL_DIRECTIVES=""
if [[ -n "$PARTITION" ]]; then
  OPTIONAL_DIRECTIVES+="#SBATCH --partition=${PARTITION}"$'\n'
fi
if [[ -n "$QOS" ]]; then
  OPTIONAL_DIRECTIVES+="#SBATCH --qos=${QOS}"$'\n'
fi
if [[ -n "$CONSTRAINT" ]]; then
  OPTIONAL_DIRECTIVES+="#SBATCH --constraint=${CONSTRAINT}"$'\n'
fi

cat > "$SBATCH_SCRIPT" <<SBATCH_EOF
#!/usr/bin/env bash
#SBATCH --job-name=mt_eval_cooldown_b1b2
#SBATCH --nodes=1
#SBATCH --ntasks=${GPUS}
#SBATCH --ntasks-per-node=${GPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEMORY}
#SBATCH --output=${SBATCH_LOG_DIR}/%x-%j.out
#SBATCH --error=${SBATCH_LOG_DIR}/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL_USER}
#SBATCH --exclude=neu303
${OPTIONAL_DIRECTIVES}
set -euo pipefail

module purge
module load anaconda3/2023.9
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate maptracker

export PROJECT_ROOT=${PROJECT_ROOT}
export PYTHONPATH="\$PROJECT_ROOT:\${PYTHONPATH:-}"
export SRUN_CPUS_PER_TASK=${CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "\$PROJECT_ROOT"
echo "[eval-cooldown] Starting cooldown eval wrapper"
${WRAP_CMD_STR}
echo "[eval-cooldown] Completed"
SBATCH_EOF

chmod +x "$SBATCH_SCRIPT"
echo "Generated: $SBATCH_SCRIPT"
echo "Logs dir:  $SBATCH_LOG_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY RUN] Would submit: sbatch $SBATCH_SCRIPT"
  exit 0
fi

sbatch "$SBATCH_SCRIPT"
