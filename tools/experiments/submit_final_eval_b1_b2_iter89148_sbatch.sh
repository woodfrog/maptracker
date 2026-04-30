#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit final B1/B2 iter_89148 evaluation as one Slurm job.

Usage:
  bash tools/experiments/submit_final_eval_b1_b2_iter89148_sbatch.sh --mail-user EMAIL [options]

Required:
  --mail-user EMAIL

Optional:
  --partition NAME
  --job-name NAME          (default: maptracker_final_eval_b1b2)
  --time LIMIT             (default: 08:00:00)
  --gpus N                 (default: 4)
  --cpus-per-task N        (default: 6)
  --mem VALUE              (default: 96G)
  --work-root DIR          (default: /n/fs/dynamicbias/tracker/work_dirs)
  --conda-env NAME         (default: maptracker)
  --seed N                 (default: 0)
  --account NAME
  --qos NAME
  --constraint EXPR
  --mail-type VALUE        (default: END,FAIL)
  --extra-wrap-args "..."  Extra args appended to run_final_eval wrapper.
  --dry-run                Generate sbatch file but do not submit.
  -h, --help

Notes:
  - This uses launcher=slurm-step inside the allocated job for robust port retry handling.
  - Eval outputs are written under:
    <work-root>/experiments/b1_b2/{b1_stage3_gpu4_short_trainonly,b2_stage3_gpu4_short_trainonly}/{b1,b2}/eval_deferred/iter_89148/clean/
USAGE
}

PARTITION=""
MAIL_USER=""
JOB_NAME="maptracker_final_eval_b1b2"
TIME_LIMIT="08:00:00"
GPUS=4
CPUS_PER_TASK=6
MEMORY="96G"
WORK_ROOT="/n/fs/dynamicbias/tracker/work_dirs"
CONDA_ENV="/n/fs/dynamicbias/tracker/env-maptracker"
SEED=0
ACCOUNT=""
QOS=""
CONSTRAINT=""
MAIL_TYPE="END,FAIL"
EXTRA_WRAP_ARGS_STR=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --mail-user) MAIL_USER="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --constraint) CONSTRAINT="$2"; shift 2 ;;
    --mail-type) MAIL_TYPE="$2"; shift 2 ;;
    --extra-wrap-args) EXTRA_WRAP_ARGS_STR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[submit_final_eval_b1_b2_iter89148_sbatch] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MAIL_USER" ]]; then
  echo "[submit_final_eval_b1_b2_iter89148_sbatch] Missing required arguments." >&2
  usage
  exit 2
fi

if (( GPUS <= 0 || CPUS_PER_TASK <= 0 )); then
  echo "[submit_final_eval_b1_b2_iter89148_sbatch] --gpus and --cpus-per-task must be positive integers." >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date -u +%Y%m%d_%H%M%S)"
SBATCH_ROOT="${WORK_ROOT%/}/sbatch/final_eval_b1_b2_iter89148/${TS}"
SBATCH_LOG_DIR="$SBATCH_ROOT/logs"
SBATCH_SCRIPT="$SBATCH_ROOT/submit.sbatch"
mkdir -p "$SBATCH_LOG_DIR"

RUN_CMD=(bash tools/experiments/run_final_eval_b1_b2_iter89148.sh
  --launcher slurm-step
  --gpus "$GPUS"
  --seed "$SEED"
  --work-root "$WORK_ROOT"
  --rerun)

if [[ -n "$EXTRA_WRAP_ARGS_STR" ]]; then
  read -r -a EXTRA_WRAP_ARGS_ARR <<< "$EXTRA_WRAP_ARGS_STR"
  RUN_CMD+=("${EXTRA_WRAP_ARGS_ARR[@]}")
fi

printf -v RUN_CMD_STR '%q ' "${RUN_CMD[@]}"

{
  echo '#!/usr/bin/env bash'
  echo "#SBATCH --job-name=${JOB_NAME}"
  echo '#SBATCH --nodes=1'
  echo "#SBATCH --ntasks=${GPUS}"
  echo "#SBATCH --ntasks-per-node=${GPUS}"
  echo "#SBATCH --gres=gpu:${GPUS}"
  echo '#SBATCH --gpus-per-task=1'
  echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}"
  echo "#SBATCH --time=${TIME_LIMIT}"
  echo "#SBATCH --mem=${MEMORY}"
  echo "#SBATCH --output=${SBATCH_LOG_DIR}/%x-%j.out"
  echo "#SBATCH --error=${SBATCH_LOG_DIR}/%x-%j.err"
  echo "#SBATCH --mail-type=${MAIL_TYPE}"
  echo "#SBATCH --mail-user=${MAIL_USER}"
  echo "#SBATCH --exclude=neu303"
  if [[ -n "$PARTITION" ]]; then
    echo "#SBATCH --partition=${PARTITION}"
  fi
  if [[ -n "$ACCOUNT" ]]; then
    echo "#SBATCH --account=${ACCOUNT}"
  fi
  if [[ -n "$QOS" ]]; then
    echo "#SBATCH --qos=${QOS}"
  fi
  if [[ -n "$CONSTRAINT" ]]; then
    echo "#SBATCH --constraint=${CONSTRAINT}"
  fi
  cat <<EOF
set -euo pipefail

cd "$PROJECT_ROOT"
eval "\$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PROJECT_ROOT:\${PYTHONPATH:-}"
export SRUN_CPUS_PER_TASK="$CPUS_PER_TASK"

# Bypass SSL verification for downloading pre-trained weights
export PYTHONHTTPSVERIFY=0

#### FILE TRANSFER ####
echo "[submit_final_eval] Preparing file transfers..."
DIR_NAME="/scratch/rc5898/tracker"
TAR_PATH="/scratch/rc5898/tracker.tar"

if [ ! -d "\$DIR_NAME" ]; then
    echo "Tracker directory not found on this node. Handling transfer..."
    if [ ! -f "\$TAR_PATH" ]; then
        echo "Tar file not found locally. Downloading from login node..."
        expect << EXPECT_EOF
set timeout -1
set send_slow {1 .1}
spawn rsync -avP "rc5898@neuronic.cs.princeton.edu:\$TAR_PATH" "/scratch/rc5898/"
expect "Passcode or option (1-3):"
sleep 3
send -s "1\r"
expect eof
EXPECT_EOF
    fi
    echo "Uncompressing \$TAR_PATH..."
    tar -xf "\$TAR_PATH" -C /scratch/rc5898/
    echo "Uncompress complete."
else
    echo "Directory \$DIR_NAME already exists on this node. Skipping transfer."
fi

echo "Setting up symlink portal..."
ln -sfn /scratch/rc5898/tracker/datasets /n/fs/dynamicbias/tracker/datasets

echo "Injecting tracking files from project directory to local scratch..."
\cp /n/fs/dynamicbias/tracker/datasets-swp/*gt_tracks.pkl /scratch/rc5898/tracker/datasets/nuscenes/

echo "[submit_final_eval] File transfer complete"

echo "[submit_final_eval] Starting final eval wrapper"
${RUN_CMD_STR}
echo "[submit_final_eval] Completed"
EOF
} > "$SBATCH_SCRIPT"

chmod +x "$SBATCH_SCRIPT"

echo "[submit_final_eval_b1_b2_iter89148_sbatch] Generated: $SBATCH_SCRIPT"
echo "[submit_final_eval_b1_b2_iter89148_sbatch] Logs dir: $SBATCH_LOG_DIR"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[submit_final_eval_b1_b2_iter89148_sbatch] Dry run, not submitting."
  exit 0
fi

SUBMIT_OUTPUT="$(sbatch "$SBATCH_SCRIPT")"
echo "$SUBMIT_OUTPUT"
