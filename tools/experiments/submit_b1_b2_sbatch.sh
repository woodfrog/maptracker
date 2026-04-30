#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: submit_b1_b2_sbatch.sh --mode MODE --base-config CFG --base-checkpoint CKPT --work-root ROOT --time LIMIT --mail-user EMAIL [options]

Required:
  --mode b1_only|b2_only|both_parallel|both_sequential
  --base-config PATH
  --base-checkpoint PATH
  --work-root DIR
  --time LIMIT                    Slurm time limit, e.g. 23:30:00 or 2-12:00:00
  --mail-user EMAIL

Optional:
  --run-id ID
  --job-name NAME                 (default: maptracker_b1_b2)
  --partition NAME                Slurm partition to request.
  --qos NAME                      Slurm QOS to request.
  --account NAME                  Slurm account.
  --constraint EXPR               Slurm constraint, e.g. gpu80 or nomig&gpu40.
  --mem VALUE                     Slurm memory request, e.g. 128G.
  --mail-type VALUE               (default: END,FAIL)
  --dependency SPEC               Slurm dependency string.
  --train-gpus N                  (default: 1)
  --eval-gpus N                   (default: train-gpus)
  --gpus-per-node N               (default: train-gpus)
  --cpus-per-task N               (default: 5)
  --seed N                        (default: 0)
  --resume                        Reuse an existing run-id and auto-resume from latest checkpoint.
  --skip-train-validation         Pass --no-validate to the training phase.
  --skip-final-eval               Skip the wrapper's post-train clean eval.
  --module-load MOD               (default: anaconda3/2023.9)
  --conda-env NAME                (default: maptracker)
  --srun-args "ARGS"             Extra args passed to inner srun steps.
  --sbatch-extra-args "ARGS"     Extra args appended to sbatch on submission.
  --cfg-options-common "k=v ..."
  --b1-cfg-options "k=v ..."
  --b2-cfg-options "k=v ..."
  --eval-cfg-options "k=v ..."
  --run-clean-cmap
  --cons-frames N
  --run-contradiction-suite
  --suite-modes "c_full c_tail"
  --suite-stale-offsets "4 8"
  --suite-onset N
  --suite-c-tail-keep-recent N
  --dry-run
  -h, --help
USAGE
}

MODE=""
BASE_CONFIG=""
BASE_CHECKPOINT=""
WORK_ROOT=""
RUN_ID=""
TIME_LIMIT=""
MAIL_USER=""
JOB_NAME="maptracker_b1_b2"
PARTITION=""
QOS=""
ACCOUNT=""
CONSTRAINT=""
MEMORY=""
MAIL_TYPE="END,FAIL"
DEPENDENCY=""
TRAIN_GPUS=1
EVAL_GPUS=""
GPUS_PER_NODE=""
CPUS_PER_TASK=5
SEED=0
RESUME=0
SKIP_TRAIN_VALIDATION=0
SKIP_FINAL_EVAL=0
MODULE_LOAD="anaconda3/2023.9"
CONDA_ENV="maptracker"
SRUN_ARGS_STR=""
SBATCH_EXTRA_ARGS_STR=""
CFG_COMMON_STR=""
B1_CFG_STR=""
B2_CFG_STR=""
EVAL_CFG_STR=""
RUN_CLEAN_CMAP=0
CONS_FRAMES=5
RUN_SUITE=0
SUITE_MODES_STR="c_full c_tail"
SUITE_STALE_OFFSETS_STR="4 8"
SUITE_ONSET=""
SUITE_C_TAIL_KEEP_RECENT=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --base-config) BASE_CONFIG="$2"; shift 2 ;;
    --base-checkpoint) BASE_CHECKPOINT="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --mail-user) MAIL_USER="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --constraint) CONSTRAINT="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --mail-type) MAIL_TYPE="$2"; shift 2 ;;
    --dependency) DEPENDENCY="$2"; shift 2 ;;
    --train-gpus) TRAIN_GPUS="$2"; shift 2 ;;
    --eval-gpus) EVAL_GPUS="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --skip-train-validation) SKIP_TRAIN_VALIDATION=1; shift ;;
    --skip-final-eval) SKIP_FINAL_EVAL=1; shift ;;
    --module-load) MODULE_LOAD="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --srun-args) SRUN_ARGS_STR="$2"; shift 2 ;;
    --sbatch-extra-args) SBATCH_EXTRA_ARGS_STR="$2"; shift 2 ;;
    --cfg-options-common) CFG_COMMON_STR="$2"; shift 2 ;;
    --b1-cfg-options) B1_CFG_STR="$2"; shift 2 ;;
    --b2-cfg-options) B2_CFG_STR="$2"; shift 2 ;;
    --eval-cfg-options) EVAL_CFG_STR="$2"; shift 2 ;;
    --run-clean-cmap) RUN_CLEAN_CMAP=1; shift ;;
    --cons-frames) CONS_FRAMES="$2"; shift 2 ;;
    --run-contradiction-suite) RUN_SUITE=1; shift ;;
    --suite-modes) SUITE_MODES_STR="$2"; shift 2 ;;
    --suite-stale-offsets) SUITE_STALE_OFFSETS_STR="$2"; shift 2 ;;
    --suite-onset) SUITE_ONSET="$2"; shift 2 ;;
    --suite-c-tail-keep-recent) SUITE_C_TAIL_KEEP_RECENT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[submit_b1_b2_sbatch] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MODE" || -z "$BASE_CONFIG" || -z "$BASE_CHECKPOINT" || -z "$WORK_ROOT" || -z "$TIME_LIMIT" || -z "$MAIL_USER" ]]; then
  echo "[submit_b1_b2_sbatch] Missing required arguments." >&2
  usage
  exit 2
fi

case "$MODE" in
  b1_only|b2_only|both_parallel|both_sequential) ;;
  *) echo "[submit_b1_b2_sbatch] Invalid mode: $MODE" >&2; exit 2 ;;
esac

if [[ -z "$EVAL_GPUS" ]]; then
  EVAL_GPUS="$TRAIN_GPUS"
fi
if [[ -z "$GPUS_PER_NODE" ]]; then
  GPUS_PER_NODE="$TRAIN_GPUS"
fi
if [[ $RESUME -eq 1 && -z "$RUN_ID" ]]; then
  echo "[submit_b1_b2_sbatch] --resume requires --run-id so the existing run can be targeted." >&2
  exit 2
fi
if [[ $RUN_SUITE -eq 1 && -z "$SUITE_ONSET" ]]; then
  echo "[submit_b1_b2_sbatch] --suite-onset is required with --run-contradiction-suite." >&2
  exit 2
fi
if [[ $SKIP_FINAL_EVAL -eq 1 && $RUN_CLEAN_CMAP -eq 1 ]]; then
  echo "[submit_b1_b2_sbatch] --skip-final-eval cannot be combined with --run-clean-cmap." >&2
  exit 2
fi
if [[ $SKIP_FINAL_EVAL -eq 1 && $RUN_SUITE -eq 1 ]]; then
  echo "[submit_b1_b2_sbatch] --skip-final-eval cannot be combined with --run-contradiction-suite." >&2
  exit 2
fi
if (( GPUS_PER_NODE <= 0 || TRAIN_GPUS <= 0 || EVAL_GPUS <= 0 )); then
  echo "[submit_b1_b2_sbatch] GPU values must be positive integers." >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_SHA="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD)"
TS="$(date -u +%Y%m%d_%H%M%S)"
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="${TS}_${GIT_SHA}_b1b2"
fi

PARALLEL_FACTOR=1
if [[ "$MODE" == "both_parallel" ]]; then
  PARALLEL_FACTOR=2
fi
MAX_STEP_GPUS=$TRAIN_GPUS
if (( EVAL_GPUS > MAX_STEP_GPUS )); then
  MAX_STEP_GPUS=$EVAL_GPUS
fi
ALLOCATED_GPUS=$(( PARALLEL_FACTOR * MAX_STEP_GPUS ))
if (( ALLOCATED_GPUS % GPUS_PER_NODE != 0 )); then
  echo "[submit_b1_b2_sbatch] Total allocated GPUs (${ALLOCATED_GPUS}) must be divisible by --gpus-per-node (${GPUS_PER_NODE}) for this sbatch wrapper." >&2
  exit 2
fi
NODES=$(( ALLOCATED_GPUS / GPUS_PER_NODE ))
SBATCH_BASE_ROOT="${WORK_ROOT%/}/sbatch/b1_b2/${RUN_ID}"
SBATCH_ROOT="$SBATCH_BASE_ROOT"
if [[ $DRY_RUN -eq 1 ]]; then
  SBATCH_ROOT="${WORK_ROOT%/}/sbatch/dry_run/b1_b2/${RUN_ID}/${TS}"
fi
SBATCH_LOG_DIR="$SBATCH_ROOT/logs"
SBATCH_SCRIPT="$SBATCH_ROOT/${MODE}.sbatch"
mkdir -p "$SBATCH_LOG_DIR"

RUN_CMD=(bash tools/experiments/run_b1_b2.sh
  --mode "$MODE"
  --base-config "$BASE_CONFIG"
  --base-checkpoint "$BASE_CHECKPOINT"
  --work-root "$WORK_ROOT"
  --run-id "$RUN_ID"
  --seed "$SEED"
  --launcher slurm-step
  --train-gpus "$TRAIN_GPUS"
  --eval-gpus "$EVAL_GPUS"
  --gpus-per-node "$GPUS_PER_NODE"
  --cpus-per-task "$CPUS_PER_TASK"
  --available-gpus "$ALLOCATED_GPUS")

if [[ $RESUME -eq 1 ]]; then
  RUN_CMD+=(--resume)
fi
if [[ $SKIP_TRAIN_VALIDATION -eq 1 ]]; then
  RUN_CMD+=(--skip-train-validation)
fi
if [[ $SKIP_FINAL_EVAL -eq 1 ]]; then
  RUN_CMD+=(--skip-final-eval)
fi
if [[ -n "$CFG_COMMON_STR" ]]; then
  RUN_CMD+=(--cfg-options-common "$CFG_COMMON_STR")
fi
if [[ -n "$B1_CFG_STR" ]]; then
  RUN_CMD+=(--b1-cfg-options "$B1_CFG_STR")
fi
if [[ -n "$B2_CFG_STR" ]]; then
  RUN_CMD+=(--b2-cfg-options "$B2_CFG_STR")
fi
if [[ -n "$EVAL_CFG_STR" ]]; then
  RUN_CMD+=(--eval-cfg-options "$EVAL_CFG_STR")
fi
if [[ $RUN_CLEAN_CMAP -eq 1 ]]; then
  RUN_CMD+=(--run-clean-cmap)
fi
if [[ "$CONS_FRAMES" != "5" ]]; then
  RUN_CMD+=(--cons-frames "$CONS_FRAMES")
fi
if [[ $RUN_SUITE -eq 1 ]]; then
  RUN_CMD+=(--run-contradiction-suite --suite-modes "$SUITE_MODES_STR" --suite-stale-offsets "$SUITE_STALE_OFFSETS_STR" --suite-onset "$SUITE_ONSET" --suite-c-tail-keep-recent "$SUITE_C_TAIL_KEEP_RECENT")
fi
if [[ $DRY_RUN -eq 1 ]]; then
  RUN_CMD+=(--dry-run)
fi

printf -v RUN_CMD_STR '%q ' "${RUN_CMD[@]}"

SBATCH_DIRECTIVES=(
  "#SBATCH --job-name=${JOB_NAME}_${MODE}_${RUN_ID}"
  "#SBATCH --nodes=${NODES}"
  "#SBATCH --ntasks=${ALLOCATED_GPUS}"
  "#SBATCH --ntasks-per-node=${GPUS_PER_NODE}"
  "#SBATCH --cpus-per-task=${CPUS_PER_TASK}"
  "#SBATCH --gres=gpu:${GPUS_PER_NODE}"
  "#SBATCH --gpus-per-task=1"
  "#SBATCH --time=${TIME_LIMIT}"
  "#SBATCH --exclude=neu303"
  "#SBATCH --output=${SBATCH_LOG_DIR}/%x-%j.out"
  "#SBATCH --error=${SBATCH_LOG_DIR}/%x-%j.err"
  "#SBATCH --mail-type=${MAIL_TYPE}"
  "#SBATCH --mail-user=${MAIL_USER}"
)

if [[ -n "$PARTITION" ]]; then
  SBATCH_DIRECTIVES+=("#SBATCH --partition=${PARTITION}")
fi
if [[ -n "$QOS" ]]; then
  SBATCH_DIRECTIVES+=("#SBATCH --qos=${QOS}")
fi
if [[ -n "$ACCOUNT" ]]; then
  SBATCH_DIRECTIVES+=("#SBATCH --account=${ACCOUNT}")
fi
if [[ -n "$CONSTRAINT" ]]; then
  SBATCH_DIRECTIVES+=("#SBATCH --constraint=${CONSTRAINT}")
fi
if [[ -n "$MEMORY" ]]; then
  SBATCH_DIRECTIVES+=("#SBATCH --mem=${MEMORY}")
fi
if [[ -n "$DEPENDENCY" ]]; then
  SBATCH_DIRECTIVES+=("#SBATCH --dependency=${DEPENDENCY}")
fi

printf '%s\n' '#!/usr/bin/env bash' > "$SBATCH_SCRIPT"
printf '%s\n' "${SBATCH_DIRECTIVES[@]}" >> "$SBATCH_SCRIPT"
printf '%s\n' 'set -euo pipefail' >> "$SBATCH_SCRIPT"

# 1. Inject the Environment Variables
cat >> "$SBATCH_SCRIPT" <<EOF

# Use direct Conda hook instead of module load for reliability
cd /n/fs/dynamicbias/tracker
eval "\$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

# Prevent PyTorch DataLoader thread explosions
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Bypass SSL verification for downloading pre-trained weights
export PYTHONHTTPSVERIFY=0

export PROJECT_ROOT=${PROJECT_ROOT}
export PYTHONPATH="\$PROJECT_ROOT:\${PYTHONPATH:-}"
export SRUN_CPUS_PER_TASK=${CPUS_PER_TASK}
EOF

# 2. Inject your literal File Transfer Script
# We use quotes around 'FILE_TRANSFER_EOF' so Bash writes it exactly as-is
cat >> "$SBATCH_SCRIPT" <<'FILE_TRANSFER_EOF'

#### FILE TRANSFER ####
DIR_NAME="/scratch/rc5898/tracker"
TAR_PATH="/scratch/rc5898/tracker.tar"

if [ ! -d "$DIR_NAME" ]; then
    echo "Tracker directory not found on this node. Handling transfer..."
    if [ ! -f "$TAR_PATH" ]; then
        echo "Tar file not found locally. Downloading from login node..."
        expect << EXPECT_EOF
set timeout -1
set send_slow {1 .1}
spawn rsync -avP "rc5898@neuronic.cs.princeton.edu:$TAR_PATH" "/scratch/rc5898/"
expect "Passcode or option (1-3):"
sleep 3
send -s "1\r"
expect eof
EXPECT_EOF
    fi
    echo "Uncompressing $TAR_PATH..."
    tar -xf "$TAR_PATH" -C /scratch/rc5898/
    echo "Uncompress complete."
else
    echo "Directory $DIR_NAME already exists on this node. Skipping transfer."
fi

echo "Setting up symlink portal..."
ln -sfn /scratch/rc5898/tracker/datasets /n/fs/dynamicbias/tracker/datasets

echo "Injecting tracking files from project directory to local scratch..."
\cp /n/fs/dynamicbias/tracker/datasets-swp/*gt_tracks.pkl /scratch/rc5898/tracker/datasets/nuscenes/

FILE_TRANSFER_EOF

# 3. Inject the SRUN arguments
if [[ -n "$SRUN_ARGS_STR" ]]; then
  printf 'export SRUN_ARGS=%q\n' "$SRUN_ARGS_STR" >> "$SBATCH_SCRIPT"
fi

# 4. Inject the final Python execution command
cat >> "$SBATCH_SCRIPT" <<EOF

cd "\$PROJECT_ROOT"
${RUN_CMD_STR}
EOF

chmod +x "$SBATCH_SCRIPT"

declare -a SBATCH_SUBMIT_CMD=(sbatch)
if [[ -n "$SBATCH_EXTRA_ARGS_STR" ]]; then
  read -r -a SBATCH_EXTRA_ARGS_ARR <<< "$SBATCH_EXTRA_ARGS_STR"
  SBATCH_SUBMIT_CMD+=("${SBATCH_EXTRA_ARGS_ARR[@]}")
fi
SBATCH_SUBMIT_CMD+=("$SBATCH_SCRIPT")

echo "[submit_b1_b2_sbatch] Generated: $SBATCH_SCRIPT"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[submit_b1_b2_sbatch] Dry-run preview kept separate from canonical path: $SBATCH_BASE_ROOT/${MODE}.sbatch"
  exit 0
fi
echo "[submit_b1_b2_sbatch] Submission command: ${SBATCH_SUBMIT_CMD[*]}"

SUBMIT_OUTPUT="$("${SBATCH_SUBMIT_CMD[@]}")"
echo "$SUBMIT_OUTPUT"
