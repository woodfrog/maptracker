#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit corruption evaluation for B1/B2 iter_89148 as four separate Slurm jobs:
  b1 × c_full,  b2 × c_full,  b1 × c_tail,  b2 × c_tail
Each job runs all stale offsets sequentially using the full eval pipeline.

Usage:
  bash tools/experiments/submit_corruption_eval_b1_b2_iter89148_sbatch.sh \
    --mail-user EMAIL [options]

Required:
  --mail-user EMAIL

Optional:
  --partition NAME
  --job-name-prefix NAME     Prefix for all four job names (default: maptracker_corrupt_b1b2)
  --time LIMIT               Wall time per job (default: 04:00:00)
  --gpus N                   GPUs per job (default: 4)
  --cpus-per-task N          (default: 6)
  --mem VALUE                (default: 96G)
  --work-root DIR            (default: /n/fs/dynamicbias/tracker/work_dirs)
  --conda-env NAME_OR_PATH   (default: /n/fs/dynamicbias/tracker/env-maptracker)
  --stale-offsets "1 2 3"    Space-separated stale offset list (default: 1 2 3)
  --keep-recent N            c_tail keep_recent (default: 1)
  --onset N                  Corruption onset frame index (default: 0)
  --seed N                   Random seed (default: 0)
  --account NAME
  --qos NAME
  --constraint EXPR
  --mail-type VALUE          (default: END,FAIL)
  --extra-wrap-args "..."    Extra args appended to each run_corruption_eval invocation.
  --dry-run                  Generate sbatch files but do not submit.
  -h, --help

Notes:
  - Uses launcher=slurm-step inside each job for robust port retry handling.
  - Each job covers one (baseline, mode) pair and iterates over all stale offsets.
  - Existing non-empty eval dirs are silently skipped (not deleted) unless --rerun
    were passed via --extra-wrap-args.
  - Eval outputs are written under:
    <work-root>/experiments/b1_b2/{b1,b2}_stage3_gpu4_short_trainonly/{b1,b2}/eval_deferred/iter_89148/
      cfull_onset<N>_stale<K>/
      ctail_onset<N>_keep<M>_stale<K>/
USAGE
}

PARTITION=""
MAIL_USER=""
JOB_NAME_PREFIX="maptracker_corrupt_b1b2"
TIME_LIMIT="04:00:00"
GPUS=4
CPUS_PER_TASK=6
MEMORY="96G"
WORK_ROOT="/n/fs/dynamicbias/tracker/work_dirs"
CONDA_ENV="/n/fs/dynamicbias/tracker/env-maptracker"
STALE_OFFSETS_STR="1 2 3"
KEEP_RECENT=1
ONSET=0
SEED=0
ACCOUNT=""
QOS=""
CONSTRAINT=""
MAIL_TYPE="END,FAIL"
EXTRA_WRAP_ARGS_STR=""
DRY_RUN=0

B1_RUN_ID="b1_stage3_gpu4_short_trainonly"
B2_RUN_ID="b2_stage3_gpu4_short_trainonly"

require_int_ge() {
  local val="$1"
  local name="$2"
  local min_val="$3"
  if ! [[ "$val" =~ ^[0-9]+$ ]]; then
    echo "[submit_corruption_eval_b1_b2_iter89148_sbatch] ${name} must be a non-negative integer." >&2
    exit 2
  fi
  if (( val < min_val )); then
    echo "[submit_corruption_eval_b1_b2_iter89148_sbatch] ${name} must be >= ${min_val}." >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --mail-user) MAIL_USER="$2"; shift 2 ;;
    --job-name-prefix) JOB_NAME_PREFIX="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --stale-offsets) STALE_OFFSETS_STR="$2"; shift 2 ;;
    --keep-recent) KEEP_RECENT="$2"; shift 2 ;;
    --onset) ONSET="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --constraint) CONSTRAINT="$2"; shift 2 ;;
    --mail-type) MAIL_TYPE="$2"; shift 2 ;;
    --extra-wrap-args) EXTRA_WRAP_ARGS_STR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[submit_corruption_eval_b1_b2_iter89148_sbatch] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MAIL_USER" ]]; then
  echo "[submit_corruption_eval_b1_b2_iter89148_sbatch] Missing required --mail-user." >&2
  usage
  exit 2
fi

require_int_ge "$GPUS" "--gpus" 1
require_int_ge "$CPUS_PER_TASK" "--cpus-per-task" 1
require_int_ge "$KEEP_RECENT" "--keep-recent" 0
require_int_ge "$ONSET" "--onset" 0
require_int_ge "$SEED" "--seed" 0

read -r -a STALE_OFFSETS_ARR <<< "$STALE_OFFSETS_STR"
if [[ ${#STALE_OFFSETS_ARR[@]} -eq 0 ]]; then
  echo "[submit_corruption_eval_b1_b2_iter89148_sbatch] --stale-offsets must contain at least one value." >&2
  exit 2
fi
for off in "${STALE_OFFSETS_ARR[@]}"; do
  require_int_ge "$off" "--stale-offsets element" 0
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Pre-flight: verify all required config + checkpoint files exist
B1_CONFIG="${WORK_ROOT%/}/experiments/b1_b2/${B1_RUN_ID}/b1/train/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py"
B2_CONFIG="${WORK_ROOT%/}/experiments/b1_b2/${B2_RUN_ID}/b2/train/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py"
B1_CHECKPOINT="${WORK_ROOT%/}/experiments/b1_b2/${B1_RUN_ID}/b1/train/iter_89148.pth"
B2_CHECKPOINT="${WORK_ROOT%/}/experiments/b1_b2/${B2_RUN_ID}/b2/train/iter_89148.pth"

for f in "$B1_CONFIG" "$B2_CONFIG" "$B1_CHECKPOINT" "$B2_CHECKPOINT"; do
  if [[ ! -f "$f" ]]; then
    echo "[submit_corruption_eval_b1_b2_iter89148_sbatch] Missing required file: $f" >&2
    exit 1
  fi
done

TS="$(date -u +%Y%m%d_%H%M%S)"
SBATCH_BASE="${WORK_ROOT%/}/sbatch/corruption_eval_b1_b2_iter89148/${TS}"

declare -a SUBMITTED_SCRIPTS=()
declare -a SUBMITTED_JOB_IDS=()

# Helper: write_file_transfer_block <log_tag>
# Writes the file transfer section into the current heredoc context.
# Must be called inside a { ... } > script block.
write_and_submit_sbatch() {
  local baseline="$1"  # b1 or b2
  local mode="$2"       # c_full or c_tail
  local run_id="$3"

  local job_name="${JOB_NAME_PREFIX}_${baseline}_${mode}"
  local sbatch_dir="${SBATCH_BASE}/${baseline}_${mode}"
  local sbatch_log_dir="${sbatch_dir}/logs"
  local sbatch_script="${sbatch_dir}/submit.sbatch"

  mkdir -p "$sbatch_log_dir"

  # Build the run_corruption_eval invocation
  local run_cmd_arr=(bash tools/experiments/run_corruption_eval_b1_b2_iter89148.sh
    --baseline "$baseline"
    --mode "$mode"
    --launcher slurm-step
    --gpus "$GPUS"
    --seed "$SEED"
    --work-root "$WORK_ROOT"
    --stale-offsets "$STALE_OFFSETS_STR"
    --onset "$ONSET"
  )

  if [[ "$mode" == "c_tail" ]]; then
    run_cmd_arr+=(--keep-recent "$KEEP_RECENT")
  fi

  if [[ -n "$EXTRA_WRAP_ARGS_STR" ]]; then
    read -r -a extra_arr <<< "$EXTRA_WRAP_ARGS_STR"
    run_cmd_arr+=("${extra_arr[@]}")
  fi

  local run_cmd_str
  printf -v run_cmd_str '%q ' "${run_cmd_arr[@]}"

  {
    echo '#!/usr/bin/env bash'
    echo "#SBATCH --job-name=${job_name}"
    echo '#SBATCH --nodes=1'
    echo "#SBATCH --ntasks=${GPUS}"
    echo "#SBATCH --ntasks-per-node=${GPUS}"
    echo "#SBATCH --gres=gpu:${GPUS}"
    echo '#SBATCH --gpus-per-task=1'
    echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}"
    echo "#SBATCH --time=${TIME_LIMIT}"
    echo "#SBATCH --mem=${MEMORY}"
    echo "#SBATCH --output=${sbatch_log_dir}/%x-%j.out"
    echo "#SBATCH --error=${sbatch_log_dir}/%x-%j.err"
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
echo "[${job_name}] Preparing file transfers..."
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

echo "[${job_name}] File transfer complete"

echo "[${job_name}] Starting corruption eval: baseline=${baseline} mode=${mode}"
${run_cmd_str}
echo "[${job_name}] Completed"
EOF
  } > "$sbatch_script"

  chmod +x "$sbatch_script"
  SUBMITTED_SCRIPTS+=("$sbatch_script")

  echo "[submit_corruption_eval_b1_b2_iter89148_sbatch] Generated: $sbatch_script"

  if [[ $DRY_RUN -eq 0 ]]; then
    local submit_out
    submit_out="$(sbatch "$sbatch_script")"
    echo "$submit_out"
    SUBMITTED_JOB_IDS+=("$submit_out")
  fi
}

echo "[submit_corruption_eval_b1_b2_iter89148_sbatch] Preparing 4 sbatch jobs (b1/b2 × c_full/c_tail)..."
echo "[submit_corruption_eval_b1_b2_iter89148_sbatch] Sbatch root: $SBATCH_BASE"
echo ""

write_and_submit_sbatch "b1" "c_full" "$B1_RUN_ID"
write_and_submit_sbatch "b2" "c_full" "$B2_RUN_ID"
write_and_submit_sbatch "b1" "c_tail" "$B1_RUN_ID"
write_and_submit_sbatch "b2" "c_tail" "$B2_RUN_ID"

echo ""
echo "[submit_corruption_eval_b1_b2_iter89148_sbatch] All 4 sbatch scripts generated under: $SBATCH_BASE"
echo "[submit_corruption_eval_b1_b2_iter89148_sbatch] Scripts:"
for s in "${SUBMITTED_SCRIPTS[@]}"; do
  echo "  $s"
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[submit_corruption_eval_b1_b2_iter89148_sbatch] Dry run, not submitting."
else
  echo "[submit_corruption_eval_b1_b2_iter89148_sbatch] Submitted jobs:"
  for j in "${SUBMITTED_JOB_IDS[@]}"; do
    echo "  $j"
  done
fi
