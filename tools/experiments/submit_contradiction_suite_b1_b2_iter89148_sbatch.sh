#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit contradiction-suite evaluation for B1/B2 iter_89148 as one Slurm job.

Usage:
  bash tools/experiments/submit_contradiction_suite_b1_b2_iter89148_sbatch.sh --mail-user EMAIL --onset N [options]

Required:
  --mail-user EMAIL

Optional:
  --onset N                         Contradiction onset frame index (default: 0; train-matched)
  --partition NAME
  --job-name NAME                   (default: maptracker_contra_suite_b1b2)
  --time LIMIT                      (default: 12:00:00)
  --gpus N                          GPUs requested for the job (default: 1)
  --cpus-per-task N                 (default: 6)
  --mem VALUE                       (default: 96G)
  --work-root DIR                   (default: /n/fs/dynamicbias/tracker/work_dirs)
  --conda-env NAME_OR_PATH          (default: /n/fs/dynamicbias/tracker/env-maptracker)
  --b1-config PATH
  --b2-config PATH
  --b1-checkpoint PATH
  --b2-checkpoint PATH
  --baselines "b1 b2"               Which baselines to run (default: b1 b2)
  --modes "c_full c_tail"           (default: c_full c_tail)
  --stale-offsets "1 2 3"           (default: 1 2 3; train-matched)
  --c-tail-keep-recent N            (default: 1)
  --suite-tag TAG                   Output suffix tag (default: trainmatched)
  --allow-overwrite                 Allow writing into existing non-empty suite dirs.
  --suite-launcher none|slurm       Launcher passed to run_contradiction_suite.py (default: none)
  --account NAME
  --qos NAME
  --constraint EXPR
  --mail-type VALUE                 (default: END,FAIL)
  --extra-suite-args "..."          Extra args appended to each suite command.
  --dry-run                         Generate sbatch script but do not submit.
  -h, --help

    bash tools/experiments/submit_contradiction_suite_b1_b2_iter89148_sbatch.sh --mail-user EMAIL [options]
  - Defaults are wired to:
    b1: /n/fs/dynamicbias/tracker/work_dirs/experiments/b1_b2/b1_stage3_gpu4_short_trainonly/b1/train/iter_89148.pth
    b2: /n/fs/dynamicbias/tracker/work_dirs/experiments/b1_b2/b2_stage3_gpu4_short_trainonly/b2/train/iter_89148.pth
  - Suite outputs are written under:
    <work-root>/experiments/b1_b2/{b1_stage3_gpu4_short_trainonly,b2_stage3_gpu4_short_trainonly}/{b1,b2}/contradiction_suite/<iter>_onset<onset>_<suite-tag>/
USAGE
}

PARTITION=""
MAIL_USER=""
ONSET=0
JOB_NAME="maptracker_contra_suite_b1b2"
TIME_LIMIT="12:00:00"
GPUS=1
CPUS_PER_TASK=6
MEMORY="96G"
WORK_ROOT="/n/fs/dynamicbias/tracker/work_dirs"
CONDA_ENV="/n/fs/dynamicbias/tracker/env-maptracker"
BASELINES_STR="b1 b2"
MODES_STR="c_full c_tail"
STALE_OFFSETS_STR="1 2 3"
C_TAIL_KEEP_RECENT=1
SUITE_TAG="trainmatched"
ALLOW_OVERWRITE=0
SUITE_LAUNCHER="none"
ACCOUNT=""
QOS=""
CONSTRAINT=""
MAIL_TYPE="END,FAIL"
EXTRA_SUITE_ARGS_STR=""
B1_RUN_ID="b1_stage3_gpu4_short_trainonly"
B2_RUN_ID="b2_stage3_gpu4_short_trainonly"
B1_BASELINE="b1"
B2_BASELINE="b2"
DRY_RUN=0

B1_CONFIG="/n/fs/dynamicbias/tracker/work_dirs/experiments/b1_b2/b1_stage3_gpu4_short_trainonly/b1/train/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py"
B2_CONFIG="/n/fs/dynamicbias/tracker/work_dirs/experiments/b1_b2/b2_stage3_gpu4_short_trainonly/b2/train/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py"

B1_CHECKPOINT="/n/fs/dynamicbias/tracker/work_dirs/experiments/b1_b2/b1_stage3_gpu4_short_trainonly/b1/train/iter_89148.pth"
B2_CHECKPOINT="/n/fs/dynamicbias/tracker/work_dirs/experiments/b1_b2/b2_stage3_gpu4_short_trainonly/b2/train/iter_89148.pth"

require_int_ge() {
  local val="$1"
  local name="$2"
  local min_val="$3"
  if ! [[ "$val" =~ ^[0-9]+$ ]]; then
    echo "[submit_contradiction_suite_b1_b2_iter89148_sbatch] ${name} must be an integer >= ${min_val}." >&2
    exit 2
  fi
  if (( val < min_val )); then
    echo "[submit_contradiction_suite_b1_b2_iter89148_sbatch] ${name} must be >= ${min_val}." >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --mail-user) MAIL_USER="$2"; shift 2 ;;
    --onset) ONSET="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --b1-config) B1_CONFIG="$2"; shift 2 ;;
    --b2-config) B2_CONFIG="$2"; shift 2 ;;
    --b1-checkpoint) B1_CHECKPOINT="$2"; shift 2 ;;
    --b2-checkpoint) B2_CHECKPOINT="$2"; shift 2 ;;
    --baselines) BASELINES_STR="$2"; shift 2 ;;
    --modes) MODES_STR="$2"; shift 2 ;;
    --stale-offsets) STALE_OFFSETS_STR="$2"; shift 2 ;;
    --c-tail-keep-recent) C_TAIL_KEEP_RECENT="$2"; shift 2 ;;
    --suite-tag) SUITE_TAG="$2"; shift 2 ;;
    --allow-overwrite) ALLOW_OVERWRITE=1; shift ;;
    --suite-launcher) SUITE_LAUNCHER="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --constraint) CONSTRAINT="$2"; shift 2 ;;
    --mail-type) MAIL_TYPE="$2"; shift 2 ;;
    --extra-suite-args) EXTRA_SUITE_ARGS_STR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[submit_contradiction_suite_b1_b2_iter89148_sbatch] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MAIL_USER" ]]; then
  echo "[submit_contradiction_suite_b1_b2_iter89148_sbatch] Missing required arguments." >&2
  usage
  exit 2
fi

require_int_ge "$ONSET" "--onset" 0
require_int_ge "$GPUS" "--gpus" 1
require_int_ge "$CPUS_PER_TASK" "--cpus-per-task" 1
require_int_ge "$C_TAIL_KEEP_RECENT" "--c-tail-keep-recent" 0

case "$SUITE_LAUNCHER" in
  none|slurm) ;;
  *) echo "[submit_contradiction_suite_b1_b2_iter89148_sbatch] --suite-launcher must be one of: none, slurm" >&2; exit 2 ;;
esac

read -r -a BASELINES_ARR <<< "$BASELINES_STR"
read -r -a MODES_ARR <<< "$MODES_STR"
read -r -a STALE_OFFSETS_ARR <<< "$STALE_OFFSETS_STR"

if [[ ${#BASELINES_ARR[@]} -eq 0 || ${#MODES_ARR[@]} -eq 0 || ${#STALE_OFFSETS_ARR[@]} -eq 0 ]]; then
  echo "[submit_contradiction_suite_b1_b2_iter89148_sbatch] --baselines, --modes and --stale-offsets must each contain at least one value." >&2
  exit 2
fi

RUN_B1=0
RUN_B2=0
for base in "${BASELINES_ARR[@]}"; do
  case "$base" in
    b1) RUN_B1=1 ;;
    b2) RUN_B2=1 ;;
    *) echo "[submit_contradiction_suite_b1_b2_iter89148_sbatch] Unsupported baseline in --baselines: $base (allowed: b1 b2)" >&2; exit 2 ;;
  esac
done

for mode in "${MODES_ARR[@]}"; do
  case "$mode" in
    c_full|c_tail) ;;
    *) echo "[submit_contradiction_suite_b1_b2_iter89148_sbatch] Unsupported mode in --modes: $mode (allowed: c_full c_tail)" >&2; exit 2 ;;
  esac
done

for off in "${STALE_OFFSETS_ARR[@]}"; do
  require_int_ge "$off" "--stale-offsets element" 0
done

for f in "$B1_CONFIG" "$B2_CONFIG" "$B1_CHECKPOINT" "$B2_CHECKPOINT"; do
  if [[ ! -f "$f" ]]; then
    echo "[submit_contradiction_suite_b1_b2_iter89148_sbatch] Missing required file: $f" >&2
    exit 1
  fi
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date -u +%Y%m%d_%H%M%S)"
SBATCH_ROOT="${WORK_ROOT%/}/sbatch/contradiction_suite_b1_b2_iter89148/${TS}"
SBATCH_LOG_DIR="$SBATCH_ROOT/logs"
SBATCH_SCRIPT="$SBATCH_ROOT/submit.sbatch"
mkdir -p "$SBATCH_LOG_DIR"

B1_ITER_TAG="$(basename "${B1_CHECKPOINT%.pth}")"
B2_ITER_TAG="$(basename "${B2_CHECKPOINT%.pth}")"
SUITE_TAG_SUFFIX=""
if [[ -n "$SUITE_TAG" ]]; then
  SUITE_TAG_SUFFIX="_${SUITE_TAG}"
fi

B1_SUITE_ROOT="${WORK_ROOT%/}/experiments/b1_b2/${B1_RUN_ID}/${B1_BASELINE}/contradiction_suite/${B1_ITER_TAG}_onset${ONSET}${SUITE_TAG_SUFFIX}"
B2_SUITE_ROOT="${WORK_ROOT%/}/experiments/b1_b2/${B2_RUN_ID}/${B2_BASELINE}/contradiction_suite/${B2_ITER_TAG}_onset${ONSET}${SUITE_TAG_SUFFIX}"

if [[ "$ALLOW_OVERWRITE" -ne 1 ]]; then
  if [[ "$RUN_B1" -eq 1 && -d "$B1_SUITE_ROOT" ]] && [[ -n "$(ls -A "$B1_SUITE_ROOT" 2>/dev/null || true)" ]]; then
    echo "[submit_contradiction_suite_b1_b2_iter89148_sbatch] Refusing to overwrite existing non-empty output dir: $B1_SUITE_ROOT" >&2
    echo "[submit_contradiction_suite_b1_b2_iter89148_sbatch] Use --suite-tag to separate outputs or --allow-overwrite to bypass." >&2
    exit 1
  fi
  if [[ "$RUN_B2" -eq 1 && -d "$B2_SUITE_ROOT" ]] && [[ -n "$(ls -A "$B2_SUITE_ROOT" 2>/dev/null || true)" ]]; then
    echo "[submit_contradiction_suite_b1_b2_iter89148_sbatch] Refusing to overwrite existing non-empty output dir: $B2_SUITE_ROOT" >&2
    echo "[submit_contradiction_suite_b1_b2_iter89148_sbatch] Use --suite-tag to separate outputs or --allow-overwrite to bypass." >&2
    exit 1
  fi
fi

B1_CMD=()
B2_CMD=()

if [[ "$RUN_B1" -eq 1 ]]; then
  B1_CMD=(python tools/tracking/run_contradiction_suite.py
    "$B1_CONFIG"
    "$B1_CHECKPOINT"
    --work-root "$B1_SUITE_ROOT"
    --modes "${MODES_ARR[@]}"
    --stale-offsets "${STALE_OFFSETS_ARR[@]}"
    --onset "$ONSET"
    --c-tail-keep-recent "$C_TAIL_KEEP_RECENT"
    --launcher "$SUITE_LAUNCHER")
fi

if [[ "$RUN_B2" -eq 1 ]]; then
  B2_CMD=(python tools/tracking/run_contradiction_suite.py
    "$B2_CONFIG"
    "$B2_CHECKPOINT"
    --work-root "$B2_SUITE_ROOT"
    --modes "${MODES_ARR[@]}"
    --stale-offsets "${STALE_OFFSETS_ARR[@]}"
    --onset "$ONSET"
    --c-tail-keep-recent "$C_TAIL_KEEP_RECENT"
    --launcher "$SUITE_LAUNCHER")
fi

if [[ -n "$EXTRA_SUITE_ARGS_STR" ]]; then
  read -r -a EXTRA_SUITE_ARGS_ARR <<< "$EXTRA_SUITE_ARGS_STR"
  if [[ "$RUN_B1" -eq 1 ]]; then
    B1_CMD+=("${EXTRA_SUITE_ARGS_ARR[@]}")
  fi
  if [[ "$RUN_B2" -eq 1 ]]; then
    B2_CMD+=("${EXTRA_SUITE_ARGS_ARR[@]}")
  fi
fi

if [[ "$RUN_B1" -eq 1 ]]; then
  printf -v B1_CMD_STR '%q ' "${B1_CMD[@]}"
fi
if [[ "$RUN_B2" -eq 1 ]]; then
  printf -v B2_CMD_STR '%q ' "${B2_CMD[@]}"
fi

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

EOF
  if [[ "$RUN_B1" -eq 1 ]]; then
    cat <<EOF
echo "[contradiction-suite] Running B1"
$B1_CMD_STR

EOF
  fi
  if [[ "$RUN_B2" -eq 1 ]]; then
    cat <<EOF
echo "[contradiction-suite] Running B2"
$B2_CMD_STR

EOF
  fi
  cat <<EOF

echo "[contradiction-suite] Done"
EOF
} > "$SBATCH_SCRIPT"

chmod +x "$SBATCH_SCRIPT"

echo "[submit_contradiction_suite_b1_b2_iter89148_sbatch] Generated: $SBATCH_SCRIPT"
echo "[submit_contradiction_suite_b1_b2_iter89148_sbatch] Logs dir: $SBATCH_LOG_DIR"

if [[ "$RUN_B1" -eq 1 ]]; then
  echo "[submit_contradiction_suite_b1_b2_iter89148_sbatch] B1 output root: $B1_SUITE_ROOT"
fi
if [[ "$RUN_B2" -eq 1 ]]; then
  echo "[submit_contradiction_suite_b1_b2_iter89148_sbatch] B2 output root: $B2_SUITE_ROOT"
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[submit_contradiction_suite_b1_b2_iter89148_sbatch] Dry run, not submitting."
  exit 0
fi

SUBMIT_OUTPUT="$(sbatch "$SBATCH_SCRIPT")"
echo "$SUBMIT_OUTPUT"
