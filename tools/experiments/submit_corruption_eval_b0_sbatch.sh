#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit a corruption-suite evaluation for the B0 (vanilla MapTracker) model as a single
Slurm job.  Runs the full contradiction matrix: modes × stale-offsets.

Usage:
  bash tools/experiments/submit_corruption_eval_b0_sbatch.sh --mail-user EMAIL [options]

Required:
  --mail-user EMAIL

Optional:
  --checkpoint PATH           Path to the B0 checkpoint.
                              Default: <work-root>/pretrained_ckpts/b0_nusc_oldsplit/latest.pth
                              (Download from https://huggingface.co/cccjc/maptracker or
                               https://www.dropbox.com/scl/fo/miulg8q9oby7q2x5vemme/ALoxX1HyxGlfR9y3xlqfzeE)
  --config PATH               Eval config.
                              Default: plugin/configs/maptracker/nuscenes_oldsplit/
                                       maptracker_nusc_oldsplit_5frame_span10_stage3_b0_eval.py
  --work-root DIR             (default: <repo-root>/work_dirs)
  --modes "c_full c_tail"     Corruption modes to include (default: c_full c_tail)
  --stale-offsets "1 2 3"     Stale offsets to sweep (default: 1 2 3 — matches B1/B2)
  --onset N                   Corruption onset frame index (default: 0 — train-matched)
  --c-tail-keep-recent N      Frames kept for c_tail mode (default: 1)
  --suite-tag TAG             Output suffix tag (default: trainmatched)
  --allow-overwrite           Overwrite a non-empty existing suite output dir.
  --gpus N                    GPUs requested for the job (default: 1)
  --cpus-per-task N           (default: 6)
  --mem VALUE                 (default: 96G)
  --time LIMIT                (default: 06:00:00)
  --job-name NAME             (default: maptracker_corruption_eval_b0)
  --partition NAME            (default: none — Della routes GPU jobs via QOS, not partition)
  --qos NAME                  (default: gpu-short — 1-day wall limit; sufficient for 6h suite)
  --account NAME
  --constraint EXPR           (default: a100&nomig — full A100 nodes; excludes MIG-sliced 3g.40gb nodes)
  --mail-type VALUE           (default: END,FAIL)
  --module-load MOD           (default: anaconda3/2023.9)
  --conda-env NAME            (default: maptracker)
  --dry-run                   Generate sbatch file but do not submit.
  -h, --help

Notes:
  - Suite outputs land under:
    <work-root>/experiments/b0/corruption_suite/<ckpt-tag>_onset<N>_<suite-tag>/
  - The sbatch script and logs are kept at:
    <work-root>/sbatch/corruption_eval_b0/<TIMESTAMP>/
  - stale-offsets default (1 2 3) matches the B1/B2 training config for a fair comparison.
USAGE
}

MAIL_USER=""
CHECKPOINT=""
CONFIG=""
WORK_ROOT=""
MODES_STR="c_full c_tail"
STALE_OFFSETS_STR="1 2 3"
ONSET=0
C_TAIL_KEEP_RECENT=1
SUITE_TAG="trainmatched"
ALLOW_OVERWRITE=0
GPUS=1
CPUS_PER_TASK=6
MEMORY="96G"
TIME_LIMIT="06:00:00"
JOB_NAME="maptracker_corruption_eval_b0"
PARTITION=""
QOS="gpu-short"
ACCOUNT=""
CONSTRAINT="a100&nomig"
MAIL_TYPE="END,FAIL"
MODULE_LOAD="anaconda3/2023.9"
CONDA_ENV="maptracker"
DRY_RUN=0
SKIP_SUMMARY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mail-user)          MAIL_USER="$2";          shift 2 ;;
    --checkpoint)         CHECKPOINT="$2";          shift 2 ;;
    --config)             CONFIG="$2";              shift 2 ;;
    --work-root)          WORK_ROOT="$2";           shift 2 ;;
    --modes)              MODES_STR="$2";           shift 2 ;;
    --stale-offsets)      STALE_OFFSETS_STR="$2";   shift 2 ;;
    --onset)              ONSET="$2";               shift 2 ;;
    --c-tail-keep-recent) C_TAIL_KEEP_RECENT="$2";  shift 2 ;;
    --suite-tag)          SUITE_TAG="$2";           shift 2 ;;
    --allow-overwrite)    ALLOW_OVERWRITE=1;         shift   ;;
    --gpus)               GPUS="$2";                shift 2 ;;
    --cpus-per-task)      CPUS_PER_TASK="$2";       shift 2 ;;
    --mem)                MEMORY="$2";              shift 2 ;;
    --time)               TIME_LIMIT="$2";          shift 2 ;;
    --job-name)           JOB_NAME="$2";            shift 2 ;;
    --partition)          PARTITION="$2";           shift 2 ;;
    --qos)                QOS="$2";                 shift 2 ;;
    --account)            ACCOUNT="$2";             shift 2 ;;
    --constraint)         CONSTRAINT="$2";          shift 2 ;;
    --mail-type)          MAIL_TYPE="$2";           shift 2 ;;
    --module-load)        MODULE_LOAD="$2";         shift 2 ;;
    --conda-env)          CONDA_ENV="$2";           shift 2 ;;
    --dry-run)            DRY_RUN=1;                shift   ;;
    --skip-summary)       SKIP_SUMMARY=1;           shift   ;;
    -h|--help)            usage; exit 0 ;;
    *) echo "[submit_corruption_eval_b0_sbatch] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ -z "$MAIL_USER" ]]; then
  echo "[submit_corruption_eval_b0_sbatch] --mail-user is required." >&2
  usage
  exit 2
fi

require_int_ge() {
  local val="$1" name="$2" min_val="$3"
  if ! [[ "$val" =~ ^[0-9]+$ ]]; then
    echo "[submit_corruption_eval_b0_sbatch] ${name} must be a non-negative integer." >&2; exit 2
  fi
  if (( val < min_val )); then
    echo "[submit_corruption_eval_b0_sbatch] ${name} must be >= ${min_val}." >&2; exit 2
  fi
}

require_int_ge "$GPUS"               "--gpus"               1
require_int_ge "$CPUS_PER_TASK"      "--cpus-per-task"      1
require_int_ge "$ONSET"              "--onset"              0
require_int_ge "$C_TAIL_KEEP_RECENT" "--c-tail-keep-recent" 0

read -r -a MODES_ARR        <<< "$MODES_STR"
read -r -a STALE_OFFSETS_ARR <<< "$STALE_OFFSETS_STR"

if [[ ${#MODES_ARR[@]} -eq 0 || ${#STALE_OFFSETS_ARR[@]} -eq 0 ]]; then
  echo "[submit_corruption_eval_b0_sbatch] --modes and --stale-offsets must each contain at least one value." >&2
  exit 2
fi

for mode in "${MODES_ARR[@]}"; do
  case "$mode" in
    c_full|c_tail) ;;
    *) echo "[submit_corruption_eval_b0_sbatch] Unsupported mode: $mode (allowed: c_full c_tail)" >&2; exit 2 ;;
  esac
done

for off in "${STALE_OFFSETS_ARR[@]}"; do
  require_int_ge "$off" "--stale-offsets element" 0
done

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -z "$WORK_ROOT" ]]; then
  WORK_ROOT="${PROJECT_ROOT}/work_dirs"
fi
if [[ -z "$CONFIG" ]]; then
  CONFIG="${PROJECT_ROOT}/plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_b0_eval.py"
fi
if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="${WORK_ROOT}/pretrained_ckpts/b0_nusc_oldsplit/latest.pth"
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "[submit_corruption_eval_b0_sbatch] Config not found: $CONFIG" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[submit_corruption_eval_b0_sbatch] Checkpoint not found: $CHECKPOINT" >&2
  echo "[submit_corruption_eval_b0_sbatch] Download the B0 checkpoint (maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune/latest.pth)" >&2
  echo "[submit_corruption_eval_b0_sbatch] from https://huggingface.co/cccjc/maptracker and place it at:" >&2
  echo "[submit_corruption_eval_b0_sbatch]   ${CHECKPOINT}" >&2
  exit 1
fi

# Derive checkpoint tag from filename (e.g. "latest" from "latest.pth").
CKPT_TAG="$(basename "$CHECKPOINT")"
CKPT_TAG="${CKPT_TAG%.pth}"

SUITE_TAG_SUFFIX=""
if [[ -n "$SUITE_TAG" ]]; then
  SUITE_TAG_SUFFIX="_${SUITE_TAG}"
fi

SUITE_ROOT="${WORK_ROOT%/}/experiments/b0/corruption_suite/${CKPT_TAG}_onset${ONSET}${SUITE_TAG_SUFFIX}"

if [[ $ALLOW_OVERWRITE -ne 1 ]]; then
  if [[ -d "$SUITE_ROOT" ]] && [[ -n "$(ls -A "$SUITE_ROOT" 2>/dev/null || true)" ]]; then
    echo "[submit_corruption_eval_b0_sbatch] Refusing to overwrite existing non-empty output dir:" >&2
    echo "[submit_corruption_eval_b0_sbatch]   $SUITE_ROOT" >&2
    echo "[submit_corruption_eval_b0_sbatch] Use --suite-tag to write to a different path, or --allow-overwrite to bypass." >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Build sbatch
# ---------------------------------------------------------------------------

TS="$(date -u +%Y%m%d_%H%M%S)"
SBATCH_ROOT="${WORK_ROOT%/}/sbatch/corruption_eval_b0/${TS}"
SBATCH_LOG_DIR="$SBATCH_ROOT/logs"
SBATCH_SCRIPT="$SBATCH_ROOT/eval.sbatch"
mkdir -p "$SBATCH_LOG_DIR"

RUN_CMD=(python tools/tracking/run_contradiction_suite.py
  "$CONFIG"
  "$CHECKPOINT"
  --work-root "$SUITE_ROOT"
  --modes "${MODES_ARR[@]}"
  --stale-offsets "${STALE_OFFSETS_ARR[@]}"
  --onset "$ONSET"
  --c-tail-keep-recent "$C_TAIL_KEEP_RECENT"
  --launcher none)

if [[ $DRY_RUN -eq 1 ]]; then
  RUN_CMD+=(--dry-run)
fi
if [[ $SKIP_SUMMARY -eq 1 ]]; then
  RUN_CMD+=(--skip-summary)
fi

printf -v RUN_CMD_STR '%q ' "${RUN_CMD[@]}"

SBATCH_DIRECTIVES=(
  "#SBATCH --job-name=${JOB_NAME}"
  "#SBATCH --nodes=1"
  "#SBATCH --ntasks=1"
  "#SBATCH --ntasks-per-node=1"
  "#SBATCH --cpus-per-task=${CPUS_PER_TASK}"
  "#SBATCH --gres=gpu:${GPUS}"
  "#SBATCH --time=${TIME_LIMIT}"
  "#SBATCH --mem=${MEMORY}"
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

printf '%s\n' '#!/usr/bin/env bash' > "$SBATCH_SCRIPT"
printf '%s\n' "${SBATCH_DIRECTIVES[@]}" >> "$SBATCH_SCRIPT"
printf '%s\n' 'set -eo pipefail' >> "$SBATCH_SCRIPT"
printf '%s\n' '# Disable unbound-variable check for the whole env-setup block: module load and' >> "$SBATCH_SCRIPT"
printf '%s\n' '# conda activate both source scripts that reference PS1 (unset in batch shells).' >> "$SBATCH_SCRIPT"
printf '%s\n' 'set +u' >> "$SBATCH_SCRIPT"

cat >> "$SBATCH_SCRIPT" <<EOF

module purge
EOF
if [[ -n "$MODULE_LOAD" ]]; then
  printf 'module load %q\n' "$MODULE_LOAD" >> "$SBATCH_SCRIPT"
fi
cat >> "$SBATCH_SCRIPT" <<EOF
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}
set -u

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PROJECT_ROOT=${PROJECT_ROOT}
export PYTHONPATH="\$PROJECT_ROOT:\${PYTHONPATH:-}"

cd "\$PROJECT_ROOT"
echo "[corruption-eval-b0] Running contradiction suite"
echo "[corruption-eval-b0] Suite root: ${SUITE_ROOT}"
${RUN_CMD_STR}
echo "[corruption-eval-b0] Done"
EOF

chmod +x "$SBATCH_SCRIPT"

echo "[submit_corruption_eval_b0_sbatch] Generated:  $SBATCH_SCRIPT"
echo "[submit_corruption_eval_b0_sbatch] Logs dir:   $SBATCH_LOG_DIR"
echo "[submit_corruption_eval_b0_sbatch] Suite root: $SUITE_ROOT"
echo "[submit_corruption_eval_b0_sbatch] Config:     $CONFIG"
echo "[submit_corruption_eval_b0_sbatch] Checkpoint: $CHECKPOINT"
echo "[submit_corruption_eval_b0_sbatch] Matrix:     ${#MODES_ARR[@]} modes x ${#STALE_OFFSETS_ARR[@]} offsets = $(( ${#MODES_ARR[@]} * ${#STALE_OFFSETS_ARR[@]} )) conditions"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[submit_corruption_eval_b0_sbatch] Dry run — not submitting."
  exit 0
fi

SUBMIT_OUTPUT="$(sbatch "$SBATCH_SCRIPT")"
echo "$SUBMIT_OUTPUT"
