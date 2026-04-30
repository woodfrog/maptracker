#!/usr/bin/env bash
# Submit B0 corruption-suite eval as N parallel single-condition GPU jobs for
# maximum backfill scheduling priority.  Each condition (mode x stale-offset)
# gets its own short wall-time job.  A CPU aggregation job is submitted as an
# afterok dependency and writes contradiction_suite_summary.json when all done.
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit B0 corruption-suite as parallel single-condition GPU jobs.

Each (mode, stale-offset) condition becomes an independent Slurm job with a short
wall time (default 01:30:00).  Short jobs are much easier for Slurm backfill to
schedule than one long job — all 6 conditions can start on different nodes within
hours rather than waiting overnight for a 6-hour window to open.

A lightweight CPU aggregation job is submitted automatically with an afterok
dependency on all 6 condition jobs.  It reads each condition's
contradiction_metrics.json and writes contradiction_suite_summary.json.

Usage:
  bash tools/experiments/submit_corruption_eval_b0_parallel.sh \
    --mail-user EMAIL [options]

Required:
  --mail-user EMAIL

Optional:
  --checkpoint PATH           (default: work_dirs/pretrained_ckpts/b0_nusc_oldsplit/latest.pth)
  --config PATH               (default: ...b0_eval.py)
  --work-root DIR             (default: <repo-root>/work_dirs)
  --modes "c_full c_tail"     (default: c_full c_tail)
  --stale-offsets "1 2 3"     (default: 1 2 3)
  --onset N                   (default: 0)
  --c-tail-keep-recent N      (default: 1)
  --suite-tag TAG             (default: trainmatched)
  --per-condition-time LIMIT  Wall time per condition job (default: 01:30:00).
                              Shorter = higher backfill priority.
  --gpus N                    GPUs per condition job (default: 1)
  --cpus-per-task N           (default: 6)
  --mem VALUE                 (default: 96G)
  --qos NAME                  GPU QOS for condition jobs (default: gpu-short)
  --constraint EXPR           (default: a100&nomig)
  --account NAME
  --mail-type VALUE           (default: END,FAIL)
  --module-load MOD           (default: anaconda3/2023.9)
  --conda-env NAME            (default: maptracker)
  --dry-run                   Generate sbatch files but do not submit.
  -h, --help

Notes:
  - All conditions share the same suite root directory; each writes to its own
    unique subdirectory, so there are no write conflicts between parallel jobs.
  - The aggregation job uses QOS=short (CPU, no GPU) and runs in minutes.
  - Cancel any existing full-suite job before running this script.
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
PER_CONDITION_TIME="01:30:00"
GPUS=1
CPUS_PER_TASK=6
MEMORY="96G"
QOS="gpu-short"
CONSTRAINT="a100&nomig"
ACCOUNT=""
MAIL_TYPE="END,FAIL"
MODULE_LOAD="anaconda3/2023.9"
CONDA_ENV="maptracker"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mail-user)          MAIL_USER="$2";         shift 2 ;;
    --checkpoint)         CHECKPOINT="$2";         shift 2 ;;
    --config)             CONFIG="$2";             shift 2 ;;
    --work-root)          WORK_ROOT="$2";          shift 2 ;;
    --modes)              MODES_STR="$2";          shift 2 ;;
    --stale-offsets)      STALE_OFFSETS_STR="$2";  shift 2 ;;
    --onset)              ONSET="$2";              shift 2 ;;
    --c-tail-keep-recent) C_TAIL_KEEP_RECENT="$2"; shift 2 ;;
    --suite-tag)          SUITE_TAG="$2";          shift 2 ;;
    --per-condition-time) PER_CONDITION_TIME="$2"; shift 2 ;;
    --gpus)               GPUS="$2";               shift 2 ;;
    --cpus-per-task)      CPUS_PER_TASK="$2";      shift 2 ;;
    --mem)                MEMORY="$2";             shift 2 ;;
    --qos)                QOS="$2";                shift 2 ;;
    --constraint)         CONSTRAINT="$2";         shift 2 ;;
    --account)            ACCOUNT="$2";            shift 2 ;;
    --mail-type)          MAIL_TYPE="$2";          shift 2 ;;
    --module-load)        MODULE_LOAD="$2";        shift 2 ;;
    --conda-env)          CONDA_ENV="$2";          shift 2 ;;
    --dry-run)            DRY_RUN=1;               shift   ;;
    -h|--help)            usage; exit 0 ;;
    *) echo "[submit_corruption_eval_b0_parallel] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MAIL_USER" ]]; then
  echo "[submit_corruption_eval_b0_parallel] --mail-user is required." >&2; usage; exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -z "$WORK_ROOT" ]];    then WORK_ROOT="${PROJECT_ROOT}/work_dirs"; fi
if [[ -z "$CONFIG" ]]; then
  CONFIG="${PROJECT_ROOT}/plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_b0_eval.py"
fi
if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="${WORK_ROOT}/pretrained_ckpts/b0_nusc_oldsplit/latest.pth"
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "[submit_corruption_eval_b0_parallel] Config not found: $CONFIG" >&2; exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[submit_corruption_eval_b0_parallel] Checkpoint not found: $CHECKPOINT" >&2; exit 1
fi

# Compute SUITE_ROOT — must match the logic in submit_corruption_eval_b0_sbatch.sh
CKPT_TAG="$(basename "$CHECKPOINT")"
CKPT_TAG="${CKPT_TAG%.pth}"
SUITE_TAG_SUFFIX=""
[[ -n "$SUITE_TAG" ]] && SUITE_TAG_SUFFIX="_${SUITE_TAG}"
SUITE_ROOT="${WORK_ROOT%/}/experiments/b0/corruption_suite/${CKPT_TAG}_onset${ONSET}${SUITE_TAG_SUFFIX}"

read -r -a MODES_ARR        <<< "$MODES_STR"
read -r -a STALE_OFFSETS_ARR <<< "$STALE_OFFSETS_STR"

N_CONDITIONS=$(( ${#MODES_ARR[@]} * ${#STALE_OFFSETS_ARR[@]} ))
echo "[submit_corruption_eval_b0_parallel] Suite root:          $SUITE_ROOT"
echo "[submit_corruption_eval_b0_parallel] Conditions:          $N_CONDITIONS  (${#MODES_ARR[@]} modes × ${#STALE_OFFSETS_ARR[@]} offsets)"
echo "[submit_corruption_eval_b0_parallel] Per-condition time:  $PER_CONDITION_TIME  (short wall time = high backfill priority)"
echo "[submit_corruption_eval_b0_parallel] QOS / constraint:    ${QOS} / ${CONSTRAINT}"
echo ""

JOB_IDS=()

for mode in "${MODES_ARR[@]}"; do
  for offset in "${STALE_OFFSETS_ARR[@]}"; do
    SUBMIT_ARGS=(
      --mail-user          "$MAIL_USER"
      --config             "$CONFIG"
      --checkpoint         "$CHECKPOINT"
      --work-root          "$WORK_ROOT"
      --modes              "$mode"
      --stale-offsets      "$offset"
      --onset              "$ONSET"
      --c-tail-keep-recent "$C_TAIL_KEEP_RECENT"
      --suite-tag          "$SUITE_TAG"
      --allow-overwrite
      --skip-summary
      --time               "$PER_CONDITION_TIME"
      --gpus               "$GPUS"
      --cpus-per-task      "$CPUS_PER_TASK"
      --mem                "$MEMORY"
      --qos                "$QOS"
      --constraint         "$CONSTRAINT"
      --mail-type          "$MAIL_TYPE"
      --module-load        "$MODULE_LOAD"
      --conda-env          "$CONDA_ENV"
    )
    [[ -n "$ACCOUNT" ]] && SUBMIT_ARGS+=(--account "$ACCOUNT")
    [[ $DRY_RUN -eq 1 ]] && SUBMIT_ARGS+=(--dry-run)

    SUBMIT_OUT="$(bash "$SCRIPT_DIR/submit_corruption_eval_b0_sbatch.sh" "${SUBMIT_ARGS[@]}")"
    echo "$SUBMIT_OUT"

    if [[ $DRY_RUN -ne 1 ]]; then
      JID="$(echo "$SUBMIT_OUT" | grep "Submitted batch job" | awk '{print $NF}')"
      if [[ -n "$JID" ]]; then
        JOB_IDS+=("$JID")
        echo "[submit_corruption_eval_b0_parallel] -> ${mode} x offset${offset}: job $JID"
      fi
    fi
    echo ""
  done
done

# Build afterok dependency string from all condition job IDs
DEPENDENCY=""
if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
  DEPENDENCY="afterok"
  for jid in "${JOB_IDS[@]}"; do
    DEPENDENCY="${DEPENDENCY}:${jid}"
  done
fi

echo "[submit_corruption_eval_b0_parallel] Submitting aggregation job (dependency: ${DEPENDENCY:-none})..."

AGGR_ARGS=(
  --mail-user     "$MAIL_USER"
  --suite-root    "$SUITE_ROOT"
  --config        "$CONFIG"
  --checkpoint    "$CHECKPOINT"
  --onset         "$ONSET"
  --modes         "$MODES_STR"
  --stale-offsets "$STALE_OFFSETS_STR"
  --module-load   "$MODULE_LOAD"
  --conda-env     "$CONDA_ENV"
  --mail-type     "$MAIL_TYPE"
)
[[ -n "$DEPENDENCY" ]] && AGGR_ARGS+=(--dependency "$DEPENDENCY")
[[ -n "$ACCOUNT"    ]] && AGGR_ARGS+=(--account    "$ACCOUNT")
[[ $DRY_RUN -eq 1  ]] && AGGR_ARGS+=(--dry-run)

bash "$SCRIPT_DIR/submit_contradiction_metrics_b0_sbatch.sh" "${AGGR_ARGS[@]}"
