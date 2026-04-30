#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run final evaluation for B1/B2 iter_89148 checkpoints.

Usage:
  bash tools/experiments/run_final_eval_b1_b2_iter89148.sh [options]

Options:
  --work-root DIR            Work root that contains experiments/ (default: /n/fs/dynamicbias/tracker/work_dirs)
  --launcher TYPE            none|pytorch|slurm|slurm-step (default: none)
  --gpus N                   Number of GPUs for launcher (default: 1)
  --partition NAME           Slurm partition (required for --launcher slurm)
  --seed N                   Random seed passed to eval (default: 0)
  --eval-options "k=v ..."   Extra eval options passed through
  --cfg-options "k=v ..."    Extra cfg overrides passed through
  --skip-cmap                Skip C-mAP post-processing
  --rerun                    Recreate eval dirs if they already exist
  --dry-run                  Print commands without executing
  -h, --help                 Show this help

Defaults are wired to:
  b1: /n/fs/dynamicbias/tracker/work_dirs/experiments/b1_b2/b1_stage3_gpu4_short_trainonly/b1/train/iter_89148.pth
  b2: /n/fs/dynamicbias/tracker/work_dirs/experiments/b1_b2/b2_stage3_gpu4_short_trainonly/b2/train/iter_89148.pth
USAGE
}

WORK_ROOT="/n/fs/dynamicbias/tracker/work_dirs"
LAUNCHER="none"
GPUS=1
PARTITION=""
SEED=0
EVAL_OPTIONS_STR=""
CFG_OPTIONS_STR=""
SKIP_CMAP=0
RERUN=0
DRY_RUN=0

B1_RUN_ID="b1_stage3_gpu4_short_trainonly"
B2_RUN_ID="b2_stage3_gpu4_short_trainonly"
B1_BASELINE="b1"
B2_BASELINE="b2"

B1_CONFIG="/n/fs/dynamicbias/tracker/work_dirs/experiments/b1_b2/b1_stage3_gpu4_short_trainonly/b1/train/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py"
B2_CONFIG="/n/fs/dynamicbias/tracker/work_dirs/experiments/b1_b2/b2_stage3_gpu4_short_trainonly/b2/train/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py"

B1_CHECKPOINT="/n/fs/dynamicbias/tracker/work_dirs/experiments/b1_b2/b1_stage3_gpu4_short_trainonly/b1/train/iter_89148.pth"
B2_CHECKPOINT="/n/fs/dynamicbias/tracker/work_dirs/experiments/b1_b2/b2_stage3_gpu4_short_trainonly/b2/train/iter_89148.pth"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --launcher) LAUNCHER="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --eval-options) EVAL_OPTIONS_STR="$2"; shift 2 ;;
    --cfg-options) CFG_OPTIONS_STR="$2"; shift 2 ;;
    --skip-cmap) SKIP_CMAP=1; shift ;;
    --rerun) RERUN=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

case "$LAUNCHER" in
  none|pytorch|slurm|slurm-step) ;;
  *) echo "Invalid --launcher: $LAUNCHER" >&2; exit 2 ;;
esac

if [[ "$LAUNCHER" == "slurm" && -z "$PARTITION" ]]; then
  echo "--partition is required when --launcher slurm" >&2
  exit 2
fi

if (( GPUS <= 0 )); then
  echo "--gpus must be a positive integer" >&2
  exit 2
fi

for f in "$B1_CONFIG" "$B2_CONFIG" "$B1_CHECKPOINT" "$B2_CHECKPOINT"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required file: $f" >&2
    exit 1
  fi
done

common_args=(
  --work-root "$WORK_ROOT"
  --launcher "$LAUNCHER"
  --gpus "$GPUS"
  --seed "$SEED"
)

if [[ -n "$PARTITION" ]]; then
  common_args+=(--partition "$PARTITION")
fi
if [[ -n "$CFG_OPTIONS_STR" ]]; then
  common_args+=(--cfg-options "$CFG_OPTIONS_STR")
fi
if [[ -n "$EVAL_OPTIONS_STR" ]]; then
  common_args+=(--eval-options "$EVAL_OPTIONS_STR")
fi
if [[ $SKIP_CMAP -eq 1 ]]; then
  common_args+=(--skip-cmap)
fi
if [[ $RERUN -eq 1 ]]; then
  common_args+=(--rerun)
fi
if [[ $DRY_RUN -eq 1 ]]; then
  common_args+=(--dry-run)
fi

echo "[1/2] Running final eval for ${B1_BASELINE} (${B1_RUN_ID})"
bash tools/experiments/run_b1_b2_deferred_eval.sh \
  --base-config "$B1_CONFIG" \
  --checkpoint "$B1_CHECKPOINT" \
  --run-id "$B1_RUN_ID" \
  --baseline "$B1_BASELINE" \
  --checkpoint-tag "iter_89148" \
  --condition-tag "clean" \
  "${common_args[@]}"

echo "[2/2] Running final eval for ${B2_BASELINE} (${B2_RUN_ID})"
bash tools/experiments/run_b1_b2_deferred_eval.sh \
  --base-config "$B2_CONFIG" \
  --checkpoint "$B2_CHECKPOINT" \
  --run-id "$B2_RUN_ID" \
  --baseline "$B2_BASELINE" \
  --checkpoint-tag "iter_89148" \
  --condition-tag "clean" \
  "${common_args[@]}"

echo "Final evaluation finished for both checkpoints."
