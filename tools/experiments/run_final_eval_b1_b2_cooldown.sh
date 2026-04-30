#!/usr/bin/env bash
# Run clean deferred evaluation for B1/B2 cooldown5ep checkpoints.
#
# This is the post-training eval script for the 5-epoch LR-cooldown runs.
# It is structurally identical to run_final_eval_b1_b2_iter89148.sh but
# points to the cooldown work-dirs and uses latest.pth as the checkpoint.
#
# Usage:
#   bash tools/experiments/run_final_eval_b1_b2_cooldown.sh [options]
#
# Options:
#   --work-root DIR            Work root (default: /scratch/gpfs/FHEIDE/yk3904/maptracker_v1/work_dirs)
#   --launcher TYPE            none|pytorch|slurm|slurm-step (default: slurm-step)
#   --gpus N                   GPUs for eval launcher (default: 4)
#   --partition NAME           Slurm partition (required for --launcher slurm)
#   --seed N                   Random seed (default: 0)
#   --eval-options "k=v ..."   Extra eval options passed through
#   --cfg-options "k=v ..."    Extra cfg overrides passed through
#   --skip-cmap                Skip C-mAP post-processing
#   --rerun                    Recreate eval dirs if they already exist
#   --dry-run                  Print commands without executing
#   -h, --help                 Show this help
set -euo pipefail

usage() {
  cat <<'USAGE'
Run clean deferred evaluation for B1/B2 cooldown5ep checkpoints (latest.pth).

Usage:
  bash tools/experiments/run_final_eval_b1_b2_cooldown.sh [options]

Options:
  --work-root DIR
  --launcher TYPE            (default: slurm-step)
  --gpus N                   (default: 4)
  --partition NAME
  --seed N                   (default: 0)
  --eval-options "k=v ..."
  --cfg-options "k=v ..."
  --skip-cmap
  --rerun
  --dry-run
  -h, --help
USAGE
}

PROJECT_ROOT="/scratch/gpfs/FHEIDE/yk3904/maptracker_v1"
WORK_ROOT="$PROJECT_ROOT/work_dirs"
LAUNCHER="slurm-step"
GPUS=4
PARTITION=""
SEED=0
EVAL_OPTIONS_STR=""
CFG_OPTIONS_STR=""
SKIP_CMAP=0
RERUN=0
DRY_RUN=0

B1_RUN_ID="b1_stage3_cooldown5ep"
B2_RUN_ID="b2_stage3_cooldown5ep"
B1_BASELINE="b1"
B2_BASELINE="b2"

# Config copies are written by run_b1_b2.sh into the train dir during training.
B1_CONFIG="$WORK_ROOT/experiments/b1_b2/${B1_RUN_ID}/${B1_BASELINE}/train/maptracker_nusc_oldsplit_5frame_span10_stage3_cooldown5ep.py"
B2_CONFIG="$WORK_ROOT/experiments/b1_b2/${B2_RUN_ID}/${B2_BASELINE}/train/maptracker_nusc_oldsplit_5frame_span10_stage3_cooldown5ep.py"

B1_CHECKPOINT="$WORK_ROOT/experiments/b1_b2/${B1_RUN_ID}/${B1_BASELINE}/train/latest.pth"
B2_CHECKPOINT="$WORK_ROOT/experiments/b1_b2/${B2_RUN_ID}/${B2_BASELINE}/train/latest.pth"

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

# Re-derive paths against the (potentially overridden) WORK_ROOT.
B1_CONFIG="$WORK_ROOT/experiments/b1_b2/${B1_RUN_ID}/${B1_BASELINE}/train/maptracker_nusc_oldsplit_5frame_span10_stage3_cooldown5ep.py"
B2_CONFIG="$WORK_ROOT/experiments/b1_b2/${B2_RUN_ID}/${B2_BASELINE}/train/maptracker_nusc_oldsplit_5frame_span10_stage3_cooldown5ep.py"
B1_CHECKPOINT="$WORK_ROOT/experiments/b1_b2/${B1_RUN_ID}/${B1_BASELINE}/train/latest.pth"
B2_CHECKPOINT="$WORK_ROOT/experiments/b1_b2/${B2_RUN_ID}/${B2_BASELINE}/train/latest.pth"

for f in "$B1_CONFIG" "$B2_CONFIG" "$B1_CHECKPOINT" "$B2_CHECKPOINT"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required file: $f" >&2
    echo "(Has training finished?  Expected files after cooldown training completes.)" >&2
    exit 1
  fi
done

common_args=(
  --work-root "$WORK_ROOT"
  --launcher "$LAUNCHER"
  --gpus "$GPUS"
  --seed "$SEED"
)
if [[ -n "$PARTITION" ]]; then common_args+=(--partition "$PARTITION"); fi
if [[ -n "$CFG_OPTIONS_STR" ]]; then common_args+=(--cfg-options "$CFG_OPTIONS_STR"); fi
if [[ -n "$EVAL_OPTIONS_STR" ]]; then common_args+=(--eval-options "$EVAL_OPTIONS_STR"); fi
if [[ $SKIP_CMAP -eq 1 ]]; then common_args+=(--skip-cmap); fi
if [[ $RERUN -eq 1 ]]; then common_args+=(--rerun); fi
if [[ $DRY_RUN -eq 1 ]]; then common_args+=(--dry-run); fi

echo "[1/2] Running cooldown eval for ${B1_BASELINE} (${B1_RUN_ID})"
bash tools/experiments/run_b1_b2_deferred_eval.sh \
  --base-config "$B1_CONFIG" \
  --checkpoint "$B1_CHECKPOINT" \
  --run-id "$B1_RUN_ID" \
  --baseline "$B1_BASELINE" \
  --checkpoint-tag "cooldown5ep_latest" \
  --condition-tag "clean" \
  "${common_args[@]}"

echo "[2/2] Running cooldown eval for ${B2_BASELINE} (${B2_RUN_ID})"
bash tools/experiments/run_b1_b2_deferred_eval.sh \
  --base-config "$B2_CONFIG" \
  --checkpoint "$B2_CHECKPOINT" \
  --run-id "$B2_RUN_ID" \
  --baseline "$B2_BASELINE" \
  --checkpoint-tag "cooldown5ep_latest" \
  --condition-tag "clean" \
  "${common_args[@]}"

echo "Cooldown evaluation finished for both checkpoints."
