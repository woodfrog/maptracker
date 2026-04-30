#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

RUN_ID="${RUN_ID-b2_stage3_gpu4_short_trainonly}"
BASE_CONFIG="${BASE_CONFIG-plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py}"
WORK_ROOT="${WORK_ROOT-$PROJECT_ROOT/work_dirs}"
TRAIN_DIR="${TRAIN_DIR-$WORK_ROOT/experiments/b1_b2/$RUN_ID/b2/train}"

# Checkpoint discovery controls
CHECKPOINT_GLOB="${CHECKPOINT_GLOB-iter_*.pth}"
INCLUDE_LATEST="${INCLUDE_LATEST-1}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS-0}"  # 0 means all

# Deferred eval job controls
TIME_LIMIT="${TIME_LIMIT-02:00:00}"
MAIL_USER="${MAIL_USER-rc5898@princeton.edu}"
MAIL_TYPE="${MAIL_TYPE-END,FAIL}"
PARTITION="${PARTITION-}"
QOS="${QOS-gpu-short}"
ACCOUNT="${ACCOUNT-}"
CONSTRAINT="${CONSTRAINT-}"
MEM="${MEM-}"
EVAL_GPUS="${EVAL_GPUS-2}"
GPUS_PER_NODE="${GPUS_PER_NODE-$EVAL_GPUS}"
CPUS_PER_TASK="${CPUS_PER_TASK-6}"
CONDITION_TAG="${CONDITION_TAG-clean}"

SKIP_CMAP="${SKIP_CMAP-0}"
CONS_FRAMES="${CONS_FRAMES-5}"
RERUN="${RERUN-0}"
DRY_RUN="${DRY_RUN-0}"

CFG_OPTIONS="${CFG_OPTIONS-}"
EVAL_OPTIONS="${EVAL_OPTIONS-}"
SRUN_ARGS="${SRUN_ARGS-}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS-}"
DEPENDENCY="${DEPENDENCY-}"

if [[ ! -d "$TRAIN_DIR" ]]; then
  echo "[submit_b2_deferred_eval_all_gpu4] Train dir not found: $TRAIN_DIR" >&2
  exit 2
fi

mapfile -t CHECKPOINTS < <(find "$TRAIN_DIR" -maxdepth 1 -type f -name "$CHECKPOINT_GLOB" | sort -V)

if [[ "$INCLUDE_LATEST" == "1" && -f "$TRAIN_DIR/latest.pth" ]]; then
  CHECKPOINTS+=("$TRAIN_DIR/latest.pth")
fi

if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "[submit_b2_deferred_eval_all_gpu4] No checkpoints found under $TRAIN_DIR matching '$CHECKPOINT_GLOB'." >&2
  exit 2
fi

if [[ "$MAX_CHECKPOINTS" =~ ^[0-9]+$ ]] && (( MAX_CHECKPOINTS > 0 )) && (( ${#CHECKPOINTS[@]} > MAX_CHECKPOINTS )); then
  CHECKPOINTS=("${CHECKPOINTS[@]: -$MAX_CHECKPOINTS}")
fi

echo "[submit_b2_deferred_eval_all_gpu4] Found ${#CHECKPOINTS[@]} checkpoint(s)."
for ckpt in "${CHECKPOINTS[@]}"; do
  checkpoint_tag="$(basename "$ckpt" .pth)"

  CMD=(bash tools/experiments/submit_b1_b2_deferred_eval.sh
    --base-config "$BASE_CONFIG"
    --checkpoint "$ckpt"
    --work-root "$WORK_ROOT"
    --run-id "$RUN_ID"
    --baseline b2
    --time "$TIME_LIMIT"
    --mail-user "$MAIL_USER"
    --mail-type "$MAIL_TYPE"
    --eval-gpus "$EVAL_GPUS"
    --gpus-per-node "$GPUS_PER_NODE"
    --cpus-per-task "$CPUS_PER_TASK"
    --condition-tag "$CONDITION_TAG"
    --checkpoint-tag "$checkpoint_tag"
    --qos "$QOS")

  if [[ -n "$PARTITION" ]]; then
    CMD+=(--partition "$PARTITION")
  fi
  if [[ -n "$ACCOUNT" ]]; then
    CMD+=(--account "$ACCOUNT")
  fi
  if [[ -n "$CONSTRAINT" ]]; then
    CMD+=(--constraint "$CONSTRAINT")
  fi
  if [[ -n "$MEM" ]]; then
    CMD+=(--mem "$MEM")
  fi
  if [[ -n "$DEPENDENCY" ]]; then
    CMD+=(--dependency "$DEPENDENCY")
  fi
  if [[ -n "$CFG_OPTIONS" ]]; then
    CMD+=(--cfg-options "$CFG_OPTIONS")
  fi
  if [[ -n "$EVAL_OPTIONS" ]]; then
    CMD+=(--eval-options "$EVAL_OPTIONS")
  fi
  if [[ -n "$SRUN_ARGS" ]]; then
    CMD+=(--srun-args "$SRUN_ARGS")
  fi
  if [[ -n "$SBATCH_EXTRA_ARGS" ]]; then
    CMD+=(--sbatch-extra-args "$SBATCH_EXTRA_ARGS")
  fi
  if [[ "$SKIP_CMAP" == "1" ]]; then
    CMD+=(--skip-cmap)
  fi
  if [[ "$CONS_FRAMES" != "5" ]]; then
    CMD+=(--cons-frames "$CONS_FRAMES")
  fi
  if [[ "$RERUN" == "1" ]]; then
    CMD+=(--rerun)
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    CMD+=(--dry-run)
  fi

  echo "[submit_b2_deferred_eval_all_gpu4] Submitting eval for $checkpoint_tag"
  "${CMD[@]}"
done

echo "[submit_b2_deferred_eval_all_gpu4] Done."
