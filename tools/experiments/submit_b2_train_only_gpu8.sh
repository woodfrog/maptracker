#!/usr/bin/env bash
set -euo pipefail

# This grabs the project root dynamically
PROJECT_ROOT="/n/fs/dynamicbias/tracker"

# ==========================================
# HARDWARE & SLURM CONFIGURATION
# ==========================================
RUN_ID="${RUN_ID-b2_stage3_gpu8_trainonly}"
PARTITION="${PARTITION-}"
TIME_LIMIT="${TIME_LIMIT-04:00:00}"
MAIL_USER="${MAIL_USER-rc5898@princeton.edu}"
MAIL_TYPE="${MAIL_TYPE-BEGIN,END,FAIL}"

# HW Fixes for the Lenovo SR670 V2 (8x L40 GPUs, 52 CPU Cores)
CONSTRAINT="${CONSTRAINT-}"                 # Removed gpu80 constraint
TRAIN_GPUS="${TRAIN_GPUS-8}"
EVAL_GPUS="${EVAL_GPUS-2}"
GPUS_PER_NODE="${GPUS_PER_NODE-8}"          # Lock to 1 node
CPUS_PER_TASK="${CPUS_PER_TASK-6}"          # 8 GPUs * 6 CPUs = 48 Cores (Fits 52)

# ==========================================
# BUILD THE COMMAND ARRAY
# ==========================================
CMD=(bash tools/experiments/submit_b1_b2_sbatch.sh
  --mode b2_only
  --run-id "$RUN_ID"
  --base-config plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py
  --base-checkpoint "$PROJECT_ROOT/work_dirs/pretrained_ckpts/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup/latest.pth"
  --work-root "$PROJECT_ROOT/work_dirs"
  --time "$TIME_LIMIT"
  --mail-user "$MAIL_USER"
  --mail-type "$MAIL_TYPE"
  --mem 450G
  --sbatch-extra-args "--exclusive"
  --conda-env /n/fs/dynamicbias/tracker/env-maptracker
  --train-gpus "$TRAIN_GPUS"
  --eval-gpus "$EVAL_GPUS"
  --gpus-per-node "$GPUS_PER_NODE"
  --cpus-per-task "$CPUS_PER_TASK")

# Append optional flags if they are set
if [[ -n "$PARTITION" ]]; then
  CMD+=(--partition "$PARTITION")
fi
if [[ -n "$CONSTRAINT" ]]; then
  CMD+=(--constraint "$CONSTRAINT")
fi

if [[ "${RESUME:-0}" == "1" ]]; then
  CMD+=(--resume)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  CMD+=(--dry-run)
fi

# ==========================================
# EXECUTE
# ==========================================
cd "$PROJECT_ROOT"

echo "Submitting b2 training job with the following command:"
echo "${CMD[@]}"
echo "---------------------------------------------------"

# Actually execute the submission engine!
"${CMD[@]}"
