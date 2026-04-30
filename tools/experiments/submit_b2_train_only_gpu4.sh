#!/usr/bin/env bash
set -euo pipefail

# This grabs the project root dynamically
PROJECT_ROOT="/n/fs/dynamicbias/tracker"

# ==========================================
# HARDWARE & SLURM CONFIGURATION
# ==========================================
RUN_ID="${RUN_ID-b2_stage3_gpu4_short_trainonly}"
PARTITION="${PARTITION-}"
TIME_LIMIT="${TIME_LIMIT-08:00:00}"
MAIL_USER="${MAIL_USER-rc5898@princeton.edu}"
MAIL_TYPE="${MAIL_TYPE-BEGIN,END,FAIL}"

# HW Fixes for the Lenovo SR670 V2 (L40s)
CONSTRAINT="${CONSTRAINT-}"                 # Removed a100 constraint
TRAIN_GPUS="${TRAIN_GPUS-4}"                # Total GPUs to use
EVAL_GPUS="${EVAL_GPUS-2}"
GPUS_PER_NODE="${GPUS_PER_NODE-4}"          # Force all 4 GPUs onto ONE node
CPUS_PER_TASK="${CPUS_PER_TASK-6}"          # 4 GPUs * 6 CPUs = 24 Cores

# Learning Rate & Batch Size Scaling for 4 GPUs
CFG_COMMON="fp16.loss_scale=512.0 num_gpus=4 data.samples_per_gpu=4 optimizer.lr=5.0e-4 lr_config.warmup_iters=500 checkpoint_config.interval=1748 evaluation.interval=10488 runner.max_iters=167808"

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
  --mem 225G                                 # Half the node's RAM
  --sbatch-extra-args "${DEPEND_FLAG-}"      # NO --exclusive, allow sharing!
  --conda-env /n/fs/dynamicbias/tracker/env-maptracker
  --train-gpus "$TRAIN_GPUS"
  --eval-gpus "$EVAL_GPUS"
  --gpus-per-node "$GPUS_PER_NODE"
  --cpus-per-task "$CPUS_PER_TASK"
  --cfg-options-common "$CFG_COMMON")

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
if [[ "${SKIP_TRAIN_VALIDATION:-0}" == "1" ]]; then
  CMD+=(--skip-train-validation)
fi
if [[ "${SKIP_FINAL_EVAL:-0}" == "1" ]]; then
  CMD+=(--skip-final-eval)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  CMD+=(--dry-run)
fi

# ==========================================
# EXECUTE
# ==========================================
cd "$PROJECT_ROOT"

echo "Submitting b2 4-GPU training job with the following command:"
echo "${CMD[@]}"
echo "---------------------------------------------------"

# Actually execute the submission engine!
"${CMD[@]}"
