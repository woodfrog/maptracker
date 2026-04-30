#!/usr/bin/env bash
set -euo pipefail

# This grabs the project root dynamically
PROJECT_ROOT="/n/fs/dynamicbias/tracker"

# ==========================================
# HARDWARE & SLURM CONFIGURATION
# ==========================================
RUN_ID="${RUN_ID-b1_stage3_gpu8_trainonly}"
PARTITION="${PARTITION-}"
TIME_LIMIT="${TIME_LIMIT-04:00:00}"
MAIL_USER="${MAIL_USER-rc5898@princeton.edu}"
MAIL_TYPE="${MAIL_TYPE-BEGIN,END,FAIL}"

# HW Fixes for the Lenovo SR670 V2 (8x L40 GPUs, 52 CPU Cores)
CONSTRAINT="${CONSTRAINT-}"                 
TRAIN_GPUS="${TRAIN_GPUS-8}"                
EVAL_GPUS="${EVAL_GPUS-2}"                  
GPUS_PER_NODE="${GPUS_PER_NODE-8}"          
CPUS_PER_TASK="${CPUS_PER_TASK-6}"          

# ==========================================
# BUILD THE COMMAND ARRAY
# ==========================================
CMD=(bash tools/experiments/submit_b1_b2_sbatch.sh
  --mode b1_only
  --run-id "$RUN_ID"
  --base-config plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py
  --base-checkpoint "$PROJECT_ROOT/work_dirs/pretrained_ckpts/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup/latest.pth"
  --work-root "$PROJECT_ROOT/work_dirs"
  --time "$TIME_LIMIT"
  --mail-user "$MAIL_USER"
  --mail-type "$MAIL_TYPE"
  --mem 450G                                 
  --sbatch-extra-args "--exclusive"          # CHANGED: Nodelist restriction removed
  --conda-env /n/fs/dynamicbias/tracker/maptracker
  --train-gpus "$TRAIN_GPUS"
  --eval-gpus "$EVAL_GPUS"
  --gpus-per-node "$GPUS_PER_NODE"
  --cpus-per-task "$CPUS_PER_TASK"
  --b1-cfg-options "model.mvp_temporal_gate_cfg.corruption_trained_no_gate_baseline=True")

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

echo "Submitting b1 training job with the following command:"
echo "${CMD[@]}"
echo "---------------------------------------------------"

# Actually run the command array!
"${CMD[@]}"
