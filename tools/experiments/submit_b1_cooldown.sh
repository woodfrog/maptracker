#!/usr/bin/env bash
# Submit the 5-epoch LR-cooldown training job for B1.
#
# Writes a self-contained sbatch file under
#   work_dirs/sbatch/b1_b2/b1_stage3_cooldown5ep/
# then submits it.
#
# Environment toggles (all optional):
#   USE_4GPU=1    Use the 4-GPU / gpu-medium fallback (14 h time limit).
#                 Default: 0  →  8 GPU / gpu-short (8 h, ~5.2 h actual).
#   RESUME=1      Add --resume to run_b1_b2.sh so the script tolerates an
#                 existing non-empty work_dir (needed after a job interruption;
#                 auto_resume in the config will pick up the latest checkpoint).
#   MAIL_USER=..  Override notification email.
#   DRY_RUN=1     Write the sbatch file but do not submit.
#
# LR resume rationale
# -------------------
# Backbone logged LR at iter 89,148 = 2.270e-5  →  base LR = 2.270e-4
# (backbone lr_mult = 0.1, so base = backbone / 0.1).
# A 50-iter linear warmup from 0.5× (1.135e-4) eases fresh AdamW moments
# before the cosine decay resumes; min_lr_ratio=3e-3 is unchanged from the
# original run, giving a final LR of 6.81e-7.
set -euo pipefail

# ── toggles ──────────────────────────────────────────────────────────────────
USE_4GPU="${USE_4GPU:-0}"
RESUME="${RESUME:-0}"
MAIL_USER="${MAIL_USER:-yk3904@princeton.edu}"
MAIL_TYPE="${MAIL_TYPE:-END,FAIL}"
DRY_RUN="${DRY_RUN:-0}"

# ── paths (all absolute, no /n/fs/dynamicbias references) ────────────────────
PROJECT_ROOT="/scratch/gpfs/FHEIDE/yk3904/maptracker_v1"
WORK_ROOT="$PROJECT_ROOT/work_dirs"
CKPT="$PROJECT_ROOT/TrainningPaths/b1_iter_89148.pth"
CONFIG="plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_cooldown5ep.py"
RUN_ID="b1_stage3_cooldown5ep"

# ── hardware selection ────────────────────────────────────────────────────────
# Option A (default): 8 A100 GPUs, gpu-short QOS, 8 h limit, ~5.2 h actual.
#   2 nodes × 4 GPUs, samples_per_gpu=2  →  total batch = 16 (unchanged).
# Option B (USE_4GPU=1): 4 A100 GPUs, gpu-medium QOS, 14 h limit, ~10.7 h actual.
#   2 nodes × 2 GPUs, samples_per_gpu=4  →  total batch = 16 (unchanged).
if [[ "$USE_4GPU" == "1" ]]; then
  TRAIN_GPUS=4; GPUS_PER_NODE=2; CPUS_PER_TASK=8
  QOS="gpu-medium"; TIME_LIMIT="14:00:00"
  CFG_COMMON="fp16.loss_scale=512.0 num_gpus=4 data.samples_per_gpu=4 optimizer.lr=2.270e-4 lr_config.warmup_iters=50 lr_config.warmup_ratio=0.5"
else
  TRAIN_GPUS=8; GPUS_PER_NODE=4; CPUS_PER_TASK=6
  QOS="gpu-short"; TIME_LIMIT="08:00:00"
  CFG_COMMON="fp16.loss_scale=512.0 num_gpus=8 data.samples_per_gpu=2 optimizer.lr=2.270e-4 lr_config.warmup_iters=50 lr_config.warmup_ratio=0.5"
fi

NODES=$(( TRAIN_GPUS / GPUS_PER_NODE ))

# ── build resume flag ─────────────────────────────────────────────────────────
RESUME_FLAG=""
if [[ "$RESUME" == "1" ]]; then
  RESUME_FLAG="--resume"
fi

# ── generate sbatch file ──────────────────────────────────────────────────────
SBATCH_DIR="$WORK_ROOT/sbatch/b1_b2/$RUN_ID"
mkdir -p "$SBATCH_DIR/logs"
SBATCH_SCRIPT="$SBATCH_DIR/b1_only.sbatch"

cat > "$SBATCH_SCRIPT" <<SBATCH_EOF
#!/usr/bin/env bash
#SBATCH --job-name=mt_b1_cooldown5ep
#SBATCH --nodes=${NODES}
#SBATCH --ntasks=${TRAIN_GPUS}
#SBATCH --ntasks-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gres=gpu:${GPUS_PER_NODE}
#SBATCH --gpus-per-task=1
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${SBATCH_DIR}/logs/%x-%j.out
#SBATCH --error=${SBATCH_DIR}/logs/%x-%j.err
#SBATCH --mail-type=${MAIL_TYPE}
#SBATCH --mail-user=${MAIL_USER}
#SBATCH --qos=${QOS}
#SBATCH --constraint=a100&nomig
set -euo pipefail

module purge
module load anaconda3/2023.9
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate maptracker

export PROJECT_ROOT=${PROJECT_ROOT}
export PYTHONPATH="\$PROJECT_ROOT:\${PYTHONPATH:-}"
export SRUN_CPUS_PER_TASK=${CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "\$PROJECT_ROOT"
bash tools/experiments/run_b1_b2.sh \
  --mode b1_only \
  --base-config ${CONFIG} \
  --base-checkpoint ${CKPT} \
  --work-root ${WORK_ROOT} \
  --run-id ${RUN_ID} \
  --seed 0 \
  --launcher slurm-step \
  --train-gpus ${TRAIN_GPUS} \
  --eval-gpus 2 \
  --gpus-per-node ${GPUS_PER_NODE} \
  --cpus-per-task ${CPUS_PER_TASK} \
  --available-gpus ${TRAIN_GPUS} \
  --skip-train-validation \
  --skip-final-eval \
  ${RESUME_FLAG} \
  --cfg-options-common "${CFG_COMMON}" \
  --b1-cfg-options "model.mvp_temporal_gate_cfg.corruption_trained_no_gate_baseline=True"
SBATCH_EOF

echo "Generated: $SBATCH_SCRIPT"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY RUN] Would submit: sbatch $SBATCH_SCRIPT"
else
  sbatch "$SBATCH_SCRIPT"
fi
