#!/usr/bin/env bash
# Submit the 5-epoch LR-cooldown training job for B2.
#
# Writes a self-contained sbatch file under
#   work_dirs/sbatch/b1_b2/b2_stage3_cooldown5ep/
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
# B2 differs from B1 in:
#   - checkpoint: b2_iter_89148.pth
#   - run-id: b2_stage3_cooldown5ep
#   - NO --b2-cfg-options gate flag (B2 has the temporal gate active)
set -euo pipefail

# ── toggles ──────────────────────────────────────────────────────────────────
USE_4GPU="${USE_4GPU:-0}"
RESUME="${RESUME:-0}"
MAIL_USER="${MAIL_USER:-yk3904@princeton.edu}"
MAIL_TYPE="${MAIL_TYPE:-END,FAIL}"
DRY_RUN="${DRY_RUN:-0}"

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT="/scratch/gpfs/FHEIDE/yk3904/maptracker_v1"
WORK_ROOT="$PROJECT_ROOT/work_dirs"
CKPT="$PROJECT_ROOT/TrainningPaths/b2_iter_89148.pth"
CONFIG="plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_cooldown5ep.py"
RUN_ID="b2_stage3_cooldown5ep"

# ── hardware selection ────────────────────────────────────────────────────────
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
SBATCH_SCRIPT="$SBATCH_DIR/b2_only.sbatch"

cat > "$SBATCH_SCRIPT" <<SBATCH_EOF
#!/usr/bin/env bash
#SBATCH --job-name=mt_b2_cooldown5ep
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
  --mode b2_only \
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
  --cfg-options-common "${CFG_COMMON}"
SBATCH_EOF

echo "Generated: $SBATCH_SCRIPT"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY RUN] Would submit: sbatch $SBATCH_SCRIPT"
else
  sbatch "$SBATCH_SCRIPT"
fi
