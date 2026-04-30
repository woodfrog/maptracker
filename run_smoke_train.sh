#!/usr/bin/env bash
#SBATCH --job-name=maptracker_smoke_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --constraint=nomig&gpu40
#SBATCH --qos=gpu-test
#SBATCH --output=/scratch/gpfs/FHEIDE/yk3904/maptracker_v1/work_dirs/smoke_train_stage3_1gpu_5iter/slurm-%j.out
#SBATCH --error=/scratch/gpfs/FHEIDE/yk3904/maptracker_v1/work_dirs/smoke_train_stage3_1gpu_5iter/slurm-%j.err

set -e

# ── environment ──────────────────────────────────────────────────────────────
module purge
module load anaconda3/2023.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate maptracker

export PROJECT_ROOT=/scratch/gpfs/FHEIDE/yk3904/maptracker_v1
export STAGE2_CKPT=$PROJECT_ROOT/work_dirs/pretrained_ckpts/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup/latest.pth
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

mkdir -p "$PROJECT_ROOT/work_dirs/smoke_train_stage3_1gpu_5iter"
cd "$PROJECT_ROOT"

# ── launch ───────────────────────────────────────────────────────────────────
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=29500 \
  tools/train.py \
    plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py \
    --launcher pytorch \
    --work-dir "$PROJECT_ROOT/work_dirs/smoke_train_stage3_1gpu_5iter" \
    --no-validate \
    --cfg-options \
      load_from="$STAGE2_CKPT" \
      data.samples_per_gpu=1 \
      data.workers_per_gpu=2 \
      optimizer.lr=1e-4 \
      log_config.interval=1 \
      checkpoint_config.interval=5 \
      runner.max_iters=5
