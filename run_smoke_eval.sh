#!/usr/bin/env bash
#SBATCH --job-name=maptracker_smoke_eval
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --constraint=nomig&gpu40
#SBATCH --qos=gpu-test
#SBATCH --output=/scratch/gpfs/FHEIDE/yk3904/maptracker_v1/work_dirs/smoke_eval_stage3_2gpu_600/slurm-%j.out
#SBATCH --error=/scratch/gpfs/FHEIDE/yk3904/maptracker_v1/work_dirs/smoke_eval_stage3_2gpu_600/slurm-%j.err
# Smoke-test evaluation: ~600 samples, gpu-test QOS, 2 GPUs
# Fixes applied vs previous attempts:
#   1. --wait=0                         prevents srun aborting rank 0 after rank 1 exits
#   2. data.test.eval_config.interval=10  (was wrongly eval_config.interval=10 before)
#   3. gts cache now includes sample count in filename (no stale-cache poisoning)
#   4. evaluate() iterates result tokens, not all 6019 GT tokens

set -e

# ── environment ──────────────────────────────────────────────────────────────
module purge
module load anaconda3/2023.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate maptracker

export PROJECT_ROOT=/scratch/gpfs/FHEIDE/yk3904/maptracker_v1
export STAGE3_CKPT=$PROJECT_ROOT/work_dirs/pretrained_ckpts/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune/latest.pth
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# ── clean old gts caches (old naming scheme, and any leftover partial caches) ─
rm -f "$PROJECT_ROOT"/tmp_gts_nusc_60x30.pkl
rm -f "$PROJECT_ROOT"/tmp_gts_nusc_60x30_newsplit.pkl

cd "$PROJECT_ROOT"

# ── launch ───────────────────────────────────────────────────────────────────
srun --wait=0 \
  python -u tools/test.py \
    plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py \
    "$STAGE3_CKPT" \
    --launcher slurm \
    --eval \
    --work-dir "$PROJECT_ROOT/work_dirs/smoke_eval_stage3_2gpu_600" \
    --cfg-options \
      data.test.interval=10 \
      data.test.eval_config.interval=10
