#!/usr/bin/env bash
# =============================================================================
# setup_maptracker.sh
# One-shot environment setup for MapTracker on a Slurm/HPC cluster.
#
# Usage:
#   bash setup_maptracker.sh
#
# What it does:
#   1. Creates (or reuses) a conda environment with all required packages
#   2. Clones + installs mmdetection3d @ v1.0.0rc6
#   3. Creates the required work_dirs directory structure
#   4. Symlinks your pre-transferred dataset(s) into datasets/
#   5. Downloads pretrained checkpoints from HuggingFace
#   6. Runs a quick import sanity check
#   7. Prints next-step commands (NOT executed automatically)
#
# Before running, fill in the ── USER CONFIG ── block below.
# =============================================================================

set -euo pipefail

# ── USER CONFIG ───────────────────────────────────────────────────────────────
# Path to your already-transferred nuScenes dataset directory.
# Leave as SET_ME_NUSCENES_PATH to skip symlinking for now.
NUSCENES_ROOT="/scratch/rc5898/tracker/datasets/nuscenes"

# Path to your already-transferred Argoverse2 dataset directory.
# Leave as SET_ME_AV2_PATH to skip symlinking for now.
AV2_ROOT="SET_ME_AV2_PATH"

# Name for the conda environment.
CONDA_ENV_NAME="maptracker"

# Anaconda module to load.  Change if your cluster uses a different module name.
ANACONDA_MODULE="anaconda3/2024.02"

# Set to 1 to download pretrained checkpoints from HuggingFace automatically.
# Set to 0 to skip (e.g. if you already have them or prefer Dropbox).
DOWNLOAD_CHECKPOINTS=1
# ─────────────────────────────────────────────────────────────────────────────

# ── helpers ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[setup]${NC} $*"; }
success() { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[setup] WARNING:${NC} $*"; }
error()   { echo -e "${RED}[setup] ERROR:${NC} $*" >&2; exit 1; }

# ── resolve project root (directory containing this script) ──────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
info "PROJECT_ROOT = $PROJECT_ROOT"

# =============================================================================
# Phase 1 — Load modules and activate / create conda environment
# =============================================================================
info "Loading $ANACONDA_MODULE ..."
module purge
module load "$ANACONDA_MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    info "Conda env '$CONDA_ENV_NAME' already exists — activating."
else
    info "Creating conda env '$CONDA_ENV_NAME' (python=3.8) ..."
    conda create -n "$CONDA_ENV_NAME" python=3.8 -y
fi

conda activate "$CONDA_ENV_NAME"
success "Conda env '$CONDA_ENV_NAME' active."

# =============================================================================
# Phase 2 — Python package installation
# =============================================================================

# ── 2a. PyTorch with CUDA 11.1 ───────────────────────────────────────────────
info "Installing PyTorch 1.9.0+cu111 ..."
pip install \
    torch==1.9.0+cu111 \
    torchvision==0.10.0+cu111 \
    torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

# ── 2b. MM* ecosystem ────────────────────────────────────────────────────────
info "Installing mmcv-full, mmdet, mmsegmentation ..."
pip install mmcv-full==1.6.0
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0

# ── 2c. mmdetection3d @ v1.0.0rc6 (editable) ────────────────────────────────
MMDET3D_DIR="$PROJECT_ROOT/mmdetection3d"

if [[ -d "$MMDET3D_DIR" ]]; then
    info "mmdetection3d directory already exists — skipping clone."
else
    info "Cloning mmdetection3d ..."
    git clone https://github.com/open-mmlab/mmdetection3d.git "$MMDET3D_DIR"
fi

info "Checking out v1.0.0rc6 ..."
git -C "$MMDET3D_DIR" checkout v1.0.0rc6

info "Installing mmdetection3d in editable mode ..."
pip install -e "$MMDET3D_DIR"

# ── 2d. MapTracker-specific requirements ─────────────────────────────────────
info "Installing maptracker requirements.txt ..."
pip install -r "$PROJECT_ROOT/requirements.txt"

# ── 2e. huggingface_hub (needed for checkpoint download) ─────────────────────
info "Installing huggingface_hub ..."
pip install huggingface_hub

success "All Python packages installed."

# =============================================================================
# Phase 3 — Directory structure
# =============================================================================
info "Creating work_dirs structure ..."
mkdir -p "$PROJECT_ROOT/work_dirs/pretrained_ckpts/maptracker_nusc_oldsplit_5frame_span10_stage1_bev_pretrain"
mkdir -p "$PROJECT_ROOT/work_dirs/pretrained_ckpts/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup"
mkdir -p "$PROJECT_ROOT/work_dirs/pretrained_ckpts/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune"
mkdir -p "$PROJECT_ROOT/work_dirs/experiments"
mkdir -p "$PROJECT_ROOT/datasets"
success "work_dirs structure ready."

# =============================================================================
# Phase 4 — Dataset symlinks
# =============================================================================

# ── nuScenes ─────────────────────────────────────────────────────────────────
if [[ "$NUSCENES_ROOT" == "SET_ME_NUSCENES_PATH" ]]; then
    warn "NUSCENES_ROOT not set — skipping nuScenes symlink."
    warn "  Once you know the path, run:"
    warn "    ln -sfn /path/to/nuscenes $PROJECT_ROOT/datasets/nuscenes"
elif [[ -e "$PROJECT_ROOT/datasets/nuscenes" ]]; then
    info "datasets/nuscenes already exists — skipping symlink."
else
    ln -sfn "$NUSCENES_ROOT" "$PROJECT_ROOT/datasets/nuscenes"
    success "Symlinked nuScenes: $NUSCENES_ROOT -> datasets/nuscenes"
fi

# ── Argoverse2 ───────────────────────────────────────────────────────────────
if [[ "$AV2_ROOT" == "SET_ME_AV2_PATH" ]]; then
    warn "AV2_ROOT not set — skipping Argoverse2 symlink."
    warn "  Once you know the path, run:"
    warn "    ln -sfn /path/to/av2 $PROJECT_ROOT/datasets/av2"
elif [[ -e "$PROJECT_ROOT/datasets/av2" ]]; then
    info "datasets/av2 already exists — skipping symlink."
else
    ln -sfn "$AV2_ROOT" "$PROJECT_ROOT/datasets/av2"
    success "Symlinked Argoverse2: $AV2_ROOT -> datasets/av2"
fi

# =============================================================================
# Phase 5 — Pretrained checkpoint download
# =============================================================================
# HuggingFace repo:  https://huggingface.co/cccjc/maptracker/tree/main
# Dropbox fallback:  https://www.dropbox.com/scl/fo/miulg8q9oby7q2x5vemme/ALoxX1HyxGlfR9y3xlqfzeE?rlkey=i3rw4mbq7lacblc7xsnjkik1u&dl=0

if [[ "$DOWNLOAD_CHECKPOINTS" -eq 1 ]]; then
    info "Downloading pretrained checkpoints from HuggingFace (cccjc/maptracker) ..."
    info "  If this is slow, use the Dropbox link as an alternative:"
    info "  https://www.dropbox.com/scl/fo/miulg8q9oby7q2x5vemme/ALoxX1HyxGlfR9y3xlqfzeE?rlkey=i3rw4mbq7lacblc7xsnjkik1u&dl=0"
    huggingface-cli download cccjc/maptracker \
        --repo-type model \
        --local-dir "$PROJECT_ROOT/work_dirs/pretrained_ckpts"
    success "Checkpoints downloaded to work_dirs/pretrained_ckpts/"
else
    info "DOWNLOAD_CHECKPOINTS=0 — skipping checkpoint download."
    info "  Download manually from HuggingFace:"
    info "    huggingface-cli download cccjc/maptracker --repo-type model --local-dir $PROJECT_ROOT/work_dirs/pretrained_ckpts"
    info "  Or from Dropbox:"
    info "    https://www.dropbox.com/scl/fo/miulg8q9oby7q2x5vemme/ALoxX1HyxGlfR9y3xlqfzeE?rlkey=i3rw4mbq7lacblc7xsnjkik1u&dl=0"
fi

# =============================================================================
# Phase 6 — Import sanity check
# =============================================================================
info "Running import sanity check ..."
PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" python -c "
import torch
import mmcv
import mmdet
import mmseg
import mmdet3d
print(f'  torch      {torch.__version__}')
print(f'  mmcv       {mmcv.__version__}')
print(f'  mmdet      {mmdet.__version__}')
print(f'  mmseg      {mmseg.__version__}')
print(f'  mmdet3d    {mmdet3d.__version__}')
print(f'  CUDA avail {torch.cuda.is_available()}')
"
success "All imports OK."

# =============================================================================
# Phase 7 — Summary and next steps
# =============================================================================
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  MapTracker environment setup complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Project root : $PROJECT_ROOT"
echo "Conda env    : $CONDA_ENV_NAME"
echo ""

echo -e "${CYAN}── Next steps (run manually in order) ──────────────────────${NC}"
echo ""
echo "1. [If not done] Set dataset symlinks:"
echo "     ln -sfn /path/to/nuscenes  $PROJECT_ROOT/datasets/nuscenes"
echo "     ln -sfn /path/to/av2       $PROJECT_ROOT/datasets/av2"
echo ""
echo "2. Generate nuScenes annotation files:"
echo "     cd $PROJECT_ROOT"
echo "     python tools/data_converter/nuscenes_converter.py --data-root ./datasets/nuscenes"
echo "     # Add --newsplit for the new geographical-based split"
echo ""
echo "3. Generate tracking ground truth (run on the training machine):"
echo "     python tools/tracking/prepare_gt_tracks.py \\"
echo "       plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py \\"
echo "       --out-dir tracking_gts/nuscenes --visualize"
echo ""
echo "4. [Optional] Smoke evaluation (2 GPUs, ~600 samples):"
echo "     sbatch $PROJECT_ROOT/run_smoke_eval.sh"
echo ""
echo "5. [Optional] Smoke training (1 GPU, 5 iterations):"
echo "     sbatch $PROJECT_ROOT/run_smoke_train.sh"
echo ""
echo "6. Full training (8 GPUs, three stages):"
echo "     bash tools/dist_train.sh plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage1_bev_pretrain.py 8"
echo "     bash tools/dist_train.sh plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup.py 8"
echo "     bash tools/dist_train.sh plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py 8"
echo ""
echo -e "${CYAN}────────────────────────────────────────────────────────────${NC}"
