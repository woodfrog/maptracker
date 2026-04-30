#!/usr/bin/env bash
# submit_alpha_dump_sbatch.sh — Re-run B1 and B2 inference on selected scenes
# with --dump-alpha-per-frame enabled, under c_full offset=1 corruption.
#
# Produces alpha_per_frame.pkl in the output eval dir, then renders alpha
# heatmaps and temporal traces via vis_alpha_heatmap.py.
#
# Usage:
#   bash tools/experiments/submit_alpha_dump_sbatch.sh \
#       --mail-user EMAIL \
#       --b1-checkpoint TrainningPaths/b1_iter_89148.pth \
#       --b2-checkpoint TrainningPaths/b2_iter_89148.pth \
#       [--scenes 'scene-0003 scene-0012 scene-0035'] \
#       [--corruption-mode c_full] [--stale-offset 1] \
#       [--dry-run]

set -euo pipefail

MAIL_USER=""
B1_CKPT=""
B2_CKPT=""
SCENES="scene-0003 scene-0012 scene-0035"
CORR_MODE="c_full"
STALE_OFFSET=1
DRY_RUN=0
GPUS=1
TIME_LIMIT="01:30:00"
MODULE_LOAD="anaconda3/2023.9"
CONDA_ENV="maptracker"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mail-user)       MAIL_USER="$2";    shift 2 ;;
    --b1-checkpoint)   B1_CKPT="$2";     shift 2 ;;
    --b2-checkpoint)   B2_CKPT="$2";     shift 2 ;;
    --scenes)          SCENES="$2";       shift 2 ;;
    --corruption-mode) CORR_MODE="$2";    shift 2 ;;
    --stale-offset)    STALE_OFFSET="$2"; shift 2 ;;
    --gpus)            GPUS="$2";         shift 2 ;;
    --time)            TIME_LIMIT="$2";   shift 2 ;;
    --module-load)     MODULE_LOAD="$2";  shift 2 ;;
    --conda-env)       CONDA_ENV="$2";    shift 2 ;;
    --dry-run)         DRY_RUN=1;         shift   ;;
    -h|--help)
      echo "Usage: $0 --mail-user EMAIL [options]"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MAIL_USER" ]]; then
  echo "[submit_alpha_dump_sbatch] --mail-user is required." >&2; exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date -u +%Y%m%d_%H%M%S)"
SBATCH_DIR="${PROJECT_ROOT}/work_dirs/sbatch/alpha_dump/${TS}"
LOG_DIR="${SBATCH_DIR}/logs"
OUT_ROOT="${PROJECT_ROOT}/work_dirs/alpha_dump_qual/${TS}"
CONFIG="${PROJECT_ROOT}/plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py"

# Default checkpoint paths (relative to PROJECT_ROOT) if not supplied
if [[ -z "$B1_CKPT" ]]; then
  B1_CKPT="${PROJECT_ROOT}/TrainningPaths/b1_iter_89148.pth"
fi
if [[ -z "$B2_CKPT" ]]; then
  B2_CKPT="${PROJECT_ROOT}/TrainningPaths/b2_iter_89148.pth"
fi

mkdir -p "$SBATCH_DIR" "$LOG_DIR"

SBATCH_FILE="${SBATCH_DIR}/alpha_dump.sbatch"

cat > "$SBATCH_FILE" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=alpha_dump_b1b2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${GPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --mem=48G
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/alpha_dump-%j.out
#SBATCH --error=${LOG_DIR}/alpha_dump-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL_USER}
#SBATCH --qos=gpu-short

set -eo pipefail
set +u
cd "${PROJECT_ROOT}"

module purge
module load ${MODULE_LOAD}
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}
set -u

echo "[alpha_dump] Starting at \$(date)"

B1_OUT="${OUT_ROOT}/b1_${CORR_MODE}_offset${STALE_OFFSET}"
B2_OUT="${OUT_ROOT}/b2_${CORR_MODE}_offset${STALE_OFFSET}"
mkdir -p "\$B1_OUT" "\$B2_OUT"

# ---- B1 (corruption-trained, no gate) ----
echo "[alpha_dump] Running B1 inference..."
python tools/test.py \\
  "${CONFIG}" \\
  "${B1_CKPT}" \\
  --work-dir "\$B1_OUT" \\
  --format-only \\
  --eval-options jsonfile_prefix="\$B1_OUT" \\
  --dump-alpha-per-frame \\
  --memory-corruption-mode ${CORR_MODE} \\
  --memory-stale-offset ${STALE_OFFSET} \\
  --memory-corruption-onset 0 \\
  --cfg-options "model.mvp_temporal_gate_cfg.corruption_trained_no_gate_baseline=True"
echo "[alpha_dump] B1 done. alpha_per_frame.pkl -> \$B1_OUT"

# ---- B2 (trained gate) ----
echo "[alpha_dump] Running B2 inference..."
python tools/test.py \\
  "${CONFIG}" \\
  "${B2_CKPT}" \\
  --work-dir "\$B2_OUT" \\
  --format-only \\
  --eval-options jsonfile_prefix="\$B2_OUT" \\
  --dump-alpha-per-frame \\
  --memory-corruption-mode ${CORR_MODE} \\
  --memory-stale-offset ${STALE_OFFSET} \\
  --memory-corruption-onset 0
echo "[alpha_dump] B2 done. alpha_per_frame.pkl -> \$B2_OUT"

# ---- Render heatmaps ----
echo "[alpha_dump] Rendering alpha heatmaps..."
SCENE_ARGS="${SCENES// / --scenes }"
# Replace spaces with proper --scenes flags
python tools/visualization/vis_alpha_heatmap.py \\
  --b1-pkl "\$B1_OUT/alpha_per_frame.pkl" \\
  --b2-pkl "\$B2_OUT/alpha_per_frame.pkl" \\
  --out-dir "${OUT_ROOT}/heatmaps_${CORR_MODE}_offset${STALE_OFFSET}" \\
  --scenes ${SCENES} \\
  --dpi 120

echo "[alpha_dump] All done at \$(date)"
echo "[alpha_dump] Heatmaps: ${OUT_ROOT}/heatmaps_${CORR_MODE}_offset${STALE_OFFSET}"
SBATCH

echo "[submit_alpha_dump_sbatch] sbatch file written: $SBATCH_FILE"

if (( DRY_RUN )); then
  echo "[submit_alpha_dump_sbatch] DRY RUN — not submitting."
else
  JOB_ID=$(sbatch --parsable "$SBATCH_FILE")
  echo "[submit_alpha_dump_sbatch] Submitted job ${JOB_ID}"
  echo "[submit_alpha_dump_sbatch] Outputs -> ${OUT_ROOT}"
fi
