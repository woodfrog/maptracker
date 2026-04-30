#!/usr/bin/env bash
# submit_bev_vis_sbatch.sh — Submit BEV polyline visualizations for B1/B2
# (clean + c_full offset=1) for selected scenes.
#
# Usage:
#   bash tools/experiments/submit_bev_vis_sbatch.sh --mail-user EMAIL [--dry-run]
#
# All prediction pkl files are read from CurrentB1B2Results/ (no GPU needed).
# Runs on CPU-only partition with short wall time.

set -euo pipefail

MAIL_USER=""
DRY_RUN=0
SCENES="scene-0003 scene-0012 scene-0035 scene-0093 scene-0096"
DPI=20

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mail-user) MAIL_USER="$2"; shift 2 ;;
    --scenes)    SCENES="$2";    shift 2 ;;
    --dpi)       DPI="$2";       shift 2 ;;
    --dry-run)   DRY_RUN=1;      shift   ;;
    -h|--help)
      echo "Usage: $0 --mail-user EMAIL [--scenes 'scene-N ...'] [--dpi N] [--dry-run]"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MAIL_USER" ]]; then
  echo "[submit_bev_vis_sbatch] --mail-user is required." >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date -u +%Y%m%d_%H%M%S)"
SBATCH_DIR="${PROJECT_ROOT}/work_dirs/sbatch/bev_vis/${TS}"
LOG_DIR="${SBATCH_DIR}/logs"
OUT_DIR="${PROJECT_ROOT}/qualitative_outputs/bev"
CONFIG="${PROJECT_ROOT}/plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py"

mkdir -p "$SBATCH_DIR" "$LOG_DIR"

# ----- build scene_id string for vis_global.py -----
SCENE_ARGS=""
for s in $SCENES; do
  SCENE_ARGS="$SCENE_ARGS $s"
done

# ---------------------------------------------------------------
# Conditions: (label, pkl_path, out_subdir)
# ---------------------------------------------------------------
declare -a LABELS=(
  "b1_clean"
  "b2_clean"
  "b1_cfull_offset1"
  "b2_cfull_offset1"
)
declare -a PKL_PATHS=(
  "${PROJECT_ROOT}/CurrentB1B2Results/b1_eval_89148/clean/pos_predictions.pkl"
  "${PROJECT_ROOT}/CurrentB1B2Results/b2_eval_89148/clean/pos_predictions.pkl"
  "${PROJECT_ROOT}/CurrentB1B2Results/b1_contra_89148/c_full_offset1_onset0/pos_predictions.pkl"
  "${PROJECT_ROOT}/CurrentB1B2Results/b2_contra_89148/c_full_offset1_onset0/pos_predictions.pkl"
)

SBATCH_FILE="${SBATCH_DIR}/bev_vis.sbatch"

cat > "$SBATCH_FILE" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=bev_vis_b1b2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=${LOG_DIR}/bev_vis-%j.out
#SBATCH --error=${LOG_DIR}/bev_vis-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL_USER}

set -eo pipefail
set +u
cd "${PROJECT_ROOT}"

module purge
module load anaconda3/2023.9
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate maptracker
set -u

echo "[bev_vis] Starting BEV visualizations at \$(date)"

LABELS=(${LABELS[*]})
PKL_PATHS=(${PKL_PATHS[*]})

for i in "\${!LABELS[@]}"; do
  LABEL="\${LABELS[\$i]}"
  PKL="\${PKL_PATHS[\$i]}"
  COND_OUT="${OUT_DIR}/\${LABEL}"

  if [[ ! -f "\$PKL" ]]; then
    echo "[bev_vis] SKIP \$LABEL — pkl not found: \$PKL" >&2
    continue
  fi

  echo "[bev_vis] Visualizing \$LABEL -> \$COND_OUT"
  for SCENE_ID in ${SCENES}; do
    echo "[bev_vis]   scene: \$SCENE_ID"
    python tools/visualization/vis_global.py \\
      "${CONFIG}" \\
      --out_dir "\$COND_OUT" \\
      --data_path "\$PKL" \\
      --option vis-pred \\
      --scene_id \$SCENE_ID \\
      --per_frame_result 1 \\
      --dpi ${DPI} \\
      --overwrite 1
  done
  echo "[bev_vis] Done: \$LABEL"
done

echo "[bev_vis] All conditions complete at \$(date)"
SBATCH

echo "[submit_bev_vis_sbatch] sbatch file written: $SBATCH_FILE"

if (( DRY_RUN )); then
  echo "[submit_bev_vis_sbatch] DRY RUN — not submitting."
else
  JOB_ID=$(sbatch --parsable "$SBATCH_FILE")
  echo "[submit_bev_vis_sbatch] Submitted job ${JOB_ID}"
  echo "[submit_bev_vis_sbatch] Outputs -> ${OUT_DIR}"
fi
