#!/usr/bin/env bash
# Submit cMAP for ALL conditions of ONE model as a single SLURM job, running each
# condition sequentially. Uses only 10 CPUs + 96G (for prepare_pred_tracks N_WORKERS=10)
# at a time. 6 conditions × ~15 min = ~90 min total.
#
# Usage:
#   bash tools/experiments/submit_cmap_parallel_sbatch.sh \
#     --mail-user EMAIL --suite-root PATH [options]
#
# Required:
#   --mail-user EMAIL
#   --suite-root PATH     Root dir containing condition subdirs (e.g. CurrentB1B2Results/b1_contra_89148)
#
# Optional:
#   --config PATH         Eval config (default: plugin/.../b0_eval.py)
#   --modes "c_full c_tail"    (default: "c_full c_tail")
#   --stale-offsets "1 2 3"    (default: "1 2 3")
#   --cons-frames N       (default: 5)
#   --time LIMIT          (default: 01:00:00)
#   --qos NAME            (default: short)
#   --account NAME
#   --job-name NAME       (default: derived from suite-root basename)
#   --dry-run
#   -h, --help
set -euo pipefail

usage() {
  sed -n '2,/^set -/p' "${BASH_SOURCE[0]}" | grep -E '^\s*#' | sed 's/^# \?//'
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MAIL_USER=""
SUITE_ROOT=""
CONFIG=""
MODES_STR="c_full c_tail"
STALE_OFFSETS_STR="1 2 3"
CONS_FRAMES=5
TIME_LIMIT="02:00:00"
QOS="short"
ACCOUNT=""
JOB_NAME=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mail-user)      MAIL_USER="$2";      shift 2 ;;
    --suite-root)     SUITE_ROOT="$2";     shift 2 ;;
    --config)         CONFIG="$2";         shift 2 ;;
    --modes)          MODES_STR="$2";      shift 2 ;;
    --stale-offsets)  STALE_OFFSETS_STR="$2"; shift 2 ;;
    --cons-frames)    CONS_FRAMES="$2";    shift 2 ;;
    --time)           TIME_LIMIT="$2";     shift 2 ;;
    --qos)            QOS="$2";            shift 2 ;;
    --account)        ACCOUNT="$2";        shift 2 ;;
    --job-name)       JOB_NAME="$2";       shift 2 ;;
    --dry-run)        DRY_RUN=1;           shift   ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "[submit_cmap_parallel_sbatch] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MAIL_USER" || -z "$SUITE_ROOT" ]]; then
  echo "[submit_cmap_parallel_sbatch] --mail-user and --suite-root are required." >&2; usage; exit 2
fi

# Resolve suite root to absolute path
if [[ "$SUITE_ROOT" != /* ]]; then
  SUITE_ROOT="${PROJECT_ROOT}/${SUITE_ROOT}"
fi
SUITE_ROOT="$(cd "$SUITE_ROOT" && pwd)"

if [[ -z "$CONFIG" ]]; then
  CONFIG="${PROJECT_ROOT}/plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_b0_eval.py"
fi

MODEL_TAG="$(basename "$SUITE_ROOT")"
if [[ -z "$JOB_NAME" ]]; then
  JOB_NAME="cmap_${MODEL_TAG}"
fi

TS="$(date +%Y%m%d_%H%M%S)"
SBATCH_ROOT="${PROJECT_ROOT}/work_dirs/sbatch/cmap_parallel/${TS}_${MODEL_TAG}"
SBATCH_LOG_DIR="${SBATCH_ROOT}/logs"
SBATCH_SCRIPT="${SBATCH_ROOT}/cmap_parallel.sbatch"
mkdir -p "$SBATCH_LOG_DIR"

# Build condition list
read -r -a MODES         <<< "$MODES_STR"
read -r -a STALE_OFFSETS <<< "$STALE_OFFSETS_STR"
CONDITIONS=()
for MODE in "${MODES[@]}"; do
  for OFFSET in "${STALE_OFFSETS[@]}"; do
    CONDITIONS+=("$(printf '%s_offset%d_onset0' "$MODE" "$OFFSET")")
  done
done
N_CONDS=${#CONDITIONS[@]}

# Validate all condition dirs exist
for COND in "${CONDITIONS[@]}"; do
  COND_DIR="${SUITE_ROOT}/${COND}"
  if [[ ! -d "$COND_DIR" ]]; then
    echo "[submit_cmap_parallel_sbatch] Condition dir not found: $COND_DIR" >&2; exit 1
  fi
  if [[ ! -f "${COND_DIR}/submission_vector.json" ]]; then
    echo "[submit_cmap_parallel_sbatch] Missing submission_vector.json in: $COND_DIR" >&2; exit 1
  fi
done

# Build the per-condition block (runs in background)
COND_BLOCKS=""
for COND in "${CONDITIONS[@]}"; do
  COND_DIR="${SUITE_ROOT}/${COND}"
  SUBMISSION_VECTOR="${COND_DIR}/submission_vector.json"
  MATCH_PKL="${COND_DIR}/pos_predictions_${CONS_FRAMES}.pkl"
  CMAP_JSON="${COND_DIR}/cmap_results.json"
  CMAP_TMP="${COND_DIR}/.cmap_raw_output.txt"
  COND_BLOCKS+="
# ── Condition: ${COND} ──────────────────────────────────────────────────────
run_condition_${COND//[-]/_}() {
  local COND=\"${COND}\"
  local COND_DIR=\"${COND_DIR}\"
  local MATCH_PKL=\"${MATCH_PKL}\"
  local CMAP_JSON=\"${CMAP_JSON}\"
  local CMAP_TMP=\"${CMAP_TMP}\"
  echo \"[\${COND}] Starting...\"
  if [[ -f \"\${MATCH_PKL}\" ]]; then
    echo \"[\${COND}] pos_predictions_${CONS_FRAMES}.pkl already exists, skipping prepare_pred_tracks.\"
  else
    echo \"[\${COND}] Running prepare_pred_tracks...\"
    python tools/tracking/prepare_pred_tracks.py \\
      \"${CONFIG}\" \\
      --result_path \"${SUBMISSION_VECTOR}\" \\
      --cons_frames \"${CONS_FRAMES}\"
    echo \"[\${COND}] prepare_pred_tracks done.\"
  fi
  echo \"[\${COND}] Running calculate_cmap...\"
  python tools/tracking/calculate_cmap.py \\
    \"${CONFIG}\" \\
    --result_path \"\${MATCH_PKL}\" \\
    --cons_frames \"${CONS_FRAMES}\" 2>&1 | tee \"\${CMAP_TMP}\"
  echo \"[\${COND}] Parsing output -> cmap_results.json...\"
  python3 - \"\${CMAP_TMP}\" \"\${CMAP_JSON}\" \"\${COND}\" \"${CONS_FRAMES}\" << 'PYEOF'
import ast, re, json, sys
raw_path    = sys.argv[1]
json_path   = sys.argv[2]
condition   = sys.argv[3]
cons_frames = int(sys.argv[4])
text = open(raw_path).read()
# per_class_per_threshold_cAP: {class: {threshold: AP}}
per_cls_thr = {}
for line in text.splitlines():
    line = line.strip()
    if "AP@" in line and line.startswith("{"):
        try:
            d = ast.literal_eval(line)
            for cls, thr_dict in d.items():
                if cls not in per_cls_thr:
                    per_cls_thr[cls] = {}
                for thr_key, val in thr_dict.items():
                    thr = thr_key.split("@")[1]
                    per_cls_thr[cls][thr] = val
        except Exception:
            pass
# per_threshold_cAP: mean across classes at each threshold
per_thr = {}
for cls, thr_vals in per_cls_thr.items():
    for thr, val in thr_vals.items():
        per_thr.setdefault(thr, []).append(val)
per_thr_mean = {thr: sum(v)/len(v) for thr, v in per_thr.items()}
cat_cap = {}
cm = re.search(r\"Category mean AP\s*(\{[^}]+\})\", text)
if cm:
    try:
        cat_cap = ast.literal_eval(cm.group(1))
    except Exception:
        pass
mm = re.search(r\"^mean AP\s+([\d.]+)\", text, re.MULTILINE)
mean_cmap = float(mm.group(1)) if mm else None
out = {\"condition\": condition, \"cons_frames\": cons_frames,
       \"per_class_per_threshold_cAP\": per_cls_thr,
       \"per_threshold_cAP\": per_thr_mean, \"category_cAP\": cat_cap, \"mean_cMAP\": mean_cmap}
with open(json_path, \"w\") as f:
    json.dump(out, f, indent=2)
print(f\"[{condition}] Wrote {json_path}  mean_cMAP={mean_cmap}\")
PYEOF
  rm -f \"\${CMAP_TMP}\"
  echo \"[\${COND}] Done.\"
}
"
done

# Build the SBATCH header
ACCOUNT_LINE=""
[[ -n "$ACCOUNT" ]] && ACCOUNT_LINE="#SBATCH --account=${ACCOUNT}"

cat > "$SBATCH_SCRIPT" << SBATCH_HEADER
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${SBATCH_LOG_DIR}/%x-%j.out
#SBATCH --error=${SBATCH_LOG_DIR}/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL_USER}
${ACCOUNT_LINE}
set -eo pipefail
set +u
module purge
module load anaconda3/2023.9
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate maptracker
set -u

export PROJECT_ROOT=${PROJECT_ROOT}
export PYTHONPATH="\${PROJECT_ROOT}:\${PYTHONPATH:-}"
cd "\${PROJECT_ROOT}"

echo "[cmap-parallel] Suite root: ${SUITE_ROOT}"
echo "[cmap-parallel] Conditions: ${CONDITIONS[*]}"
echo "[cmap-parallel] Running ${N_CONDS} conditions sequentially..."
echo ""

SBATCH_HEADER

# Append function definitions
echo "$COND_BLOCKS" >> "$SBATCH_SCRIPT"

# Launch conditions sequentially
echo "" >> "$SBATCH_SCRIPT"
echo "FAILED=0" >> "$SBATCH_SCRIPT"
for COND in "${CONDITIONS[@]}"; do
  SAFE="${COND//[-]/_}"
  cat >> "$SBATCH_SCRIPT" << SEQBLOCK
echo "[cmap-parallel] === Starting ${COND} ==="
if run_condition_${SAFE} 2>&1 | tee "${SBATCH_LOG_DIR}/${COND}.log"; then
  echo "[cmap-parallel] === Finished ${COND} ==="
else
  echo "[cmap-parallel] === FAILED ${COND} ===" >&2
  FAILED=1
fi
SEQBLOCK
done

cat >> "$SBATCH_SCRIPT" << 'FOOTER'

echo ""
echo "[cmap-parallel] All conditions finished. Results:"
FOOTER

for COND in "${CONDITIONS[@]}"; do
  echo "  [[ -f \"${SUITE_ROOT}/${COND}/cmap_results.json\" ]] && echo \"  DONE: ${COND}\" || echo \"  FAIL: ${COND}\"" >> "$SBATCH_SCRIPT"
done

cat >> "$SBATCH_SCRIPT" << 'FOOTER2'

if [[ $FAILED -ne 0 ]]; then
  echo "[cmap-parallel] One or more conditions failed." >&2
  exit 1
fi
echo "[cmap-parallel] All done."
FOOTER2

echo "[submit_cmap_parallel_sbatch] Generated: $SBATCH_SCRIPT"
echo "[submit_cmap_parallel_sbatch] Suite:     $SUITE_ROOT"
echo "[submit_cmap_parallel_sbatch] Conditions: ${CONDITIONS[*]}"
echo "[submit_cmap_parallel_sbatch] CPUs: 10, Mem: 96G, Time: ${TIME_LIMIT} (${N_CONDS} conditions sequential)"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[submit_cmap_parallel_sbatch] Dry run — not submitting."
  exit 0
fi

JID="$(sbatch "$SBATCH_SCRIPT" | awk '{print $NF}')"
echo "[submit_cmap_parallel_sbatch] Submitted job ${JID} for ${MODEL_TAG}"
echo "$JID"
