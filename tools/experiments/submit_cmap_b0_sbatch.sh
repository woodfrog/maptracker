#!/usr/bin/env bash
# CPU-only job: compute cMAP for one B0 corruption condition.
# Runs:
#   1. prepare_pred_tracks.py  →  pos_predictions_<CONS_FRAMES>.pkl
#   2. calculate_cmap.py       →  prints cAP + mean_cMAP
# Saves parsed cMAP metrics to <cond_dir>/cmap_results.json.
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit a CPU-only job to compute cMAP for one B0 corruption condition.

Usage:
  bash tools/experiments/submit_cmap_b0_sbatch.sh \
    --mail-user EMAIL --condition COND_TAG [options]

Required:
  --mail-user EMAIL
  --condition COND_TAG   e.g. c_full_offset1_onset0

Optional:
  --suite-root PATH    Root dir of corruption suite
                       (default: work_dirs/experiments/b0/corruption_suite/latest_onset0_trainmatched)
  --config PATH        B0 eval config  (default: plugin/.../b0_eval.py)
  --work-root DIR      (default: <repo-root>/work_dirs)
  --cons-frames N      Consecutive frames for cMAP (default: 5)
  --dependency STR     Slurm --dependency string
  --time LIMIT         (default: 00:45:00)
  --qos NAME           (default: short)
  --account NAME
  --mail-type VALUE    (default: END,FAIL)
  --module-load MOD    (default: anaconda3/2023.9)
  --conda-env NAME     (default: maptracker)
  --dry-run
  -h, --help
USAGE
}

MAIL_USER=""
CONDITION=""
SUITE_ROOT=""
CONFIG=""
WORK_ROOT=""
CONS_FRAMES=5
DEPENDENCY=""
TIME_LIMIT="00:45:00"
QOS="short"
ACCOUNT=""
MAIL_TYPE="END,FAIL"
MODULE_LOAD="anaconda3/2023.9"
CONDA_ENV="maptracker"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mail-user)    MAIL_USER="$2";   shift 2 ;;
    --condition)    CONDITION="$2";   shift 2 ;;
    --suite-root)   SUITE_ROOT="$2";  shift 2 ;;
    --config)       CONFIG="$2";      shift 2 ;;
    --work-root)    WORK_ROOT="$2";   shift 2 ;;
    --cons-frames)  CONS_FRAMES="$2"; shift 2 ;;
    --dependency)   DEPENDENCY="$2";  shift 2 ;;
    --time)         TIME_LIMIT="$2";  shift 2 ;;
    --qos)          QOS="$2";         shift 2 ;;
    --account)      ACCOUNT="$2";     shift 2 ;;
    --mail-type)    MAIL_TYPE="$2";   shift 2 ;;
    --module-load)  MODULE_LOAD="$2"; shift 2 ;;
    --conda-env)    CONDA_ENV="$2";   shift 2 ;;
    --dry-run)      DRY_RUN=1;        shift   ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "[submit_cmap_b0_sbatch] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MAIL_USER" ]]; then
  echo "[submit_cmap_b0_sbatch] --mail-user is required." >&2; usage; exit 2
fi
if [[ -z "$CONDITION" ]]; then
  echo "[submit_cmap_b0_sbatch] --condition is required." >&2; usage; exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "$WORK_ROOT" ]]; then
  WORK_ROOT="${PROJECT_ROOT}/work_dirs"
fi
if [[ -z "$SUITE_ROOT" ]]; then
  SUITE_ROOT="${WORK_ROOT}/experiments/b0/corruption_suite/latest_onset0_trainmatched"
fi
if [[ -z "$CONFIG" ]]; then
  CONFIG="${PROJECT_ROOT}/plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_b0_eval.py"
fi

COND_DIR="${SUITE_ROOT}/${CONDITION}"
SUBMISSION_VECTOR="${COND_DIR}/submission_vector.json"
MATCH_PKL="${COND_DIR}/pos_predictions_${CONS_FRAMES}.pkl"
CMAP_JSON="${COND_DIR}/cmap_results.json"

if [[ $DRY_RUN -eq 0 ]]; then
  [[ -f "$CONFIG" ]]            || { echo "[submit_cmap_b0_sbatch] Config not found: $CONFIG" >&2; exit 1; }
  [[ -d "$COND_DIR" ]]          || { echo "[submit_cmap_b0_sbatch] Condition dir not found: $COND_DIR" >&2; exit 1; }
  [[ -f "$SUBMISSION_VECTOR" ]] || { echo "[submit_cmap_b0_sbatch] submission_vector.json missing: $SUBMISSION_VECTOR" >&2; exit 1; }
fi

TS="$(date -u +%Y%m%d_%H%M%S)"
SBATCH_ROOT="${WORK_ROOT}/sbatch/cmap_b0/${TS}_${CONDITION}"
SBATCH_LOG_DIR="${SBATCH_ROOT}/logs"
SBATCH_SCRIPT="${SBATCH_ROOT}/cmap.sbatch"
mkdir -p "$SBATCH_LOG_DIR"

SBATCH_DIRECTIVES=(
  "#SBATCH --job-name=cmap_b0_${CONDITION}"
  "#SBATCH --nodes=1"
  "#SBATCH --ntasks=1"
  "#SBATCH --cpus-per-task=10"
  "#SBATCH --mem=40G"
  "#SBATCH --time=${TIME_LIMIT}"
  "#SBATCH --output=${SBATCH_LOG_DIR}/%x-%j.out"
  "#SBATCH --error=${SBATCH_LOG_DIR}/%x-%j.err"
  "#SBATCH --mail-type=${MAIL_TYPE}"
  "#SBATCH --mail-user=${MAIL_USER}"
)
if [[ -n "$QOS"        ]]; then SBATCH_DIRECTIVES+=("#SBATCH --qos=${QOS}"); fi
if [[ -n "$ACCOUNT"    ]]; then SBATCH_DIRECTIVES+=("#SBATCH --account=${ACCOUNT}"); fi
if [[ -n "$DEPENDENCY" ]]; then SBATCH_DIRECTIVES+=("#SBATCH --dependency=${DEPENDENCY}"); fi

# ── Write the sbatch script ─────────────────────────────────────────────────
printf '%s\n' '#!/usr/bin/env bash' > "$SBATCH_SCRIPT"
printf '%s\n' "${SBATCH_DIRECTIVES[@]}" >> "$SBATCH_SCRIPT"

# Append the job body using a plain cat with variable expansion disabled inside Python
cat >> "$SBATCH_SCRIPT" << ENDBODY
set -eo pipefail
# Disable unbound-variable check: module load / conda activate reference PS1.
set +u

module purge
module load ${MODULE_LOAD}
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}
set -u

export PROJECT_ROOT=${PROJECT_ROOT}
export PYTHONPATH="\${PROJECT_ROOT}:\${PYTHONPATH:-}"
cd "\${PROJECT_ROOT}"

CONDITION="${CONDITION}"
COND_DIR="${COND_DIR}"
SUBMISSION_VECTOR="${SUBMISSION_VECTOR}"
MATCH_PKL="${MATCH_PKL}"
CMAP_JSON="${CMAP_JSON}"
CONFIG="${CONFIG}"
CONS_FRAMES="${CONS_FRAMES}"

echo "[cmap-b0] Condition:  \${CONDITION}"
echo "[cmap-b0] Config:     \${CONFIG}"
echo "[cmap-b0] Submission: \${SUBMISSION_VECTOR}"

# ── Step 1: prepare_pred_tracks.py → pos_predictions_\${CONS_FRAMES}.pkl ──
if [[ -f "\${MATCH_PKL}" ]]; then
  echo "[cmap-b0] Step 1/2: \${MATCH_PKL} already exists, skipping."
else
  echo "[cmap-b0] Step 1/2: prepare_pred_tracks ..."
  python tools/tracking/prepare_pred_tracks.py \\
    "\${CONFIG}" \\
    --result_path "\${SUBMISSION_VECTOR}" \\
    --cons_frames "\${CONS_FRAMES}"
  echo "[cmap-b0] Step 1 done."
fi

# ── Step 2: calculate_cmap.py → capture output + parse → cmap_results.json ──
echo "[cmap-b0] Step 2/2: calculate_cmap ..."
CMAP_TMP="\${COND_DIR}/.cmap_raw_output.txt"
python tools/tracking/calculate_cmap.py \\
  "\${CONFIG}" \\
  --result_path "\${MATCH_PKL}" \\
  --cons_frames "\${CONS_FRAMES}" 2>&1 | tee "\${CMAP_TMP}"
echo "[cmap-b0] Step 2 done."

# ── Step 3: parse output → cmap_results.json ──────────────────────────────
echo "[cmap-b0] Saving cmap_results.json ..."
python3 - "\${CMAP_TMP}" "\${CMAP_JSON}" "\${CONDITION}" "\${CONS_FRAMES}" << 'PYEOF'
import re, json, sys

raw_path    = sys.argv[1]
json_path   = sys.argv[2]
condition   = sys.argv[3]
cons_frames = int(sys.argv[4])

text = open(raw_path).read()

# Per-threshold dicts: {'ped_crossing': {'AP@0.5': 0.58...}, ...}
per_thr = {}
for m in re.finditer(r"\{[^{}]*'AP@([\d.]+)'[^{}]*\}", text):
    thr_str = m.group(1)
    try:
        d = json.loads(m.group(0).replace("'", '"'))
        per_thr[thr_str] = d
    except Exception:
        pass

# Category mean AP
cat_cap = {}
cm = re.search(r"Category mean AP\s*(\{[^}]+\})", text)
if cm:
    try:
        cat_cap = json.loads(cm.group(1).replace("'", '"'))
    except Exception:
        pass

# mean AP = mean_cMAP
mm = re.search(r"^mean AP\s+([\d.]+)", text, re.MULTILINE)
mean_cmap = float(mm.group(1)) if mm else None

out = {
    "condition":         condition,
    "cons_frames":       cons_frames,
    "per_threshold_cAP": per_thr,
    "category_cAP":      cat_cap,
    "mean_cMAP":         mean_cmap,
}
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"[cmap-b0] Wrote: {json_path}")
if mean_cmap is not None:
    print(f"[cmap-b0] mean_cMAP = {mean_cmap:.4f}")
else:
    print("[cmap-b0] WARNING: mean_cMAP could not be parsed", file=sys.stderr)
for cat, cap in cat_cap.items():
    print(f"[cmap-b0]   {cat}: cAP = {cap:.4f}")
PYEOF

rm -f "\${CMAP_TMP}"
echo "[cmap-b0] Done: \${CONDITION}"
ENDBODY

chmod +x "$SBATCH_SCRIPT"

echo "[submit_cmap_b0_sbatch] Generated:  $SBATCH_SCRIPT"
echo "[submit_cmap_b0_sbatch] Logs dir:   $SBATCH_LOG_DIR"
echo "[submit_cmap_b0_sbatch] Condition:  $CONDITION"
echo "[submit_cmap_b0_sbatch] cmap out:   $CMAP_JSON"
[[ -n "$DEPENDENCY" ]] && echo "[submit_cmap_b0_sbatch] Dependency: $DEPENDENCY"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[submit_cmap_b0_sbatch] Dry run — not submitting."
  exit 0
fi

SUBMIT_OUTPUT="$(sbatch "$SBATCH_SCRIPT")"
echo "$SUBMIT_OUTPUT"
JID="${SUBMIT_OUTPUT##* }"
echo "[submit_cmap_b0_sbatch] Submitted job $JID for ${CONDITION}"
# Last line = JID, used by the parallel launcher
echo "$JID"
