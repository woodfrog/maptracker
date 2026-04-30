#!/usr/bin/env bash
# CPU-only aggregation job: reads per-condition contradiction_metrics.json files
# already produced by parallel corruption eval jobs and writes the final
# contradiction_suite_summary.json.  Typically submitted as an afterok dependency.
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit a CPU-only job to aggregate per-condition contradiction_metrics.json files
into contradiction_suite_summary.json.  Intended to run after all parallel
corruption eval jobs finish (submit as --dependency afterok:JID1:JID2:...).

Usage:
  bash tools/experiments/submit_contradiction_metrics_b0_sbatch.sh \
    --mail-user EMAIL --suite-root PATH [options]

Required:
  --mail-user EMAIL
  --suite-root PATH     Root dir of the corruption suite, e.g.:
                        work_dirs/experiments/b0/corruption_suite/latest_onset0_trainmatched

Optional:
  --config PATH         Eval config (positional arg for run_contradiction_suite.py;
                        not used in aggregate-only mode but required by the CLI).
                        Default: plugin/configs/.../b0_eval.py
  --checkpoint PATH     Checkpoint (same note as --config).
                        Default: work_dirs/pretrained_ckpts/b0_nusc_oldsplit/latest.pth
  --work-root DIR       (default: <repo-root>/work_dirs)
  --onset N             Must match the suite (default: 0)
  --modes "..."         Space-separated modes (default: "c_full c_tail")
  --stale-offsets "..."  Space-separated offsets (default: "1 2 3")
  --dependency STR      Slurm --dependency string, e.g. afterok:JID1:JID2
  --time LIMIT          (default: 00:15:00)
  --qos NAME            (default: short)
  --account NAME
  --mail-type VALUE     (default: END,FAIL)
  --module-load MOD     (default: anaconda3/2023.9)
  --conda-env NAME      (default: maptracker)
  --dry-run
  -h, --help
USAGE
}

MAIL_USER=""
SUITE_ROOT=""
CONFIG=""
CHECKPOINT=""
WORK_ROOT=""
ONSET=0
MODES_STR="c_full c_tail"
STALE_OFFSETS_STR="1 2 3"
DEPENDENCY=""
TIME_LIMIT="00:15:00"
QOS="short"
ACCOUNT=""
MAIL_TYPE="END,FAIL"
MODULE_LOAD="anaconda3/2023.9"
CONDA_ENV="maptracker"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mail-user)      MAIL_USER="$2";         shift 2 ;;
    --suite-root)     SUITE_ROOT="$2";        shift 2 ;;
    --config)         CONFIG="$2";            shift 2 ;;
    --checkpoint)     CHECKPOINT="$2";        shift 2 ;;
    --work-root)      WORK_ROOT="$2";         shift 2 ;;
    --onset)          ONSET="$2";             shift 2 ;;
    --modes)          MODES_STR="$2";         shift 2 ;;
    --stale-offsets)  STALE_OFFSETS_STR="$2"; shift 2 ;;
    --dependency)     DEPENDENCY="$2";        shift 2 ;;
    --time)           TIME_LIMIT="$2";        shift 2 ;;
    --qos)            QOS="$2";               shift 2 ;;
    --account)        ACCOUNT="$2";           shift 2 ;;
    --mail-type)      MAIL_TYPE="$2";         shift 2 ;;
    --module-load)    MODULE_LOAD="$2";       shift 2 ;;
    --conda-env)      CONDA_ENV="$2";         shift 2 ;;
    --dry-run)        DRY_RUN=1;              shift   ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "[submit_contradiction_metrics_b0_sbatch] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MAIL_USER" ]]; then
  echo "[submit_contradiction_metrics_b0_sbatch] --mail-user is required." >&2; usage; exit 2
fi
if [[ -z "$SUITE_ROOT" ]]; then
  echo "[submit_contradiction_metrics_b0_sbatch] --suite-root is required." >&2; usage; exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "$WORK_ROOT" ]]; then
  WORK_ROOT="${PROJECT_ROOT}/work_dirs"
fi
if [[ -z "$CONFIG" ]]; then
  CONFIG="${PROJECT_ROOT}/plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_b0_eval.py"
fi
if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="${WORK_ROOT}/pretrained_ckpts/b0_nusc_oldsplit/latest.pth"
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "[submit_contradiction_metrics_b0_sbatch] Config not found: $CONFIG" >&2; exit 1
fi

read -r -a MODES_ARR        <<< "$MODES_STR"
read -r -a STALE_OFFSETS_ARR <<< "$STALE_OFFSETS_STR"

TS="$(date -u +%Y%m%d_%H%M%S)"
SBATCH_ROOT="${WORK_ROOT%/}/sbatch/contradiction_metrics_b0/${TS}"
SBATCH_LOG_DIR="$SBATCH_ROOT/logs"
SBATCH_SCRIPT="$SBATCH_ROOT/aggregate.sbatch"
mkdir -p "$SBATCH_LOG_DIR"

RUN_CMD=(python tools/tracking/run_contradiction_suite.py
  "$CONFIG"
  "$CHECKPOINT"
  --work-root "$SUITE_ROOT"
  --modes "${MODES_ARR[@]}"
  --stale-offsets "${STALE_OFFSETS_ARR[@]}"
  --onset "$ONSET"
  --aggregate-only)

if [[ $DRY_RUN -eq 1 ]]; then
  RUN_CMD+=(--dry-run)
fi

printf -v RUN_CMD_STR '%q ' "${RUN_CMD[@]}"

SBATCH_DIRECTIVES=(
  "#SBATCH --job-name=contradiction_metrics_b0"
  "#SBATCH --nodes=1"
  "#SBATCH --ntasks=1"
  "#SBATCH --cpus-per-task=2"
  "#SBATCH --time=${TIME_LIMIT}"
  "#SBATCH --output=${SBATCH_LOG_DIR}/%x-%j.out"
  "#SBATCH --error=${SBATCH_LOG_DIR}/%x-%j.err"
  "#SBATCH --mail-type=${MAIL_TYPE}"
  "#SBATCH --mail-user=${MAIL_USER}"
)
if [[ -n "$QOS"        ]]; then SBATCH_DIRECTIVES+=("#SBATCH --qos=${QOS}"); fi
if [[ -n "$ACCOUNT"    ]]; then SBATCH_DIRECTIVES+=("#SBATCH --account=${ACCOUNT}"); fi
if [[ -n "$DEPENDENCY" ]]; then SBATCH_DIRECTIVES+=("#SBATCH --dependency=${DEPENDENCY}"); fi

printf '%s\n' '#!/usr/bin/env bash' > "$SBATCH_SCRIPT"
printf '%s\n' "${SBATCH_DIRECTIVES[@]}" >> "$SBATCH_SCRIPT"
printf '%s\n' 'set -eo pipefail' >> "$SBATCH_SCRIPT"
printf '%s\n' '# Disable unbound-variable check for the whole env-setup block: module load and' >> "$SBATCH_SCRIPT"
printf '%s\n' '# conda activate both source scripts that reference PS1 (unset in batch shells).' >> "$SBATCH_SCRIPT"
printf '%s\n' 'set +u' >> "$SBATCH_SCRIPT"

cat >> "$SBATCH_SCRIPT" <<EOF

module purge
EOF
if [[ -n "$MODULE_LOAD" ]]; then
  printf 'module load %q\n' "$MODULE_LOAD" >> "$SBATCH_SCRIPT"
fi
cat >> "$SBATCH_SCRIPT" <<EOF
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}
set -u

export PROJECT_ROOT=${PROJECT_ROOT}
export PYTHONPATH="\$PROJECT_ROOT:\${PYTHONPATH:-}"

cd "\$PROJECT_ROOT"
echo "[contradiction-metrics-b0] Aggregating suite: ${SUITE_ROOT}"
${RUN_CMD_STR}
echo "[contradiction-metrics-b0] Done — summary: ${SUITE_ROOT}/contradiction_suite_summary.json"
EOF

chmod +x "$SBATCH_SCRIPT"

echo "[submit_contradiction_metrics_b0_sbatch] Generated:   $SBATCH_SCRIPT"
echo "[submit_contradiction_metrics_b0_sbatch] Logs dir:    $SBATCH_LOG_DIR"
echo "[submit_contradiction_metrics_b0_sbatch] Suite root:  $SUITE_ROOT"
[[ -n "$DEPENDENCY" ]] && echo "[submit_contradiction_metrics_b0_sbatch] Dependency:  $DEPENDENCY"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[submit_contradiction_metrics_b0_sbatch] Dry run — not submitting."
  exit 0
fi

SUBMIT_OUTPUT="$(sbatch "$SBATCH_SCRIPT")"
echo "$SUBMIT_OUTPUT"
