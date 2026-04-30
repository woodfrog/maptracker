#!/usr/bin/env bash
# Submit cMAP computation for all 6 B0 corruption conditions in parallel.
# Each condition runs as an independent CPU-only job (~30 min).
# After all finish, re-runs the consolidation script to add cMAP to the CSV.
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit cMAP computation for all B0 corruption conditions in parallel.

Usage:
  bash tools/experiments/submit_cmap_b0_all.sh --mail-user EMAIL [options]

Required:
  --mail-user EMAIL

Optional:
  --suite-root PATH      (default: work_dirs/experiments/b0/corruption_suite/latest_onset0_trainmatched)
  --work-root PATH       (default: work_dirs) — sbatch scripts written under PATH/sbatch/cmap_b0/
  --modes "c_full c_tail"      (default: "c_full c_tail")
  --stale-offsets "1 2 3"      (default: "1 2 3")
  --qos NAME             (default: short)
  --account NAME
  --time LIMIT           Per-condition time limit (default: 00:45:00)
  --cons-frames N        (default: 5)
  --dry-run
  -h, --help
USAGE
}

MAIL_USER=""
SUITE_ROOT=""
WORK_ROOT_OVERRIDE=""
MODES_STR="c_full c_tail"
STALE_OFFSETS_STR="1 2 3"
QOS="short"
ACCOUNT=""
TIME_LIMIT="00:45:00"
CONS_FRAMES=5
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mail-user)      MAIL_USER="$2";         shift 2 ;;
    --suite-root)     SUITE_ROOT="$2";        shift 2 ;;
    --work-root)      WORK_ROOT_OVERRIDE="$2"; shift 2 ;;
    --modes)          MODES_STR="$2";         shift 2 ;;
    --stale-offsets)  STALE_OFFSETS_STR="$2"; shift 2 ;;
    --qos)            QOS="$2";               shift 2 ;;
    --account)        ACCOUNT="$2";           shift 2 ;;
    --time)           TIME_LIMIT="$2";        shift 2 ;;
    --cons-frames)    CONS_FRAMES="$2";       shift 2 ;;
    --dry-run)        DRY_RUN=1;              shift   ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "[submit_cmap_b0_all] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MAIL_USER" ]]; then
  echo "[submit_cmap_b0_all] --mail-user is required." >&2; usage; exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_ROOT="${PROJECT_ROOT}/work_dirs"
if [[ -n "$WORK_ROOT_OVERRIDE" ]]; then
  if [[ "$WORK_ROOT_OVERRIDE" == /* ]]; then
    WORK_ROOT="$WORK_ROOT_OVERRIDE"
  else
    WORK_ROOT="${PROJECT_ROOT}/${WORK_ROOT_OVERRIDE}"
  fi
  mkdir -p "$WORK_ROOT"
fi
if [[ -z "$SUITE_ROOT" ]]; then
  SUITE_ROOT="${WORK_ROOT}/experiments/b0/corruption_suite/latest_onset0_trainmatched"
fi

read -r -a MODES          <<< "$MODES_STR"
read -r -a STALE_OFFSETS  <<< "$STALE_OFFSETS_STR"

SUBMIT_SCRIPT="${PROJECT_ROOT}/tools/experiments/submit_cmap_b0_sbatch.sh"

JIDS=()
echo "[submit_cmap_b0_all] Submitting cMAP jobs for ${#MODES[@]} modes x ${#STALE_OFFSETS[@]} offsets ..."
echo ""

for MODE in "${MODES[@]}"; do
  for OFFSET in "${STALE_OFFSETS[@]}"; do
    COND="$(printf '%s_offset%d_onset0' "$MODE" "$OFFSET")"

    SUBMIT_ARGS=(
      --mail-user   "$MAIL_USER"
      --condition   "$COND"
      --suite-root  "$SUITE_ROOT"
      --work-root   "$WORK_ROOT"
      --time        "$TIME_LIMIT"
      --qos         "$QOS"
      --cons-frames "$CONS_FRAMES"
    )
    [[ -n "$ACCOUNT" ]] && SUBMIT_ARGS+=(--account "$ACCOUNT")
    [[ $DRY_RUN -eq 1 ]] && SUBMIT_ARGS+=(--dry-run)

    OUTPUT="$(bash "$SUBMIT_SCRIPT" "${SUBMIT_ARGS[@]}" 2>&1)"
    echo "$OUTPUT"
    echo ""

    if [[ $DRY_RUN -eq 0 ]]; then
      JID="$(echo "$OUTPUT" | tail -1)"
      JIDS+=("$JID")
      echo "[submit_cmap_b0_all] -> ${COND}: job ${JID}"
    fi
  done
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[submit_cmap_b0_all] Dry run complete."
  exit 0
fi

echo ""
echo "[submit_cmap_b0_all] All ${#JIDS[@]} cMAP jobs submitted: ${JIDS[*]}"
echo ""
echo "[submit_cmap_b0_all] When jobs finish, regenerate CSVs with:"
echo "  cd ${PROJECT_ROOT}"
echo "  python3 tools/experiments/consolidate_b0_corruption_results.py \\"
echo "    --suite-root ${SUITE_ROOT} \\"
echo "    --log-dir <sbatch-log-dir-with-.out-files>"
