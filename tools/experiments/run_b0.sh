#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_b0.sh --config CFG --checkpoint CKPT --work-root ROOT [options]

Required:
  --config PATH
  --checkpoint PATH
  --work-root DIR

Optional:
  --run-id ID
  --gpus N                           (default: 1)
  --launcher none|pytorch|slurm      (default: none)
  --partition NAME                   (required for launcher=slurm)
  --gpus-per-node N                  (default: gpus)
  --cpus-per-task N                  (default: 5)
  --seed N                           (default: 0)
  --condition-tag TAG                (default: clean)
  --cfg-options "k=v ..."
  --eval-options "k=v ..."
  --cons-frames N                    (default: 5 for C-mAP preparation)
  --skip-cmap
  --dry-run
  -h, --help
USAGE
}

CONFIG=""
CHECKPOINT=""
WORK_ROOT=""
RUN_ID=""
GPUS=1
LAUNCHER="none"
PARTITION=""
GPUS_PER_NODE=""
CPUS_PER_TASK=5
SEED=0
CONDITION_TAG="clean"
CFG_OPTIONS_STR=""
EVAL_OPTIONS_STR=""
SKIP_CMAP=0
CONS_FRAMES=5
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --launcher) LAUNCHER="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --condition-tag) CONDITION_TAG="$2"; shift 2 ;;
    --cfg-options) CFG_OPTIONS_STR="$2"; shift 2 ;;
    --eval-options) EVAL_OPTIONS_STR="$2"; shift 2 ;;
    --cons-frames) CONS_FRAMES="$2"; shift 2 ;;
    --skip-cmap) SKIP_CMAP=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[run_b0] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$CONFIG" || -z "$CHECKPOINT" || -z "$WORK_ROOT" ]]; then
  echo "[run_b0] Missing required args." >&2
  usage
  exit 2
fi

if [[ "$LAUNCHER" == "slurm" && -z "$PARTITION" ]]; then
  echo "[run_b0] --partition is required when --launcher slurm" >&2
  exit 2
fi

if [[ -z "$GPUS_PER_NODE" ]]; then
  GPUS_PER_NODE="$GPUS"
fi

GIT_SHA="$(git rev-parse --short HEAD)"
TS="$(date -u +%Y%m%d_%H%M%S)"
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="${TS}_${GIT_SHA}_b0"
fi

BASE_DIR="${WORK_ROOT%/}/experiments/b0/${RUN_ID}"
LOG_DIR="$BASE_DIR/logs"
EVAL_DIR="$BASE_DIR/eval_clean"
TRACK_DIR="$BASE_DIR/tracking"
MANIFEST_DIR="$BASE_DIR/manifests"
LOCK_FILE="$BASE_DIR/.lock"
MAIN_LOG="$LOG_DIR/run_b0.log"

mkdir -p "$WORK_ROOT"
if [[ -e "$BASE_DIR" ]]; then
  if [[ -n "$(find "$BASE_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)" ]]; then
    echo "[run_b0] Refusing to overwrite non-empty run dir: $BASE_DIR" >&2
    exit 1
  fi
fi

VERIFY_CMD=(bash tools/experiments/verify_assets.sh
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --work-root "$WORK_ROOT"
  --run-dir "$BASE_DIR")

run_or_print() {
  if [[ -d "$(dirname "$MAIN_LOG")" ]]; then
    echo "[cmd] $*" | tee -a "$MAIN_LOG"
  else
    echo "[cmd] $*"
  fi
  if [[ $DRY_RUN -eq 0 ]]; then
    if [[ -d "$(dirname "$MAIN_LOG")" ]]; then
      "$@" 2>&1 | tee -a "$MAIN_LOG"
    else
      "$@"
    fi
  fi
}

echo "[cmd] ${VERIFY_CMD[*]}"
if [[ $DRY_RUN -eq 0 ]]; then
  "${VERIFY_CMD[@]}"
fi

mkdir -p "$LOG_DIR" "$EVAL_DIR" "$TRACK_DIR" "$MANIFEST_DIR"
if [[ -e "$LOCK_FILE" ]]; then
  echo "[run_b0] Lock exists: $LOCK_FILE" >&2
  exit 1
fi
echo "$$" > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

TEST_CMD=(python tools/test.py "$CONFIG" "$CHECKPOINT"
  --work-dir "$EVAL_DIR"
  --eval
  --seed "$SEED"
  --condition-tag "$CONDITION_TAG"
  --launcher "$LAUNCHER")

if [[ -n "$CFG_OPTIONS_STR" ]]; then
  read -r -a CFG_OPTIONS_ARR <<< "$CFG_OPTIONS_STR"
  TEST_CMD+=(--cfg-options "${CFG_OPTIONS_ARR[@]}")
fi
if [[ -n "$EVAL_OPTIONS_STR" ]]; then
  read -r -a EVAL_OPTIONS_ARR <<< "$EVAL_OPTIONS_STR"
  TEST_CMD+=(--eval-options "${EVAL_OPTIONS_ARR[@]}")
fi

if [[ "$LAUNCHER" == "slurm" ]]; then
  SLURM_CMD=(bash tools/slurm_test.sh "$PARTITION" "b0_eval_${RUN_ID}" "$CONFIG" "$CHECKPOINT"
    --work-dir "$EVAL_DIR" --eval --seed "$SEED" --condition-tag "$CONDITION_TAG")
  if [[ -n "$CFG_OPTIONS_STR" ]]; then
    SLURM_CMD+=(--cfg-options "${CFG_OPTIONS_ARR[@]}")
  fi
  if [[ -n "$EVAL_OPTIONS_STR" ]]; then
    SLURM_CMD+=(--eval-options "${EVAL_OPTIONS_ARR[@]}")
  fi
  run_or_print env GPUS="$GPUS" GPUS_PER_NODE="$GPUS_PER_NODE" CPUS_PER_TASK="$CPUS_PER_TASK" "${SLURM_CMD[@]}"
elif [[ "$LAUNCHER" == "pytorch" ]]; then
  DIST_CMD=(bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" --work-dir "$EVAL_DIR" --eval --seed "$SEED" --condition-tag "$CONDITION_TAG")
  if [[ -n "$CFG_OPTIONS_STR" ]]; then
    DIST_CMD+=(--cfg-options "${CFG_OPTIONS_ARR[@]}")
  fi
  if [[ -n "$EVAL_OPTIONS_STR" ]]; then
    DIST_CMD+=(--eval-options "${EVAL_OPTIONS_ARR[@]}")
  fi
  run_or_print "${DIST_CMD[@]}"
else
  run_or_print "${TEST_CMD[@]}"
fi

PRED_JSON="$EVAL_DIR/submission_vector.json"
if [[ $SKIP_CMAP -eq 0 ]]; then
  if [[ $DRY_RUN -eq 0 ]]; then
    [[ -f "$PRED_JSON" ]] || { echo "[run_b0] Missing expected submission_vector.json in $EVAL_DIR" >&2; exit 1; }
  fi
  MATCH_PKL="$EVAL_DIR/pos_predictions_${CONS_FRAMES}.pkl"
  run_or_print python tools/tracking/prepare_pred_tracks.py "$CONFIG" --result_path "$PRED_JSON" --cons_frames "$CONS_FRAMES"
  run_or_print python tools/tracking/calculate_cmap.py "$CONFIG" --result_path "$MATCH_PKL" --cons_frames "$CONS_FRAMES"
fi

MANIFEST_JSON="$MANIFEST_DIR/manifest.json"
INDEX_CSV="$MANIFEST_DIR/results_index.csv"
STATUS="success"
if [[ $DRY_RUN -eq 1 ]]; then
  STATUS="dry_run"
fi
cat > "$MANIFEST_JSON" <<JSON
{
  "run_id": "$RUN_ID",
  "baseline": "b0",
  "config": "$CONFIG",
  "checkpoint": "$CHECKPOINT",
  "launcher": "$LAUNCHER",
  "gpus": $GPUS,
  "seed": $SEED,
  "condition_tag": "$CONDITION_TAG",
  "skip_cmap": $SKIP_CMAP,
  "status": "$STATUS",
  "eval_dir": "$EVAL_DIR",
  "tracking_dir": "$TRACK_DIR",
  "log": "$MAIN_LOG"
}
JSON

echo "run_id,baseline,config,checkpoint,seed,launcher,gpus,status,eval_dir,log" > "$INDEX_CSV"
echo "$RUN_ID,b0,$CONFIG,$CHECKPOINT,$SEED,$LAUNCHER,$GPUS,$STATUS,$EVAL_DIR,$MAIN_LOG" >> "$INDEX_CSV"

echo "[run_b0] Done. Manifest: $MANIFEST_JSON"
