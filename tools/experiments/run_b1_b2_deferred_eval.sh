#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_b1_b2_deferred_eval.sh --base-config CFG --checkpoint CKPT --work-root ROOT --run-id ID --baseline BASELINE [options]

Required:
  --base-config PATH
  --checkpoint PATH
  --work-root DIR
  --run-id ID
  --baseline b1|b2

Optional:
  --checkpoint-tag TAG
  --condition-tag TAG                (default: derived from eval mode, else clean)
  --rerun                            Reuse and replace an existing eval dir.
  --launcher none|pytorch|slurm|slurm-step  (default: none)
  --partition NAME                   (required for launcher=slurm)
  --gpus N                           (default: 2)
  --gpus-per-node N                  (default: gpus)
  --cpus-per-task N                  (default: 4)
  --seed N                           (default: 0)
  --cfg-options "k=v ..."
  --eval-options "k=v ..."
  --cons-frames N                    (default: 5)
  --skip-cmap
  --memory-corruption-mode clean|c_full|c_tail
  --memory-stale-offset N
  --memory-c-tail-keep-recent N
  --memory-corruption-onset N
  --dry-run
  -h, --help
USAGE
}

BASE_CONFIG=""
CHECKPOINT=""
WORK_ROOT=""
RUN_ID=""
BASELINE=""
CHECKPOINT_TAG=""
CONDITION_TAG=""
LAUNCHER="none"
PARTITION=""
GPUS=2
GPUS_PER_NODE=""
CPUS_PER_TASK=4
SEED=0
CFG_OPTIONS_STR=""
EVAL_OPTIONS_STR=""
CONS_FRAMES=5
SKIP_CMAP=0
MEMORY_CORRUPTION_MODE=""
MEMORY_STALE_OFFSET=""
MEMORY_C_TAIL_KEEP_RECENT=""
MEMORY_CORRUPTION_ONSET=""
DUMP_ALPHA_PER_FRAME=0
RERUN=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-config) BASE_CONFIG="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --checkpoint-tag) CHECKPOINT_TAG="$2"; shift 2 ;;
    --condition-tag) CONDITION_TAG="$2"; shift 2 ;;
    --rerun) RERUN=1; shift ;;
    --launcher) LAUNCHER="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --cfg-options) CFG_OPTIONS_STR="$2"; shift 2 ;;
    --eval-options) EVAL_OPTIONS_STR="$2"; shift 2 ;;
    --cons-frames) CONS_FRAMES="$2"; shift 2 ;;
    --skip-cmap) SKIP_CMAP=1; shift ;;
    --memory-corruption-mode) MEMORY_CORRUPTION_MODE="$2"; shift 2 ;;
    --memory-stale-offset) MEMORY_STALE_OFFSET="$2"; shift 2 ;;
    --memory-c-tail-keep-recent) MEMORY_C_TAIL_KEEP_RECENT="$2"; shift 2 ;;
    --memory-corruption-onset) MEMORY_CORRUPTION_ONSET="$2"; shift 2 ;;
    --dump-alpha-per-frame) DUMP_ALPHA_PER_FRAME=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[run_b1_b2_deferred_eval] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$BASE_CONFIG" || -z "$CHECKPOINT" || -z "$WORK_ROOT" || -z "$RUN_ID" || -z "$BASELINE" ]]; then
  echo "[run_b1_b2_deferred_eval] Missing required args." >&2
  usage
  exit 2
fi

case "$BASELINE" in
  b1|b2) ;;
  *) echo "[run_b1_b2_deferred_eval] Invalid baseline: $BASELINE" >&2; exit 2 ;;
esac

case "$LAUNCHER" in
  none|pytorch|slurm|slurm-step) ;;
  *) echo "[run_b1_b2_deferred_eval] Invalid launcher: $LAUNCHER" >&2; exit 2 ;;
esac

if [[ "$LAUNCHER" == "slurm" && -z "$PARTITION" ]]; then
  echo "[run_b1_b2_deferred_eval] --partition is required when --launcher slurm" >&2
  exit 2
fi

if [[ -z "$GPUS_PER_NODE" ]]; then
  GPUS_PER_NODE="$GPUS"
fi

if (( GPUS <= 0 || GPUS_PER_NODE <= 0 || CPUS_PER_TASK <= 0 )); then
  echo "[run_b1_b2_deferred_eval] GPU/CPU values must be positive integers." >&2
  exit 2
fi

normalize_tag() {
  local raw="$1"
  printf '%s' "$raw" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/_/g; s/_+/_/g; s/^_+//; s/_+$//'
}

derive_condition_tag() {
  local mode="$1"
  local onset="$2"
  local stale="$3"
  local keep_recent="$4"
  local tag=""

  if [[ -z "$mode" || "$mode" == "clean" ]]; then
    tag="clean"
  elif [[ "$mode" == "c_full" ]]; then
    tag="cfull"
    if [[ -n "$onset" ]]; then
      tag+="_onset${onset}"
    fi
    if [[ -n "$stale" ]]; then
      tag+="_stale${stale}"
    fi
  elif [[ "$mode" == "c_tail" ]]; then
    tag="ctail"
    if [[ -n "$onset" ]]; then
      tag+="_onset${onset}"
    fi
    if [[ -n "$keep_recent" ]]; then
      tag+="_keep${keep_recent}"
    fi
    if [[ -n "$stale" ]]; then
      tag+="_stale${stale}"
    fi
  fi

  normalize_tag "$tag"
}

if [[ -n "$MEMORY_STALE_OFFSET" || -n "$MEMORY_C_TAIL_KEEP_RECENT" || -n "$MEMORY_CORRUPTION_ONSET" ]]; then
  if [[ -z "$MEMORY_CORRUPTION_MODE" ]]; then
    echo "[run_b1_b2_deferred_eval] --memory-corruption-mode is required when using other memory corruption overrides." >&2
    exit 2
  fi
fi

if [[ "$MEMORY_CORRUPTION_MODE" == "clean" && ( -n "$MEMORY_STALE_OFFSET" || -n "$MEMORY_C_TAIL_KEEP_RECENT" || -n "$MEMORY_CORRUPTION_ONSET" ) ]]; then
  echo "[run_b1_b2_deferred_eval] clean mode cannot be combined with stale/keep_recent/onset overrides." >&2
  exit 2
fi

if [[ -z "$CHECKPOINT_TAG" ]]; then
  CHECKPOINT_TAG="$(basename "$CHECKPOINT")"
  CHECKPOINT_TAG="${CHECKPOINT_TAG%.pth}"
fi
CHECKPOINT_TAG="$(normalize_tag "$CHECKPOINT_TAG")"
if [[ -z "$CHECKPOINT_TAG" ]]; then
  echo "[run_b1_b2_deferred_eval] checkpoint tag resolved to empty value." >&2
  exit 2
fi

if [[ -z "$CONDITION_TAG" ]]; then
  CONDITION_TAG="$(derive_condition_tag "$MEMORY_CORRUPTION_MODE" "$MEMORY_CORRUPTION_ONSET" "$MEMORY_STALE_OFFSET" "$MEMORY_C_TAIL_KEEP_RECENT")"
fi
CONDITION_TAG="$(normalize_tag "$CONDITION_TAG")"
if [[ -z "$CONDITION_TAG" ]]; then
  echo "[run_b1_b2_deferred_eval] condition tag resolved to empty value." >&2
  exit 2
fi

BASE_DIR="${WORK_ROOT%/}/experiments/b1_b2/${RUN_ID}/${BASELINE}"
EVAL_DIR="$BASE_DIR/eval_deferred/$CHECKPOINT_TAG/$CONDITION_TAG"
LOG_DIR="$EVAL_DIR/logs"
LOCK_FILE="$EVAL_DIR/.lock"
MAIN_LOG="$LOG_DIR/run_b1_b2_deferred_eval.log"
MANIFEST_JSON="$EVAL_DIR/manifest.json"
INDEX_CSV="$EVAL_DIR/results_index.csv"
EVAL_DIR_NONEMPTY=0

mkdir -p "$WORK_ROOT"
if [[ -e "$EVAL_DIR" ]]; then
  if [[ -n "$(find "$EVAL_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)" ]]; then
    EVAL_DIR_NONEMPTY=1
    if [[ $RERUN -eq 0 ]]; then
      echo "[run_b1_b2_deferred_eval] Refusing to overwrite non-empty eval dir: $EVAL_DIR" >&2
      exit 1
    fi
  fi
fi

VERIFY_CMD=(bash tools/experiments/verify_assets.sh
  --config "$BASE_CONFIG"
  --checkpoint "$CHECKPOINT"
  --work-root "$WORK_ROOT"
  --run-dir "$EVAL_DIR")
if [[ $RERUN -eq 1 ]]; then
  VERIFY_CMD+=(--allow-nonempty-run-dir)
fi

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

log_message() {
  local log_file="$1"
  shift
  if [[ $DRY_RUN -eq 0 && -d "$(dirname "$log_file")" ]]; then
    echo "$*" | tee -a "$log_file"
  else
    echo "$*"
  fi
}

if [[ $DRY_RUN -eq 1 ]]; then
  VERIFY_CMD+=(--dry-run)
fi
echo "[cmd] ${VERIFY_CMD[*]}"
"${VERIFY_CMD[@]}"

if [[ $DRY_RUN -eq 0 ]]; then
  if [[ $RERUN -eq 1 && -e "$LOCK_FILE" ]]; then
    lock_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
    if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
      echo "[run_b1_b2_deferred_eval] Lock exists and is active: $LOCK_FILE (pid=$lock_pid)" >&2
      exit 1
    fi
  fi
  if [[ $RERUN -eq 1 && $EVAL_DIR_NONEMPTY -eq 1 ]]; then
    echo "[run_b1_b2_deferred_eval] Clearing existing eval dir for rerun: $EVAL_DIR"
    rm -rf "$EVAL_DIR"
  fi
  mkdir -p "$LOG_DIR"
  if [[ -e "$LOCK_FILE" ]]; then
    lock_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
    if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
      echo "[run_b1_b2_deferred_eval] Lock exists and is active: $LOCK_FILE (pid=$lock_pid)" >&2
      exit 1
    fi
    log_message "$MAIN_LOG" "[run_b1_b2_deferred_eval] Removing stale lock: $LOCK_FILE"
    rm -f "$LOCK_FILE"
  fi
  echo "$$" > "$LOCK_FILE"
  trap 'rm -f "$LOCK_FILE"' EXIT
fi

declare -a CFG_OPTIONS_ARR=() EVAL_OPTIONS_ARR=()
if [[ -n "$CFG_OPTIONS_STR" ]]; then
  read -r -a CFG_OPTIONS_ARR <<< "$CFG_OPTIONS_STR"
fi
if [[ -n "$EVAL_OPTIONS_STR" ]]; then
  read -r -a EVAL_OPTIONS_ARR <<< "$EVAL_OPTIONS_STR"
fi

declare -a MEMORY_ARGS=()
if [[ -n "$MEMORY_CORRUPTION_MODE" ]]; then
  MEMORY_ARGS+=(--memory-corruption-mode "$MEMORY_CORRUPTION_MODE")
fi
if [[ -n "$MEMORY_STALE_OFFSET" ]]; then
  MEMORY_ARGS+=(--memory-stale-offset "$MEMORY_STALE_OFFSET")
fi
if [[ -n "$MEMORY_C_TAIL_KEEP_RECENT" ]]; then
  MEMORY_ARGS+=(--memory-c-tail-keep-recent "$MEMORY_C_TAIL_KEEP_RECENT")
fi
if [[ -n "$MEMORY_CORRUPTION_ONSET" ]]; then
  MEMORY_ARGS+=(--memory-corruption-onset "$MEMORY_CORRUPTION_ONSET")
fi

TEST_ARGS=(--work-dir "$EVAL_DIR" --eval --seed "$SEED" --condition-tag "$CONDITION_TAG")
TEST_ARGS+=("${MEMORY_ARGS[@]}")
if (( DUMP_ALPHA_PER_FRAME )); then
  TEST_ARGS+=(--dump-alpha-per-frame)
fi
if [[ ${#CFG_OPTIONS_ARR[@]} -gt 0 ]]; then
  TEST_ARGS+=(--cfg-options "${CFG_OPTIONS_ARR[@]}")
fi
if [[ ${#EVAL_OPTIONS_ARR[@]} -gt 0 ]]; then
  TEST_ARGS+=(--eval-options "${EVAL_OPTIONS_ARR[@]}")
fi

if [[ "$LAUNCHER" == "slurm" ]]; then
  TEST_CMD=(env GPUS="$GPUS" GPUS_PER_NODE="$GPUS_PER_NODE" CPUS_PER_TASK="$CPUS_PER_TASK"
    bash tools/slurm_test.sh "$PARTITION" "${BASELINE}_eval_${RUN_ID}_${CHECKPOINT_TAG}" "$BASE_CONFIG" "$CHECKPOINT"
    "${TEST_ARGS[@]}")
elif [[ "$LAUNCHER" == "slurm-step" ]]; then
  TEST_CMD=(env GPUS="$GPUS" GPUS_PER_NODE="$GPUS_PER_NODE" CPUS_PER_TASK="$CPUS_PER_TASK"
    bash tools/slurm_test_step.sh "$BASE_CONFIG" "$CHECKPOINT"
    "${TEST_ARGS[@]}")
elif [[ "$LAUNCHER" == "pytorch" ]]; then
  TEST_CMD=(bash tools/dist_test.sh "$BASE_CONFIG" "$CHECKPOINT" "$GPUS" "${TEST_ARGS[@]}")
else
  TEST_CMD=(python tools/test.py "$BASE_CONFIG" "$CHECKPOINT" --launcher none "${TEST_ARGS[@]}")
fi

run_or_print "${TEST_CMD[@]}"

PRED_JSON="$EVAL_DIR/submission_vector.json"
if [[ $SKIP_CMAP -eq 0 ]]; then
  if [[ $DRY_RUN -eq 0 ]]; then
    [[ -f "$PRED_JSON" ]] || { echo "[run_b1_b2_deferred_eval] Missing expected submission_vector.json in $EVAL_DIR" >&2; exit 1; }
  fi
  MATCH_PKL="$EVAL_DIR/pos_predictions_${CONS_FRAMES}.pkl"
  run_or_print python tools/tracking/prepare_pred_tracks.py "$BASE_CONFIG" --result_path "$PRED_JSON" --cons_frames "$CONS_FRAMES"
  run_or_print python tools/tracking/calculate_cmap.py "$BASE_CONFIG" --result_path "$MATCH_PKL" --cons_frames "$CONS_FRAMES"
fi

STATUS="success"
if [[ $DRY_RUN -eq 1 ]]; then
  STATUS="dry_run"
fi
if [[ $DRY_RUN -eq 0 ]]; then
  cat > "$MANIFEST_JSON" <<JSON
{
  "run_id": "$RUN_ID",
  "baseline": "$BASELINE",
  "config": "$BASE_CONFIG",
  "checkpoint": "$CHECKPOINT",
  "checkpoint_tag": "$CHECKPOINT_TAG",
  "condition_tag": "$CONDITION_TAG",
  "rerun": $RERUN,
  "launcher": "$LAUNCHER",
  "gpus": $GPUS,
  "seed": $SEED,
  "skip_cmap": $SKIP_CMAP,
  "cons_frames": $CONS_FRAMES,
  "memory_corruption_mode": "${MEMORY_CORRUPTION_MODE}",
  "memory_stale_offset": "${MEMORY_STALE_OFFSET}",
  "memory_c_tail_keep_recent": "${MEMORY_C_TAIL_KEEP_RECENT}",
  "memory_corruption_onset": "${MEMORY_CORRUPTION_ONSET}",
  "status": "$STATUS",
  "eval_dir": "$EVAL_DIR",
  "log": "$MAIN_LOG"
}
JSON

  echo "run_id,baseline,checkpoint_tag,condition_tag,config,checkpoint,seed,launcher,gpus,status,eval_dir,log" > "$INDEX_CSV"
  echo "$RUN_ID,$BASELINE,$CHECKPOINT_TAG,$CONDITION_TAG,$BASE_CONFIG,$CHECKPOINT,$SEED,$LAUNCHER,$GPUS,$STATUS,$EVAL_DIR,$MAIN_LOG" >> "$INDEX_CSV"

  echo "[run_b1_b2_deferred_eval] Done. Manifest: $MANIFEST_JSON"
else
  echo "[run_b1_b2_deferred_eval] Dry run complete."
fi