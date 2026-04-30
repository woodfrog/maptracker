#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_b1_b2.sh --mode MODE --base-config CFG --base-checkpoint CKPT --work-root ROOT [options]

Required:
  --mode b1_only|b2_only|both_parallel|both_sequential
  --base-config PATH
  --base-checkpoint PATH
  --work-root DIR

Optional:
  --run-id ID
  --seed N                           (default: 0)
  --launcher none|pytorch|slurm|slurm-step  (default: none)
  --partition NAME                   (required for launcher=slurm)
  --train-gpus N                     (default: 1)
  --eval-gpus N                      (default: train-gpus)
  --gpus-per-node N                  (default: train-gpus)
  --cpus-per-task N                  (default: 5)
  --available-gpus N                 (for auto fallback in both_parallel)
  --resume                           (reuse an existing run dir and auto-resume train)
  --skip-train-validation            (pass --no-validate to training)
  --skip-final-eval                  (skip wrapper's post-train clean eval)

  --cfg-options-common "k=v ..."     (applied to both)
  --b1-cfg-options "k=v ..."
  --b2-cfg-options "k=v ..."
  --eval-cfg-options "k=v ..."
  --run-clean-cmap                  (after clean eval, run prepare_pred_tracks + calculate_cmap)
  --cons-frames N                   (default: 5 for clean C-mAP path)

  --run-contradiction-suite
  --suite-modes "c_full c_tail"      (default: c_full c_tail)
  --suite-stale-offsets "4 8"        (default: 4 8)
  --suite-onset N                     (required if suite enabled)
  --suite-c-tail-keep-recent N        (default: 1)

  --dry-run
  -h, --help
USAGE
}

MODE=""
BASE_CONFIG=""
BASE_CHECKPOINT=""
WORK_ROOT=""
RUN_ID=""
SEED=0
LAUNCHER="none"
PARTITION=""
TRAIN_GPUS=1
EVAL_GPUS=""
GPUS_PER_NODE=""
CPUS_PER_TASK=5
AVAILABLE_GPUS=0

CFG_COMMON_STR=""
B1_CFG_STR=""
B2_CFG_STR=""
EVAL_CFG_STR=""

RUN_SUITE=0
SUITE_MODES_STR="c_full c_tail"
SUITE_STALE_OFFSETS_STR="4 8"
SUITE_ONSET=""
SUITE_C_TAIL_KEEP_RECENT=1
RUN_CLEAN_CMAP=0
CONS_FRAMES=5

DRY_RUN=0
RESUME=0
SKIP_TRAIN_VALIDATION=0
SKIP_FINAL_EVAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --base-config) BASE_CONFIG="$2"; shift 2 ;;
    --base-checkpoint) BASE_CHECKPOINT="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --launcher) LAUNCHER="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --train-gpus) TRAIN_GPUS="$2"; shift 2 ;;
    --eval-gpus) EVAL_GPUS="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
    --available-gpus) AVAILABLE_GPUS="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --skip-train-validation) SKIP_TRAIN_VALIDATION=1; shift ;;
    --skip-final-eval) SKIP_FINAL_EVAL=1; shift ;;
    --cfg-options-common) CFG_COMMON_STR="$2"; shift 2 ;;
    --b1-cfg-options) B1_CFG_STR="$2"; shift 2 ;;
    --b2-cfg-options) B2_CFG_STR="$2"; shift 2 ;;
    --eval-cfg-options) EVAL_CFG_STR="$2"; shift 2 ;;
    --run-clean-cmap) RUN_CLEAN_CMAP=1; shift ;;
    --cons-frames) CONS_FRAMES="$2"; shift 2 ;;
    --run-contradiction-suite) RUN_SUITE=1; shift ;;
    --suite-modes) SUITE_MODES_STR="$2"; shift 2 ;;
    --suite-stale-offsets) SUITE_STALE_OFFSETS_STR="$2"; shift 2 ;;
    --suite-onset) SUITE_ONSET="$2"; shift 2 ;;
    --suite-c-tail-keep-recent) SUITE_C_TAIL_KEEP_RECENT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[run_b1_b2] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MODE" || -z "$BASE_CONFIG" || -z "$BASE_CHECKPOINT" || -z "$WORK_ROOT" ]]; then
  echo "[run_b1_b2] Missing required args" >&2
  usage
  exit 2
fi

if [[ "$LAUNCHER" == "slurm" && -z "$PARTITION" ]]; then
  echo "[run_b1_b2] --partition required with --launcher slurm" >&2
  exit 2
fi
if [[ -z "$EVAL_GPUS" ]]; then
  EVAL_GPUS="$TRAIN_GPUS"
fi
if [[ -z "$GPUS_PER_NODE" ]]; then
  GPUS_PER_NODE="$TRAIN_GPUS"
fi
if [[ $RUN_SUITE -eq 1 && -z "$SUITE_ONSET" ]]; then
  echo "[run_b1_b2] --suite-onset is required with --run-contradiction-suite" >&2
  exit 2
fi
if [[ $SKIP_FINAL_EVAL -eq 1 && $RUN_CLEAN_CMAP -eq 1 ]]; then
  echo "[run_b1_b2] --skip-final-eval cannot be combined with --run-clean-cmap" >&2
  exit 2
fi
if [[ $SKIP_FINAL_EVAL -eq 1 && $RUN_SUITE -eq 1 ]]; then
  echo "[run_b1_b2] --skip-final-eval cannot be combined with --run-contradiction-suite" >&2
  exit 2
fi

case "$MODE" in
  b1_only|b2_only|both_parallel|both_sequential) ;;
  *) echo "[run_b1_b2] Invalid mode: $MODE" >&2; exit 2 ;;
esac

GIT_SHA="$(git rev-parse --short HEAD)"
TS="$(date -u +%Y%m%d_%H%M%S)"
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="${TS}_${GIT_SHA}_b1b2"
fi
BASE_DIR="${WORK_ROOT%/}/experiments/b1_b2/${RUN_ID}"
LOG_DIR="$BASE_DIR/logs"
MANIFEST_DIR="$BASE_DIR/manifests"
LOCK_FILE="$BASE_DIR/.lock"
MAIN_LOG="$LOG_DIR/run_b1_b2.log"

mkdir -p "$WORK_ROOT"
if [[ $RESUME -eq 0 && -e "$BASE_DIR" ]]; then
  if [[ -n "$(find "$BASE_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)" ]]; then
    echo "[run_b1_b2] Refusing to overwrite non-empty run dir: $BASE_DIR" >&2
    exit 1
  fi
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

declare -a CFG_COMMON_ARR=() B1_CFG_ARR=() B2_CFG_ARR=() EVAL_CFG_ARR=() SUITE_MODES_ARR=() SUITE_OFFSETS_ARR=()
if [[ -n "$CFG_COMMON_STR" ]]; then read -r -a CFG_COMMON_ARR <<< "$CFG_COMMON_STR"; fi
if [[ -n "$B1_CFG_STR" ]]; then read -r -a B1_CFG_ARR <<< "$B1_CFG_STR"; fi
if [[ -n "$B2_CFG_STR" ]]; then read -r -a B2_CFG_ARR <<< "$B2_CFG_STR"; fi
if [[ -n "$EVAL_CFG_STR" ]]; then read -r -a EVAL_CFG_ARR <<< "$EVAL_CFG_STR"; fi
if [[ -n "$SUITE_MODES_STR" ]]; then read -r -a SUITE_MODES_ARR <<< "$SUITE_MODES_STR"; fi
if [[ -n "$SUITE_STALE_OFFSETS_STR" ]]; then read -r -a SUITE_OFFSETS_ARR <<< "$SUITE_STALE_OFFSETS_STR"; fi

# parity safety: parity-controlled keys must be in common args, not per-baseline.
for key in seed runner.max_epochs data.samples_per_gpu data.workers_per_gpu model.mvp_temporal_gate_cfg.eval_corruption_cfg.memory_corruption_mode model.mvp_temporal_gate_cfg.eval_corruption_cfg.memory_stale_offset model.mvp_temporal_gate_cfg.eval_corruption_cfg.memory_c_tail_keep_recent model.mvp_temporal_gate_cfg.eval_corruption_cfg.memory_corruption_onset; do
  if printf '%s\n' "${B1_CFG_ARR[@]}" | grep -q "^${key}=" || printf '%s\n' "${B2_CFG_ARR[@]}" | grep -q "^${key}="; then
    echo "[run_b1_b2] Parity key '$key' must not be baseline-specific. Use --cfg-options-common." >&2
    exit 1
  fi
done

VERIFY_CMD=(bash tools/experiments/verify_assets.sh
  --config "$BASE_CONFIG"
  --checkpoint "$BASE_CHECKPOINT"
  --work-root "$WORK_ROOT"
  --run-dir "$BASE_DIR")
if [[ $RESUME -eq 1 ]]; then
  VERIFY_CMD+=(--allow-nonempty-run-dir)
fi
if [[ $DRY_RUN -eq 1 ]]; then
  VERIFY_CMD+=(--dry-run)
fi
echo "[cmd] ${VERIFY_CMD[*]}"
"${VERIFY_CMD[@]}"

if [[ $DRY_RUN -eq 0 ]]; then
  mkdir -p "$LOG_DIR" "$MANIFEST_DIR"
  if [[ -e "$LOCK_FILE" ]]; then
    lock_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
    if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
      echo "[run_b1_b2] Lock exists and is active: $LOCK_FILE (pid=$lock_pid)" >&2
      exit 1
    fi
    log_message "$MAIN_LOG" "[run_b1_b2] Removing stale lock: $LOCK_FILE"
    rm -f "$LOCK_FILE"
  fi
  echo "$$" > "$LOCK_FILE"
  trap 'rm -f "$LOCK_FILE"' EXIT
fi

run_one() {
  local baseline="$1"
  local train_dir="$BASE_DIR/${baseline}/train"
  local eval_dir="$BASE_DIR/${baseline}/eval_clean"
  local suite_dir="$BASE_DIR/${baseline}/contradiction_suite"
  local b_log="$LOG_DIR/${baseline}.log"
  local ckpt_path="$BASE_CHECKPOINT"
  local final_ckpt=""

  if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$train_dir" "$eval_dir"
  fi

  local -a baseline_cfg=("${CFG_COMMON_ARR[@]}")
  if [[ "$baseline" == "b1" ]]; then
    baseline_cfg+=("${B1_CFG_ARR[@]}")
  else
    baseline_cfg+=("${B2_CFG_ARR[@]}")
  fi

  # force deterministic seed in cfg and CLI for reproducibility
  baseline_cfg+=("seed=${SEED}")
  baseline_cfg+=("auto_resume=True")
  if ! printf '%s\n' "${baseline_cfg[@]}" | grep -q '^load_from='; then
    baseline_cfg+=("load_from=${BASE_CHECKPOINT}")
  fi

  local -a train_cmd
  if [[ "$LAUNCHER" == "slurm" ]]; then
    train_cmd=(env GPUS="$TRAIN_GPUS" GPUS_PER_NODE="$GPUS_PER_NODE" CPUS_PER_TASK="$CPUS_PER_TASK" bash tools/slurm_train.sh "$PARTITION" "${baseline}_train_${RUN_ID}" "$BASE_CONFIG" "$train_dir" --seed "$SEED")
    if [[ ${#baseline_cfg[@]} -gt 0 ]]; then
      train_cmd+=(--cfg-options "${baseline_cfg[@]}")
    fi
  elif [[ "$LAUNCHER" == "slurm-step" ]]; then
    train_cmd=(env GPUS="$TRAIN_GPUS" GPUS_PER_NODE="$GPUS_PER_NODE" CPUS_PER_TASK="$CPUS_PER_TASK" bash tools/slurm_train_step.sh "$BASE_CONFIG" "$train_dir" --seed "$SEED")
    if [[ ${#baseline_cfg[@]} -gt 0 ]]; then
      train_cmd+=(--cfg-options "${baseline_cfg[@]}")
    fi
  elif [[ "$LAUNCHER" == "pytorch" ]]; then
    train_cmd=(bash tools/dist_train.sh "$BASE_CONFIG" "$TRAIN_GPUS" --work-dir "$train_dir" --seed "$SEED")
    if [[ ${#baseline_cfg[@]} -gt 0 ]]; then
      train_cmd+=(--cfg-options "${baseline_cfg[@]}")
    fi
  else
    train_cmd=(python tools/train.py "$BASE_CONFIG" --work-dir "$train_dir" --seed "$SEED" --launcher none)
    if [[ ${#baseline_cfg[@]} -gt 0 ]]; then
      train_cmd+=(--cfg-options "${baseline_cfg[@]}")
    fi
  fi
  if [[ $SKIP_TRAIN_VALIDATION -eq 1 ]]; then
    train_cmd+=(--no-validate)
  fi

  log_message "$b_log" "[${baseline}] train start"
  run_or_print "${train_cmd[@]}"

  # Safe checkpoint intake after train: prefer latest.pth then iter_*.pth
  if [[ -f "$train_dir/latest.pth" ]]; then
    final_ckpt="$train_dir/latest.pth"
  elif compgen -G "$train_dir/iter_*.pth" > /dev/null; then
    final_ckpt="$(ls -1 "$train_dir"/iter_*.pth | sort -V | tail -n 1)"
  else
    if [[ $SKIP_FINAL_EVAL -eq 0 || $RUN_CLEAN_CMAP -eq 1 || $RUN_SUITE -eq 1 ]]; then
      echo "[${baseline}] ERROR: no training checkpoint found in $train_dir" >&2
      exit 1
    fi
    final_ckpt=""
    log_message "$b_log" "[${baseline}] WARNING: no train checkpoint found in $train_dir"
  fi

  if [[ $SKIP_FINAL_EVAL -eq 0 ]]; then
    local -a eval_cmd
    if [[ "$LAUNCHER" == "slurm" ]]; then
      eval_cmd=(env GPUS="$EVAL_GPUS" GPUS_PER_NODE="$GPUS_PER_NODE" CPUS_PER_TASK="$CPUS_PER_TASK" bash tools/slurm_test.sh "$PARTITION" "${baseline}_eval_${RUN_ID}" "$BASE_CONFIG" "$final_ckpt" --work-dir "$eval_dir" --eval --seed "$SEED" --condition-tag "${baseline}_clean")
      if [[ ${#EVAL_CFG_ARR[@]} -gt 0 ]]; then
        eval_cmd+=(--cfg-options "${EVAL_CFG_ARR[@]}")
      fi
    elif [[ "$LAUNCHER" == "slurm-step" ]]; then
      eval_cmd=(env GPUS="$EVAL_GPUS" GPUS_PER_NODE="$GPUS_PER_NODE" CPUS_PER_TASK="$CPUS_PER_TASK" bash tools/slurm_test_step.sh "$BASE_CONFIG" "$final_ckpt" --work-dir "$eval_dir" --eval --seed "$SEED" --condition-tag "${baseline}_clean")
      if [[ ${#EVAL_CFG_ARR[@]} -gt 0 ]]; then
        eval_cmd+=(--cfg-options "${EVAL_CFG_ARR[@]}")
      fi
    elif [[ "$LAUNCHER" == "pytorch" ]]; then
      eval_cmd=(bash tools/dist_test.sh "$BASE_CONFIG" "$final_ckpt" "$EVAL_GPUS" --work-dir "$eval_dir" --eval --seed "$SEED" --condition-tag "${baseline}_clean")
      if [[ ${#EVAL_CFG_ARR[@]} -gt 0 ]]; then
        eval_cmd+=(--cfg-options "${EVAL_CFG_ARR[@]}")
      fi
    else
      eval_cmd=(python tools/test.py "$BASE_CONFIG" "$final_ckpt" --work-dir "$eval_dir" --eval --seed "$SEED" --condition-tag "${baseline}_clean" --launcher none)
      if [[ ${#EVAL_CFG_ARR[@]} -gt 0 ]]; then
        eval_cmd+=(--cfg-options "${EVAL_CFG_ARR[@]}")
      fi
    fi

    log_message "$b_log" "[${baseline}] eval start"
    run_or_print "${eval_cmd[@]}"
  else
    log_message "$b_log" "[${baseline}] final eval skipped"
  fi

  if [[ $RUN_CLEAN_CMAP -eq 1 ]]; then
    local pred_json="$eval_dir/submission_vector.json"
    local match_pkl="$eval_dir/pos_predictions_${CONS_FRAMES}.pkl"
    if [[ $DRY_RUN -eq 0 && ! -f "$pred_json" ]]; then
      echo "[${baseline}] ERROR: expected clean eval output missing: $pred_json" | tee -a "$b_log"
      exit 1
    fi
    log_message "$b_log" "[${baseline}] clean C-mAP start"
    run_or_print python tools/tracking/prepare_pred_tracks.py "$BASE_CONFIG" --result_path "$pred_json" --cons_frames "$CONS_FRAMES"
    run_or_print python tools/tracking/calculate_cmap.py "$BASE_CONFIG" --result_path "$match_pkl" --cons_frames "$CONS_FRAMES"
  fi

  if [[ $RUN_SUITE -eq 1 ]]; then
    if [[ $DRY_RUN -eq 0 ]]; then
      mkdir -p "$suite_dir"
    fi
    local -a suite_cmd=(python tools/tracking/run_contradiction_suite.py "$BASE_CONFIG" "$final_ckpt" --work-root "$suite_dir" --onset "$SUITE_ONSET" --c-tail-keep-recent "$SUITE_C_TAIL_KEEP_RECENT" --launcher "$LAUNCHER" --gpus "$EVAL_GPUS")
    if [[ ${#SUITE_MODES_ARR[@]} -gt 0 ]]; then
      suite_cmd+=(--modes "${SUITE_MODES_ARR[@]}")
    fi
    if [[ ${#SUITE_OFFSETS_ARR[@]} -gt 0 ]]; then
      suite_cmd+=(--stale-offsets "${SUITE_OFFSETS_ARR[@]}")
    fi
    if [[ ${#EVAL_CFG_ARR[@]} -gt 0 ]]; then
      suite_cmd+=(--extra-cfg-options "${EVAL_CFG_ARR[@]}")
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
      suite_cmd+=(--dry-run)
    fi
    log_message "$b_log" "[${baseline}] contradiction suite start"
    run_or_print "${suite_cmd[@]}"
  fi

  if [[ $DRY_RUN -eq 0 ]]; then
    local effective_json="$BASE_DIR/${baseline}/effective_args.json"
    cat > "$effective_json" <<JSON
{
  "baseline": "${baseline}",
  "seed": ${SEED},
  "train_gpus": ${TRAIN_GPUS},
  "eval_gpus": ${EVAL_GPUS},
  "launcher": "${LAUNCHER}",
  "base_config": "${BASE_CONFIG}",
  "base_checkpoint": "${BASE_CHECKPOINT}",
  "final_checkpoint": "${final_ckpt}",
  "skip_train_validation": ${SKIP_TRAIN_VALIDATION},
  "skip_final_eval": ${SKIP_FINAL_EVAL},
  "cfg_options_common": "${CFG_COMMON_STR}",
  "cfg_options_baseline": "$( [[ "$baseline" == "b1" ]] && echo "$B1_CFG_STR" || echo "$B2_CFG_STR" )",
  "eval_cfg_options": "${EVAL_CFG_STR}",
  "run_suite": ${RUN_SUITE},
  "run_clean_cmap": ${RUN_CLEAN_CMAP},
  "cons_frames": ${CONS_FRAMES}
}
JSON
  fi
}

TARGETS=()
if [[ "$MODE" == "b1_only" ]]; then
  TARGETS=(b1)
elif [[ "$MODE" == "b2_only" ]]; then
  TARGETS=(b2)
else
  TARGETS=(b1 b2)
fi

EFFECTIVE_MODE="$MODE"
if [[ "$MODE" == "both_parallel" && "$AVAILABLE_GPUS" -gt 0 ]]; then
  needed=$(( TRAIN_GPUS * 2 ))
  if [[ "$AVAILABLE_GPUS" -lt "$needed" ]]; then
    log_message "$MAIN_LOG" "[run_b1_b2] Insufficient available-gpus=${AVAILABLE_GPUS} for parallel need=${needed}; falling back to sequential."
    EFFECTIVE_MODE="both_sequential"
  fi
fi

if [[ "$EFFECTIVE_MODE" == "both_parallel" ]]; then
  run_one b1 &
  pid1=$!
  run_one b2 &
  pid2=$!
  wait "$pid1"
  wait "$pid2"
else
  for b in "${TARGETS[@]}"; do
    run_one "$b"
  done
fi

STATUS="success"
if [[ $DRY_RUN -eq 1 ]]; then
  STATUS="dry_run"
fi
if [[ $DRY_RUN -eq 0 ]]; then
  MANIFEST_JSON="$MANIFEST_DIR/manifest.json"
  INDEX_CSV="$MANIFEST_DIR/results_index.csv"
  cat > "$MANIFEST_JSON" <<JSON
{
  "run_id": "${RUN_ID}",
  "mode_requested": "${MODE}",
  "mode_effective": "${EFFECTIVE_MODE}",
  "targets": "${TARGETS[*]}",
  "base_config": "${BASE_CONFIG}",
  "base_checkpoint": "${BASE_CHECKPOINT}",
  "seed": ${SEED},
  "launcher": "${LAUNCHER}",
  "resume": ${RESUME},
  "skip_train_validation": ${SKIP_TRAIN_VALIDATION},
  "skip_final_eval": ${SKIP_FINAL_EVAL},
  "train_gpus": ${TRAIN_GPUS},
  "eval_gpus": ${EVAL_GPUS},
  "run_contradiction_suite": ${RUN_SUITE},
  "run_clean_cmap": ${RUN_CLEAN_CMAP},
  "cons_frames": ${CONS_FRAMES},
  "status": "${STATUS}",
  "log": "${MAIN_LOG}"
}
JSON

  echo "run_id,baseline,mode,seed,launcher,train_gpus,eval_gpus,status,base_dir" > "$INDEX_CSV"
  for b in "${TARGETS[@]}"; do
    echo "$RUN_ID,$b,$EFFECTIVE_MODE,$SEED,$LAUNCHER,$TRAIN_GPUS,$EVAL_GPUS,$STATUS,$BASE_DIR/$b" >> "$INDEX_CSV"
  done

  echo "[run_b1_b2] Done. Manifest: $MANIFEST_JSON"
else
  echo "[run_b1_b2] Dry run complete."
fi
