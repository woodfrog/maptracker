#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run corruption evaluation for a single baseline × mode over all stale offsets.
Calls run_b1_b2_deferred_eval.sh once per stale offset using the full eval pipeline.

Usage:
  bash tools/experiments/run_corruption_eval_b1_b2_iter89148.sh \
    --baseline b1|b2 --mode c_full|c_tail [options]

Required:
  --baseline b1|b2
  --mode c_full|c_tail

Optional:
  --work-root DIR            Work root containing experiments/ (default: /n/fs/dynamicbias/tracker/work_dirs)
  --launcher TYPE            none|pytorch|slurm|slurm-step (default: none)
  --gpus N                   Number of GPUs for launcher (default: 4)
  --seed N                   Random seed (default: 0)
  --stale-offsets "1 2 3"    Space-separated list of stale offsets (default: 1 2 3)
  --keep-recent N            c_tail: number of recent frames to keep uncorrupted (default: 1)
  --onset N                  Corruption onset frame index (default: 0)
  --skip-cmap                Skip C-mAP post-processing
  --rerun                    Re-run even if eval dir already exists and is non-empty
  --eval-options "k=v ..."   Extra eval options passed through
  --cfg-options "k=v ..."    Extra cfg overrides passed through
  --dry-run                  Print commands without executing
  -h, --help                 Show this help

Defaults are wired to:
  b1: <work-root>/experiments/b1_b2/b1_stage3_gpu4_short_trainonly/b1/train/iter_89148.pth
  b2: <work-root>/experiments/b1_b2/b2_stage3_gpu4_short_trainonly/b2/train/iter_89148.pth

Condition tags are auto-derived from corruption parameters (e.g. cfull_onset0_stale1,
ctail_onset0_keep1_stale2) and match the derivation in run_b1_b2_deferred_eval.sh.

Existing non-empty eval dirs are skipped (not overwritten) unless --rerun is given.
USAGE
}

BASELINE=""
MODE=""
WORK_ROOT="/n/fs/dynamicbias/tracker/work_dirs"
LAUNCHER="none"
GPUS=4
SEED=0
STALE_OFFSETS_STR="1 2 3"
KEEP_RECENT=1
ONSET=0
SKIP_CMAP=0
RERUN=0
EVAL_OPTIONS_STR=""
CFG_OPTIONS_STR=""
DRY_RUN=0

B1_RUN_ID="b1_stage3_gpu4_short_trainonly"
B2_RUN_ID="b2_stage3_gpu4_short_trainonly"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline) BASELINE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --launcher) LAUNCHER="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --stale-offsets) STALE_OFFSETS_STR="$2"; shift 2 ;;
    --keep-recent) KEEP_RECENT="$2"; shift 2 ;;
    --onset) ONSET="$2"; shift 2 ;;
    --skip-cmap) SKIP_CMAP=1; shift ;;
    --rerun) RERUN=1; shift ;;
    --eval-options) EVAL_OPTIONS_STR="$2"; shift 2 ;;
    --cfg-options) CFG_OPTIONS_STR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[run_corruption_eval_b1_b2_iter89148] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

# Validate required args
if [[ -z "$BASELINE" || -z "$MODE" ]]; then
  echo "[run_corruption_eval_b1_b2_iter89148] --baseline and --mode are required." >&2
  usage
  exit 2
fi

case "$BASELINE" in
  b1|b2) ;;
  *) echo "[run_corruption_eval_b1_b2_iter89148] Invalid --baseline: $BASELINE (must be b1 or b2)" >&2; exit 2 ;;
esac

case "$MODE" in
  c_full|c_tail) ;;
  *) echo "[run_corruption_eval_b1_b2_iter89148] Invalid --mode: $MODE (must be c_full or c_tail)" >&2; exit 2 ;;
esac

case "$LAUNCHER" in
  none|pytorch|slurm|slurm-step) ;;
  *) echo "[run_corruption_eval_b1_b2_iter89148] Invalid --launcher: $LAUNCHER" >&2; exit 2 ;;
esac

if ! [[ "$GPUS" =~ ^[0-9]+$ ]] || (( GPUS <= 0 )); then
  echo "[run_corruption_eval_b1_b2_iter89148] --gpus must be a positive integer." >&2
  exit 2
fi

if ! [[ "$ONSET" =~ ^[0-9]+$ ]]; then
  echo "[run_corruption_eval_b1_b2_iter89148] --onset must be a non-negative integer." >&2
  exit 2
fi

if ! [[ "$KEEP_RECENT" =~ ^[0-9]+$ ]]; then
  echo "[run_corruption_eval_b1_b2_iter89148] --keep-recent must be a non-negative integer." >&2
  exit 2
fi

read -r -a STALE_OFFSETS_ARR <<< "$STALE_OFFSETS_STR"
if [[ ${#STALE_OFFSETS_ARR[@]} -eq 0 ]]; then
  echo "[run_corruption_eval_b1_b2_iter89148] --stale-offsets must contain at least one value." >&2
  exit 2
fi
for off in "${STALE_OFFSETS_ARR[@]}"; do
  if ! [[ "$off" =~ ^[0-9]+$ ]] || (( off < 0 )); then
    echo "[run_corruption_eval_b1_b2_iter89148] --stale-offsets element '$off' must be a non-negative integer." >&2
    exit 2
  fi
done

# Resolve hardwired paths
if [[ "$BASELINE" == "b1" ]]; then
  RUN_ID="$B1_RUN_ID"
else
  RUN_ID="$B2_RUN_ID"
fi

BASE_CONFIG="${WORK_ROOT%/}/experiments/b1_b2/${RUN_ID}/${BASELINE}/train/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py"
CHECKPOINT="${WORK_ROOT%/}/experiments/b1_b2/${RUN_ID}/${BASELINE}/train/iter_89148.pth"

if [[ $DRY_RUN -eq 0 ]]; then
  for f in "$BASE_CONFIG" "$CHECKPOINT"; do
    if [[ ! -f "$f" ]]; then
      echo "[run_corruption_eval_b1_b2_iter89148] Missing required file: $f" >&2
      exit 1
    fi
  done
else
  echo "[run_corruption_eval_b1_b2_iter89148] Dry run: skipping file existence checks for:"
  echo "  config:     $BASE_CONFIG"
  echo "  checkpoint: $CHECKPOINT"
fi

# Replicate normalize_tag + derive_condition_tag from run_b1_b2_deferred_eval.sh
# Used to pre-compute eval dir path for skip detection.
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

CHECKPOINT_TAG="iter_89148"

SKIPPED=()
RAN=()

echo "[run_corruption_eval_b1_b2_iter89148] Starting corruption eval: baseline=${BASELINE} mode=${MODE} offsets=(${STALE_OFFSETS_STR}) onset=${ONSET}"
if [[ "$MODE" == "c_tail" ]]; then
  echo "[run_corruption_eval_b1_b2_iter89148]   keep_recent=${KEEP_RECENT}"
fi

for STALE_OFFSET in "${STALE_OFFSETS_ARR[@]}"; do
  CONDITION_TAG="$(derive_condition_tag "$MODE" "$ONSET" "$STALE_OFFSET" "$KEEP_RECENT")"
  EVAL_DIR="${WORK_ROOT%/}/experiments/b1_b2/${RUN_ID}/${BASELINE}/eval_deferred/${CHECKPOINT_TAG}/${CONDITION_TAG}"

  # Pre-flight skip check: if eval dir is non-empty and --rerun is not set, skip
  if [[ $RERUN -eq 0 && -e "$EVAL_DIR" ]]; then
    if [[ -n "$(find "$EVAL_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)" ]]; then
      echo "[run_corruption_eval_b1_b2_iter89148] Skipping stale_offset=${STALE_OFFSET}: eval dir already exists and is non-empty: $EVAL_DIR"
      SKIPPED+=("$STALE_OFFSET")
      continue
    fi
  fi

  echo "[run_corruption_eval_b1_b2_iter89148] Running stale_offset=${STALE_OFFSET} (condition_tag=${CONDITION_TAG})"

  RUN_ARGS=(
    --base-config "$BASE_CONFIG"
    --checkpoint "$CHECKPOINT"
    --work-root "$WORK_ROOT"
    --run-id "$RUN_ID"
    --baseline "$BASELINE"
    --checkpoint-tag "$CHECKPOINT_TAG"
    --launcher "$LAUNCHER"
    --gpus "$GPUS"
    --seed "$SEED"
    --memory-corruption-mode "$MODE"
    --memory-stale-offset "$STALE_OFFSET"
    --memory-corruption-onset "$ONSET"
  )

  if [[ "$MODE" == "c_tail" ]]; then
    RUN_ARGS+=(--memory-c-tail-keep-recent "$KEEP_RECENT")
  fi

  if [[ $SKIP_CMAP -eq 1 ]]; then
    RUN_ARGS+=(--skip-cmap)
  fi

  if [[ $RERUN -eq 1 ]]; then
    RUN_ARGS+=(--rerun)
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    RUN_ARGS+=(--dry-run)
  fi

  if [[ -n "$CFG_OPTIONS_STR" ]]; then
    RUN_ARGS+=(--cfg-options "$CFG_OPTIONS_STR")
  fi

  if [[ -n "$EVAL_OPTIONS_STR" ]]; then
    RUN_ARGS+=(--eval-options "$EVAL_OPTIONS_STR")
  fi

  bash tools/experiments/run_b1_b2_deferred_eval.sh "${RUN_ARGS[@]}"

  RAN+=("$STALE_OFFSET")
done

echo ""
echo "[run_corruption_eval_b1_b2_iter89148] Summary for baseline=${BASELINE} mode=${MODE}:"
if [[ ${#RAN[@]} -gt 0 ]]; then
  echo "  Ran:     ${RAN[*]}"
fi
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
  echo "  Skipped: ${SKIPPED[*]} (eval dirs already exist; use --rerun to overwrite)"
fi
if [[ ${#RAN[@]} -eq 0 && ${#SKIPPED[@]} -eq 0 ]]; then
  echo "  No offsets processed."
fi
