#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: submit_b1_b2_deferred_eval_sweep.sh --base-config CFG --work-root ROOT --run-id ID --baseline BASELINE --time LIMIT --mail-user EMAIL [options]

Required:
  --base-config PATH
  --work-root DIR
  --run-id ID
  --baseline b1|b2
  --time LIMIT
  --mail-user EMAIL

Optional:
  --train-dir DIR                  (default: work-root/experiments/b1_b2/run-id/baseline/train)
  --checkpoint-glob GLOB           (default: iter_*.pth)
  --checkpoints "A B C"           Explicit checkpoint filenames or paths; overrides --checkpoint-glob.
  --include-latest                 Include latest.pth after iter checkpoints.
  --condition-tag TAG
  --rerun                          Reuse and replace existing eval dirs.
  --job-name NAME                  (default: maptracker_b1_b2_eval)
  --partition NAME
  --qos NAME                       (default: gpu-short)
  --account NAME
  --constraint EXPR                (default: nomig&gpu40)
  --mem VALUE
  --mail-type VALUE                (default: END,FAIL)
  --dependency SPEC
  --eval-gpus N                    (default: 2)
  --gpus-per-node N                (default: eval-gpus)
  --cpus-per-task N                (default: 4)
  --seed N                         (default: 0)
  --module-load MOD                (default: anaconda3/2023.9)
  --conda-env NAME                 (default: maptracker)
  --srun-args "ARGS"
  --sbatch-extra-args "ARGS"
  --cfg-options "k=v ..."
  --eval-options "k=v ..."
  --cons-frames N                  (default: 5)
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
WORK_ROOT=""
RUN_ID=""
BASELINE=""
TIME_LIMIT=""
MAIL_USER=""
TRAIN_DIR=""
CHECKPOINT_GLOB="iter_*.pth"
CHECKPOINTS_STR=""
INCLUDE_LATEST=0
CONDITION_TAG=""
JOB_NAME="maptracker_b1_b2_eval"
PARTITION=""
QOS="gpu-short"
ACCOUNT=""
CONSTRAINT="nomig&gpu40"
MEMORY=""
MAIL_TYPE="END,FAIL"
DEPENDENCY=""
EVAL_GPUS=2
GPUS_PER_NODE=""
CPUS_PER_TASK=4
SEED=0
MODULE_LOAD="anaconda3/2023.9"
CONDA_ENV="maptracker"
SRUN_ARGS_STR=""
SBATCH_EXTRA_ARGS_STR=""
CFG_OPTIONS_STR=""
EVAL_OPTIONS_STR=""
CONS_FRAMES=5
SKIP_CMAP=0
MEMORY_CORRUPTION_MODE=""
MEMORY_STALE_OFFSET=""
MEMORY_C_TAIL_KEEP_RECENT=""
MEMORY_CORRUPTION_ONSET=""
RERUN=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-config) BASE_CONFIG="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --mail-user) MAIL_USER="$2"; shift 2 ;;
    --train-dir) TRAIN_DIR="$2"; shift 2 ;;
    --checkpoint-glob) CHECKPOINT_GLOB="$2"; shift 2 ;;
    --checkpoints) CHECKPOINTS_STR="$2"; shift 2 ;;
    --include-latest) INCLUDE_LATEST=1; shift ;;
    --condition-tag) CONDITION_TAG="$2"; shift 2 ;;
    --rerun) RERUN=1; shift ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --constraint) CONSTRAINT="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --mail-type) MAIL_TYPE="$2"; shift 2 ;;
    --dependency) DEPENDENCY="$2"; shift 2 ;;
    --eval-gpus) EVAL_GPUS="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --module-load) MODULE_LOAD="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --srun-args) SRUN_ARGS_STR="$2"; shift 2 ;;
    --sbatch-extra-args) SBATCH_EXTRA_ARGS_STR="$2"; shift 2 ;;
    --cfg-options) CFG_OPTIONS_STR="$2"; shift 2 ;;
    --eval-options) EVAL_OPTIONS_STR="$2"; shift 2 ;;
    --cons-frames) CONS_FRAMES="$2"; shift 2 ;;
    --skip-cmap) SKIP_CMAP=1; shift ;;
    --memory-corruption-mode) MEMORY_CORRUPTION_MODE="$2"; shift 2 ;;
    --memory-stale-offset) MEMORY_STALE_OFFSET="$2"; shift 2 ;;
    --memory-c-tail-keep-recent) MEMORY_C_TAIL_KEEP_RECENT="$2"; shift 2 ;;
    --memory-corruption-onset) MEMORY_CORRUPTION_ONSET="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[submit_b1_b2_deferred_eval_sweep] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$BASE_CONFIG" || -z "$WORK_ROOT" || -z "$RUN_ID" || -z "$BASELINE" || -z "$TIME_LIMIT" || -z "$MAIL_USER" ]]; then
  echo "[submit_b1_b2_deferred_eval_sweep] Missing required arguments." >&2
  usage
  exit 2
fi

case "$BASELINE" in
  b1|b2) ;;
  *) echo "[submit_b1_b2_deferred_eval_sweep] Invalid baseline: $BASELINE" >&2; exit 2 ;;
esac

if [[ -z "$GPUS_PER_NODE" ]]; then
  GPUS_PER_NODE="$EVAL_GPUS"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "$TRAIN_DIR" ]]; then
  TRAIN_DIR="${WORK_ROOT%/}/experiments/b1_b2/${RUN_ID}/${BASELINE}/train"
fi

if [[ ! -d "$TRAIN_DIR" ]]; then
  echo "[submit_b1_b2_deferred_eval_sweep] Training directory not found: $TRAIN_DIR" >&2
  exit 1
fi

resolve_checkpoint_path() {
  local candidate="$1"
  if [[ "$candidate" == /* ]]; then
    printf '%s\n' "$candidate"
  else
    printf '%s\n' "$TRAIN_DIR/$candidate"
  fi
}

declare -a CHECKPOINTS=()
declare -A SEEN=()

add_checkpoint() {
  local checkpoint="$1"
  if [[ ! -f "$checkpoint" ]]; then
    echo "[submit_b1_b2_deferred_eval_sweep] Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  if [[ -z "${SEEN[$checkpoint]:-}" ]]; then
    CHECKPOINTS+=("$checkpoint")
    SEEN[$checkpoint]=1
  fi
}

if [[ -n "$CHECKPOINTS_STR" ]]; then
  read -r -a EXPLICIT_ARR <<< "$CHECKPOINTS_STR"
  for checkpoint in "${EXPLICIT_ARR[@]}"; do
    add_checkpoint "$(resolve_checkpoint_path "$checkpoint")"
  done
else
  while IFS= read -r checkpoint; do
    add_checkpoint "$checkpoint"
  done < <(find "$TRAIN_DIR" -maxdepth 1 -type f -name "$CHECKPOINT_GLOB" | sort -V)
fi

if [[ $INCLUDE_LATEST -eq 1 && -f "$TRAIN_DIR/latest.pth" ]]; then
  add_checkpoint "$TRAIN_DIR/latest.pth"
fi

if (( ${#CHECKPOINTS[@]} == 0 )); then
  echo "[submit_b1_b2_deferred_eval_sweep] No checkpoints selected." >&2
  exit 1
fi

for checkpoint in "${CHECKPOINTS[@]}"; do
  CMD=(bash tools/experiments/submit_b1_b2_deferred_eval.sh
    --base-config "$BASE_CONFIG"
    --checkpoint "$checkpoint"
    --work-root "$WORK_ROOT"
    --run-id "$RUN_ID"
    --baseline "$BASELINE"
    --time "$TIME_LIMIT"
    --mail-user "$MAIL_USER"
    --job-name "$JOB_NAME"
    --qos "$QOS"
    --eval-gpus "$EVAL_GPUS"
    --gpus-per-node "$GPUS_PER_NODE"
    --cpus-per-task "$CPUS_PER_TASK"
    --seed "$SEED"
    --module-load "$MODULE_LOAD"
    --conda-env "$CONDA_ENV")

  if [[ -n "$PARTITION" ]]; then
    CMD+=(--partition "$PARTITION")
  fi
  if [[ -n "$ACCOUNT" ]]; then
    CMD+=(--account "$ACCOUNT")
  fi
  if [[ -n "$CONSTRAINT" ]]; then
    CMD+=(--constraint "$CONSTRAINT")
  fi
  if [[ -n "$MEMORY" ]]; then
    CMD+=(--mem "$MEMORY")
  fi
  if [[ -n "$MAIL_TYPE" ]]; then
    CMD+=(--mail-type "$MAIL_TYPE")
  fi
  if [[ -n "$DEPENDENCY" ]]; then
    CMD+=(--dependency "$DEPENDENCY")
  fi
  if [[ -n "$SRUN_ARGS_STR" ]]; then
    CMD+=(--srun-args "$SRUN_ARGS_STR")
  fi
  if [[ -n "$SBATCH_EXTRA_ARGS_STR" ]]; then
    CMD+=(--sbatch-extra-args "$SBATCH_EXTRA_ARGS_STR")
  fi
  if [[ -n "$CFG_OPTIONS_STR" ]]; then
    CMD+=(--cfg-options "$CFG_OPTIONS_STR")
  fi
  if [[ -n "$EVAL_OPTIONS_STR" ]]; then
    CMD+=(--eval-options "$EVAL_OPTIONS_STR")
  fi
  if [[ -n "$CONDITION_TAG" ]]; then
    CMD+=(--condition-tag "$CONDITION_TAG")
  fi
  if [[ $RERUN -eq 1 ]]; then
    CMD+=(--rerun)
  fi
  if [[ "$CONS_FRAMES" != "5" ]]; then
    CMD+=(--cons-frames "$CONS_FRAMES")
  fi
  if [[ $SKIP_CMAP -eq 1 ]]; then
    CMD+=(--skip-cmap)
  fi
  if [[ -n "$MEMORY_CORRUPTION_MODE" ]]; then
    CMD+=(--memory-corruption-mode "$MEMORY_CORRUPTION_MODE")
  fi
  if [[ -n "$MEMORY_STALE_OFFSET" ]]; then
    CMD+=(--memory-stale-offset "$MEMORY_STALE_OFFSET")
  fi
  if [[ -n "$MEMORY_C_TAIL_KEEP_RECENT" ]]; then
    CMD+=(--memory-c-tail-keep-recent "$MEMORY_C_TAIL_KEEP_RECENT")
  fi
  if [[ -n "$MEMORY_CORRUPTION_ONSET" ]]; then
    CMD+=(--memory-corruption-onset "$MEMORY_CORRUPTION_ONSET")
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    CMD+=(--dry-run)
  fi

  echo "[submit_b1_b2_deferred_eval_sweep] Submitting checkpoint: $checkpoint"
  "${CMD[@]}"
done