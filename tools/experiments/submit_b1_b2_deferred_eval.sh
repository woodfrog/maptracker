#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: submit_b1_b2_deferred_eval.sh --base-config CFG --checkpoint CKPT --work-root ROOT --run-id ID --baseline BASELINE --time LIMIT --mail-user EMAIL [options]

Required:
  --base-config PATH
  --checkpoint PATH
  --work-root DIR
  --run-id ID
  --baseline b1|b2
  --time LIMIT                    Slurm time limit, e.g. 23:30:00 or 1-00:00:00
  --mail-user EMAIL

Optional:
  --checkpoint-tag TAG
  --condition-tag TAG             (default: derived from eval mode, else clean)
  --rerun                         Reuse and replace an existing eval dir.
  --job-name NAME                 (default: maptracker_b1_b2_eval)
  --partition NAME
  --qos NAME                      (default: gpu-short)
  --account NAME
  --constraint EXPR               (default: nomig&gpu40)
  --mem VALUE
  --mail-type VALUE               (default: END,FAIL)
  --dependency SPEC
  --eval-gpus N                   (default: 2)
  --gpus-per-node N               (default: eval-gpus)
  --cpus-per-task N               (default: 4)
  --seed N                        (default: 0)
  --module-load MOD               (default: anaconda3/2023.9)
  --conda-env NAME                (default: maptracker)
  --srun-args "ARGS"
  --sbatch-extra-args "ARGS"
  --cfg-options "k=v ..."
  --eval-options "k=v ..."
  --cons-frames N                 (default: 5)
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
TIME_LIMIT=""
MAIL_USER=""
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
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --checkpoint-tag) CHECKPOINT_TAG="$2"; shift 2 ;;
    --condition-tag) CONDITION_TAG="$2"; shift 2 ;;
    --rerun) RERUN=1; shift ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --mail-user) MAIL_USER="$2"; shift 2 ;;
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
    *) echo "[submit_b1_b2_deferred_eval] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$BASE_CONFIG" || -z "$CHECKPOINT" || -z "$WORK_ROOT" || -z "$RUN_ID" || -z "$BASELINE" || -z "$TIME_LIMIT" || -z "$MAIL_USER" ]]; then
  echo "[submit_b1_b2_deferred_eval] Missing required arguments." >&2
  usage
  exit 2
fi

case "$BASELINE" in
  b1|b2) ;;
  *) echo "[submit_b1_b2_deferred_eval] Invalid baseline: $BASELINE" >&2; exit 2 ;;
esac

if [[ -z "$GPUS_PER_NODE" ]]; then
  GPUS_PER_NODE="$EVAL_GPUS"
fi

if (( EVAL_GPUS <= 0 || GPUS_PER_NODE <= 0 || CPUS_PER_TASK <= 0 )); then
  echo "[submit_b1_b2_deferred_eval] GPU/CPU values must be positive integers." >&2
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
    echo "[submit_b1_b2_deferred_eval] --memory-corruption-mode is required when using other memory corruption overrides." >&2
    exit 2
  fi
fi

if [[ "$MEMORY_CORRUPTION_MODE" == "clean" && ( -n "$MEMORY_STALE_OFFSET" || -n "$MEMORY_C_TAIL_KEEP_RECENT" || -n "$MEMORY_CORRUPTION_ONSET" ) ]]; then
  echo "[submit_b1_b2_deferred_eval] clean mode cannot be combined with stale/keep_recent/onset overrides." >&2
  exit 2
fi

if [[ -z "$CHECKPOINT_TAG" ]]; then
  CHECKPOINT_TAG="$(basename "$CHECKPOINT")"
  CHECKPOINT_TAG="${CHECKPOINT_TAG%.pth}"
fi
CHECKPOINT_TAG="$(normalize_tag "$CHECKPOINT_TAG")"
if [[ -z "$CHECKPOINT_TAG" ]]; then
  echo "[submit_b1_b2_deferred_eval] checkpoint tag resolved to empty value." >&2
  exit 2
fi

if [[ -z "$CONDITION_TAG" ]]; then
  CONDITION_TAG="$(derive_condition_tag "$MEMORY_CORRUPTION_MODE" "$MEMORY_CORRUPTION_ONSET" "$MEMORY_STALE_OFFSET" "$MEMORY_C_TAIL_KEEP_RECENT")"
fi
CONDITION_TAG="$(normalize_tag "$CONDITION_TAG")"
if [[ -z "$CONDITION_TAG" ]]; then
  echo "[submit_b1_b2_deferred_eval] condition tag resolved to empty value." >&2
  exit 2
fi

if (( EVAL_GPUS % GPUS_PER_NODE != 0 )); then
  echo "[submit_b1_b2_deferred_eval] Total eval GPUs (${EVAL_GPUS}) must be divisible by --gpus-per-node (${GPUS_PER_NODE})." >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREVIEW_TAG="$(date -u +%Y%m%d_%H%M%S)"
SBATCH_BASE_ROOT="${WORK_ROOT%/}/sbatch/b1_b2_deferred_eval/${RUN_ID}/${BASELINE}/${CHECKPOINT_TAG}/${CONDITION_TAG}"
SBATCH_ROOT="$SBATCH_BASE_ROOT"
if [[ $DRY_RUN -eq 1 ]]; then
  SBATCH_ROOT="${WORK_ROOT%/}/sbatch/dry_run/b1_b2_deferred_eval/${RUN_ID}/${BASELINE}/${CHECKPOINT_TAG}/${CONDITION_TAG}/${PREVIEW_TAG}"
fi
SBATCH_LOG_DIR="$SBATCH_ROOT/logs"
SBATCH_SCRIPT="$SBATCH_ROOT/eval.sbatch"
mkdir -p "$SBATCH_LOG_DIR"

RUN_CMD=(bash tools/experiments/run_b1_b2_deferred_eval.sh
  --base-config "$BASE_CONFIG"
  --checkpoint "$CHECKPOINT"
  --work-root "$WORK_ROOT"
  --run-id "$RUN_ID"
  --baseline "$BASELINE"
  --checkpoint-tag "$CHECKPOINT_TAG"
  --condition-tag "$CONDITION_TAG"
  --launcher slurm-step
  --gpus "$EVAL_GPUS"
  --gpus-per-node "$GPUS_PER_NODE"
  --cpus-per-task "$CPUS_PER_TASK"
  --seed "$SEED")

if [[ $RERUN -eq 1 ]]; then
  RUN_CMD+=(--rerun)
fi

if [[ -n "$CFG_OPTIONS_STR" ]]; then
  RUN_CMD+=(--cfg-options "$CFG_OPTIONS_STR")
fi
if [[ -n "$EVAL_OPTIONS_STR" ]]; then
  RUN_CMD+=(--eval-options "$EVAL_OPTIONS_STR")
fi
if [[ "$CONS_FRAMES" != "5" ]]; then
  RUN_CMD+=(--cons-frames "$CONS_FRAMES")
fi
if [[ $SKIP_CMAP -eq 1 ]]; then
  RUN_CMD+=(--skip-cmap)
fi
if [[ -n "$MEMORY_CORRUPTION_MODE" ]]; then
  RUN_CMD+=(--memory-corruption-mode "$MEMORY_CORRUPTION_MODE")
fi
if [[ -n "$MEMORY_STALE_OFFSET" ]]; then
  RUN_CMD+=(--memory-stale-offset "$MEMORY_STALE_OFFSET")
fi
if [[ -n "$MEMORY_C_TAIL_KEEP_RECENT" ]]; then
  RUN_CMD+=(--memory-c-tail-keep-recent "$MEMORY_C_TAIL_KEEP_RECENT")
fi
if [[ -n "$MEMORY_CORRUPTION_ONSET" ]]; then
  RUN_CMD+=(--memory-corruption-onset "$MEMORY_CORRUPTION_ONSET")
fi
if [[ $DRY_RUN -eq 1 ]]; then
  RUN_CMD+=(--dry-run)
fi

printf -v RUN_CMD_STR '%q ' "${RUN_CMD[@]}"

NODES=$(( EVAL_GPUS / GPUS_PER_NODE ))
SBATCH_DIRECTIVES=(
  "#SBATCH --job-name=${JOB_NAME}_${BASELINE}_${CHECKPOINT_TAG}"
  "#SBATCH --nodes=${NODES}"
  "#SBATCH --ntasks=${EVAL_GPUS}"
  "#SBATCH --ntasks-per-node=${GPUS_PER_NODE}"
  "#SBATCH --cpus-per-task=${CPUS_PER_TASK}"
  "#SBATCH --gres=gpu:${GPUS_PER_NODE}"
  "#SBATCH --gpus-per-task=1"
  "#SBATCH --time=${TIME_LIMIT}"
  "#SBATCH --output=${SBATCH_LOG_DIR}/%x-%j.out"
  "#SBATCH --error=${SBATCH_LOG_DIR}/%x-%j.err"
  "#SBATCH --mail-type=${MAIL_TYPE}"
  "#SBATCH --mail-user=${MAIL_USER}"
)

if [[ -n "$PARTITION" ]]; then
  SBATCH_DIRECTIVES+=("#SBATCH --partition=${PARTITION}")
fi
if [[ -n "$QOS" ]]; then
  SBATCH_DIRECTIVES+=("#SBATCH --qos=${QOS}")
fi
if [[ -n "$ACCOUNT" ]]; then
  SBATCH_DIRECTIVES+=("#SBATCH --account=${ACCOUNT}")
fi
if [[ -n "$CONSTRAINT" ]]; then
  SBATCH_DIRECTIVES+=("#SBATCH --constraint=${CONSTRAINT}")
fi
if [[ -n "$MEMORY" ]]; then
  SBATCH_DIRECTIVES+=("#SBATCH --mem=${MEMORY}")
fi
if [[ -n "$DEPENDENCY" ]]; then
  SBATCH_DIRECTIVES+=("#SBATCH --dependency=${DEPENDENCY}")
fi

printf '%s\n' '#!/usr/bin/env bash' > "$SBATCH_SCRIPT"
printf '%s\n' "${SBATCH_DIRECTIVES[@]}" >> "$SBATCH_SCRIPT"
printf '%s\n' 'set -euo pipefail' >> "$SBATCH_SCRIPT"
cat >> "$SBATCH_SCRIPT" <<EOF

module purge
EOF
if [[ -n "$MODULE_LOAD" ]]; then
  printf 'module load %q\n' "$MODULE_LOAD" >> "$SBATCH_SCRIPT"
fi
cat >> "$SBATCH_SCRIPT" <<EOF
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

export PROJECT_ROOT=${PROJECT_ROOT}
export PYTHONPATH="\$PROJECT_ROOT:\${PYTHONPATH:-}"
export SRUN_CPUS_PER_TASK=${CPUS_PER_TASK}
EOF
if [[ -n "$SRUN_ARGS_STR" ]]; then
  printf 'export SRUN_ARGS=%q\n' "$SRUN_ARGS_STR" >> "$SBATCH_SCRIPT"
fi
cat >> "$SBATCH_SCRIPT" <<EOF

cd "\$PROJECT_ROOT"
${RUN_CMD_STR}
EOF

chmod +x "$SBATCH_SCRIPT"

declare -a SBATCH_SUBMIT_CMD=(sbatch)
if [[ -n "$SBATCH_EXTRA_ARGS_STR" ]]; then
  read -r -a SBATCH_EXTRA_ARGS_ARR <<< "$SBATCH_EXTRA_ARGS_STR"
  SBATCH_SUBMIT_CMD+=("${SBATCH_EXTRA_ARGS_ARR[@]}")
fi
SBATCH_SUBMIT_CMD+=("$SBATCH_SCRIPT")

echo "[submit_b1_b2_deferred_eval] Generated: $SBATCH_SCRIPT"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[submit_b1_b2_deferred_eval] Dry-run preview kept separate from canonical path: $SBATCH_BASE_ROOT/eval.sbatch"
fi
echo "[submit_b1_b2_deferred_eval] Submission command: ${SBATCH_SUBMIT_CMD[*]}"
if [[ $DRY_RUN -eq 1 ]]; then
  exit 0
fi

SUBMIT_OUTPUT="$("${SBATCH_SUBMIT_CMD[@]}")"
echo "$SUBMIT_OUTPUT"