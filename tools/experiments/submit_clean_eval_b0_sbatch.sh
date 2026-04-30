#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit a clean (no corruption) B0 evaluation as a single Slurm job.

Usage:
  bash tools/experiments/submit_clean_eval_b0_sbatch.sh --mail-user EMAIL [options]

Required:
  --mail-user EMAIL

Optional:
  --checkpoint PATH           Path to the B0 checkpoint.
                              Default: <work-root>/pretrained_ckpts/b0_nusc_oldsplit/latest.pth
                              (Download from https://huggingface.co/cccjc/maptracker or
                               https://www.dropbox.com/scl/fo/miulg8q9oby7q2x5vemme/ALoxX1HyxGlfR9y3xlqfzeE)
  --config PATH               Eval config.
                              Default: plugin/configs/maptracker/nuscenes_oldsplit/
                                       maptracker_nusc_oldsplit_5frame_span10_stage3_b0_eval.py
  --work-root DIR             (default: <repo-root>/work_dirs)
  --gpus N                    GPUs for torchrun distributed eval (default: 4)
                              All GPUs are allocated to a single Slurm task; torchrun
                              spawns N worker processes.  Must fit on one node.
  --gpus-per-node N           (default: gpus — single-node only)
  --cpus-per-task N           CPUs per GPU worker (default: 6).
                              Slurm task receives gpus x cpus-per-task total CPUs.
  --time LIMIT                (default: 02:00:00)
  --job-name NAME             (default: maptracker_clean_eval_b0)
  --partition NAME            (default: none — Della routes GPU jobs via QOS, not partition)
  --qos NAME                  (default: gpu-short — 1-day wall limit; sufficient for 2h eval)
  --account NAME
  --constraint EXPR           (default: a100&nomig — full A100 nodes; excludes MIG-sliced 3g.40gb nodes)
  --mem VALUE                 (default: 96G)
  --mail-type VALUE           (default: END,FAIL)
  --seed N                    (default: 0)
  --condition-tag TAG         (default: clean)
  --module-load MOD           (default: anaconda3/2023.9)
  --conda-env NAME            (default: maptracker)
  --cons-frames N             (default: 5)
  --skip-cmap                 Skip C-mAP computation after eval.
  --cfg-options "k=v ..."     Extra cfg-options forwarded to run_b0.sh.
  --dry-run                   Generate sbatch file but do not submit.
  -h, --help

Notes:
  - Outputs land under: <work-root>/experiments/b0/<RUN_ID>/eval_clean/
  - The sbatch script and logs are kept at:
    <work-root>/sbatch/clean_eval_b0/<TIMESTAMP>/
USAGE
}

MAIL_USER=""
CHECKPOINT=""
CONFIG=""
WORK_ROOT=""
GPUS=4
GPUS_PER_NODE=""
CPUS_PER_TASK=6
TIME_LIMIT="02:00:00"
JOB_NAME="maptracker_clean_eval_b0"
PARTITION=""
QOS="gpu-short"
ACCOUNT=""
CONSTRAINT="a100&nomig"
MEMORY="96G"
MAIL_TYPE="END,FAIL"
SEED=0
CONDITION_TAG="clean"
MODULE_LOAD="anaconda3/2023.9"
CONDA_ENV="maptracker"
CONS_FRAMES=5
SKIP_CMAP=0
CFG_OPTIONS_STR=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mail-user)     MAIL_USER="$2";      shift 2 ;;
    --checkpoint)    CHECKPOINT="$2";     shift 2 ;;
    --config)        CONFIG="$2";         shift 2 ;;
    --work-root)     WORK_ROOT="$2";      shift 2 ;;
    --gpus)          GPUS="$2";           shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2";  shift 2 ;;
    --cpus-per-task) CPUS_PER_TASK="$2";  shift 2 ;;
    --time)          TIME_LIMIT="$2";     shift 2 ;;
    --job-name)      JOB_NAME="$2";       shift 2 ;;
    --partition)     PARTITION="$2";      shift 2 ;;
    --qos)           QOS="$2";            shift 2 ;;
    --account)       ACCOUNT="$2";        shift 2 ;;
    --constraint)    CONSTRAINT="$2";     shift 2 ;;
    --mem)           MEMORY="$2";         shift 2 ;;
    --mail-type)     MAIL_TYPE="$2";      shift 2 ;;
    --seed)          SEED="$2";           shift 2 ;;
    --condition-tag) CONDITION_TAG="$2";  shift 2 ;;
    --module-load)   MODULE_LOAD="$2";    shift 2 ;;
    --conda-env)     CONDA_ENV="$2";      shift 2 ;;
    --cons-frames)   CONS_FRAMES="$2";    shift 2 ;;
    --skip-cmap)     SKIP_CMAP=1;         shift   ;;
    --cfg-options)   CFG_OPTIONS_STR="$2"; shift 2 ;;
    --dry-run)       DRY_RUN=1;           shift   ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "[submit_clean_eval_b0_sbatch] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MAIL_USER" ]]; then
  echo "[submit_clean_eval_b0_sbatch] --mail-user is required." >&2
  usage
  exit 2
fi

if (( GPUS <= 0 || CPUS_PER_TASK <= 0 )); then
  echo "[submit_clean_eval_b0_sbatch] --gpus and --cpus-per-task must be positive integers." >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Apply defaults that depend on PROJECT_ROOT / WORK_ROOT.
if [[ -z "$WORK_ROOT" ]]; then
  WORK_ROOT="${PROJECT_ROOT}/work_dirs"
fi
if [[ -z "$CONFIG" ]]; then
  CONFIG="${PROJECT_ROOT}/plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_b0_eval.py"
fi
if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="${WORK_ROOT}/pretrained_ckpts/b0_nusc_oldsplit/latest.pth"
fi
if [[ -z "$GPUS_PER_NODE" ]]; then
  GPUS_PER_NODE="$GPUS"
fi

if (( GPUS != GPUS_PER_NODE )); then
  echo "[submit_clean_eval_b0_sbatch] Multi-node eval is not supported (run_b0.sh --launcher pytorch uses torchrun on one node)." >&2
  echo "[submit_clean_eval_b0_sbatch] --gpus (${GPUS}) must equal --gpus-per-node (${GPUS_PER_NODE})." >&2
  exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "[submit_clean_eval_b0_sbatch] Config not found: $CONFIG" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[submit_clean_eval_b0_sbatch] Checkpoint not found: $CHECKPOINT" >&2
  echo "[submit_clean_eval_b0_sbatch] Download the B0 checkpoint (maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune/latest.pth)" >&2
  echo "[submit_clean_eval_b0_sbatch] from https://huggingface.co/cccjc/maptracker and place it at:" >&2
  echo "[submit_clean_eval_b0_sbatch]   ${CHECKPOINT}" >&2
  exit 1
fi

NODES=$(( GPUS / GPUS_PER_NODE ))
TS="$(date -u +%Y%m%d_%H%M%S)"
SBATCH_ROOT="${WORK_ROOT%/}/sbatch/clean_eval_b0/${TS}"
SBATCH_LOG_DIR="$SBATCH_ROOT/logs"
SBATCH_SCRIPT="$SBATCH_ROOT/eval.sbatch"
mkdir -p "$SBATCH_LOG_DIR"

RUN_CMD=(bash tools/experiments/run_b0.sh
  --config   "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --work-root "$WORK_ROOT"
  --launcher pytorch
  --gpus "$GPUS"
  --seed "$SEED"
  --condition-tag "$CONDITION_TAG"
  --cons-frames "$CONS_FRAMES")

if [[ $SKIP_CMAP -eq 1 ]]; then
  RUN_CMD+=(--skip-cmap)
fi
if [[ -n "$CFG_OPTIONS_STR" ]]; then
  RUN_CMD+=(--cfg-options "$CFG_OPTIONS_STR")
fi
if [[ $DRY_RUN -eq 1 ]]; then
  RUN_CMD+=(--dry-run)
fi

printf -v RUN_CMD_STR '%q ' "${RUN_CMD[@]}"

# --ntasks=1: torchrun (invoked by dist_test.sh) spawns all worker processes from
# a single master process and needs to see all GPUs via CUDA_VISIBLE_DEVICES.
# --gpus-per-task would restrict each Slurm task to 1 GPU and break nproc_per_node.
# cpus-per-task = GPUS * CPUS_PER_TASK so all torchrun workers share the full allocation.
SBATCH_DIRECTIVES=(
  "#SBATCH --job-name=${JOB_NAME}"
  "#SBATCH --nodes=1"
  "#SBATCH --ntasks=1"
  "#SBATCH --ntasks-per-node=1"
  "#SBATCH --cpus-per-task=$((GPUS * CPUS_PER_TASK))"
  "#SBATCH --gres=gpu:${GPUS}"
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

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PROJECT_ROOT=${PROJECT_ROOT}
export PYTHONPATH="\$PROJECT_ROOT:\${PYTHONPATH:-}"

cd "\$PROJECT_ROOT"
${RUN_CMD_STR}
EOF

chmod +x "$SBATCH_SCRIPT"

echo "[submit_clean_eval_b0_sbatch] Generated: $SBATCH_SCRIPT"
echo "[submit_clean_eval_b0_sbatch] Logs dir:  $SBATCH_LOG_DIR"
echo "[submit_clean_eval_b0_sbatch] Config:     $CONFIG"
echo "[submit_clean_eval_b0_sbatch] Checkpoint: $CHECKPOINT"
echo "[submit_clean_eval_b0_sbatch] Work root:  $WORK_ROOT"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[submit_clean_eval_b0_sbatch] Dry run — not submitting."
  exit 0
fi

SUBMIT_OUTPUT="$(sbatch "$SBATCH_SCRIPT")"
echo "$SUBMIT_OUTPUT"
