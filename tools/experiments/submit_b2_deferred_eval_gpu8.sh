#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CHECKPOINT="${CHECKPOINT-}"
if [[ -z "$CHECKPOINT" ]]; then
  echo "[submit_b2_deferred_eval_gpu8] CHECKPOINT must be set to the checkpoint path to evaluate." >&2
  exit 2
fi

RUN_ID="${RUN_ID-b2_stage3_gpu8_trainonly}"
PARTITION="${PARTITION-}"
QOS="${QOS-gpu-short}"
TIME_LIMIT="${TIME_LIMIT-02:00:00}"
MAIL_USER="${MAIL_USER-yk3904@princeton.edu}"
MAIL_TYPE="${MAIL_TYPE-END,FAIL}"
CONSTRAINT="${CONSTRAINT-nomig&gpu40}"
EVAL_GPUS="${EVAL_GPUS-2}"
GPUS_PER_NODE="${GPUS_PER_NODE-2}"
CPUS_PER_TASK="${CPUS_PER_TASK-4}"
CONDITION_TAG="${CONDITION_TAG-clean}"

CMD=(bash tools/experiments/submit_b1_b2_deferred_eval.sh
  --base-config plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py
  --checkpoint "$CHECKPOINT"
  --work-root "$PROJECT_ROOT/work_dirs"
  --run-id "$RUN_ID"
  --baseline b2
  --time "$TIME_LIMIT"
  --mail-user "$MAIL_USER"
  --mail-type "$MAIL_TYPE"
  --qos "$QOS"
  --eval-gpus "$EVAL_GPUS"
  --gpus-per-node "$GPUS_PER_NODE"
  --cpus-per-task "$CPUS_PER_TASK"
  --condition-tag "$CONDITION_TAG")

if [[ -n "$PARTITION" ]]; then
  CMD+=(--partition "$PARTITION")
fi
if [[ -n "$CONSTRAINT" ]]; then
  CMD+=(--constraint "$CONSTRAINT")
fi
if [[ -n "${CHECKPOINT_TAG-}" ]]; then
  CMD+=(--checkpoint-tag "$CHECKPOINT_TAG")
fi
if [[ "${RERUN:-0}" == "1" ]]; then
  CMD+=(--rerun)
fi
if [[ -n "${ACCOUNT-}" ]]; then
  CMD+=(--account "$ACCOUNT")
fi
if [[ -n "${MEM-}" ]]; then
  CMD+=(--mem "$MEM")
fi
if [[ -n "${DEPENDENCY-}" ]]; then
  CMD+=(--dependency "$DEPENDENCY")
fi
if [[ -n "${SRUN_ARGS-}" ]]; then
  CMD+=(--srun-args "$SRUN_ARGS")
fi
if [[ -n "${SBATCH_EXTRA_ARGS-}" ]]; then
  CMD+=(--sbatch-extra-args "$SBATCH_EXTRA_ARGS")
fi
if [[ -n "${CFG_OPTIONS-}" ]]; then
  CMD+=(--cfg-options "$CFG_OPTIONS")
fi
if [[ -n "${EVAL_OPTIONS-}" ]]; then
  CMD+=(--eval-options "$EVAL_OPTIONS")
fi
if [[ -n "${CONS_FRAMES-}" ]]; then
  CMD+=(--cons-frames "$CONS_FRAMES")
fi
if [[ "${SKIP_CMAP:-0}" == "1" ]]; then
  CMD+=(--skip-cmap)
fi
if [[ -n "${MEMORY_CORRUPTION_MODE-}" ]]; then
  CMD+=(--memory-corruption-mode "$MEMORY_CORRUPTION_MODE")
fi
if [[ -n "${MEMORY_STALE_OFFSET-}" ]]; then
  CMD+=(--memory-stale-offset "$MEMORY_STALE_OFFSET")
fi
if [[ -n "${MEMORY_C_TAIL_KEEP_RECENT-}" ]]; then
  CMD+=(--memory-c-tail-keep-recent "$MEMORY_C_TAIL_KEEP_RECENT")
fi
if [[ -n "${MEMORY_CORRUPTION_ONSET-}" ]]; then
  CMD+=(--memory-corruption-onset "$MEMORY_CORRUPTION_ONSET")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  CMD+=(--dry-run)
fi

cd "$PROJECT_ROOT"
"${CMD[@]}"