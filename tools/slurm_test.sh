#!/usr/bin/env bash

set -euo pipefail

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
CHECKPOINT=$4
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PY_ARGS=("${@:5}")
SRUN_ARGS=${SRUN_ARGS:-""}

declare -a SRUN_ARGS_ARR=()
if [[ -n "$SRUN_ARGS" ]]; then
    read -r -a SRUN_ARGS_ARR <<< "$SRUN_ARGS"
fi

PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}" \
srun -p "$PARTITION" \
        --job-name="$JOB_NAME" \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --gpus-per-task=1 \
        --kill-on-bad-exit=1 \
        "${SRUN_ARGS_ARR[@]}" \
        python -u tools/test.py "$CONFIG" "$CHECKPOINT" --launcher="slurm" "${PY_ARGS[@]}"
