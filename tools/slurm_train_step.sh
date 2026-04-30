#!/usr/bin/env bash

set -euo pipefail

CONFIG=$1
WORK_DIR=$2
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=("${@:3}")
STEP_NODES=$(( (GPUS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
MAX_PORT_RETRIES=${MAX_PORT_RETRIES:-8}

select_master_port() {
    local attempt="$1"
    local base_seed i candidate_port
    base_seed=$(( (${SLURM_JOB_ID:-0} + ${SLURM_STEP_ID:-0} + $$ + attempt * 997) % 50000 ))
    for i in $(seq 0 63); do
        candidate_port=$((10000 + ((base_seed + i * 97) % 50000)))
        if python - "$candidate_port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("", port))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
        then
            echo "$candidate_port"
            return 0
        fi
    done

    echo $((10000 + (base_seed % 50000)))
    return 1
}

# Make distributed rendezvous robust per job step to reduce intermittent startup hangs.
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)}"
fi

MASTER_PORT_USER_SET=0
if [[ -n "${MASTER_PORT:-}" ]]; then
    MASTER_PORT_USER_SET=1
fi

export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}

declare -a SRUN_ARGS_ARR=()
if [[ -n "$SRUN_ARGS" ]]; then
    read -r -a SRUN_ARGS_ARR <<< "$SRUN_ARGS"
fi

export SRUN_CPUS_PER_TASK=${CPUS_PER_TASK}

attempt=1
while (( attempt <= MAX_PORT_RETRIES )); do
    if (( MASTER_PORT_USER_SET == 0 )); then
        if MASTER_PORT=$(select_master_port "$attempt"); then
            export MASTER_PORT
        else
            export MASTER_PORT
            echo "[slurm_train_step] Warning: no verified free port found; trying fallback MASTER_PORT=${MASTER_PORT}" >&2
        fi
    fi

    echo "[slurm_train_step] attempt=${attempt}/${MAX_PORT_RETRIES} MASTER_ADDR=${MASTER_ADDR:-unset} MASTER_PORT=${MASTER_PORT:-unset} GPUS=${GPUS} GPUS_PER_NODE=${GPUS_PER_NODE}"

    step_log=$(mktemp)
    set +e
    PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}" \
    srun --nodes=${STEP_NODES} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --gpus-per-task=1 \
        --label \
        --kill-on-bad-exit=1 \
        --wait=0 \
        "${SRUN_ARGS_ARR[@]}" \
        python -u tools/train.py "$CONFIG" --work-dir="$WORK_DIR" --launcher="slurm" "${PY_ARGS[@]}" 2>&1 | tee "$step_log"
    rc=${PIPESTATUS[0]}
    set -e

    if (( rc == 0 )); then
        rm -f "$step_log"
        exit 0
    fi

    if (( MASTER_PORT_USER_SET == 0 )) && grep -qi "Address already in use" "$step_log"; then
        rm -f "$step_log"
        echo "[slurm_train_step] Port collision detected, retrying with a new MASTER_PORT..." >&2
        attempt=$((attempt + 1))
        continue
    fi

    rm -f "$step_log"
    exit "$rc"
done

echo "[slurm_train_step] Failed after ${MAX_PORT_RETRIES} attempts due to repeated rendezvous port collisions." >&2
exit 1