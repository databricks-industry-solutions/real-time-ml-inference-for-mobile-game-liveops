#!/usr/bin/env bash
# Ray-on-AI-Runtime bootstrap shared by the training and tuning `ai_runtime_task`s
# (resources/jobs.yml). AI Runtime runs the task's command.sh on every node; that
# script sets RAY_ENTRYPOINT and execs this one. Node 0 starts the Ray head and runs
# $RAY_ENTRYPOINT; the other nodes join it. AI Runtime sets NODE_RANK, NUM_NODES,
# LOCAL_WORLD_SIZE and MASTER_ADDR; RAY_HEAD_PORT and GPUS_PER_NODE are ours.
#
# Workers retry ONLY head-port reachability (the start-order race where a worker
# boots before the head), then `ray start --block` ONCE and exit when the head goes
# away. Do NOT retry around `--block`: it returns when node 0 finishes and
# `ray stop`s the head, so a retry loop would re-join a torn-down head forever and
# hang the job past training.
set -euo pipefail
: "${RAY_ENTRYPOINT:?RAY_ENTRYPOINT must be set (the python driver to run on node 0)}"

# Deploy-time config (catalog, schema, epochs) rendered by jobs/gpu_task_config.py
# into the hyperparameters.yaml beside command.sh; export each key.
eval "$(python3 -c 'import json, os, shlex
for k, v in json.load(open(os.environ["HYPERPARAMETERS_PATH"])).items():
    print(f"export {k}={shlex.quote(v)}")')"

# The bundle syncs the repo to the workspace, which every GPU node can read.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/$(dirname "$RAY_ENTRYPOINT"):${PYTHONPATH:-}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
GPUS_PER_NODE="${LOCAL_WORLD_SIZE:-1}"
export RECO_TOTAL_GPUS=$(( ${NUM_NODES:-1} * GPUS_PER_NODE ))

if [ "${NODE_RANK:-0}" = "0" ]; then
  ray start --head --port="$RAY_HEAD_PORT" --num-gpus="$GPUS_PER_NODE" --dashboard-host=0.0.0.0
  python "$RAY_ENTRYPOINT"
  ray stop
else
  for i in $(seq 1 60); do
    python3 -c "import socket,sys; s=socket.socket(); s.settimeout(3); sys.exit(0 if s.connect_ex(('${MASTER_ADDR}', ${RAY_HEAD_PORT}))==0 else 1)" && break
    echo "worker ${NODE_RANK} waiting for head ${MASTER_ADDR}:${RAY_HEAD_PORT} (try ${i})"
    sleep 5
  done
  ray start --address="${MASTER_ADDR}:${RAY_HEAD_PORT}" --num-gpus="$GPUS_PER_NODE" --block 2>/dev/null || true
fi
