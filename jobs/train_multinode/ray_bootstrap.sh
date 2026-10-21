#!/usr/bin/env bash
# Canonical Ray-on-AI-Runtime bootstrap (shared across air workloads), matching the
# official ray-hello-world / ray-train-distributed examples: node 0 starts the Ray
# head, the other nodes join it, and node 0 runs $RAY_ENTRYPOINT once the cluster is
# formed. AIR sets the rendezvous env vars (NODE_RANK, MASTER_ADDR, LOCAL_WORLD_SIZE).
#
# The worker path ports in our hard-won clean-teardown fix: retry ONLY head-port
# reachability (the start-order race where a worker boots before the head), then
# `ray start --block` ONCE and exit when the head goes away. Do NOT retry around
# `--block`: it returns when node 0 finishes and `ray stop`s the head, so a retry
# loop would re-join a torn-down head forever and hang the job past training.
set -euo pipefail

RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
GPUS_PER_NODE="${LOCAL_WORLD_SIZE:-8}"
: "${RAY_ENTRYPOINT:?RAY_ENTRYPOINT must be set (the python driver to run on node 0)}"

if [ "${NODE_RANK:-0}" = "0" ]; then
  ray start --head --port="$RAY_HEAD_PORT" --num-gpus="$GPUS_PER_NODE" --dashboard-host=0.0.0.0
  python "$RAY_ENTRYPOINT"
  ray stop
else
  for i in $(seq 1 60); do
    python3 -c "import socket,sys; s=socket.socket(); s.settimeout(3); sys.exit(0 if s.connect_ex(('${MASTER_ADDR}', ${RAY_HEAD_PORT}))==0 else 1)" && break
    echo "worker ${NODE_RANK:-?} waiting for head ${MASTER_ADDR}:${RAY_HEAD_PORT} (try ${i})"
    sleep 5
  done
  ray start --address="${MASTER_ADDR}:${RAY_HEAD_PORT}" --num-gpus="$GPUS_PER_NODE" --block 2>/dev/null || true
fi
