#!/usr/bin/env bash
# ai_runtime_task entry point for the Ray Tune sweep (resources/jobs.yml); runs on every node.
set -euo pipefail
export RAY_ENTRYPOINT=jobs/tune/tune_ray.py
exec bash "$(dirname "$0")/../train_multinode/ray_bootstrap.sh"
