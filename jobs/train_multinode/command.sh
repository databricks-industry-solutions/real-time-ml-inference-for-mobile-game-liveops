#!/usr/bin/env bash
# ai_runtime_task entry point for Ray Train (resources/jobs.yml); runs on every node.
set -euo pipefail
export RAY_ENTRYPOINT=jobs/train_multinode/train_ray.py
exec bash "$(dirname "$0")/ray_bootstrap.sh"
