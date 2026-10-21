"""Launcher: render + run the Ray Tune HPO showcase via the `air` CLI.

Mirrors jobs/train_multinode_launcher.py (reusing its proven air-launch helpers)
but renders jobs/tune/workload.yaml and runs jobs/tune/tune_ray.py. Standalone,
manual-run job (offer_recommender_tune) -- NOT part of the daily pipeline.

Args (argv): catalog, schema, tune_gpus, gpu_type.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from offer_recommender.config import configure_from_argv, get_config
from offer_recommender.logging_config import get_logger
from offer_recommender.training.air_launch import (
    context_host_token,
    ensure_air,
    run_air,
    stage_snapshot,
)

logger = get_logger(__name__)


def main() -> None:
    """Render the tune workload and launch it via air."""
    args = sys.argv[1:]
    configure_from_argv(args)
    tune_gpus = args[2] if len(args) > 2 else "4"
    gpu_type = args[3] if len(args) > 3 else "GPU_1xA10"

    repo_root = Path(sys.argv[0]).resolve().parent.parent
    ensure_air()
    host, token = context_host_token()
    staged = stage_snapshot(repo_root)
    uc = get_config()
    tmpl = (repo_root / "jobs" / "tune" / "workload.yaml").read_text()
    rendered = (
        tmpl.replace("{{EXPERIMENT_NAME}}", "offer_recommender_tune")
        .replace("{{TUNE_GPUS}}", tune_gpus)
        .replace("{{GPU_TYPE}}", gpu_type)
        .replace("{{REPO_ROOT}}", str(staged))
        .replace("{{RECO_CATALOG}}", uc.catalog)
        .replace("{{RECO_SCHEMA}}", uc.schema)
    )
    wl = Path(tempfile.mkdtemp()) / "tune.rendered.yaml"
    wl.write_text(rendered)
    logger.info(
        "TUNE_WORKLOAD_RENDERED path=%s gpus=%s type=%s", wl, tune_gpus, gpu_type
    )
    run_air(wl, staged, host, token)


if __name__ == "__main__":
    main()
