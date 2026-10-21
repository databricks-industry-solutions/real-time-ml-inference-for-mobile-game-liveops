"""Launcher task: render the air workload and run `air run` from a serverless job.

All Ray training on the AI Runtime uses the sanctioned `air` CLI + workload-YAML +
`ray_bootstrap.sh` pattern (single OR multi node -- a small run is just
`num_accelerators: 1`). A DAB task cannot itself span nodes, so this serverless
spark_python_task invokes `air`: ensure the CLI, authenticate it from the job's
notebook context, render `jobs/train_multinode/workload.yaml` with the DAB-injected
GPU count / accelerator type / UC location / epochs, and `air run -f <rendered>
--watch`.

The workload uses the named AI env `databricks_ai_v5` (ray + torch preinstalled, no
per-run install) and the GPU node reads training data as parquet off the
FUSE-mounted UC Volume (no SQL warehouse / host / token in the workload).

Task parameters (argv): catalog, schema, train_gpus, gpu_accelerator_type, epochs.
catalog/schema follow the usual `configure_from_argv` convention so the rendered
workload derives the same per-target UC Volume path.
"""

from __future__ import annotations

import os
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

_EXPERIMENT_NAME = "offer_recommender_multinode"


def _repo_root() -> Path:
    """Resolve the synced repo root (.../dev/files) from argv[0].

    A serverless spark_python_task has NO `__file__`, but sys.argv[0] IS the
    deployed script path (.../dev/files/jobs/train_multinode_launcher.py), so the
    repo root is its parent's parent.
    """
    return Path(sys.argv[0]).resolve().parent.parent


def _workload_template() -> Path:
    """Path to the air workload template (repo_root/jobs/train_multinode)."""
    return _repo_root() / "jobs" / "train_multinode" / "workload.yaml"


def _render_workload(
    template: str,
    *,
    snapshot_root: Path,
    train_gpus: str,
    gpu_accelerator_type: str,
    epochs: str,
) -> str:
    """Substitute the {{PLACEHOLDER}} tokens in the workload template.

    Uses str.replace (not str.format) so the YAML's own `${...}` shell expansions
    in the bootstrap `command` block are left untouched. No warehouse/host/token is
    rendered: the GPU node reads training data as parquet off the FUSE-mounted UC
    Volume (host/token still authenticate the `air` CLI itself, in run_air, not the
    workload).
    """
    uc = get_config()
    replacements = {
        "{{EXPERIMENT_NAME}}": _EXPERIMENT_NAME,
        "{{TRAIN_GPUS}}": train_gpus,
        "{{GPU_ACCELERATOR_TYPE}}": gpu_accelerator_type,
        # code_source.root_path must be ABSOLUTE (air requirement); point at the
        # STAGED local copy (not the churny workspace FUSE mount).
        "{{REPO_ROOT}}": str(snapshot_root),
        "{{RECO_CATALOG}}": uc.catalog,
        "{{RECO_SCHEMA}}": uc.schema,
        "{{RECO_EPOCHS}}": epochs,
    }
    rendered = template
    for token_key, value in replacements.items():
        rendered = rendered.replace(token_key, value)
    return rendered


def main() -> None:
    """Render the workload and launch Ray Train via the air CLI."""
    args = sys.argv[1:]
    configure_from_argv(args)

    # argv: catalog, schema, train_gpus, gpu_accelerator_type, epochs
    train_gpus = args[2] if len(args) > 2 else "16"
    gpu_accelerator_type = args[3] if len(args) > 3 else "GPU_8xH100"
    epochs = args[4] if len(args) > 4 else os.environ.get("RECO_EPOCHS", "40")

    ensure_air()
    host, token = context_host_token()
    template = _workload_template()
    snapshot_root = stage_snapshot(_repo_root())
    rendered = _render_workload(
        template.read_text(),
        snapshot_root=snapshot_root,
        train_gpus=train_gpus,
        gpu_accelerator_type=gpu_accelerator_type,
        epochs=epochs,
    )

    rendered_path = Path(tempfile.mkdtemp()) / "workload.rendered.yaml"
    rendered_path.write_text(rendered)
    logger.info(
        "WORKLOAD_RENDERED path=%s gpus=%s type=%s snapshot_root=%s",
        rendered_path,
        train_gpus,
        gpu_accelerator_type,
        snapshot_root,
    )

    run_air(rendered_path, snapshot_root, host, token)


if __name__ == "__main__":
    main()
