"""Shared helpers to launch an `air` workload from a serverless Databricks job task.

Both the multi-node training launcher and the Ray Tune launcher are plain
serverless `spark_python_task`s that shell out to the `air` CLI (a DAB task cannot
itself span GPU nodes; `air` provisions them). These helpers are the common
plumbing: ensure the CLI is present, authenticate it from the job's notebook
context, stage a quiescent local snapshot of the code, and run the workload.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from offer_recommender.logging_config import get_logger

logger = get_logger(__name__)

# Only what the GPU nodes need: the package source + the job drivers/bootstrap.
# Keeps the staged snapshot tiny and off the churny workspace FUSE mount.
SNAPSHOT_DIRS = ("src", "jobs")


def ensure_air() -> None:
    """Ensure the `air` CLI is on PATH; pip-install as a fallback only if absent.

    `databricks-air` is a declared env dependency (installed at serverless env-build
    time), so the common path pays no runtime install cost.
    """
    if shutil.which("air") is not None:
        logger.info("air CLI present from declared env dependency; skipping install")
        return
    logger.info("air CLI not found on PATH; pip-installing databricks-air (fallback)")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "databricks-air"],
        check=True,
    )


def context_host_token() -> tuple[str, str]:
    """Return (host, short-lived token) from the running job's notebook context.

    A serverless job task authenticates the `air` CLI as its own principal via the
    runtime-issued API token; nothing is read from a secret or committed.
    """
    from databricks.sdk.runtime import dbutils

    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    return ctx.apiUrl().get(), ctx.apiToken().get()


def stage_snapshot(repo_root: Path, dirs: tuple[str, ...] = SNAPSHOT_DIRS) -> Path:
    """Copy the code the nodes need to a stable LOCAL temp dir; return its path.

    `air` snapshots `code_source.root_path` with a strict `tar` that aborts on "file
    changed as we read it". The live-synced workspace files dir is a FUSE mount the
    workspace background-syncs, so tar races the sync. Copying to a real local tmpdir
    (nothing else writes it) removes the race; air then tars a quiescent tree.
    """
    staged = Path(tempfile.mkdtemp(prefix="air_snapshot_"))
    for name in dirs:
        src = repo_root / name
        if src.exists():
            shutil.copytree(src, staged / name)
    return staged


def run_air(workload_path: Path, cwd: Path, host: str, token: str) -> None:
    """Run `air run -f <workload> --watch` from `cwd`, streaming output to the log."""
    env = {**os.environ, "DATABRICKS_HOST": host, "DATABRICKS_TOKEN": token}
    logger.info("AIR_RUN workload=%s cwd=%s", workload_path, cwd)
    proc = subprocess.run(
        ["air", "run", "-f", str(workload_path), "--watch"],
        env=env,
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    if proc.stdout:
        logger.info("AIR_STDOUT\n%s", proc.stdout)
    if proc.stderr:
        logger.info("AIR_STDERR\n%s", proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"air run failed with exit code {proc.returncode}")
    logger.info("AIR_RUN_COMPLETE")
