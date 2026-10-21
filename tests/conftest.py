"""Pytest fixtures and Spark setup for local tests.

Pins PYSPARK_PYTHON/PYSPARK_DRIVER_PYTHON to the current interpreter so Spark
workers use the same Python minor version as the driver (otherwise Spark raises
PYTHON_VERSION_MISMATCH when the system Python differs from the venv). Also
sets JAVA_HOME on macOS if unset.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-mark every test that requests the `spark` fixture as `slow`.

    The Spark-session tests each spin up a local JVM and dominate wall time, so the
    fast local gate (`pytest -m "not slow"`) excludes them without per-test decorators
    that could drift. Data-generation tests that are slow WITHOUT Spark are marked
    explicitly with `@pytest.mark.slow` at their definition (see tests/test_units.py).
    """
    for item in items:
        if "spark" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.slow)


def _ensure_spark_env() -> None:
    """Pin worker/driver Python and JAVA_HOME before any SparkSession is built."""
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    if "JAVA_HOME" not in os.environ and sys.platform == "darwin":
        try:
            java_home = subprocess.run(
                ["/usr/libexec/java_home"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            os.environ["JAVA_HOME"] = java_home
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass


@pytest.fixture(scope="session")
def spark():
    """Session-scoped local SparkSession for preprocessing tests.

    Master defaults to ``local[*]`` (all cores) so the exploded-event Spark tests
    parallelise across the box instead of running single-threaded; override with
    ``SPARK_TEST_MASTER`` (e.g. ``local[4]``) to cap threads on a busy machine.
    """
    _ensure_spark_env()
    import os

    from pyspark.sql import SparkSession

    master = os.environ.get("SPARK_TEST_MASTER", "local[*]")
    session = (
        SparkSession.builder.master(master)
        .appName("offer-recommender-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
