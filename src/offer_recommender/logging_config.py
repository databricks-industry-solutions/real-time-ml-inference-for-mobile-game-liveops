"""Central logging configuration for all offer-recommender jobs and modules.

One place configures the format and level so job tasks, the Ray driver and the
serving code all log consistently (and to stdout, which Databricks captures in
the run/driver logs). Call `get_logger(__name__)` at module top; the first call
installs a single stdout handler on the package root logger, and subsequent
calls reuse it (idempotent, so repeated imports never double-log).

Level is overridable with the RECO_LOG_LEVEL env var (default INFO).
"""

from __future__ import annotations

import logging
import os
import sys

_ROOT_NAME = "offer_recommender"
_CONFIGURED = False


def _configure_root() -> None:
    """Attach a single stdout handler + formatter to the package root logger."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(os.environ.get("RECO_LOG_LEVEL", "INFO").upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.propagate = False  # don't double-log through the Python root logger
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger under the `offer_recommender` root.

    Args:
        name: usually `__name__`; namespaced under the package root so all
            project loggers share the single configured handler.
    """
    _configure_root()
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
