"""Unit tests for jobs/gpu_task_config.py (the deploy-time GPU task config writer)."""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest


def _load_module():
    """Import jobs/gpu_task_config.py (jobs/ is not an importable package)."""
    path = (
        pathlib.Path(__file__).resolve().parent.parent / "jobs" / "gpu_task_config.py"
    )
    spec = importlib.util.spec_from_file_location("gpu_task_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_writes_pairs_as_json(tmp_path: pathlib.Path) -> None:
    # Arrange
    module = _load_module()
    out = tmp_path / "hyperparameters.yaml"

    # Act
    module.main([str(out), "RECO_CATALOG=main", "RECO_EPOCHS=3", "URL=a=b"])

    # Assert
    assert json.loads(out.read_text()) == {
        "RECO_CATALOG": "main",
        "RECO_EPOCHS": "3",
        "URL": "a=b",
    }


@pytest.mark.parametrize("bad", ["NOEQUALS", "=value"])
def test_rejects_malformed_pairs(bad: str) -> None:
    # Arrange
    module = _load_module()

    # Act / Assert
    with pytest.raises(ValueError):
        module.parse_pairs([bad])
