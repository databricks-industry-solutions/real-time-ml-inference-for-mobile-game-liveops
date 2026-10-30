"""Unit tests for the Genie Agent definition (resources/genie.yml) and jobs/smoke_genie.py."""

from __future__ import annotations

import importlib.util
import json
import pathlib
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_smoke_genie() -> ModuleType:
    """Import jobs/smoke_genie.py (jobs/ is not an importable package)."""
    spec = importlib.util.spec_from_file_location(
        "smoke_genie", ROOT / "jobs" / "smoke_genie.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _serialized_space() -> dict:
    """Return the inline serialized_space of the bundle's Genie Agent."""
    resources = yaml.safe_load((ROOT / "resources" / "genie.yml").read_text())
    genie = resources["resources"]["genie_spaces"]["offer_performance_genie"]
    return genie["serialized_space"]


def test_genie_id_keyed_arrays_are_sorted_and_unique() -> None:
    """The Genie API rejects id-keyed arrays that are unsorted or reuse an id."""
    space = _serialized_space()
    instructions = space["instructions"]
    arrays = {
        "sample_questions": space["config"]["sample_questions"],
        "example_question_sqls": instructions["example_question_sqls"],
        "text_instructions": instructions["text_instructions"],
        "benchmarks": space["benchmarks"]["questions"],
    }
    for name, items in arrays.items():
        ids = [item["id"] for item in items]
        assert ids == sorted(ids), f"{name} must be sorted by id"
        assert all(len(i) == 32 and int(i, 16) >= 0 for i in ids), name
    all_ids = [i["id"] for items in arrays.values() for i in items]
    assert len(all_ids) == len(set(all_ids))
    columns = [
        c["column_name"]
        for c in space["data_sources"]["metric_views"][0]["column_configs"]
    ]
    assert columns == sorted(columns), "column_configs must be sorted by column_name"


def test_markdown_table_renders_nulls_as_blank() -> None:
    """Rows become pipe-delimited lines under a header and separator."""
    smoke = _load_smoke_genie()

    lines = smoke.markdown_table(["a", "b"], [["1", None]])

    assert lines == ["| a | b |", "|---|---|", "| 1 |  |"]


def test_sample_questions_read_from_space() -> None:
    """Sample questions come from the deployed space, in configured order."""
    smoke = _load_smoke_genie()
    config = {
        "config": {"sample_questions": [{"question": ["q1"]}, {"question": ["q2"]}]}
    }
    ws = MagicMock()
    ws.genie.get_space.return_value = SimpleNamespace(
        serialized_space=json.dumps(config)
    )

    questions = smoke._sample_questions(ws, "space")

    assert questions == ["q1", "q2"]
    ws.genie.get_space.assert_called_once_with("space", include_serialized_space=True)
