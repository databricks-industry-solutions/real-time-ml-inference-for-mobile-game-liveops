"""Contract tests for the Lakeflow ingest pipeline (pipelines/ingest/).

The pipeline must publish exactly the refined tables the downstream jobs read, and
its bronze tables must ingest from the landing folder generate_data writes.
"""

from __future__ import annotations

import pathlib
import re

import yaml

from offer_recommender.config import LANDING_DIR, TABLE_IMPRESSIONS, TABLE_PLAYERS

ROOT = pathlib.Path(__file__).resolve().parent.parent
PIPELINE_DIR = ROOT / "pipelines" / "ingest"
DATASET = re.compile(
    r"CREATE OR REFRESH (STREAMING TABLE|MATERIALIZED VIEW) (\w+)", re.IGNORECASE
)


def _datasets() -> dict[str, str]:
    """Map each dataset the pipeline defines to the SQL file that defines it."""
    found: dict[str, str] = {}
    for path in sorted(PIPELINE_DIR.glob("*.sql")):
        for _, name in DATASET.findall(path.read_text()):
            found[name] = path.read_text()
    return found


def test_pipeline_publishes_the_tables_downstream_reads() -> None:
    """impressions, players and offers come from the pipeline, one file each."""
    datasets = _datasets()

    assert {TABLE_IMPRESSIONS, TABLE_PLAYERS, "offers"} <= set(datasets)
    assert {path.stem for path in PIPELINE_DIR.glob("*.sql")} == set(datasets)


def test_bronze_tables_use_auto_loader_on_the_landing_folder() -> None:
    """Raw tables stream the landed files; refined impressions carry expectations."""
    datasets = _datasets()

    for name in ("impressions_raw", "offers_raw"):
        source = name.removesuffix("_raw")
        assert f"STREAM read_files('${{landing_path}}/{source}'" in datasets[name]
    assert datasets[TABLE_IMPRESSIONS].count("ON VIOLATION DROP ROW") >= 1


def test_landing_path_points_at_the_generator_landing_dir() -> None:
    """The pipeline's landing_path ends in the folder generate_data writes."""
    resource = yaml.safe_load((ROOT / "resources" / "ingest.pipeline.yml").read_text())
    pipeline = resource["resources"]["pipelines"]["offer_recommender_ingest"]

    assert pipeline["configuration"]["landing_path"].endswith(f"/{LANDING_DIR}")
    assert pipeline["serverless"] is True
