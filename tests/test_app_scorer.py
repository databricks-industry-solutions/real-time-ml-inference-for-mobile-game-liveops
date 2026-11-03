"""The live board only records logins that came back with a scored offer."""

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# The app is a flat Databricks App dir with sibling-module imports, and imports the
# usage tracker that is only installed in the app runtime.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
_tracker = types.ModuleType("dbdemos_tracker")
_tracker.Tracker = object
sys.modules.setdefault("dbdemos_tracker", _tracker)

import app


def _event(revenue: float | None) -> dict:
    """A feed event whose chosen offer carries `revenue` (None = unscored)."""
    offer = {"offer_id": "offer_000", "expected_offer_revenue": revenue}
    return {"player_id": "player_1", "chosen_offer": offer, "ranked_offers": [offer]}


@pytest.mark.parametrize(("revenue", "recorded"), [(1.5, True), (None, False)])
def test_score_and_record_drops_unscored(revenue: float | None, recorded: bool) -> None:
    # Arrange
    async def fake_rank(player_id: str, realtime: dict | None = None) -> dict:
        return _event(revenue)

    # Act
    with (
        patch.object(app, "_rank_event_async", fake_rank),
        patch.object(app.aggregate, "record") as record,
    ):
        asyncio.run(app._score_and_record({"player_id": "player_1"}))

    # Assert
    assert record.called is recorded
