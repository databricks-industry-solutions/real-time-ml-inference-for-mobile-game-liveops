"""The scorer's async ranking call: predictions sort the slate, errors degrade."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

# The app is a flat Databricks App dir with sibling-module imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import serving_client

_SLATE = [{"offer_id": "offer_000"}, {"offer_id": "offer_001"}]
_TARGET = ("https://data-plane.example/invocations", "Bearer test-token")


class _FakeResponse:
    """Async-context-manager stand-in for an aiohttp response."""

    def __init__(self, status: int, payload: dict) -> None:
        self.status = status
        self._payload = payload

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    async def json(self) -> dict:
        return self._payload

    async def text(self) -> str:
        return "endpoint error body"

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


class _FakeSession:
    """Records each POST and returns a fixed response."""

    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.posts: list[tuple[str, dict, dict]] = []

    def post(self, url: str, json: dict, headers: dict) -> _FakeResponse:
        self.posts.append((url, json, headers))
        return self.response


def _rank(session: _FakeSession) -> tuple[list[dict], float, int]:
    """Run rank_offers_async against `session`; also return latency records made."""
    with (
        patch.object(serving_client, "_session", return_value=session),
        patch.object(serving_client, "_data_plane_target", return_value=_TARGET),
        patch.object(serving_client.metrics, "record_ranking") as record,
    ):
        result = asyncio.run(
            serving_client.rank_offers_async("player_1", _SLATE, {"gem_balance": 10})
        )
    offers, latency_ms = result
    return offers, latency_ms, record.call_count


def test_predictions_sort_offers_and_record_latency() -> None:
    # Arrange
    payload = {
        "predictions": [
            {"expected_offer_revenue": 1.0, "purchase_probability": 0.2},
            {"expected_offer_revenue": 3.0, "purchase_probability": 0.6},
        ]
    }
    session = _FakeSession(_FakeResponse(200, payload))

    # Act
    offers, latency_ms, recorded = _rank(session)

    # Assert
    assert [o["offer_id"] for o in offers] == ["offer_001", "offer_000"]
    assert latency_ms >= 0.0
    assert recorded == 1
    url, body, headers = session.posts[0]
    assert url == _TARGET[0]
    assert headers == {"Authorization": _TARGET[1]}
    assert len(body["dataframe_records"]) == len(_SLATE)


def test_http_error_returns_unscored_offers() -> None:
    # Arrange
    session = _FakeSession(_FakeResponse(503, {}))

    # Act
    offers, latency_ms, recorded = _rank(session)

    # Assert
    assert all(o["expected_offer_revenue"] is None for o in offers)
    assert latency_ms == 0.0
    assert recorded == 0
