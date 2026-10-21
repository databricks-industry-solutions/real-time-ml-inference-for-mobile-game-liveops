"""Smoke test: query the deployed offer-ranker endpoint end to end.

Confirms the serving contract works on a live endpoint: send a real `player_id`
plus a candidate-offer slate, and verify the endpoint resolves the player's
offline + real-time features from the Lakebase online store (via the model's
embedded FeatureSpec) and returns a sane expected-revenue score per candidate.

This exercises the piece unit tests cannot: that features auto-resolve from the
online store at inference. Run locally against the workspace once the endpoint
is READY:

    RECO_CATALOG=<catalog> RECO_WAREHOUSE_ID=<warehouse-id> \
    python jobs/smoke_serving_inference.py

Uses the CLI profile's auth (WorkspaceClient()). Prints PASS/FAIL and the scored
slate. Not wired into the DAB pipeline; a manual post-deploy verification.
"""

from __future__ import annotations

from offer_recommender.config import get_config, get_warehouse_id
from offer_recommender.logging_config import get_logger

logger = get_logger(__name__)

ENDPOINT = "offer-ranker-rt"


def _sql(warehouse_id: str, statement: str) -> list[dict]:
    """Run a SQL statement on the warehouse and return rows as dicts."""
    from databricks.sdk import WorkspaceClient

    ws = WorkspaceClient()
    resp = ws.statement_execution.execute_statement(
        warehouse_id=warehouse_id, statement=statement, wait_timeout="30s"
    )
    cols = [c.name for c in resp.manifest.schema.columns]
    return [dict(zip(cols, row, strict=True)) for row in (resp.result.data_array or [])]


def _candidate_slate(warehouse_id: str, catalog: str, schema: str) -> list[dict]:
    """Pull a few real offers to score (offer-side columns the request needs)."""
    rows = _sql(
        warehouse_id,
        f"SELECT offer_id, offer_type, price_tier, offer_price_usd, "
        f"offer_gem_amount, offer_discount_pct FROM {catalog}.{schema}.offers LIMIT 5",
    )
    if not rows:
        raise RuntimeError("no offers found; run generate_data first")
    return rows


def _a_player(warehouse_id: str, catalog: str, schema: str) -> str:
    """Return one real player_id present in the offline feature table."""
    rows = _sql(
        warehouse_id,
        f"SELECT player_id FROM {catalog}.{schema}.player_features_offline LIMIT 1",
    )
    if not rows:
        raise RuntimeError("no players in offline features; run publish_features first")
    return rows[0]["player_id"]


def main() -> None:
    """Query the endpoint for one player x candidate slate and validate output."""
    from databricks.sdk import WorkspaceClient

    uc = get_config()
    warehouse_id = get_warehouse_id()
    player_id = _a_player(warehouse_id, uc.catalog, uc.schema)
    offers = _candidate_slate(warehouse_id, uc.catalog, uc.schema)
    records = [{"player_id": player_id, **offer} for offer in offers]

    ws = WorkspaceClient()
    resp = ws.serving_endpoints.query(name=ENDPOINT, dataframe_records=records)
    preds = resp.predictions

    revenues = [
        p.get("expected_offer_revenue") if isinstance(p, dict) else p for p in preds
    ]
    logger.info("player_id=%s", player_id)
    for offer, rev in zip(offers, revenues, strict=True):
        logger.info(
            "  offer=%s type=%s -> $%.4f", offer["offer_id"], offer["offer_type"], rev
        )

    # Validate: one score per candidate, all finite non-negative dollars, and not
    # all identical (the model should differentiate offers for a player).
    assert len(revenues) == len(offers), "prediction count != candidate count"
    assert all(r is not None and r >= 0 for r in revenues), f"bad revenues: {revenues}"
    assert len(set(round(r, 6) for r in revenues)) > 1, "all offers scored identically"
    logger.info("PASS: endpoint resolved features and scored %d offers", len(offers))


if __name__ == "__main__":
    main()
