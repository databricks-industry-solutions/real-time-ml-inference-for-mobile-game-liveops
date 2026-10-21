"""Offer catalogue, player sampling and feature-split helpers for the app.

On Databricks these read the project's UC tables and the online feature store;
in local dev (no Databricks SDK / warehouse) they fall back to a small
synthetic catalogue so the console still renders and demos offline. Keeps the
app free of heavy ML deps: it only shapes data for the UI and delegates scoring
to the serving endpoint.
"""

from __future__ import annotations

import logging
import os
import random

import serving_client

logger = logging.getLogger(__name__)

# Feature-group membership for partitioning the Feature Serving response into the
# drill-down's offline (daily batch) vs real-time (live stream) groups. Kept in
# sync with offer_recommender.schema; duplicated here so the app has no ML deps.
_OFFLINE_FEATURE_NAMES = (
    "lifetime_spend",
    "tenure_days",
    "historical_ltv",
    "avg_session_len_min",
    "purchases_30d",
    "country",
    "segment",
)
_REALTIME_FEATURE_NAMES = (
    "gem_balance",
    "minutes_since_last_purchase",
    "match_count",
)
_STREAMING_FEATURE_NAMES = (
    "gems_spent_15m",
    "avg_gem_balance_15m",
    "offers_seen_15m",
    "match_wins_15m",
    "match_losses_15m",
    "match_events_15m",
)

_SEGMENTS = ("minnow", "dolphin", "whale", "new", "lapsed")
_OFFER_TYPES = ("starter", "value", "whale", "seasonal", "comeback")
_PRICE_TIERS = {
    "tier_1": (0.99, 80),
    "tier_2": (4.99, 500),
    "tier_3": (19.99, 2400),
    "tier_4": (49.99, 6500),
    "tier_5": (99.99, 14000),
}


def _on_databricks() -> bool:
    """True when running inside a Databricks App (use the real serving endpoints).

    The app needs NO warehouse/Postgres of its own -- it reads everything through
    the offer-ranker-rt / offer-features-fs serving endpoints via a WorkspaceClient.
    The Databricks Apps runtime always sets DATABRICKS_APP_NAME, so that (or an
    explicitly-configured serving endpoint) is the signal we're live, not a
    warehouse. (The old DATABRICKS_WAREHOUSE_ID/PGHOST check was a leftover from a
    removed warehouse path and kept the app stuck in synthetic-demo mode.)
    """
    return bool(
        os.environ.get("DATABRICKS_APP_NAME")
        or os.environ.get("SERVING_ENDPOINT_NAME")
        or os.environ.get("DATABRICKS_WAREHOUSE_ID")
        or os.environ.get("PGHOST")
    )


_N_OFFERS = 5


def _offer(offer_num: int) -> dict:
    """Build one offer with attributes seeded by its id, so the same offer_id
    always has the same price/type/tier (coherent across slates)."""
    rng = random.Random(offer_num)
    tier, (price, gems) = rng.choice(list(_PRICE_TIERS.items()))
    return {
        "offer_id": f"offer_{offer_num:03d}",
        "offer_type": rng.choice(_OFFER_TYPES),
        "price_tier": tier,
        "offer_price_usd": price,
        "offer_gem_amount": float(gems),
        "offer_discount_pct": round(rng.uniform(0, 0.4), 3),
    }


# The full catalogue is a stable set of coherent offers built once.
_CATALOGUE = [_offer(i) for i in range(_N_OFFERS)]


def candidate_slate(player_id: str, n: int = 5) -> list[dict]:
    """Return a candidate offer slate of `n` DISTINCT offers for the player.

    With only 5 offers in the catalogue, every login is scored against the FULL
    set (n defaults to the catalogue size), so the recommendation-mix chart shows
    exactly 5 lines. Sampled without replacement so no offer_id repeats in a slate.
    Each offer carries the offer-side columns the serving endpoint needs (customer
    values are looked up by the endpoint from the online store).
    """
    rng = random.Random(hash(player_id) & 0xFFFFFFFF)
    return rng.sample(_CATALOGUE, min(n, len(_CATALOGUE)))


def random_active_player() -> str:
    """Return a random player id for the live feed."""
    return f"player_{random.randrange(20000):06d}"


def player_segment(player_id: str) -> str:
    """Return the player's segment (deterministic per player in local mode)."""
    rng = random.Random(hash(player_id) & 0xFFFFFFFF)
    return rng.choice(_SEGMENTS)


def feature_split(player_id: str) -> dict:
    """Return the player's features grouped into offline (batch) vs real-time.

    On Databricks the real values are fetched from the Feature Serving endpoint
    (`serving_client.fetch_features`, keyed by player_id) and partitioned into the
    two groups by name; this is the point of the drill-down: which signals are
    slow-moving daily-batch vs live-from-the-session streaming. Falls back to
    deterministic synthetic values when the endpoint is unreachable (local dev) so
    the console still renders. Each group carries a `_source` label for the UI.
    """
    served = serving_client.fetch_features(player_id)
    if served:
        offline = {k: served[k] for k in _OFFLINE_FEATURE_NAMES if k in served}
        # Real-time features are RequestSource PASSTHROUGH -- the model receives them
        # in the live login's scoring request; they are NOT materialized online, so
        # offer-features-fs does not return them and a drill-down on an
        # arbitrarily-clicked player (no live login) has no real value for them. We
        # do NOT invent them. The drill-down shows only the two groups that ARE
        # recoverable by player_id: offline (batch online lookup) and streaming (SFV).
        # Streaming online aggregates are NULL until their rolling window has elapsed
        # (or for a player with no in-window events) -> coalesce to 0.0 (truthful "no
        # activity", matching the model's transform.py fill).
        streaming = {
            k: (0.0 if served[k] is None else round(float(served[k]), 2))
            for k in _STREAMING_FEATURE_NAMES
            if k in served
        }
        if offline or streaming:
            return {
                "offline": offline,
                "streaming": streaming,
                "offline_source": "daily batch · online lookup",
                "streaming_source": "15-min rolling window · SFV online lookup",
            }

    rng = random.Random(hash(player_id) & 0xFFFFFFFF)
    offline = {
        "lifetime_spend": round(rng.uniform(0, 800), 2),
        "tenure_days": rng.randrange(1, 900),
        "historical_ltv": round(rng.uniform(0, 1200), 2),
        "avg_session_len_min": round(rng.uniform(3, 40), 1),
        "purchases_30d": rng.randrange(0, 12),
    }
    streaming = {
        "gems_spent_15m": round(rng.uniform(0, 150), 2),
        "avg_gem_balance_15m": round(rng.uniform(0, 3000)),
        "offers_seen_15m": rng.randrange(0, 60),
        "match_wins_15m": rng.randrange(0, 150),
        "match_losses_15m": rng.randrange(0, 150),
        "match_events_15m": rng.randrange(0, 300),
    }
    return {
        "offline": offline,
        "streaming": streaming,
        "offline_source": "daily batch · online lookup (demo)",
        "streaming_source": "15-min rolling window · SFV online lookup (demo)",
    }


def model_status() -> dict:
    """Return model/run metadata for the run panel."""
    return {
        "version": os.environ.get("RECO_MODEL_VERSION", "dev"),
        "mode": "databricks" if _on_databricks() else "local-demo",
    }


_PROVENANCE_CACHE: dict | None = None


def provenance() -> dict:
    """Return training provenance for the header strip (cached).

    Surfaces the AIR+Ray headline from the MLflow `train_air` experiment: the
    latest finished run's params (epochs, workers, lr) + final loss, plus the
    served model version. Read via the SDK; degrades to static text off-workspace.
    """
    global _PROVENANCE_CACHE
    if _PROVENANCE_CACHE is not None:
        return _PROVENANCE_CACHE

    base = {
        "model_version": os.environ.get("RECO_MODEL_VERSION", "dev"),
        "trained_on": "Ray Train · serverless GPU (Databricks AI Runtime)",
        "loss": None,
        "epochs": None,
        "num_workers": None,
    }
    if not _on_databricks():
        _PROVENANCE_CACHE = base
        return base
    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        user = w.current_user.me().user_name
        exp = w.experiments.get_by_name(f"/Users/{user}/train_air").experiment
        runs = w.experiments.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=20,
        )
        for r in runs:
            params = {p.key: p.value for p in (r.data.params or [])}
            metrics = {m.key: m.value for m in (r.data.metrics or [])}
            if "epochs" in params and "loss" in metrics:
                base.update(
                    loss=round(float(metrics["loss"]), 3),
                    epochs=int(params["epochs"]),
                    num_workers=int(params.get("num_workers", 1)),
                )
                break
    except Exception as e:  # observability, never break the app
        logger.warning("Provenance lookup failed: %s", e)
    _PROVENANCE_CACHE = base
    return base
