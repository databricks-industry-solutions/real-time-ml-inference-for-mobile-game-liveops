"""Domain schema for the Harvest Haven offer recommender.

Single source of truth for feature names, categorical vocabularies and the
prediction targets. Every stage (data generation, SparkML preprocessing, the
PyTorch model, the feature store and serving) imports from here so the feature
contract cannot drift between training and serving.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class OfferType(StrEnum):
    """Categories of gem offers shown in the Harvest Haven shop."""

    STARTER = "starter"
    VALUE = "value"
    WHALE = "whale"
    SEASONAL = "seasonal"
    COMEBACK = "comeback"


class PriceTier(StrEnum):
    """Coarse price bucket of an offer, used as a categorical feature."""

    TIER_1 = "tier_1"  # ~0.99
    TIER_2 = "tier_2"  # ~4.99
    TIER_3 = "tier_3"  # ~19.99
    TIER_4 = "tier_4"  # ~49.99
    TIER_5 = "tier_5"  # ~99.99


# --- Feature groups ----------------------------------------------------------
# Categorical features become integer indices (StringIndexer) then embeddings.
# Split by serving provenance so the feature contract holds at inference:
#   - OFFLINE_CATEGORICAL: slow-moving player attributes, looked up from the
#     offline feature store by player_id (the app does NOT send these).
#   - OFFER_CATEGORICAL: attributes of the candidate offer, sent in the request
#     alongside player_id (passthrough, not a feature-store lookup).
# CATEGORICAL_FEATURES concatenates them in the canonical order used everywhere
# downstream (data gen, SparkML pipeline, the model's embedding order).
OFFLINE_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "country",
    "segment",
)
OFFER_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "offer_id",
    "offer_type",
    "price_tier",
)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    OFFLINE_CATEGORICAL_FEATURES + OFFER_CATEGORICAL_FEATURES
)

# Offline (batch, slow-moving) numerical features. Computed daily by Spark.
OFFLINE_NUMERICAL_FEATURES: Final[tuple[str, ...]] = (
    "lifetime_spend",
    "tenure_days",
    "historical_ltv",
    "avg_session_len_min",
    "purchases_30d",
)

# Real-time numerical features: the player's instantaneous snapshot at the moment a
# login triggers scoring. REQUEST-TIME PASSTHROUGH (RequestSource), NOT an online
# lookup -- the app sends the just-consumed login's values in the serving request
# (millisecond-fresh), so they are never materialised. gem_balance (instantaneous
# balance) and minutes_since_last_purchase (historical recency fact) are genuinely
# known at a login. match_count is KEPT: the label's urgency term (match outcome
# drives purchase) leans on the instantaneous count, so the model must see it at
# inference. Dropped as not-knowable-at-login: session_depth (belongs in SFV
# RollingWindow only).
REALTIME_NUMERICAL_FEATURES: Final[tuple[str, ...]] = (
    "gem_balance",
    "minutes_since_last_purchase",
    "match_count",
)

# Offer-side numerical features (attributes of the candidate offer).
OFFER_NUMERICAL_FEATURES: Final[tuple[str, ...]] = (
    "offer_price_usd",
    "offer_gem_amount",
    "offer_discount_pct",
)

# Streaming rolling-window aggregates the MODEL trains on (5). A strict subset of
# WINDOWS (below): match_events_15m is materialized and served and shown in the app but
# NOT a model input (it == match_wins_15m and match_losses_15m, pure collinearity).
STREAMING_NUMERICAL_FEATURES: Final[tuple[str, ...]] = (
    "gems_spent_15m",
    "avg_gem_balance_15m",
    "offers_seen_15m",
    "match_wins_15m",
    "match_losses_15m",
)

# Rolling-window aggregate spec: feature column -> (operator, source signal,
# window seconds). Single source of truth shared by the SFV feature builder
# (features.store._streaming_agg_features / _operator). All signals are real
# per-event values (AVG or SUM) except match_events_15m which counts raw login rows.
# Kept here (not in data/) so the feature layer imports it without depending on the
# data-generation module. Windows are deliberately SHORT (<=15 min): a managed
# streaming online aggregation is empty/inaccurate until its window has fully elapsed,
# so the online store only serves real values ~max-window after the stream is
# (re)created. Short windows make that warm-up demo-viable (~15 min) and the features
# read as CURRENT-SESSION intensity, which is what a shop-open ranker actually wants.
#
# CAUSAL DESIGN:
#   gems_spent_15m        : spend velocity; correlates positively with purchase intent.
#   avg_gem_balance_15m   : gem-block proximity; LOW balance is the buy trigger
#                           (propensity term uses 1 - clipped(balance/cap)).
#   offers_seen_15m       : offer fatigue (negative signal in propensity).
#   match_wins_15m        : recent match win count; wins lift purchase propensity.
#   match_losses_15m      : recent match loss count; losses suppress purchase propensity.
#   match_events_15m      : raw login event count over the window (not a model input;
#                           collinear with wins and losses).
_M: Final[float] = 60.0
WINDOWS: Final[dict[str, tuple[str, str, float]]] = {
    "gems_spent_15m": ("sum", "gem_spend", 15 * _M),
    "avg_gem_balance_15m": ("avg", "gem_balance", 15 * _M),
    "offers_seen_15m": ("sum", "offer_shown", 15 * _M),
    "match_wins_15m": ("sum", "match_won", 15 * _M),
    "match_losses_15m": ("sum", "match_lost", 15 * _M),
    "match_events_15m": ("count", "login", 15 * _M),
}

NUMERICAL_FEATURES: Final[tuple[str, ...]] = (
    OFFLINE_NUMERICAL_FEATURES
    + REALTIME_NUMERICAL_FEATURES
    + OFFER_NUMERICAL_FEATURES
    + STREAMING_NUMERICAL_FEATURES
)

# Numerical features that are log1p-transformed before scaling (heavy right
# skew: money and counts). Applied by SQLTransformer in the SparkML pipeline
# and mirrored in the in-model transform.
LOG1P_FEATURES: Final[tuple[str, ...]] = (
    "lifetime_spend",
    "historical_ltv",
    "gem_balance",
    "offer_price_usd",
    "offer_gem_amount",
    "gems_spent_15m",
    "match_wins_15m",
    "match_losses_15m",
)

# --- Targets -----------------------------------------------------------------
# Primary head: expected offer revenue at a d7 horizon (regression, drives
# ranking). Auxiliary head: did the player purchase the offer (binary).
TARGET_REVENUE_D7: Final[str] = "offer_revenue_d7"
TARGET_PURCHASED: Final[str] = "purchased"

# --- Keys --------------------------------------------------------------------
ENTITY_KEY: Final[str] = "player_id"
EVENT_TIMESTAMP: Final[str] = "event_ts"
