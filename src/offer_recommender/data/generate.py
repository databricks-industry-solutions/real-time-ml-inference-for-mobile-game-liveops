"""Synthetic Harvest Haven data generator.

Produces three reproducible datasets with no real PII:

- players: one row per player with slow-moving offline attributes.
- offers: the gem-offer catalogue (candidate items to rank).
- impressions: one row per (player, shop-open, candidate offer) with the
  features seen at decision time and the two targets (offer revenue at a d7
  horizon, and whether the offer was purchased).

The numpy core (`generate_frames`) runs locally for tests and small demos and
returns pandas DataFrames. `write_delta` is the thin Spark wrapper used by the
Databricks job to produce the genuinely large impression log that makes
data-parallel training honest.

Causal structure (post data-realism pass):
- Impression `event_ts` is drawn FROM each player's session timeline (md5-indexed
  choice among that player's session events over the spine window), so every shop-open
  sits inside an active session and rolling-window counts evaluated AT that ts are
  naturally non-zero.
- `_purchase_propensity` includes a MODEST session-intensity term using the 6 WINDOWS
  streaming features: gem-block proximity (low avg_gem_balance_15m lifts propensity),
  losing streak (match_losses_15m LIFTS: a frustrated player buys gems for an edge),
  offer fatigue (offers_seen_15m suppresses), gems_spent_15m mild positive.
  A recency term (minutes_since_last_purchase) further modulates propensity: smaller
  values (recent purchase) lift it, reflecting cross-sell behaviour.
- The 6 WINDOWS streaming counts are computed internally at label-generation time only
  (via `sessions.window_counts_batch`, chunked in ~50k-row batches to bound memory)
  and are DISCARDED from the returned impression frame. Training values come from the
  Kafka stream via preprocess's PIT join; writing a second source would introduce
  train/serve skew.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass

import numpy as np
import pandas as pd

from offer_recommender.data.hashing import md5_uniform_np
from offer_recommender.logging_config import get_logger
from offer_recommender.schema import (
    TARGET_PURCHASED,
    TARGET_REVENUE_D7,
    OfferType,
    PriceTier,
)

# Chunk size for window_counts_batch calls (bounds peak memory on the full spine).
_BATCH_CHUNK = 50_000

# Local numpy generation parallelism (the Databricks job uses generate_frames_spark,
# so this only affects local runs + tests). generate_frames is embarrassingly parallel
# per player -- each player seeds its own RNG from md5_seed(player_id) -- so chunks run
# in separate processes and concatenate in order to a bit-identical result (guarded by
# tests/test_generate_distributed_parity.py's shard-invariance). The per-cell md5 work
# in window_counts_batch (Python hashlib loop over ~6000 events/player) is CPU-bound and
# does not release the GIL, so processes (not threads) are the win. Threshold is low so
# even the ~150-player test tier fans out across cores; only trivially small runs stay
# single-process (spawn + pickle overhead is not worth it below it).
_PARALLEL_MIN_PLAYERS = 64
# Target ~2 chunks per worker so a straggler chunk cannot leave a core idle; the real
# chunk size is derived per run from n_players / (2 * workers), floored at this.
_PARALLEL_MIN_CHUNK = 24


def _gen_workers(n_players: int) -> int:
    """Return the process count for local generation.

    ``RECO_GEN_WORKERS`` overrides (0/1 forces single-process, e.g. inside a Spark
    worker or to avoid a nested pool). Default: all cores minus one, only above
    ``_PARALLEL_MIN_PLAYERS`` and capped so each worker gets a meaningful chunk.
    """
    override = os.environ.get("RECO_GEN_WORKERS")
    if override is not None:
        return max(1, int(override))
    if n_players < _PARALLEL_MIN_PLAYERS:
        return 1
    max_by_chunk = max(1, n_players // _PARALLEL_MIN_CHUNK)
    return max(1, min((os.cpu_count() or 2) - 1, max_by_chunk))


def _gen_chunk_size(n_players: int, n_workers: int) -> int:
    """Chunk size that spreads n_players over ~2 rounds per worker (load balance)."""
    return max(_PARALLEL_MIN_CHUNK, -(-n_players // (n_workers * 2)))


logger = get_logger(__name__)

# Countries and player segments used as categorical features. Kept small and
# fixed so the vocab is stable across runs.
COUNTRIES: tuple[str, ...] = (
    "US",
    "FI",
    "DE",
    "GB",
    "FR",
    "JP",
    "KR",
    "BR",
    "SE",
    "CA",
)
SEGMENTS: tuple[str, ...] = ("minnow", "dolphin", "whale", "new", "lapsed")

# Price tier -> (usd, gem_amount) anchor. Gems scale a little better than
# linearly with price (bulk discount), which the discount feature encodes.
_TIER_PRICE: dict[str, tuple[float, int]] = {
    PriceTier.TIER_1: (0.99, 80),
    PriceTier.TIER_2: (4.99, 500),
    PriceTier.TIER_3: (19.99, 2400),
    PriceTier.TIER_4: (49.99, 6500),
    PriceTier.TIER_5: (99.99, 14000),
}

# Segment -> baseline daily spend propensity (drives lifetime spend + convert).
_SEGMENT_PROPENSITY: dict[str, float] = {
    "new": 0.15,
    "minnow": 0.25,
    "dolphin": 0.55,
    "whale": 0.9,
    "lapsed": 0.1,
}

# Ordered tier values matching PriceTier enum (tier_1 .. tier_5).
_TIERS: tuple[str, ...] = tuple(t.value for t in PriceTier)
_TIER_IDX: dict[str, int] = {t: i for i, t in enumerate(_TIERS)}

# Segment x tier unnormalized affinity weights (SEGMENTS order x _TIERS order).
# Every entry is strictly positive so every offer has nonzero probability for
# every segment — vocab stability is preserved regardless of which tiers the
# random catalogue happens to contain.
# Design intent: minnow/new concentrate on cheap tiers; whale on expensive
# tiers; dolphin prefers mid; lapsed is flat with a slight mid preference.
_SEGMENT_TIER_AFFINITY: dict[str, list[float]] = {
    "minnow": [8.0, 4.0, 2.0, 1.0, 0.5],
    "dolphin": [2.0, 4.0, 5.0, 3.0, 1.0],
    "whale": [0.5, 1.0, 2.0, 5.0, 8.0],
    "new": [8.0, 4.0, 2.0, 1.0, 0.5],
    "lapsed": [1.5, 3.0, 3.0, 2.0, 1.5],
}


@dataclass(frozen=True)
class GenConfig:
    """Sizes and knobs for a synthetic-data run."""

    n_players: int = 5_000
    n_offers: int = 5
    impressions_per_player: int = 8
    seed: int = 17
    span_days: int = 14
    # Anchor the impression window to end `end_offset_hours` before "now" so the
    # spine sits inside the live-stream-covered range (SFV PIT recompute needs
    # ingestion events at those timestamps). Ignored when `end_ts` is set.
    end_offset_hours: float = 0.0
    # Fixed window end for deterministic runs (tests). When None, derived from
    # utcnow() minus end_offset_hours so production runs get a recent window.
    end_ts: pd.Timestamp | None = None


def _build_offers(rng: np.random.Generator, n_offers: int) -> pd.DataFrame:
    """Build the gem-offer catalogue."""
    offer_types = np.array([t.value for t in OfferType])
    tiers = np.array([t.value for t in PriceTier])
    type_idx = rng.integers(0, len(offer_types), n_offers)
    tier_idx = rng.integers(0, len(tiers), n_offers)
    price = np.array([_TIER_PRICE[tiers[i]][0] for i in tier_idx])
    gems = np.array([_TIER_PRICE[tiers[i]][1] for i in tier_idx], dtype=float)
    # Seasonal/comeback offers carry a discount; others rarely do.
    discount = np.where(
        np.isin(offer_types[type_idx], [OfferType.SEASONAL, OfferType.COMEBACK]),
        rng.uniform(0.1, 0.5, n_offers),
        rng.uniform(0.0, 0.1, n_offers),
    )
    return pd.DataFrame(
        {
            "offer_id": [f"offer_{i:03d}" for i in range(n_offers)],
            "offer_type": offer_types[type_idx],
            "price_tier": tiers[tier_idx],
            "offer_price_usd": price.round(2),
            "offer_gem_amount": gems,
            "offer_discount_pct": discount.round(3),
        }
    )


def _build_players(rng: np.random.Generator, n_players: int) -> pd.DataFrame:
    """Build the player table with slow-moving offline attributes."""
    segment = rng.choice(SEGMENTS, n_players, p=[0.35, 0.3, 0.05, 0.2, 0.1])
    propensity = np.array([_SEGMENT_PROPENSITY[s] for s in segment])
    tenure = rng.integers(1, 900, n_players)
    # Lifetime spend correlates with propensity and tenure, heavy right tail.
    lifetime_spend = (
        rng.gamma(shape=1.5, scale=1.0, size=n_players) * propensity * tenure * 0.5
    ).round(2)
    return pd.DataFrame(
        {
            "player_id": [f"player_{i:06d}" for i in range(n_players)],
            "country": rng.choice(COUNTRIES, n_players),
            "segment": segment,
            "tenure_days": tenure,
            "lifetime_spend": lifetime_spend,
            "historical_ltv": (lifetime_spend * rng.uniform(1.0, 1.6, n_players)).round(
                2
            ),
            "avg_session_len_min": (
                propensity * 20 + rng.uniform(2, 15, n_players)
            ).round(2),
            "purchases_30d": rng.poisson(propensity * 6, n_players),
            "_propensity": propensity,  # latent, dropped before returning
        }
    )


def _offer_weights_for_segments(
    offers: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Build a per-segment probability distribution over offer indices.

    Each offer's weight for a segment equals the segment's tier-affinity score
    for that offer's price_tier, then normalised to sum to 1. Every weight is
    strictly positive, so no offer is ever excluded from any segment — the
    StringIndexer vocab cannot lose categories due to exposure bias.

    Args:
        offers: The offer catalogue produced by ``_build_offers``. Must contain
            a ``price_tier`` column with ``PriceTier`` values.

    Returns:
        Mapping from segment name to a probability array of length
        ``len(offers)``, one entry per row.
    """
    tier_indices = np.array([_TIER_IDX[t] for t in offers["price_tier"]])
    result: dict[str, np.ndarray] = {}
    for seg, affinity in _SEGMENT_TIER_AFFINITY.items():
        af = np.array(affinity)
        weights = af[tier_indices]
        result[seg] = weights / weights.sum()
    return result


def _resolve_end(cfg: GenConfig) -> pd.Timestamp:
    """Return the observation-window end timestamp.

    Uses ``cfg.end_ts`` when set (deterministic / test mode). Falls back to
    ``utcnow() - end_offset_hours`` for production runs. Extracted so tests can
    monkeypatch this single callsite instead of ``pd.Timestamp.now``.
    """
    if cfg.end_ts is not None:
        return cfg.end_ts
    return pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(
        hours=cfg.end_offset_hours
    )


def _purchase_propensity(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Latent per-impression purchase probability in [0, 1].

    Combines segment propensity, offer fit (whales like big tiers), real-time
    urgency (low gem balance, active match count), offer discount, a MODEST
    session-intensity term using the 6 WINDOWS streaming features, and a
    recency term.

    Session-intensity (reads schema.WINDOWS columns):
    - Gem-block proximity LIFTS: low avg_gem_balance_15m means the player is near
      zero gems and primed to buy. Term uses (1 - clipped(balance/GEM_BALANCE_CAP)).
    - Losing streak LIFTS: match_losses_15m captures a frustrated player who buys
      gems for an edge. Positive coefficient.
    - Offer fatigue SUPPRESSES: offers_seen_15m (negative coefficient).
    - Spend momentum adds mild positive: gems_spent_15m (log1p-normalised).

    Recency: minutes_since_last_purchase is converted to a [0,1] engagement score
    (recent -> high) so short inter-purchase gaps lift propensity (cross-sell).

    Weights are intentionally small (<0.15) so segment/offer-fit remains the
    dominant signal; the streaming counts add learnable but not overwhelming lift.

    Parameters
    ----------
    df:
        Impression frame with columns: _propensity, offer_price_usd, gem_balance,
        match_count, offer_discount_pct, minutes_since_last_purchase, and all 6
        schema.WINDOWS columns.
    rng:
        NumPy random generator for the logit noise term.

    Returns
    -------
    np.ndarray
        Per-row purchase probability in (0, 1), shape ``(len(df),)``.
    """
    from offer_recommender.data.sessions import GEM_BALANCE_CAP

    base = df["_propensity"].to_numpy()
    price = df["offer_price_usd"].to_numpy()
    gem_balance = df["gem_balance"].to_numpy()
    match_count = df["match_count"].to_numpy()
    discount = df["offer_discount_pct"].to_numpy()
    minutes_recency = df["minutes_since_last_purchase"].to_numpy()

    # Streaming rolling-window counts (label-side only; not returned in imp frame).
    gems_spent_15m = df["gems_spent_15m"].to_numpy()
    avg_gem_balance_15m = df["avg_gem_balance_15m"].to_numpy()
    offers_seen_15m = df["offers_seen_15m"].to_numpy()
    match_losses_15m = df["match_losses_15m"].to_numpy()

    # Offer-fit: high-propensity players tolerate higher prices; low-propensity
    # players are put off by them. Centred so mid prices are neutral.
    price_fit = (base - 0.5) * np.log1p(price) * 0.4
    # Urgency: a player low on gems mid-match is primed to buy.
    low_gems = np.clip(1.0 - gem_balance / 2000.0, 0, 1)
    urgency = low_gems * np.clip(match_count / 10.0, 0, 1) * 0.6

    # Session-intensity term (small weights).
    # avg_gem_balance_15m: LOW balance is the buy trigger (gem-block proximity).
    # Term is DECREASING in balance — opposite direction to the raw urgency term.
    gem_block = np.clip(1.0 - avg_gem_balance_15m / GEM_BALANCE_CAP, 0.0, 1.0) * 0.12
    # Losing streak LIFTS: a frustrated player on a losing streak buys gems for an edge.
    losing_streak = np.clip(match_losses_15m / 20.0, 0, 1) * 0.12
    # Offer fatigue: negative coefficient — each recent shown offer reduces odds.
    fatigue = np.clip(offers_seen_15m / 5.0, 0, 1) * 0.12
    # Spend momentum: log1p-normalised, mild positive.
    spend_momentum = np.log1p(gems_spent_15m) / np.log1p(100.0) * 0.07

    # Recency: sigmoid-like decay from 0..1 over a 2-hour window (120 min).
    # Recent purchase (short gap) -> score near 1; stale -> near 0.
    recency_score = np.exp(-minutes_recency / 120.0) * 0.12

    logit = (
        -1.6
        + 2.4 * base
        + price_fit
        + urgency
        + 1.2 * discount
        + gem_block
        + losing_streak
        - fatigue
        + spend_momentum
        + recency_score
        + rng.normal(0, 0.15, len(df))
    )
    return 1.0 / (1.0 + np.exp(-logit))


def _generate_player_block(
    player_ids: list[str],
    offers: pd.DataFrame,
    ext_window_start_s: int,
    ext_window_end_s: int,
    spine_start_s: int,
    spine_end_s: int,
    cfg: GenConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate players and impressions for a given set of player IDs.

    Uses per-player md5-seeded RNGs so generation is independent per player
    and shardable across Spark executors.
    """
    from offer_recommender import schema as _schema
    from offer_recommender.data import sessions as _sessions
    from offer_recommender.data.hashing import md5_seed

    n_players = len(player_ids)
    if n_players == 0:
        return pd.DataFrame(), pd.DataFrame()

    player_rows = []
    for player_id in player_ids:
        rng_p = np.random.default_rng(md5_seed(f"{player_id}:{cfg.seed}"))
        segment = str(rng_p.choice(SEGMENTS, 1, p=[0.35, 0.3, 0.05, 0.2, 0.1])[0])
        propensity = _SEGMENT_PROPENSITY[segment]
        tenure = int(rng_p.integers(1, 900))
        lifetime_spend = round(rng_p.gamma(1.5, 1.0) * propensity * tenure * 0.5, 2)
        historical_ltv = round(lifetime_spend * rng_p.uniform(1.0, 1.6), 2)
        avg_session_len_min = round(propensity * 20 + rng_p.uniform(2, 15), 2)
        purchases_30d = int(rng_p.poisson(propensity * 6))
        country = str(rng_p.choice(COUNTRIES))

        player_rows.append(
            {
                "player_id": player_id,
                "country": country,
                "segment": segment,
                "tenure_days": tenure,
                "lifetime_spend": lifetime_spend,
                "historical_ltv": historical_ltv,
                "avg_session_len_min": avg_session_len_min,
                "purchases_30d": purchases_30d,
                "_propensity": propensity,
            }
        )

    players_pdf = pd.DataFrame(player_rows)
    ext_span_s: float = float(ext_window_end_s - ext_window_start_s)
    seg_weights = _offer_weights_for_segments(offers)

    # Precompute the shared shape once (segment-independent).
    max_k = _sessions._segment_max_events()
    event_idx_full = np.arange(max_k)
    session_col = event_idx_full // _sessions.EVENTS_PER_SESSION
    n_imp = cfg.impressions_per_player

    per_player_frames: list[pd.DataFrame] = []
    for player_idx, player_id in enumerate(player_ids):
        rng_p = np.random.default_rng(md5_seed(f"{player_id}:{cfg.seed}"))
        player_data = players_pdf.iloc[player_idx]
        segment = player_data["segment"]
        propensity = player_data["_propensity"]

        # --- Per-impression scalar draws, in the SAME rng sequence as the
        # reference (offer, gem_balance, minutes_since_last_purchase, match_count)
        # per impression, so the per-player RNG stream is byte-identical. ---
        weights = seg_weights[segment]
        cum_weights = np.cumsum(weights)
        offer_idxs = np.empty(n_imp, dtype=np.int64)
        gem_balances = np.empty(n_imp)
        minutes_arr = np.empty(n_imp)
        match_counts = np.empty(n_imp, dtype=np.int64)
        for i in range(n_imp):
            u = rng_p.uniform(0.0, 1.0)
            offer_idxs[i] = int(np.clip((cum_weights < u).sum(), 0, cfg.n_offers - 1))
            gem_balances[i] = round(rng_p.gamma(2.0, 400.0), 0)
            minutes_arr[i] = round(rng_p.exponential(4000), 1)
            match_counts[i] = int(rng_p.integers(1, _sessions.MATCH_COUNT_MAX))

        # --- Session timeline built ONCE per player (not per impression). Pure md5,
        # vectorised; this is the hot path the old per-impression loop rebuilt 20x. ---
        sess_keys = np.array(
            [f"{player_id}:{s}:sess:{cfg.seed}" for s in session_col], dtype=object
        )
        session_start_ts = float(ext_window_start_s) + md5_uniform_np(sess_keys) * (
            ext_span_s - _sessions.BURST_SPAN_S
        )
        ts_keys = np.array(
            [f"{player_id}:{e}:ts:{cfg.seed}" for e in event_idx_full], dtype=object
        )
        T = session_start_ts + md5_uniform_np(ts_keys) * _sessions.BURST_SPAN_S
        event_count_val = int(
            _sessions.event_count(np.array([player_id]), np.array([segment]), cfg.seed)[
                0
            ]
        )
        valid_mask = event_idx_full < event_count_val
        in_spine_mask = (
            valid_mask & (float(spine_start_s) <= T) & (float(spine_end_s) >= T)
        )
        n_qualifying = int(in_spine_mask.sum())
        cum_valid = np.cumsum(in_spine_mask.astype(np.int32))

        # --- Per-impression event_ts (just AFTER the selected session's burst). ---
        ets_arr = np.empty(n_imp)
        for i in range(n_imp):
            if n_qualifying > 0:
                sel_u = md5_uniform_np(
                    np.array([f"{player_id}:{i}:impsel:{cfg.seed}"])
                )[0]
                sel_col = int(np.clip(sel_u * n_qualifying, 0, n_qualifying - 1))
                pos = int(np.clip((cum_valid < sel_col + 1).sum(), 0, max_k - 1))
                ets_arr[i] = float(
                    np.ceil(session_start_ts[pos] + _sessions.BURST_SPAN_S + 1.0)
                )
            else:
                fb_u = md5_uniform_np(np.array([f"{player_id}:{i}:impfb:{cfg.seed}"]))[
                    0
                ]
                ets_arr[i] = float(spine_start_s) + fb_u * float(
                    spine_end_s - spine_start_s
                )

        # --- ONE window_counts_batch call for ALL of this player's impressions. ---
        pid_col = np.array([player_id] * n_imp)
        seg_col = np.array([segment] * n_imp)
        wc = _sessions.window_counts_batch(
            pid_col,
            seg_col,
            ets_arr,
            window_start=float(ext_window_start_s),
            window_end=float(ext_window_end_s),
            seed=cfg.seed,
        )

        off = offers.iloc[offer_idxs].reset_index(drop=True)
        block = pd.DataFrame(
            {
                "player_id": player_id,
                "country": player_data["country"],
                "segment": segment,
                "tenure_days": int(player_data["tenure_days"]),
                "lifetime_spend": player_data["lifetime_spend"],
                "historical_ltv": player_data["historical_ltv"],
                "avg_session_len_min": player_data["avg_session_len_min"],
                "purchases_30d": int(player_data["purchases_30d"]),
                "_propensity": propensity,
                "offer_id": off["offer_id"].to_numpy(),
                "offer_type": off["offer_type"].to_numpy(),
                "price_tier": off["price_tier"].to_numpy(),
                "offer_price_usd": off["offer_price_usd"].to_numpy(),
                "offer_gem_amount": off["offer_gem_amount"].to_numpy(),
                "offer_discount_pct": off["offer_discount_pct"].to_numpy(),
                "gem_balance": gem_balances,
                "minutes_since_last_purchase": minutes_arr,
                "match_count": match_counts,
                "event_ts": pd.to_datetime(ets_arr * 1e9, unit="ns"),
            }
        )
        for feat in _schema.WINDOWS:
            block[feat] = wc[feat]

        # Label: propensity noise + purchase + revenue, drawn in the SAME rng order
        # (normal per row, then uniform per row, then lognormal per row) as the ref.
        prob = _purchase_propensity(block, rng_p)
        purchased = (rng_p.uniform(0, 1, n_imp) < prob).astype(int)
        downstream_mean = (
            propensity * 6.0
            + np.log1p(block["offer_gem_amount"].to_numpy()) * propensity * 1.5
            + np.clip(match_counts / 10.0, 0, 2) * 3.0
        )
        downstream = purchased * downstream_mean * rng_p.lognormal(0.0, 0.35, n_imp)
        revenue = (purchased * block["offer_price_usd"].to_numpy() + downstream).round(
            2
        )
        block[TARGET_PURCHASED] = purchased
        block[TARGET_REVENUE_D7] = revenue

        block = block.drop(columns=[*_schema.WINDOWS, "_propensity"])
        per_player_frames.append(block)

    impressions_pdf = (
        pd.concat(per_player_frames, ignore_index=True)
        if per_player_frames
        else pd.DataFrame()
    )
    players_pdf = players_pdf.drop(columns=["_propensity"])

    return players_pdf, impressions_pdf


def _generate_player_block_chunk(
    args: tuple[list[str], pd.DataFrame, tuple[int, int, int, int], GenConfig],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top-level ProcessPoolExecutor worker: unpack args and run one player chunk.

    Must be module-level (not a closure) so it is picklable for the process pool.
    ``bounds`` is ``(ext_window_start_s, ext_window_end_s, spine_start_s, spine_end_s)``.
    """
    player_ids, offers, bounds, cfg = args
    return _generate_player_block(player_ids, offers, *bounds, cfg)


def _generate_parallel(
    player_ids: list[str],
    offers: pd.DataFrame,
    bounds: tuple[int, int, int, int],
    cfg: GenConfig,
    n_workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run per-player block generation across a process pool and concat in order.

    Order-preserving concat makes the result bit-identical to the single-process
    path (per-player md5-seeded RNG => shard-invariant, see the distributed-parity
    test). Falls back to single-process on ``BrokenProcessPool`` -- the spawn start
    method (macOS default) re-imports the entry module, which fails when the caller's
    ``__main__`` is not importable (``python -c``, a REPL, some notebook setups). The
    fallback keeps generation correct there, just single-threaded.
    """
    chunk_size = _gen_chunk_size(len(player_ids), n_workers)
    chunk_args = [
        (player_ids[i : i + chunk_size], offers, bounds, cfg)
        for i in range(0, len(player_ids), chunk_size)
    ]
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_generate_player_block_chunk, chunk_args))
    except BrokenProcessPool:
        logger.warning(
            "ProcessPool unavailable (non-importable __main__?); "
            "generating single-process."
        )
        results = [_generate_player_block_chunk(a) for a in chunk_args]
    players = pd.concat([p for p, _ in results], ignore_index=True)
    impressions = pd.concat([imp for _, imp in results], ignore_index=True)
    return players, impressions


def generate_frames(
    config: GenConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate (players, offers, impressions) as pandas DataFrames.

    Deterministic for a given `config.seed`. Player generation fans out across
    processes for large runs (see `_gen_workers`); small runs stay single-process.
    """
    from offer_recommender.data.hashing import md5_seed

    cfg = config or GenConfig()

    rng_offers = np.random.default_rng(md5_seed(f"offers:{cfg.seed}"))
    offers = _build_offers(rng_offers, cfg.n_offers)

    end = _resolve_end(cfg)
    start = end - pd.Timedelta(days=cfg.span_days)
    spine_start_s: int = int(start.timestamp())
    spine_end_s: int = int(end.timestamp())
    ext_window_start_s: int = spine_start_s - 15 * 24 * 3600
    ext_window_end_s: int = spine_end_s

    player_ids = [f"player_{i:06d}" for i in range(cfg.n_players)]
    bounds = (ext_window_start_s, ext_window_end_s, spine_start_s, spine_end_s)
    n_workers = _gen_workers(cfg.n_players)

    if n_workers <= 1:
        players, impressions = _generate_player_block(player_ids, offers, *bounds, cfg)
    else:
        players, impressions = _generate_parallel(
            player_ids, offers, bounds, cfg, n_workers
        )

    impressions.attrs["_session_window"] = {
        "window_start_s": float(ext_window_start_s),
        "window_end_s": float(ext_window_end_s),
        "seed": cfg.seed,
        "span_days": cfg.span_days,
    }

    return players, offers, impressions


def generate_frames_spark(spark, config: GenConfig | None = None):
    """Distributed twin of generate_frames: return (players, offers, impressions)
    as SPARK DataFrames + the resolved window bounds, generated across executors.

    Shards the player ids and runs `_generate_player_block` per shard via
    mapInPandas, so the (up to 400k-row) impression log is NEVER materialised on
    the driver (the old generate_frames-on-driver path OOM'd -> DRIVER_UNREACHABLE).
    `_generate_player_block` is per-player independent (per-player md5-seeded RNG),
    so sharding does not change any player's output -- guarded by
    tests/test_generate_distributed_parity.py against the numpy `generate_frames`.

    Returns (players_sdf, offers_sdf, impressions_sdf, ext_window_start_s,
    ext_window_end_s). The offers catalogue + window bounds are resolved ONCE on the
    driver and captured into the workers so every shard uses IDENTICAL values
    (resolving _resolve_end/now() inside a worker would drift per-task and desync
    the backfill from the label).
    """
    from offer_recommender.data.hashing import md5_seed

    cfg = config or GenConfig()

    rng_offers = np.random.default_rng(md5_seed(f"offers:{cfg.seed}"))
    offers = _build_offers(rng_offers, cfg.n_offers)

    end = _resolve_end(cfg)
    start = end - pd.Timedelta(days=cfg.span_days)
    spine_start_s = int(start.timestamp())
    spine_end_s = int(end.timestamp())
    ext_window_start_s = spine_start_s - 15 * 24 * 3600
    ext_window_end_s = spine_end_s

    _offers, _cfg = offers, cfg
    _ws, _we, _ss, _se = (
        ext_window_start_s,
        ext_window_end_s,
        spine_start_s,
        spine_end_s,
    )

    def _gen_impressions(iterator):
        for pdf_ids in iterator:
            ids = pdf_ids["player_id"].tolist()
            _, imp = _generate_player_block(ids, _offers, _ws, _we, _ss, _se, _cfg)
            if len(imp):
                yield imp

    # Schema from a 1-player NON-EMPTY sample (an empty pandas frame has no
    # inferable types -> CANNOT_INFER_EMPTY_SCHEMA); reuse for mapInPandas.
    _sample_pl, _sample_imp = _generate_player_block(
        ["player_000000"],
        offers,
        ext_window_start_s,
        ext_window_end_s,
        spine_start_s,
        spine_end_s,
        cfg,
    )
    impressions_schema = spark.createDataFrame(_sample_imp).schema
    _player_cols = list(_sample_pl.columns)

    # mapInPandas runs ONE task per input partition (it preserves input
    # partitioning), so parallelism == the repartition count. On serverless we
    # cannot set executor count / spark.sql.shuffle.partitions (auto-tuned), so the
    # ONLY lever is making MANY MORE, SMALLER partitions: the autoscaler then has
    # lots of short tasks to spread as it scales up (a coarse ~40-partition split
    # left most tasks serialized -> 6+ min wall-clock while the cluster sat near
    # idle). ~50 players/partition keeps each task's window matrices small
    # (50 x max_k, chunked by _BATCH_CHUNK) and short. Databricks-recommended
    # pattern for a CPU-bound per-key pandas UDF on serverless.
    n_parts = max(1, (cfg.n_players + 49) // 50)
    pid_sdf = spark.createDataFrame(
        [(f"player_{i:06d}",) for i in range(cfg.n_players)], ["player_id"]
    ).repartition(n_parts)

    # ONE distributed generation pass. Impressions carry every player column, so
    # `players` is derived by dedup -- but to avoid re-running the mapInPandas a
    # second time (lazy DataFrames re-execute per action; .cache() is rejected on
    # serverless: NOT_SUPPORTED_WITH_SERVERLESS PERSIST TABLE), write_delta writes
    # impressions to Delta FIRST, then reads that persisted table back to derive
    # players. Here we just return the lazy impressions DF + the player-column list.
    impressions_sdf = pid_sdf.mapInPandas(_gen_impressions, schema=impressions_schema)
    offers_sdf = spark.createDataFrame(offers)
    return (
        _player_cols,
        offers_sdf,
        impressions_sdf,
        float(ext_window_start_s),
        float(ext_window_end_s),
    )


def write_delta(
    spark, catalog: str, schema: str, config: GenConfig | None = None
) -> None:
    """Generate the datasets (distributed) and write them as Delta tables.

    Also writes a one-row ``session_window_bounds`` table with the exact
    ``ext_window_start_s``, ``ext_window_end_s``, ``seed`` and ``span_days`` so
    ``setup_stream.py`` anchors the backfill window bounds exactly, closing the
    label-feature alignment gap.
    """
    from pyspark.sql import functions as sf

    from offer_recommender.config import TABLE_SESSION_WINDOW_BOUNDS

    cfg = config or GenConfig()
    logger.info(
        "Generating synthetic data (distributed): n_players=%d, n_offers=%d, "
        "impressions_per_player=%d, seed=%d -> %s.%s",
        cfg.n_players,
        cfg.n_offers,
        cfg.impressions_per_player,
        cfg.seed,
        catalog,
        schema,
    )

    (
        player_cols,
        offers_sdf,
        impressions_sdf,
        _ext_window_start_s,
        _ext_window_end_s,
    ) = generate_frames_spark(spark, cfg)

    imp_fqn = f"{catalog}.{schema}.impressions"

    total_start = time.perf_counter()
    # Write impressions FIRST (the ONE distributed generation), then derive players
    # by DEDUP of the PERSISTED impressions table -- reading it back is cheap and
    # avoids re-running the mapInPandas (.cache() is unsupported on serverless).
    t0 = time.perf_counter()
    (
        impressions_sdf.withColumn("event_date", sf.to_date("event_ts"))
        .write.mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("event_date")
        .format("delta")
        .saveAsTable(imp_fqn)
    )
    logger.info(
        "Wrote %s in %.1fs (distributed, partitioned by event_date).",
        imp_fqn,
        time.perf_counter() - t0,
    )

    players_sdf = (
        spark.table(imp_fqn).select(*player_cols).dropDuplicates(["player_id"])
    )
    for name, sdf, partition in (
        ("players", players_sdf, None),
        ("offers", offers_sdf, None),
    ):
        fqn = f"{catalog}.{schema}.{name}"
        t0 = time.perf_counter()
        writer = (
            sdf.write.mode("overwrite")
            .option("overwriteSchema", "true")
            .format("delta")
        )
        writer.saveAsTable(fqn)
        logger.info(
            "Wrote %s%s in %.1fs (distributed).",
            fqn,
            f" (partitioned by {partition})" if partition else "",
            time.perf_counter() - t0,
        )

    # Persist window bounds metadata so setup_stream reads the exact values.
    bounds_fqn = f"{catalog}.{schema}.{TABLE_SESSION_WINDOW_BOUNDS}"
    bounds_pdf = pd.DataFrame(
        [
            {
                "window_start_s": float(_ext_window_start_s),
                "window_end_s": float(_ext_window_end_s),
                "seed": cfg.seed,
                "span_days": cfg.span_days,
            }
        ]
    )
    spark.createDataFrame(bounds_pdf).write.mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(bounds_fqn)
    logger.info(
        "Wrote %s: window_start_s=%.0f, window_end_s=%.0f, seed=%d.",
        bounds_fqn,
        _ext_window_start_s,
        _ext_window_end_s,
        cfg.seed,
    )

    logger.info(
        "generate_data complete: 4 tables written in %.1fs total.",
        time.perf_counter() - total_start,
    )
