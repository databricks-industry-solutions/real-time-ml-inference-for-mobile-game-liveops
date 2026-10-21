"""Pure-python unit tests for correctness-bearing logic (no Spark, no GPU).

Covers branches the Spark-dependent end-to-end tests only exercise transitively:
config env-override + caching discipline, UC FQN formatting, data-generator
invariants and determinism, the log1p target contract, the ranking-lift math
and its divide-by-zero guard, embedding sizing/wiring, and the zero-variance
scaler guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from offer_recommender.config import UCConfig, configure_from_argv, get_config
from offer_recommender.data.generate import GenConfig, generate_frames
from offer_recommender.model.net import OfferRankerNet, multitask_loss
from offer_recommender.model.transform import transform_numeric
from offer_recommender.preprocessing.pipeline import TransformParams
from offer_recommender.schema import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    STREAMING_NUMERICAL_FEATURES,
    TARGET_PURCHASED,
    TARGET_REVENUE_D7,
)
from offer_recommender.training.core import TrainConfig, make_dataset, rank_offers
from offer_recommender.training.evaluate import RankingEval


# --- config ------------------------------------------------------------------
def test_ucconfig_fqn_formatting():
    """table/model_name build catalog.schema.name; volume_path is a path."""
    uc = UCConfig(catalog="c", schema="s", volume="v")
    assert uc.table("players") == "c.s.players"
    assert uc.model_name("m") == "c.s.m"
    assert uc.volume_path == "/Volumes/c/s/v"


def test_get_config_env_override_and_cache(monkeypatch):
    """Env vars override defaults; cache_clear discipline makes it deterministic."""
    get_config.cache_clear()
    try:
        default = get_config()
        assert default.catalog == "main"
        assert default.schema == "offer_recommender"
        assert default.volume == "artifacts"

        monkeypatch.setenv("RECO_CATALOG", "cat2")
        monkeypatch.setenv("RECO_SCHEMA", "sch2")
        monkeypatch.setenv("RECO_VOLUME", "vol2")
        get_config.cache_clear()  # required: the value is cached from import time
        overridden = get_config()
        assert (overridden.catalog, overridden.schema, overridden.volume) == (
            "cat2",
            "sch2",
            "vol2",
        )
    finally:
        get_config.cache_clear()


def test_configure_from_argv_sets_schema(monkeypatch):
    """DAB [catalog, schema] argv is applied; explicit env still wins."""
    monkeypatch.delenv("RECO_CATALOG", raising=False)
    monkeypatch.delenv("RECO_SCHEMA", raising=False)
    get_config.cache_clear()
    try:
        configure_from_argv(["dab_cat", "dev_user_offer_recommender"])
        cfg = get_config()
        assert cfg.catalog == "dab_cat"
        assert cfg.schema == "dev_user_offer_recommender"
    finally:
        monkeypatch.delenv("RECO_CATALOG", raising=False)
        monkeypatch.delenv("RECO_SCHEMA", raising=False)
        get_config.cache_clear()


# --- data generator ----------------------------------------------------------
@pytest.mark.slow
def test_generate_frames_deterministic_and_schema():
    """Same seed -> identical frames; expected shapes/columns; no latent leak."""
    # Fix end_ts so utcnow() inside generate_frames produces the same window
    # across both calls (without it the two calls land in different sub-second
    # buckets and event_ts diverges).
    fixed_end = pd.Timestamp("2026-07-15")
    cfg = GenConfig(
        n_players=300, n_offers=20, impressions_per_player=5, seed=7, end_ts=fixed_end
    )
    players_a, offers_a, imp_a = generate_frames(cfg)
    players_b, _, imp_b = generate_frames(cfg)
    assert imp_a.equals(imp_b)
    assert players_a.equals(players_b)

    _, _, imp_diff = generate_frames(GenConfig(n_players=300, seed=8))
    assert not imp_a.equals(imp_diff)

    assert len(players_a) == 300
    assert len(offers_a) == 20
    assert len(imp_a) == 300 * 5
    # Latent propensity must not leak into emitted frames.
    assert "_propensity" not in players_a.columns
    assert "_propensity" not in imp_a.columns
    # Streaming columns (STREAMING_NUMERICAL_FEATURES) are now owned by the
    # Streaming Feature View at train/serve time and are NOT emitted by
    # generate_frames. Assert only the features generate IS responsible for.
    generated_numericals = set(NUMERICAL_FEATURES) - set(STREAMING_NUMERICAL_FEATURES)
    expected = (
        set(CATEGORICAL_FEATURES)
        | generated_numericals
        | {
            TARGET_REVENUE_D7,
            TARGET_PURCHASED,
            "event_ts",
        }
    )
    assert expected <= set(imp_a.columns)
    # Streaming columns must be absent (they come from the SFV, not generation).
    for col in STREAMING_NUMERICAL_FEATURES:
        assert col not in imp_a.columns
    assert "match_events_15m" not in imp_a.columns
    assert "match_count" in imp_a.columns
    assert "session_depth" not in imp_a.columns


@pytest.mark.slow
def test_revenue_invariants():
    """d7 revenue is non-negative and exactly zero when not purchased."""
    _, _, imp = generate_frames(
        GenConfig(n_players=500, n_offers=25, impressions_per_player=6, seed=3)
    )
    assert (imp[TARGET_REVENUE_D7] >= 0).all()
    not_bought = imp.loc[imp[TARGET_PURCHASED] == 0, TARGET_REVENUE_D7]
    assert (not_bought == 0).all()
    assert imp[TARGET_PURCHASED].isin({0, 1}).all()


@pytest.mark.slow
def test_segment_purchase_rate_monotonicity():
    """Whales convert more than minnows; dolphins more than lapsed."""
    _, _, imp = generate_frames(
        GenConfig(n_players=150, n_offers=30, impressions_per_player=8, seed=5)
    )
    rate = imp.groupby("segment")[TARGET_PURCHASED].mean()
    assert rate["whale"] > rate["minnow"]
    assert rate["dolphin"] > rate["lapsed"]


@pytest.mark.slow
def test_offer_exposure_is_player_correlated():
    """Whale segment sees high-tier offers materially more than minnows (>1.5x).

    Uses tier_4/tier_5 as the canonical "high tier". Falls back to the highest
    tier actually present if this seed's small catalogue has none; skips if
    there is only one unique tier (no spread to measure).
    """
    _, offers, imp = generate_frames(GenConfig(n_players=150, n_offers=5, seed=17))

    _TIER_RANK = {
        "tier_1": 1,
        "tier_2": 2,
        "tier_3": 3,
        "tier_4": 4,
        "tier_5": 5,
    }
    hi = set(offers[offers["price_tier"].isin(["tier_4", "tier_5"])]["offer_id"])
    if not hi:
        ranked = offers["price_tier"].map(_TIER_RANK)
        if ranked.nunique() == 1:
            pytest.skip("All offers share one price tier — no tier spread to measure")
        top = ranked.max()
        hi = set(offers[offers["price_tier"].map(_TIER_RANK) == top]["offer_id"])

    whale_hi = imp[imp["segment"] == "whale"]["offer_id"].isin(hi).mean()
    minnow_hi = imp[imp["segment"] == "minnow"]["offer_id"].isin(hi).mean()
    assert whale_hi > minnow_hi * 1.5


# --- training core -----------------------------------------------------------
def test_make_dataset_shapes_and_log1p_target():
    """make_dataset yields the right dtypes/shapes and log1p-transforms revenue."""
    n, n_num, n_cat = 12, len(NUMERICAL_FEATURES), len(CATEGORICAL_FEATURES)
    numeric = np.random.default_rng(0).standard_normal((n, n_num)).astype(np.float32)
    categorical = np.random.default_rng(1).integers(0, 3, (n, n_cat))
    revenue = np.random.default_rng(2).uniform(0, 100, n).astype(np.float32)
    purchased = np.random.default_rng(3).integers(0, 2, n).astype(np.float32)

    ds = make_dataset(numeric, categorical, revenue, purchased)
    num_t, cat_t, rev_t, buy_t = ds.tensors
    assert num_t.shape == (n, n_num) and num_t.dtype == torch.float32
    assert cat_t.shape == (n, n_cat) and cat_t.dtype == torch.int64
    assert torch.allclose(rev_t, torch.log1p(torch.tensor(revenue)))
    assert buy_t.dtype == torch.float32


def _toy_params() -> TransformParams:
    """Toy fitted-params covering every categorical, unit scaler."""
    return TransformParams(
        vocabs={f: [f"{f}_a", f"{f}_b", f"{f}_c"] for f in CATEGORICAL_FEATURES},
        mean=[0.0] * len(NUMERICAL_FEATURES),
        std=[1.0] * len(NUMERICAL_FEATURES),
        numerical_features=list(NUMERICAL_FEATURES),
        categorical_features=list(CATEGORICAL_FEATURES),
        log1p_features=[],
    )


def test_train_step_reduces_loss_on_single_batch():
    """Overfitting one batch must drive the multitask loss down."""
    params = _toy_params()
    torch.manual_seed(0)
    net = OfferRankerNet(params)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    rng = np.random.default_rng(0)
    numeric = torch.tensor(
        rng.standard_normal((64, len(NUMERICAL_FEATURES))), dtype=torch.float32
    )
    categorical = torch.tensor(
        rng.integers(0, 3, (64, len(CATEGORICAL_FEATURES))), dtype=torch.long
    )
    revenue = torch.tensor(rng.uniform(0, 3, 64), dtype=torch.float32)
    purchased = torch.tensor(rng.integers(0, 2, 64), dtype=torch.float32)

    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = multitask_loss(net(numeric, categorical), revenue, purchased)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.7


def test_rank_offers_is_expm1_of_head():
    """rank_offers returns expm1 of the log-space revenue head."""
    params = _toy_params()
    net = OfferRankerNet(params).eval()
    rng = np.random.default_rng(0)
    numeric = rng.standard_normal((10, len(NUMERICAL_FEATURES))).astype(np.float32)
    categorical = rng.integers(0, 3, (10, len(CATEGORICAL_FEATURES)))
    with torch.no_grad():
        head = net(torch.tensor(numeric), torch.tensor(categorical, dtype=torch.long))[
            "revenue_d7"
        ].numpy()
    np.testing.assert_allclose(
        rank_offers(net, numeric, categorical), np.expm1(head), rtol=1e-5
    )


def test_train_config_defaults():
    """TrainConfig carries the documented defaults."""
    cfg = TrainConfig()
    assert cfg.epochs == 40
    assert cfg.batch_size == 256
    assert cfg.embedding_dim == 16


# --- model wiring ------------------------------------------------------------
def test_embedding_sizes_and_forward_contract():
    """One embedding per categorical sized len(vocab)+1; forward output shapes."""
    params = _toy_params()
    net = OfferRankerNet(params).eval()
    sizes = params.embedding_sizes()
    assert len(net.embeddings) == len(CATEGORICAL_FEATURES)
    for i, feature in enumerate(CATEGORICAL_FEATURES):
        assert net.embeddings[i].num_embeddings == sizes[feature] == 4  # 3 + OOV

    numeric = torch.zeros((5, len(NUMERICAL_FEATURES)))
    categorical = torch.zeros((5, len(CATEGORICAL_FEATURES)), dtype=torch.long)
    with torch.no_grad():
        out = net(numeric, categorical)
    assert set(out) == {"revenue_d7", "purchase_logit"}
    assert out["revenue_d7"].shape == (5,)
    assert (out["revenue_d7"] >= 0).all()  # softplus non-negativity


# --- transform zero-variance guard -------------------------------------------
def test_transform_numeric_zero_variance_guard():
    """A std==0 feature is centered but not divided by zero (no NaN/inf)."""
    params = TransformParams(
        vocabs={},
        mean=[5.0, 2.0],
        std=[0.0, 2.0],  # first feature constant
        numerical_features=["a", "b"],
        categorical_features=[],
        log1p_features=[],
    )
    df = pd.DataFrame({"a": [5.0, 5.0, 5.0], "b": [2.0, 4.0, 6.0]})
    out = transform_numeric(df, params)
    assert np.isfinite(out).all()
    # constant feature - mean == 0, scale left at 1
    np.testing.assert_allclose(out[:, 0], [0.0, 0.0, 0.0])


def test_transform_numeric_fills_streaming_nulls_with_zero():
    """A NaN streaming feature (no online row) is filled to 0.0, matching the
    training-side fillna, so serving does not emit NaN predictions."""
    stream_col = STREAMING_NUMERICAL_FEATURES[0]
    params = TransformParams(
        vocabs={},
        mean=[0.0, 3.0],
        std=[1.0, 2.0],
        numerical_features=[stream_col, "other"],
        categorical_features=[],
        log1p_features=[],
    )
    df = pd.DataFrame({stream_col: [None, 4.0], "other": [3.0, 5.0]})
    out = transform_numeric(df, params)
    assert np.isfinite(out).all()
    # NaN streaming value -> 0.0 -> (0 - mean)/std == 0.0
    np.testing.assert_allclose(out[0, 0], 0.0)


def test_transform_numeric_nonstreaming_null_surfaces():
    """A NaN in a NON-streaming feature is NOT silently filled (real-bug guard)."""
    params = TransformParams(
        vocabs={},
        mean=[0.0],
        std=[1.0],
        numerical_features=["other"],
        categorical_features=[],
        log1p_features=[],
    )
    df = pd.DataFrame({"other": [None, 4.0]})
    out = transform_numeric(df, params)
    assert np.isnan(out[0, 0])


# --- ranking-lift math -------------------------------------------------------
@pytest.mark.parametrize(
    ("model", "oracle", "random", "expected"),
    [
        (7.0, 10.0, 5.0, 0.4),  # normal case
        (10.0, 10.0, 5.0, 1.0),  # model == oracle
        (5.0, 10.0, 5.0, 0.0),  # model == random
        (7.0, 5.0, 5.0, 0.0),  # denom == 0 guard (no ZeroDivisionError)
        (7.0, 4.0, 5.0, 0.0),  # denom < 0 degenerate -> guard
    ],
)
def test_oracle_lift_captured(model, oracle, random, expected):
    """oracle_lift_captured math + the divide-by-zero/degenerate guard."""
    result = RankingEval(
        model_revenue=model, oracle_revenue=oracle, random_revenue=random
    )
    assert result.oracle_lift_captured == pytest.approx(expected)


# --- Task-5: session-driven impressions + causal label -----------------------


def _base_frame(n: int = 200) -> pd.DataFrame:
    """Minimal frame with every column _purchase_propensity reads after Task 5.

    Provides segment propensity, offer features, real-time features and all 6
    WINDOWS rolling-window counts (all zeroed so tests can vary them independently).
    """
    df = pd.DataFrame(
        {
            "_propensity": np.full(n, 0.5),
            "offer_price_usd": np.full(n, 4.99),
            "offer_gem_amount": np.full(n, 500.0),
            "gem_balance": np.full(n, 500.0),
            "match_count": np.full(n, 5),
            "offer_discount_pct": np.zeros(n),
            "minutes_since_last_purchase": np.full(n, 1000.0),
        }
    )
    from offer_recommender import schema

    for c in schema.WINDOWS:
        df[c] = 0.0
    return df


def test_offer_fatigue_lowers_purchase_prob() -> None:
    """High offers_seen_15m (fatigue) must lower mean purchase probability."""
    from offer_recommender.data.generate import _purchase_propensity

    rng = np.random.default_rng(0)
    lo = _base_frame()
    lo["offers_seen_15m"] = 0.0
    hi = _base_frame()
    hi["offers_seen_15m"] = 8.0
    assert _purchase_propensity(hi, rng).mean() < _purchase_propensity(lo, rng).mean()


def test_losing_streak_raises_purchase_prob() -> None:
    """High match_losses_15m (frustration) must lift mean purchase probability."""
    from offer_recommender.data.generate import _purchase_propensity

    rng = np.random.default_rng(0)
    tilted = _base_frame()
    tilted["match_losses_15m"] = 40.0
    calm = _base_frame()
    calm["match_losses_15m"] = 0.0
    assert (
        _purchase_propensity(tilted, rng).mean()
        > _purchase_propensity(calm, rng).mean()
    )


def test_low_avg_gem_balance_raises_purchase_prob() -> None:
    """Low avg_gem_balance_15m (gem-block proximity) lifts purchase probability.

    A player near zero gems is more likely to buy more — the gem-block buy trigger.
    """
    from offer_recommender.data.generate import _purchase_propensity

    rng = np.random.default_rng(0)
    depleted = _base_frame()
    depleted["avg_gem_balance_15m"] = 0.0  # near zero gems -> buy trigger
    flush = _base_frame()
    flush["avg_gem_balance_15m"] = 2800.0  # gem-rich -> less urgent
    assert (
        _purchase_propensity(depleted, rng).mean()
        > _purchase_propensity(flush, rng).mean()
    )


def test_recency_raises_purchase_prob() -> None:
    """Small minutes_since_last_purchase (recent) must lift mean purchase probability."""
    from offer_recommender.data.generate import _purchase_propensity

    rng = np.random.default_rng(0)
    recent = _base_frame()
    recent["minutes_since_last_purchase"] = 10.0
    stale = _base_frame()
    stale["minutes_since_last_purchase"] = 8000.0
    assert (
        _purchase_propensity(recent, rng).mean()
        > _purchase_propensity(stale, rng).mean()
    )


@pytest.mark.slow
def test_streaming_counts_nonzero_fraction_is_high() -> None:
    """Impression event_ts fires just AFTER its session burst -> the 15-min window
    captures the burst, so effectively every impression has avg_gem_balance_15m > 0.

    Recomputes the streaming windows at each impression's event_ts via
    window_counts_batch (the same function the label uses), rather than reading an
    internal attr -- this asserts the observable causal-alignment property directly
    and survives the distributed refactor (attrs don't cross Spark executors).
    """
    import numpy as np
    import pandas as pd

    from offer_recommender.data import sessions as s

    cfg = GenConfig(n_players=150, seed=17)
    _, _, imp = generate_frames(cfg)
    sw = imp.attrs["_session_window"]
    at = (
        (pd.to_datetime(imp["event_ts"]).astype("int64") // 10**9)
        .to_numpy()
        .astype(float)
    )
    wc = s.window_counts_batch(
        imp["player_id"].to_numpy(),
        imp["segment"].to_numpy(),
        at,
        window_start=float(sw["window_start_s"]),
        window_end=float(sw["window_end_s"]),
        seed=cfg.seed,
    )
    assert float((wc["avg_gem_balance_15m"] > 0).mean()) > 0.5
    # Windows should hold a full burst (~EVENTS_PER_SESSION events), not ~0.
    assert float(np.mean(wc["match_events_15m"])) > 5.0


def test_is_bursty_fraction_matches_target() -> None:
    """is_bursty flags ~BURSTY_FRACTION of players (md5-uniform, orthogonal to segment)."""
    from offer_recommender.data import sessions as s

    ids = [f"player_{i:06d}" for i in range(20000)]
    frac = sum(s.is_bursty(pid, 17) for pid in ids) / len(ids)
    assert abs(frac - s.BURSTY_FRACTION) < 0.005


def test_bursty_players_get_more_events() -> None:
    """A bursty player has BURSTY_FACTOR x its segment's base events; non-bursty 0.5x."""
    import numpy as np

    from offer_recommender.data import sessions as s

    # find one bursty + one non-bursty whale-id deterministically
    bursty_id = next(
        f"player_{i:06d}" for i in range(100000) if s.is_bursty(f"player_{i:06d}", 17)
    )
    plain_id = next(
        f"player_{i:06d}"
        for i in range(100000)
        if not s.is_bursty(f"player_{i:06d}", 17)
    )
    base = s.SESSIONS_PER_SEGMENT["whale"] * s.EVENTS_PER_SESSION
    counts = s.event_count(
        np.array([bursty_id, plain_id]), np.array(["whale", "whale"]), 17
    )
    assert counts[0] == round(base * s.BURSTY_FACTOR)
    assert counts[1] == round(base * s.NONBURSTY_FACTOR)


def test_window_counts_returns_all_windows() -> None:
    """window_counts_batch returns every WINDOWS key incl. match_events_15m."""
    import numpy as np

    from offer_recommender import schema
    from offer_recommender.data import sessions as s

    pid = np.array(["player_000001"])
    seg = np.array(["whale"])
    at = np.array([10 * 24 * 3600.0])
    wc = s.window_counts_batch(pid, seg, at, 0.0, 15 * 24 * 3600.0, 17)
    assert set(wc) == set(schema.WINDOWS)
    # wins + losses == events (every match is won XOR lost).
    assert (
        abs(
            (wc["match_wins_15m"][0] + wc["match_losses_15m"][0])
            - wc["match_events_15m"][0]
        )
        < 1e-9
    )


# --- session_end_ts pin (Task R1) --------------------------------------------


@pytest.mark.slow
def test_two_generate_runs_without_explicit_end_ts_produce_identical_window_bounds(
    monkeypatch,
) -> None:
    """Two independent config builds must yield identical _session_window bounds.

    ASSERTION 1 — DETERMINISM / no-drift: even if pd.Timestamp.now() returns
    different values on each call (simulating hours between setup_stream and a
    later generate_data run), get_session_end_ts() is never consulted for now()
    so bounds are identical.  The monkeypatch returning DIFFERENT timestamps is
    intentional: if now() WERE consulted the two bounds would differ, so the
    assertion of equality is the real invariant.  If get_session_end_ts is ever
    reverted to call pd.Timestamp.now(), now() will be called with different
    return values and the bounds will diverge — this assertion will FAIL.

    ASSERTION 2 — ENV PLUMBING: setting RECO_SESSION_END_TS to two different
    epochs produces two different bounds, each matching its respective pin.
    Proves the env override flows all the way through to the label window.
    """
    from offer_recommender.config import get_config, get_session_end_ts

    # --- Assertion 1: now() is never consulted ---

    def _make_cfg() -> GenConfig:
        """Mirror what generate_data.py does: read get_session_end_ts(), build GenConfig."""
        return GenConfig(
            n_players=200,
            n_offers=5,
            impressions_per_player=4,
            seed=17,
            end_ts=get_session_end_ts(),
        )

    # Return DIFFERENT timestamps on successive now() calls to simulate drift.
    t0 = pd.Timestamp("2026-07-30 10:00:00")
    t1 = pd.Timestamp("2026-07-30 12:30:00")  # 2.5 hours later
    call_count = {"n": 0}

    def _fake_now(tz=None):
        call_count["n"] += 1
        ts = t0 if call_count["n"] <= 1 else t1
        return ts.tz_localize(tz) if tz is not None else ts

    monkeypatch.setattr(pd.Timestamp, "now", staticmethod(_fake_now))

    get_config.cache_clear()
    _, _, imp_first = generate_frames(_make_cfg())
    bounds_first = imp_first.attrs["_session_window"]

    get_config.cache_clear()
    _, _, imp_second = generate_frames(_make_cfg())
    bounds_second = imp_second.attrs["_session_window"]

    assert bounds_first["window_start_s"] == bounds_second["window_start_s"], (
        f"window_start_s drifted: {bounds_first['window_start_s']} != "
        f"{bounds_second['window_start_s']}. get_session_end_ts is calling now()."
    )
    assert bounds_first["window_end_s"] == bounds_second["window_end_s"], (
        f"window_end_s drifted: {bounds_first['window_end_s']} != "
        f"{bounds_second['window_end_s']}. get_session_end_ts is calling now()."
    )

    # --- Assertion 2: env plumbing flows through to label bounds ---

    pin_a = pd.Timestamp("2026-06-01 00:00:00")
    pin_b = pd.Timestamp("2026-08-01 00:00:00")

    monkeypatch.setenv("RECO_SESSION_END_TS", str(pin_a.timestamp()))
    get_config.cache_clear()
    _, _, imp_a = generate_frames(_make_cfg())
    bounds_a = imp_a.attrs["_session_window"]

    monkeypatch.setenv("RECO_SESSION_END_TS", str(pin_b.timestamp()))
    get_config.cache_clear()
    _, _, imp_b = generate_frames(_make_cfg())
    bounds_b = imp_b.attrs["_session_window"]

    assert bounds_a["window_end_s"] != bounds_b["window_end_s"], (
        "window_end_s is the same for pin_a and pin_b — env override is not flowing "
        "through to the label window."
    )
    assert bounds_a["window_end_s"] == pytest.approx(pin_a.timestamp(), abs=1), (
        f"window_end_s {bounds_a['window_end_s']} != pin_a {pin_a.timestamp()}"
    )
    assert bounds_b["window_end_s"] == pytest.approx(pin_b.timestamp(), abs=1), (
        f"window_end_s {bounds_b['window_end_s']} != pin_b {pin_b.timestamp()}"
    )
