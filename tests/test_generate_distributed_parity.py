"""Parity test for distributed player/impression generation.

Verifies that _generate_player_block produces the same output when called
once for all players vs when called per-shard (distributed pattern).
"""

from __future__ import annotations

import numpy as np

from offer_recommender.data.generate import (
    GenConfig,
    _build_offers,
    _generate_player_block,
)
from offer_recommender.data.hashing import md5_seed


def test_generate_player_block_single_vs_sharded():
    """Single-shard generation must match multi-shard concatenation."""
    cfg = GenConfig(n_players=40, impressions_per_player=4, seed=17)

    # Generate offers (shared).
    rng_offers = np.random.default_rng(md5_seed(f"offers:{cfg.seed}"))
    offers = _build_offers(rng_offers, cfg.n_offers)

    # Window bounds.
    import pandas as pd

    end = pd.Timestamp("2026-08-02T00:00:00Z").tz_localize(None)
    start = end - pd.Timedelta(days=cfg.span_days)
    spine_start_s = int(start.timestamp())
    spine_end_s = int(end.timestamp())
    ext_window_start_s = spine_start_s - 15 * 24 * 3600
    ext_window_end_s = spine_end_s

    # Single-shard: all players at once.
    player_ids_all = [f"player_{i:06d}" for i in range(cfg.n_players)]
    players_single, impressions_single = _generate_player_block(
        player_ids_all,
        offers,
        ext_window_start_s,
        ext_window_end_s,
        spine_start_s,
        spine_end_s,
        cfg,
    )

    # Multi-shard: split into shards and concatenate.
    shard_size = 10
    players_shards = []
    impressions_shards = []

    for shard_start in range(0, cfg.n_players, shard_size):
        shard_end = min(shard_start + shard_size, cfg.n_players)
        player_ids_shard = player_ids_all[shard_start:shard_end]

        players_shard, impressions_shard = _generate_player_block(
            player_ids_shard,
            offers,
            ext_window_start_s,
            ext_window_end_s,
            spine_start_s,
            spine_end_s,
            cfg,
        )
        players_shards.append(players_shard)
        impressions_shards.append(impressions_shard)

    import pandas as pd

    players_multi = pd.concat(players_shards, ignore_index=True)
    impressions_multi = pd.concat(impressions_shards, ignore_index=True)

    # Assert: same row counts.
    assert len(players_single) == len(players_multi), (
        f"Player count mismatch: single={len(players_single)} "
        f"multi={len(players_multi)}"
    )
    assert len(impressions_single) == len(impressions_multi), (
        f"Impression count mismatch: single={len(impressions_single)} "
        f"multi={len(impressions_multi)}"
    )

    # Assert: per-player output is identical (sort by player_id before compare).
    for player_id in player_ids_all:
        p_single = players_single[players_single["player_id"] == player_id]
        p_multi = players_multi[players_multi["player_id"] == player_id]

        assert len(p_single) == 1
        assert len(p_multi) == 1

        for col in [
            "country",
            "segment",
            "tenure_days",
            "lifetime_spend",
            "historical_ltv",
            "avg_session_len_min",
            "purchases_30d",
        ]:
            v_s = p_single[col].iloc[0]
            v_m = p_multi[col].iloc[0]
            assert v_s == v_m, f"{player_id}/{col}: single={v_s} multi={v_m}"

    # Assert: impression output is identical per (player_id, offer_id).
    for player_id in player_ids_all:
        imp_single = (
            impressions_single[impressions_single["player_id"] == player_id]
            .sort_values("offer_id")
            .reset_index(drop=True)
        )
        imp_multi = (
            impressions_multi[impressions_multi["player_id"] == player_id]
            .sort_values("offer_id")
            .reset_index(drop=True)
        )

        assert len(imp_single) == len(imp_multi), (
            f"{player_id}: single={len(imp_single)} multi={len(imp_multi)}"
        )

        # Compare numeric/categorical columns.
        for col in [
            "segment",
            "gem_balance",
            "minutes_since_last_purchase",
            "match_count",
            "offer_id",
            "offer_type",
            "price_tier",
            "offer_price_usd",
            "offer_gem_amount",
            "offer_discount_pct",
            "purchased",
        ]:
            v_s = imp_single[col].values
            v_m = imp_multi[col].values

            if col in ["gem_balance", "minutes_since_last_purchase", "offer_price_usd"]:
                # Float tolerance.
                assert np.allclose(v_s, v_m, rtol=1e-5, atol=1e-6), (
                    f"{player_id}/{col}: single={v_s[:3]} multi={v_m[:3]}"
                )
            elif col in ["offer_gem_amount", "offer_discount_pct"]:
                # Float, tighter tolerance.
                assert np.allclose(v_s, v_m, rtol=1e-5, atol=1e-9), (
                    f"{player_id}/{col}: single={v_s[:3]} multi={v_m[:3]}"
                )
            else:
                # Exact match.
                assert (v_s == v_m).all(), (
                    f"{player_id}/{col}: single={v_s} multi={v_m}"
                )

        # Revenue is a derived float, use tolerance.
        rev_s = imp_single["offer_revenue_d7"].values
        rev_m = imp_multi["offer_revenue_d7"].values
        assert np.allclose(rev_s, rev_m, rtol=1e-5, atol=1e-6), (
            f"{player_id}/offer_revenue_d7: single={rev_s[:3]} multi={rev_m[:3]}"
        )

        # event_ts: check they're equal as timestamps.
        ts_s = imp_single["event_ts"].values
        ts_m = imp_multi["event_ts"].values
        for i, (t_s, t_m) in enumerate(zip(ts_s, ts_m, strict=True)):
            assert t_s == t_m, f"{player_id}/event_ts[{i}]: single={t_s} multi={t_m}"


def test_spark_distributed_matches_numpy(spark):
    """The DISTRIBUTED mapInPandas path must equal numpy generate_frames.

    This is the real gate: generate_frames_spark shards player generation across
    Spark (mapInPandas, never collecting to the driver). Its per-player output must
    be bit-identical to the single-process numpy path (per-player md5 RNG ->
    shard-invariant), so the distributed job produces the SAME data. Exercises the
    real mapInPandas distribution; the subsequent Delta write in write_delta is
    plain I/O (not tested here -- local Spark has no Delta data source).
    """
    import pandas as pd

    from offer_recommender.data.generate import (
        GenConfig,
        generate_frames,
        generate_frames_spark,
    )

    cfg = GenConfig(
        n_players=40,
        n_offers=5,
        impressions_per_player=8,
        seed=17,
        end_ts=pd.Timestamp("2026-08-02T00:00:00"),
    )
    # numpy reference
    players_np, _offers_np, imp_np = generate_frames(cfg)

    # distributed Spark path (mapInPandas), materialised to pandas ONLY for the
    # assert (production writes the Spark DataFrame straight to Delta, no collect).
    # Arrow off-heap (mapInPandas/toPandas) is blocked on JDK 17+ without the
    # --add-opens flags that Databricks sets but a local pyspark 3.5/JDK-21 session
    # cannot always apply; skip here rather than fail on a pure local-JVM artifact
    # (the numpy shard-invariance test above already proves per-player determinism
    # is shard-independent, and the real pipeline run exercises mapInPandas E2E).
    import pytest

    player_cols, _offers_sdf, impressions_sdf, _ws, _we = generate_frames_spark(
        spark, cfg
    )
    try:
        imp_sp = impressions_sdf.toPandas()
        # players = dedup of impressions (same derivation write_delta uses).
        players_sp = (
            imp_sp[player_cols].drop_duplicates("player_id").reset_index(drop=True)
        )
    except Exception as exc:  # local Arrow/JDK incompat only
        if "sun.misc.Unsafe" in str(exc) or "DirectByteBuffer" in str(exc):
            pytest.skip(
                f"local Arrow off-heap unavailable (JDK opens): {str(exc)[:80]}"
            )
        raise

    assert len(imp_sp) == len(imp_np), f"imp rows {len(imp_sp)} != {len(imp_np)}"
    assert len(players_sp) == len(players_np)

    # Per-player, per-offer comparison (order-independent).
    key = ["player_id", "offer_id"]
    a = imp_np.sort_values(key).reset_index(drop=True)
    b = imp_sp.sort_values(key).reset_index(drop=True)
    assert (a["player_id"].values == b["player_id"].values).all()
    for col in (
        "gem_balance",
        "match_count",
        "minutes_since_last_purchase",
        "purchased",
        "offer_revenue_d7",
    ):
        assert np.allclose(
            a[col].astype(float).values, b[col].astype(float).values, atol=1e-6
        ), f"col {col} diverged between numpy and Spark paths"
    # event_ts equal to the second.
    assert (
        pd.to_datetime(a["event_ts"]).astype("int64").values
        == pd.to_datetime(b["event_ts"]).astype("int64").values
    ).all(), "event_ts diverged between numpy and Spark paths"
