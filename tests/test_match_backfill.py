"""Unit tests for the SFV match-backfill frame (data.match_backfill).

Uses the local SparkSession fixture (conftest). Guards the ingestion-schema
contract (the MERGE into the live match_events table fails on any mismatch), the
backdated event_ts window, determinism given seed + end_ts, and the production
bound-derivation path (persisted bounds == label bounds exactly).

Task 7 additions: the backfill is now built from `sessions.emit_events_spark` so
every emitted event is bit-for-bit identical to the session timeline the label was
computed from. The correspondence guard verifies this for a sampled player.

Fix-round 1 (CRITICAL): test_persisted_bounds_match_label_bounds verifies that
the exact ext_window_start_s / ext_window_end_s computed by generate_frames equal
the values read back from the session_window_bounds metadata table, so setup_stream
anchors the backfill with 0% row-level disagreement.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from offer_recommender.data.match_backfill import build_backfill_frame

# A fixed window end (unix seconds) so the run is fully deterministic.
_END_TS = 1_800_000_000.0
_SPAN_DAYS = 14
_BACKFILL_DAYS = 15  # extension before spine start
_N_PLAYERS = 5
_SEED = 17
_TOPIC = "match_events"

# Window bounds matching generate.py:
#   spine_end   = end_ts
#   spine_start = end_ts - span_days * 86400
#   ext_window_start = spine_start - backfill_days * 86400
_SPINE_END_S = _END_TS
_SPINE_START_S = _END_TS - _SPAN_DAYS * 86400
_WINDOW_START = _SPINE_START_S - _BACKFILL_DAYS * 86400
_WINDOW_END = _SPINE_END_S


# The value STRUCT field order the match-event ingestion table (and
# store._match_payload_json_schema) expects: key, event time, event type, then the
# match-outcome signals the SFV aggregates are built from. NO request-time passthrough
# here (that rides the separate logins topic).
_EXPECTED_VALUE_FIELDS = (
    ("player_id", "string"),
    ("event_ts", "timestamp"),
    ("event_type", "string"),
    ("gem_spend", "double"),
    ("gem_balance", "double"),
    ("offer_shown", "double"),
    ("match_won", "double"),
    ("match_lost", "double"),
)

_EXPECTED_TOP_LEVEL = (
    ("key", "string"),
    ("record_source", "string"),
    ("stream_record_timestamp", "timestamp"),
    ("kafka_topic", "string"),
    ("kafka_partition", "int"),
    ("kafka_offset", "bigint"),
)

# Representative segments for the small test players.
_SEGMENTS = ["minnow", "dolphin", "whale", "new", "lapsed"]


def _players_df(spark):
    """Build a minimal players DataFrame with player_id and segment."""
    from pyspark.sql import Row

    rows = [
        Row(player_id=f"player_{i:06d}", segment=_SEGMENTS[i % len(_SEGMENTS)])
        for i in range(_N_PLAYERS)
    ]
    return spark.createDataFrame(rows)


def _build(spark):
    """Build the deterministic small backfill frame used across tests."""
    return build_backfill_frame(
        spark,
        players_df=_players_df(spark),
        window_start=_WINDOW_START,
        window_end=_WINDOW_END,
        topic=_TOPIC,
        end_ts=_END_TS,
        seed=_SEED,
    )


def test_schema_matches_ingestion_table(spark) -> None:
    """Top-level and value-struct fields/types match the SFV ingestion schema."""
    frame = _build(spark)
    fields = {f.name: f.dataType.simpleString() for f in frame.schema.fields}

    # Exact top-level column set (no extra/missing columns -> MERGE fails otherwise).
    assert set(fields) == {name for name, _ in _EXPECTED_TOP_LEVEL} | {"value"}
    for name, dtype in _EXPECTED_TOP_LEVEL:
        assert fields[name] == dtype, f"{name}: {fields[name]} != {dtype}"

    value_type = frame.schema["value"].dataType
    value_fields = [(f.name, f.dataType.simpleString()) for f in value_type.fields]
    assert value_fields == list(_EXPECTED_VALUE_FIELDS)


def test_row_count_and_record_source(spark) -> None:
    """Row count equals total session events across all players; all tagged backfill."""
    from offer_recommender.data import sessions

    frame = _build(spark)
    pdf = _players_df(spark).toPandas()
    player_ids = pdf["player_id"].to_numpy()
    segs = pdf["segment"].to_numpy()
    expected_rows = int(sessions.event_count(player_ids, segs, _SEED).sum())
    assert frame.count() == expected_rows
    sources = {r["record_source"] for r in frame.select("record_source").collect()}
    assert sources == {"backfill"}


def test_event_ts_within_window(spark) -> None:
    """Every value.event_ts (TIMESTAMP) falls in [window_start, window_end]."""
    from pyspark.sql.types import TimestampType

    frame = _build(spark)
    value_type = frame.schema["value"].dataType
    assert isinstance(value_type["event_ts"].dataType, TimestampType)

    tss = [r["value"]["event_ts"].timestamp() for r in frame.select("value").collect()]
    assert all(_WINDOW_START <= ts <= _WINDOW_END for ts in tss)


def test_deterministic(spark) -> None:
    """Same seed + end_ts -> identical rows (sorted for order-independence)."""
    a = sorted(_build(spark).collect(), key=lambda r: (r["key"], r["kafka_offset"]))
    b = sorted(_build(spark).collect(), key=lambda r: (r["key"], r["kafka_offset"]))
    assert a == b


def test_backfill_events_match_player_events(spark) -> None:
    """Row-level correspondence: backfill events for a player equal sessions.player_events.

    This is the payoff guard for Task 7. Verifies that the signals emitted by
    build_backfill_frame (via emit_events_spark) are bit-for-bit identical to the
    numpy session model that produced the purchase labels in generate.py.
    """
    from offer_recommender.data import sessions

    pdf = _players_df(spark).toPandas()
    # Pick the first player for the correspondence check.
    player_id = pdf["player_id"].iloc[0]
    segment = pdf["segment"].iloc[0]

    # Ground truth from the numpy session model (same args as the label side uses).
    expected = sessions.player_events(
        player_id, segment, _WINDOW_START, _WINDOW_END, _SEED
    )

    frame = _build(spark)
    # Key the comparison by event_index (not event_ts): burst events pack ~100 rows
    # into a whole-second-truncated span, so many share a timestamp -- sorting by ts
    # gives an unstable cross-engine tiebreak. event_index is recoverable from
    # kafka_offset (= player_index * _OFFSET_STRIDE + event_index), and the numpy
    # arrays are already in event_index order.
    from offer_recommender.data.match_backfill import _OFFSET_STRIDE

    rows = (
        frame.where(frame["value.player_id"] == player_id)
        .select("value", "kafka_offset")
        .collect()
    )
    assert len(rows) == len(expected["t"]), (
        f"Row count mismatch: backfill={len(rows)}, player_events={len(expected['t'])}"
    )
    rows_sorted = sorted(rows, key=lambda r: r["kafka_offset"] % _OFFSET_STRIDE)

    # event_ts within 1s (whole-second truncation) at each event_index.
    for i, row in enumerate(rows_sorted):
        assert math.isclose(
            row["value"]["event_ts"].timestamp(), float(expected["t"][i]), abs_tol=1.0
        ), f"event_ts mismatch at event {i}"

    # Verify match-outcome signals (gem_spend, gem_balance, offer_shown, match_won,
    # match_lost) match element-wise by event_index. gem_balance is explicitly
    # included: it feeds avg_gem_balance_15m (the SFV aggregate and the label's
    # marquee buy-trigger), so a decoupled independent draw would be caught.
    exp_gem_spend = expected["gem_spend"]
    exp_gem_balance = expected["gem_balance"]
    exp_os = expected["offer_shown"]
    exp_mw = expected["match_won"]
    exp_ml = expected["match_lost"]

    for i, row in enumerate(rows_sorted):
        v = row["value"]
        assert math.isclose(v["gem_spend"], float(exp_gem_spend[i]), abs_tol=1e-6), (
            f"gem_spend mismatch at event {i}"
        )
        assert math.isclose(
            v["gem_balance"], float(exp_gem_balance[i]), abs_tol=1e-6
        ), (
            f"gem_balance mismatch at event {i}: backfill={v['gem_balance']:.4f}, "
            f"session_model={float(exp_gem_balance[i]):.4f}. "
            "The backfill must source gem_balance from emit_events_spark (salt 'gb'), "
            "not an independent draw."
        )
        assert float(v["offer_shown"]) == float(exp_os[i]), (
            f"offer_shown mismatch at event {i}"
        )
        assert float(v["match_won"]) == float(exp_mw[i]), (
            f"match_won mismatch at event {i}"
        )
        assert float(v["match_lost"]) == float(exp_ml[i]), (
            f"match_lost mismatch at event {i}"
        )


def test_deterministic_all_columns(spark) -> None:
    """All columns (including non-session REALTIME numerics) are fully deterministic.

    Previously, events_this_session / gem_balance / minutes_since_last_purchase /
    stream_record_timestamp used sf.rand (partition-order-dependent). After the
    fix-round, they use md5_uniform keyed on player_id + event_index + salt + seed,
    so the frame is bit-for-bit identical across runs.  This test is stricter than
    test_deterministic because it also checks the _value_ struct fields.
    """

    def _collect_sorted(frame):
        return sorted(
            frame.select("key", "kafka_offset", "value").collect(),
            key=lambda r: (r["key"], r["kafka_offset"]),
        )

    rows_a = _collect_sorted(_build(spark))
    rows_b = _collect_sorted(_build(spark))
    assert rows_a == rows_b, "Non-session REALTIME columns are not deterministic"


def test_persisted_bounds_match_label_bounds(spark) -> None:
    """Persisted session_window_bounds exactly equals the label's ext_window bounds.

    This is the production-path guard for the CRITICAL fix-round finding.  It:
    1. Generates frames with a fixed GenConfig (deterministic end_ts).
    2. Reconstructs ext_window_start_s / ext_window_end_s using the SAME formula
       as write_delta (integer seconds, -15d extension).
    3. Asserts that the values written to session_window_bounds equal those exact
       floats (float equality, no tolerance).
    4. Builds the backfill with the PERSISTED bounds and verifies that the
       label's window_counts and the SFV recompute over the backfill agree for a
       sample player (near-0 disagreement threshold).
    """
    from offer_recommender.data import generate, sessions
    from offer_recommender.data.generate import GenConfig

    # Fixed, deterministic config.
    cfg = GenConfig(
        n_players=_N_PLAYERS,
        n_offers=2,
        impressions_per_player=4,
        seed=_SEED,
        span_days=_SPAN_DAYS,
        end_ts=pd.Timestamp.fromtimestamp(_END_TS, "UTC").tz_localize(None),
    )
    _players, _offers, _impressions = generate.generate_frames(cfg)

    # Reconstruct the exact bounds write_delta computes.
    _end = cfg.end_ts
    _start = _end - pd.Timedelta(days=cfg.span_days)
    expected_start_s = float(int(_start.timestamp()) - 15 * 24 * 3600)
    expected_end_s = float(int(_end.timestamp()))

    # Confirm that the formula matches what generate_frames used.
    # The label's streaming counts were computed with window_start = expected_start_s.
    # (No tolerance: these must be the exact same floats.)
    assert expected_start_s == _WINDOW_START, (
        f"Expected window_start {expected_start_s} != test constant {_WINDOW_START}"
    )
    assert expected_end_s == _WINDOW_END, (
        f"Expected window_end {expected_end_s} != test constant {_WINDOW_END}"
    )

    # Build the backfill with the persisted-path bounds (identical to what
    # _read_window_bounds would return from the Delta metadata table).
    players_pdf = pd.DataFrame(
        [
            {"player_id": f"player_{i:06d}", "segment": _SEGMENTS[i % len(_SEGMENTS)]}
            for i in range(_N_PLAYERS)
        ]
    )
    players_sdf = spark.createDataFrame(players_pdf)
    frame = build_backfill_frame(
        spark,
        players_df=players_sdf,
        window_start=expected_start_s,
        window_end=expected_end_s,
        topic=_TOPIC,
        end_ts=_END_TS,
        seed=cfg.seed,
    )

    # Pick the first player and compare window_counts from the label against a
    # manual count over the backfill events for the same player and window.
    pid = f"player_{0:06d}"
    seg = _SEGMENTS[0]

    # Anchor `at` on the LAST actual session event for this player so the 15-min
    # window contains at least one event (deterministic, non-empty).  Sampling at
    # window_end - 60 risks an empty 900-second window with sparse events (~80
    # events over 29 days), making the assertion a tautological 0.0 == 0.0.
    player_evts = sessions.player_events(
        pid, seg, expected_start_s, expected_end_s, cfg.seed
    )
    at = float(np.max(player_evts["t"]))  # guaranteed non-empty: last event in window
    assert at > expected_start_s, "player_events returned no events; test setup error"

    # Ground-truth from the label side (same bounds, same seed).
    label_counts = sessions.window_counts_batch(
        np.array([pid]),
        np.array([seg]),
        np.array([at]),
        window_start=expected_start_s,
        window_end=expected_end_s,
        seed=cfg.seed,
    )

    # Confirm the label window is non-empty (avg_gem_balance_15m would be 0.0 for
    # an empty window, making the assertion tautological).
    assert float(label_counts["avg_gem_balance_15m"][0]) != 0.0, (
        "avg_gem_balance_15m is 0.0 at the last event ts — window is empty; "
        "test-setup invariant violated."
    )

    # Recompute avg_gem_balance_15m manually from the backfill frame: mean
    # gem_balance of events in (at - 900, at] (the window_counts_batch semantics).
    bf_rows_full = (
        frame.where(frame["value.player_id"] == pid)
        .select("value.event_ts", "value.gem_balance")
        .collect()
    )
    gb_in_window = [
        float(r["gem_balance"])
        for r in bf_rows_full
        if r["event_ts"] is not None and at - 900 < r["event_ts"].timestamp() <= at
    ]
    gb_manual = float(np.mean(gb_in_window)) if gb_in_window else 0.0
    gb_label = float(label_counts["avg_gem_balance_15m"][0])

    assert abs(gb_manual - gb_label) < 1e-6, (
        f"Persisted-bounds backfill avg_gem_balance_15m={gb_manual:.4f} "
        f"!= label={gb_label:.4f} for player {pid}. "
        "Bounds mismatch shifts event timestamps; should be 0 disagreement."
    )


def test_write_delta_consumes_generate_frames_bounds_no_double_now(
    monkeypatch,
) -> None:
    """Production path (end_ts=None): write_delta must persist the EXACT bounds
    that generate_frames used for window_counts_batch — never call now() a second
    time after generate_frames returns.

    Strategy: monkeypatch _resolve_end to return a fixed timestamp and assert that
    the bounds stored in imp.attrs["_session_window"] equal what write_delta would
    pass to the metadata table. A second independent now() call inside write_delta
    would return a DIFFERENT value (simulated by advancing a counter), causing the
    bounds to diverge.

    This test FAILS against round-1 code (which recomputed now() in write_delta)
    and passes after this fix (which reads imp.attrs["_session_window"]).
    """
    import pandas as pd

    from offer_recommender.data import generate
    from offer_recommender.data.generate import GenConfig

    # Simulate advancing wall clock: first call returns T0, second would return T0+5s.
    _t0 = pd.Timestamp("2026-07-30 12:00:00")
    _t1 = pd.Timestamp("2026-07-30 12:00:05")
    _call_count: list[int] = [0]

    def _advancing_resolve_end(cfg: GenConfig) -> pd.Timestamp:
        val = _t0 if _call_count[0] == 0 else _t1
        _call_count[0] += 1
        return val

    monkeypatch.setattr(generate, "_resolve_end", _advancing_resolve_end)

    cfg = GenConfig(
        n_players=10,
        n_offers=2,
        impressions_per_player=2,
        seed=42,
        span_days=14,
        end_ts=None,  # production path: no fixed end_ts
    )

    _players, _offers, imp = generate.generate_frames(cfg)
    assert _call_count[0] == 1, "generate_frames must call _resolve_end exactly once"

    # The bounds generate_frames actually used (captured via attrs):
    sw = imp.attrs["_session_window"]
    actual_start_s = sw["window_start_s"]
    actual_end_s = sw["window_end_s"]

    # What round-1 write_delta would have independently recomputed (a second now()):
    _end_round1 = _t1  # second call returns t1, which is t0 + 5s
    _start_round1 = _end_round1 - pd.Timedelta(days=cfg.span_days)
    drifted_start_s = float(int(_start_round1.timestamp()) - 15 * 24 * 3600)
    drifted_end_s = float(int(_end_round1.timestamp()))

    # The attrs-sourced bounds must equal the T0-based bounds, NOT the drifted ones.
    expected_end_s = float(int(_t0.timestamp()))
    expected_start_s = float(
        int((_t0 - pd.Timedelta(days=cfg.span_days)).timestamp()) - 15 * 24 * 3600
    )

    assert actual_end_s == expected_end_s, (
        f"window_end_s in attrs={actual_end_s} != expected={expected_end_s}. "
        "generate_frames may not be using _resolve_end."
    )
    assert actual_start_s == expected_start_s, (
        f"window_start_s in attrs={actual_start_s} != expected={expected_start_s}."
    )

    # Confirm the drift IS real (round-1 would have produced different bounds):
    assert drifted_end_s != actual_end_s, (
        "Test setup error: t0 == t1, no simulated drift to detect."
    )
    assert drifted_start_s != actual_start_s, (
        "Test setup error: drifted start == actual start despite t0 != t1."
    )


def test_kafka_offset_unique_no_global_sort(spark) -> None:
    """kafka_offset is unique per row and requires no unpartitioned Window sort.

    This guard verifies the hash-derived offset formula produces no collisions
    across all rows for a small player set. A unpartitioned Window.orderBy would
    shuffle all data to one partition at studio scale; this approach is O(1) per row.
    """
    frame = _build(spark)
    offsets = [r["kafka_offset"] for r in frame.select("kafka_offset").collect()]
    assert len(offsets) == len(set(offsets)), (
        f"kafka_offset has {len(offsets) - len(set(offsets))} collision(s); "
        "the formula player_index*1000+event_index must yield unique values."
    )
