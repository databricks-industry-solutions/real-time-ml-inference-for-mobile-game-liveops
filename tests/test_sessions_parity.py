"""Numpy/Spark parity test for the shared session-event model.

For each (player_id, segment) pair, collects ``emit_events_spark`` rows and asserts
element-wise equality against ``player_events`` (numpy) on every signal.  This is the
cross-engine anti-skew guarantee: if the md5 key strings or arithmetic diverge between
engines, a failing assertion names the exact player, event index, and field.
"""

from __future__ import annotations

import pytest

from offer_recommender.data import sessions


@pytest.mark.usefixtures("spark")
def test_spark_and_numpy_events_agree(spark):
    """Spark emit_events_spark must agree with numpy player_events on all signals."""
    # player_000004 is bursty at seed=17 (is_bursty True) -> exercises the 3x path.
    players = [
        ("player_000001", "whale"),
        ("player_000042", "minnow"),
        ("player_000004", "lapsed"),
    ]
    ws, we, seed = 0.0, 15 * 24 * 3600.0, 17
    players_df = spark.createDataFrame(players, ["player_id", "segment"])
    sdf = sessions.emit_events_spark(spark, players_df, ws, we, seed)
    got = {(r["player_id"], r["event_index"]): r for r in sdf.collect()}

    for pid, seg in players:
        ev = sessions.player_events(pid, seg, ws, we, seed)
        n = len(ev["t"])

        # Verify total event count matches numpy.
        spark_keys = [(pid, i) for i in range(n)]
        assert all(k in got for k in spark_keys), (
            f"{pid}: expected {n} events but Spark dict missing some keys"
        )

        for i in range(n):
            r = got[(pid, i)]
            # event_ts: cross-engine double arithmetic; 1e-6 absolute tolerance.
            assert abs(r["event_ts"] - ev["t"][i]) < 1e-6, (
                f"{pid}[{i}] event_ts: spark={r['event_ts']} numpy={ev['t'][i]}"
            )
            # Binary signals: must agree exactly (same threshold comparison).
            assert r["offer_shown"] == int(ev["offer_shown"][i]), (
                f"{pid}[{i}] offer_shown: spark={r['offer_shown']} numpy={ev['offer_shown'][i]}"
            )
            assert r["match_won"] == int(ev["match_won"][i]), (
                f"{pid}[{i}] match_won: spark={r['match_won']} numpy={ev['match_won'][i]}"
            )
            assert r["match_lost"] == int(ev["match_lost"][i]), (
                f"{pid}[{i}] match_lost: spark={r['match_lost']} numpy={ev['match_lost'][i]}"
            )
            # gem_spend: zero branch must be exact; non-zero branch tight.
            np_gs = float(ev["gem_spend"][i])
            sp_gs = float(r["gem_spend"])
            assert abs(sp_gs - np_gs) < 1e-9, (
                f"{pid}[{i}] gem_spend: spark={sp_gs} numpy={np_gs}"
            )
            # gem_balance: continuous float, tight tolerance.
            np_gb = float(ev["gem_balance"][i])
            sp_gb = float(r["gem_balance"])
            assert abs(sp_gb - np_gb) < 1e-6, (
                f"{pid}[{i}] gem_balance: spark={sp_gb} numpy={np_gb}"
            )


def test_emit_events_spark_row_count(spark):
    """emit_events_spark must produce exactly event_count rows per player."""
    players = [
        ("player_000001", "whale"),
        ("player_000002", "minnow"),
        ("player_000003", "lapsed"),
    ]
    players_df = spark.createDataFrame(players, ["player_id", "segment"])
    sdf = sessions.emit_events_spark(spark, players_df, 0.0, 86400.0, seed=42)

    counts = {
        r["player_id"]: r["cnt"]
        for r in sdf.groupBy("player_id")
        .count()
        .withColumnRenamed("count", "cnt")
        .collect()
    }
    for pid, seg in players:
        expected = sessions._event_count_for_segment(seg, pid, 42)
        assert counts[pid] == expected, (
            f"{pid} ({seg}): spark={counts[pid]} expected={expected}"
        )
