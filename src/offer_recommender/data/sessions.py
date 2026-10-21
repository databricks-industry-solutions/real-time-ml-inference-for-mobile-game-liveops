"""Shared per-player session timeline — numpy core and Spark evaluator.

A single deterministic event model that both the impression-label side (numpy,
generate.py) and the Kafka-stream side (Spark, match_backfill.py / Task 3)
evaluate identically, so window aggregates at the label and at the stream agree
without any runtime data handoff.

Key design:
- Every event is identified by a string key ``f"{player_id}:{i}:{salt}:{seed}"``
  hashed through md5_uniform_str, giving cross-engine reproducibility.
- ``player_events`` reconstructs all events for one player (per-event Python loop
  is fine for debugging / invariant checks).
- ``window_counts_batch`` is vectorized numpy over the full impression spine:
  it reconstructs events via broadcasted md5 arithmetic and reduces in matrix
  form (no per-row Python replay).
- ``emit_events_spark`` mirrors ``player_events`` / ``event_count`` in Spark SQL,
  using ``SPARK_MD5_UNIFORM_SQL`` with identical key strings for cross-engine
  bit-for-bit parity (see ``tests/test_sessions_parity.py``).

The aggregation semantics MUST match ``features.store._operator`` exactly:
  - ``("count", "login", W)``: row count in window (every event is a login).
    NOTE: this branch is unused by the current WINDOWS set (all ops are avg/sum
    over real signals) but retained in _operator for future use.
  - ``("count", signal, W)`` or ``("sum", signal, W)`` where signal != "login":
    Sum of the signal's 0/1 (or float) values in window.
  - ``("avg", signal, W)``: mean of signal values in window (NaN when no events).

Per-event signals (R2 value-centric set):
  gem_spend     : float spend amount (mostly 0; positive when gem purchase occurs).
  gem_balance   : continuous gem balance at event time; avg over window gives
                  avg_gem_balance_15m (the gem-block proximity buy trigger).
  offer_shown   : 0/1 indicator; sum -> offers_seen_15m (fatigue signal).
  match_won     : 0/1; sum -> match_wins_15m.
  match_lost    : 0/1, complement of match_won; sum -> match_losses_15m.

gem_balance ALSO stays a REALTIME_NUMERICAL_FEATURES passthrough (the instantaneous
snapshot at the moment of the request). avg_gem_balance_15m is the windowed average
and is DISTINCT from that passthrough (different name, different semantics).

Tasks 6 and 7 import the signal thresholds + SESSION_SIGNALS from this module so
the live producer and the backfill use the exact same per-event distributions.
"""

from __future__ import annotations

import numpy as np

from offer_recommender import schema
from offer_recommender.data.hashing import (
    SPARK_MD5_UNIFORM_SQL,
    md5_uniform_np,
    md5_uniform_str,
)

# ---------------------------------------------------------------------------
# Module constants (single source of truth for Tasks 6 and 7)
# ---------------------------------------------------------------------------

# --- Session-burst structure --------------------------------------------------
# A player's activity is a series of SESSIONS. Each session fires a dense BURST of
# EVENTS_PER_SESSION match-events scattered across the BURST_SPAN_S seconds after
# the session start (a login opens a session, then ~100 gameplay events over the
# next ~4 min). This is what makes a 15-min rolling window hold ~a full burst of
# events -- so the streaming SUM/AVG features are dense and non-degenerate, and
# identical between training (this timeline) and live serving (the match_events
# producer reproduces the same burst density). Windows spread uniformly over ~29
# days would instead leave ~0 events per 15-min window (the density skew that made
# the model score chatty players at $0).
EVENTS_PER_SESSION: int = 100
"""Match-events per session burst (fixed, so event_index -> session is i // this)."""

BURST_SPAN_S: float = 30.0
"""Seconds over which a session's events scatter (uniform). Short (30s, was 240s) so
the live producer's per-event heap stays bounded at ~1100 logins/sec; still << the
900s SFV window, so a burst fully populates the window."""

# Sessions per player over the observation window, by segment (the ~10-20 band:
# whales play most, lapsed least). event_count = sessions * EVENTS_PER_SESSION.
SESSIONS_PER_SEGMENT: dict[str, int] = {
    "whale": 20,
    "dolphin": 16,
    "minnow": 12,
    "new": 10,
    "lapsed": 8,
}
_DEFAULT_SESSIONS: int = 10

# Bursty-player dimension (orthogonal to segment). A md5-uniform ~BURSTY_FRACTION (1%)
# of players are "very bursty" (BURSTY_FACTOR x their segment's session count -> dense
# 15-min windows that blip in the app); the rest get NONBURSTY_FACTOR x (less active).
# Shared bit-for-bit by the numpy label side (event_count here), the Spark backfill
# (emit_events_spark), and the live producer (jobs/produce_events.py), and stamped on the
# login event so the app feed filters to the 1% bursty players.
BURSTY_FRACTION: float = 0.01
BURSTY_FACTOR: float = 3.0
NONBURSTY_FACTOR: float = 0.5

# Gem-spend signal: probability an event includes a gem purchase and the max amount.
GEM_SPEND_PROB: float = 0.15
GEM_SPEND_MAX: float = 200.0

# Gem-balance: continuous gem balance at event time.
# Distribution: uniform(0, GEM_BALANCE_CAP) via md5_uniform (deterministic hashing).
# Cap is used by _purchase_propensity to normalise the gem-block term.
# jobs/produce_events.py mirrors GEM_BALANCE_CAP.
GEM_BALANCE_CAP: float = 3000.0

# Offer-shown signal: probability an offer was shown at this event.
OFFER_SHOWN_PROB: float = 0.60

# Match-outcome signal: probability a match is WON (match_lost = 1 - match_won).
MATCH_WON_PROB: float = 0.5

# match_count passthrough: instantaneous matches-played-this-session (the old
# session_depth analog); uniform int in [1, MATCH_COUNT_MAX].
MATCH_COUNT_MAX: int = 20

# Number of distinct offer IDs in the synthetic catalogue.
N_OFFERS: int = 5

# Ordered listing of per-event signal fields. Tasks 6/7 import this to verify parity.
SESSION_SIGNALS: tuple[str, ...] = (
    "t",
    "gem_spend",
    "gem_balance",
    "offer_shown",
    "match_won",
    "match_lost",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def is_bursty(player_id: str, seed: int) -> bool:
    """True for the ~BURSTY_FRACTION of players flagged as very bursty (md5-uniform)."""
    return md5_uniform_str(f"{player_id}:bursty:{seed}") < BURSTY_FRACTION


def _sessions_for_segment(segment: str, player_id: str, seed: int) -> int:
    """Deterministic session count: segment base x the player's burst factor."""
    base = SESSIONS_PER_SEGMENT.get(segment, _DEFAULT_SESSIONS)
    factor = BURSTY_FACTOR if is_bursty(player_id, seed) else NONBURSTY_FACTOR
    return round(base * factor)


def _event_count_for_segment(segment: str, player_id: str, seed: int) -> int:
    """event_count = sessions(segment, player) * EVENTS_PER_SESSION."""
    return _sessions_for_segment(segment, player_id, seed) * EVENTS_PER_SESSION


def _segment_max_events() -> int:
    """Max event count across segments for the BURSTIEST player (matrix width bound)."""
    top = max(SESSIONS_PER_SEGMENT.values())
    return round(top * BURSTY_FACTOR) * EVENTS_PER_SESSION


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def event_count(
    player_ids: np.ndarray,
    segments: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Return the deterministic event count for each player/segment pair.

    Parameters
    ----------
    player_ids:
        1-D array of player-id strings (shape ``(n,)``).
    segments:
        1-D array of segment strings, same length as ``player_ids``.
    seed:
        Integer seed embedded in the hash key for reproducibility.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n,)`` with each player's event count
        (sessions * EVENTS_PER_SESSION, from SESSIONS_PER_SEGMENT).
    """
    return np.array(
        [
            _event_count_for_segment(str(seg), str(pid), seed)
            for pid, seg in zip(player_ids, segments, strict=True)
        ],
        dtype=np.int64,
    )


def player_events(
    player_id: str,
    segment: str,
    window_start: float,
    window_end: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Reconstruct the full event timeline for one player.

    Each event is determined by ``md5_uniform_str(f"{player_id}:{i}:{salt}:{seed}")``
    so the output is fully deterministic and cross-engine reproducible.

    Parameters
    ----------
    player_id:
        Player identifier string.
    segment:
        Player segment (e.g. ``"whale"``, ``"minnow"``).
    window_start:
        Start of the observation window in unix seconds (inclusive).
    window_end:
        End of the observation window in unix seconds (exclusive).
    seed:
        Integer seed for reproducibility.

    Returns
    -------
    dict with keys matching ``SESSION_SIGNALS``:
        ``t`` — event timestamps (float, unix seconds) \n
        ``gem_spend`` — float, >0 when the player spent gems \n
        ``gem_balance`` — float in ``[0, GEM_BALANCE_CAP)``, gem balance at event time \n
        ``offer_shown`` — int 0/1 \n
        ``match_won`` — int 0/1, 1 when the match was won \n
        ``match_lost`` — int 0/1, complement of match_won
    """
    k = _event_count_for_segment(segment, player_id, seed)
    span = window_end - window_start
    idx = np.arange(k)

    # Build all hash keys for each signal per event index.
    def _keys(salt: str) -> np.ndarray:
        return np.array([f"{player_id}:{i}:{salt}:{seed}" for i in idx], dtype=object)

    # Burst-structured event time: event i belongs to session (i // EVENTS_PER_SESSION);
    # each session starts uniformly in the window (leaving room for the burst span) and
    # its events scatter uniformly across the next BURST_SPAN_S seconds. So ~100 events
    # cluster within a ~4-min burst, filling a 15-min rolling window. Session-start uses
    # a "sess" salt keyed on the SESSION index (constant within a burst); the per-event
    # "ts" offset stays keyed on the event index.
    session_idx = idx // EVENTS_PER_SESSION
    sess_keys = np.array(
        [f"{player_id}:{s}:sess:{seed}" for s in session_idx], dtype=object
    )
    session_start_ts = window_start + md5_uniform_np(sess_keys) * (span - BURST_SPAN_S)
    t = session_start_ts + md5_uniform_np(_keys("ts")) * BURST_SPAN_S
    gsflag = md5_uniform_np(_keys("gsflag")) < GEM_SPEND_PROB
    gem_spend = np.where(gsflag, md5_uniform_np(_keys("gs")) * GEM_SPEND_MAX, 0.0)
    # gem_balance: uniform in [0, GEM_BALANCE_CAP). Salt "gb" — does not collide with
    # existing salts: ts, gsflag, gs, os, mw.
    gem_balance = md5_uniform_np(_keys("gb")) * GEM_BALANCE_CAP
    offer_shown = (md5_uniform_np(_keys("os")) < OFFER_SHOWN_PROB).astype(np.int32)
    match_won = (md5_uniform_np(_keys("mw")) < MATCH_WON_PROB).astype(np.int32)
    match_lost = 1 - match_won

    return {
        "t": t,
        "gem_spend": gem_spend,
        "gem_balance": gem_balance,
        "offer_shown": offer_shown,
        "match_won": match_won,
        "match_lost": match_lost,
    }


def window_counts_batch(
    player_ids: np.ndarray,
    segments: np.ndarray,
    at_ts: np.ndarray,
    window_start: float,
    window_end: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Vectorized rolling-window aggregates over the impression spine.

    For each row ``r``, computes the 6 ``schema.WINDOWS`` aggregates ending at
    ``at_ts[r]``.  Reconstructs each player's events via md5 arithmetic without
    any per-row Python session replay (fully vectorized numpy).

    Aggregation semantics mirror ``features.store._operator`` exactly (R2 set):
    - ``gems_spent_15m``      : sum of ``gem_spend`` in ``(at-900, at]``.
    - ``avg_gem_balance_15m`` : mean of ``gem_balance`` in ``(at-900, at]``;
      0.0 when window is empty (matching transform.py nan-fill).
    - ``offers_seen_15m``     : sum of ``offer_shown`` (0/1) in ``(at-900, at]``.
    - ``match_wins_15m``      : sum of ``match_won`` (0/1) in ``(at-900, at]``.
    - ``match_losses_15m``    : sum of ``match_lost`` (0/1) in ``(at-900, at]``.
    - ``match_events_15m``    : row count (every event is a login) in ``(at-900, at]``.

    Parameters
    ----------
    player_ids:
        1-D array of player-id strings, shape ``(n,)``.
    segments:
        1-D array of segment strings, shape ``(n,)``.
    at_ts:
        1-D float array of the timestamp at which to evaluate the window,
        shape ``(n,)``; unix seconds.
    window_start:
        Earliest possible event timestamp (passed to ``player_events``).
    window_end:
        Latest possible event timestamp (passed to ``player_events``).
    seed:
        Integer seed for reproducibility (same as used in ``player_events``).

    Returns
    -------
    dict[str, np.ndarray]
        Keys == ``schema.WINDOWS``, each value is a float array of shape ``(n,)``.
    """
    n = len(player_ids)
    span = window_end - window_start
    max_k = _segment_max_events()

    # Per-row event counts (determined purely by segment, no md5).
    counts = event_count(player_ids, segments, seed)  # (n,)

    # Build (n, max_k) matrices for each signal by vectorizing all md5 calls.
    # Key format: f"{player_id}:{event_idx}:{salt}:{seed}"
    # We broadcast: rows x events -> flattened key array -> md5 -> reshape.
    row_idx = np.arange(n)
    event_idx = np.arange(max_k)

    # (n, max_k) string key matrix built via list comprehension (unavoidable for
    # variable-length string formatting; md5_uniform_np handles the vector call).
    def _build_signal(salt: str) -> np.ndarray:
        """Build (n, max_k) uniform float matrix for one signal salt."""
        keys = np.array(
            [f"{player_ids[r]}:{e}:{salt}:{seed}" for r in row_idx for e in event_idx],
            dtype=object,
        )
        return md5_uniform_np(keys).reshape(n, max_k)

    # Burst-structured timestamps (mirror player_events): each event's session index
    # is (event_idx // EVENTS_PER_SESSION); the session starts uniformly in the window
    # (keyed on the session index, "sess" salt) and events scatter across the next
    # BURST_SPAN_S seconds (keyed on event index, "ts" salt). (n, max_k).
    session_col = event_idx // EVENTS_PER_SESSION  # (max_k,)
    sess_keys = np.array(
        [f"{player_ids[r]}:{s}:sess:{seed}" for r in row_idx for s in session_col],
        dtype=object,
    )
    session_start_mat = window_start + md5_uniform_np(sess_keys).reshape(n, max_k) * (
        span - BURST_SPAN_S
    )
    ts_u = _build_signal("ts")
    t_mat = session_start_mat + ts_u * BURST_SPAN_S  # event times, unix seconds

    # Signal matrices: (n, max_k)
    gsflag_mat = _build_signal("gsflag") < GEM_SPEND_PROB
    gs_mat = _build_signal("gs") * GEM_SPEND_MAX * gsflag_mat
    # gem_balance: uniform in [0, GEM_BALANCE_CAP). Salt "gb" — no collision with
    # existing salts (ts, gsflag, gs, os, mw). Matches player_events.
    gb_mat = _build_signal("gb") * GEM_BALANCE_CAP
    os_mat = (_build_signal("os") < OFFER_SHOWN_PROB).astype(np.float64)
    mw_mat = (_build_signal("mw") < MATCH_WON_PROB).astype(np.float64)
    ml_mat = 1.0 - mw_mat

    # Event-valid mask: event index < this player's count.
    # Shape (n, max_k); True where the event exists for this player.
    valid_mask = event_idx[np.newaxis, :] < counts[:, np.newaxis]  # (n, max_k)

    # at_ts as (n, 1) for broadcasting against (n, max_k).
    at_col = at_ts[:, np.newaxis]

    results: dict[str, np.ndarray] = {}
    for feat, (op, signal, W) in schema.WINDOWS.items():
        # Window mask: strict lower bound, non-strict upper (matches the test's
        # bruteforce: (t > at - W) & (t <= at)).
        win_mask = valid_mask & (t_mat > at_col - W) & (t_mat <= at_col)

        if op == "count" and signal == "login":
            # Row count: every event is a login (match_events_15m).
            results[feat] = win_mask.sum(axis=1).astype(np.float64)
        elif op in ("count", "sum"):
            # Sum of a 0/1 or float signal over in-window events.
            if signal == "gem_spend":
                sig_mat = gs_mat
            elif signal == "offer_shown":
                sig_mat = os_mat
            elif signal == "match_won":
                sig_mat = mw_mat
            elif signal == "match_lost":
                sig_mat = ml_mat
            else:
                raise ValueError(f"Unknown signal {signal!r} for op {op!r}")
            results[feat] = (sig_mat * win_mask).sum(axis=1)
        elif op == "avg":
            # Mean of signal over in-window events; 0.0 when window is empty
            # (matches training NaN-fill in transform.py: np.nan_to_num(nan=0.0)).
            if signal == "gem_balance":
                sig_mat = gb_mat
            else:
                raise ValueError(f"Unknown signal {signal!r} for op 'avg'")
            n_in = win_mask.sum(axis=1).astype(np.float64)
            total = (sig_mat * win_mask).sum(axis=1)
            # Avoid divide-by-zero warning: use safe denominator, then zero out.
            safe_n = np.where(n_in > 0, n_in, 1.0)
            results[feat] = np.where(n_in > 0, total / safe_n, 0.0)
        else:
            raise ValueError(f"Unknown operator {op!r}")

    return results


def emit_events_spark(
    spark,  # pyspark.sql.SparkSession
    players_df,  # pyspark.sql.DataFrame with columns (player_id: str, segment: str)
    window_start: float,
    window_end: float,
    seed: int,
):  # -> pyspark.sql.DataFrame
    """Spark evaluator for the shared per-player session timeline.

    Mirrors ``player_events`` / ``event_count`` exactly: uses the same md5 key
    strings and arithmetic so every signal is bit-for-bit identical to numpy.

    Each output row represents one (player, event) pair with columns:
        - ``player_id``   : string
        - ``event_index`` : int (0-based)
        - ``event_ts``    : double, unix seconds
        - ``gem_spend``   : float (>0 with probability GEM_SPEND_PROB)
        - ``gem_balance`` : float in [0, GEM_BALANCE_CAP), uniform via md5 "gb" salt
        - ``offer_shown`` : int 0/1
        - ``match_won``   : int 0/1, 1 when the match was won
        - ``match_lost``  : int 0/1, complement of match_won

    Parameters
    ----------
    spark:
        Active ``SparkSession``.
    players_df:
        DataFrame with at least ``player_id`` (string) and ``segment`` (string).
    window_start:
        Start of the observation window in unix seconds.
    window_end:
        End of the observation window in unix seconds.
    seed:
        Integer seed; must match the numpy ``player_events`` call to ensure parity.

    Returns
    -------
    pyspark.sql.DataFrame
        One row per (player, event_index).  Order within a player is arbitrary.

    Notes
    -----
    Uses ``SPARK_MD5_UNIFORM_SQL`` from ``hashing.py`` for every random draw so
    Spark's md5/conv and numpy's hashlib.md5 produce the same [0,1) values.
    Do NOT substitute ``sf.rand()`` — it uses a different RNG and will not match.
    """
    import pyspark.sql.functions as sf

    span = window_end - window_start

    # -----------------------------------------------------------------------
    # Step 1: attach per-player event count via segment base * burst factor.
    # Mirrors _event_count_for_segment: round(base * factor) * EVENTS_PER_SESSION,
    # where factor = BURSTY_FACTOR if md5(player:bursty:seed) < BURSTY_FRACTION
    # else NONBURSTY_FACTOR. The bursty key uses NO event index (player-level).
    # sf.round uses HALF_UP; Python round uses banker's. base*{0.5,3.0} for integer
    # bases {8,10,12,16,20} yields no half-integer ties, so they agree.
    # -----------------------------------------------------------------------
    seg_sessions_map = sf.create_map(
        *[
            item
            for seg, ns in SESSIONS_PER_SEGMENT.items()
            for item in (sf.lit(seg), sf.lit(ns))
        ]
    )
    seed_str = str(seed)
    bursty_key = f"concat_ws(':', player_id, 'bursty', '{seed_str}')"
    bursty_u = sf.expr(SPARK_MD5_UNIFORM_SQL.format(col=bursty_key)).cast("double")
    factor = sf.when(
        bursty_u < sf.lit(BURSTY_FRACTION), sf.lit(BURSTY_FACTOR)
    ).otherwise(sf.lit(NONBURSTY_FACTOR))
    players_with_k = players_df.withColumn(
        "_sessions",
        sf.round(
            sf.coalesce(seg_sessions_map[sf.col("segment")], sf.lit(_DEFAULT_SESSIONS))
            * factor
        ).cast("int"),
    ).withColumn(
        "_k",
        sf.col("_sessions") * sf.lit(EVENTS_PER_SESSION),
    )

    # -----------------------------------------------------------------------
    # Step 2: explode each player into (player_id, event_index) rows.
    # sequence(0, k-1) produces an array of ints 0..k-1; explode flattens.
    # -----------------------------------------------------------------------
    events_df = players_with_k.withColumn(
        "event_index",
        sf.explode(sf.sequence(sf.lit(0), sf.col("_k") - 1)),
    ).select("player_id", "segment", "event_index")

    # -----------------------------------------------------------------------
    # Step 3: compute each signal via SPARK_MD5_UNIFORM_SQL.
    # Key pattern: concat_ws(':', player_id, cast(event_index as string),
    #              lit(salt), lit(str(seed)))
    # Must be byte-identical to f"{player_id}:{i}:{salt}:{seed}" in numpy.
    # Integer event_index casts to e.g. "0", "1" — no ".0" suffix.
    # -----------------------------------------------------------------------
    def _uniform(salt: str):
        """Build a Spark column [0,1) uniform using md5 on the canonical key.

        Key string is ``{player_id}:{event_index}:{salt}:{seed}`` — byte-identical
        to the numpy key ``f"{player_id}:{i}:{salt}:{seed}"``.
        """
        key_sql = (
            f"concat_ws(':', player_id, CAST(event_index AS STRING),"
            f" '{salt}', '{seed_str}')"
        )
        return sf.expr(SPARK_MD5_UNIFORM_SQL.format(col=key_sql)).cast("double")

    def _uniform_session():
        """Session-start [0,1) uniform, keyed on the SESSION index (constant within a
        burst): f"{player_id}:{event_index // EVENTS_PER_SESSION}:sess:{seed}" --
        byte-identical to the numpy sess key (integer session index, no ".0")."""
        sess_sql = f"CAST(event_index AS INT) DIV {EVENTS_PER_SESSION}"
        key_sql = (
            f"concat_ws(':', player_id, CAST({sess_sql} AS STRING),"
            f" 'sess', '{seed_str}')"
        )
        return sf.expr(SPARK_MD5_UNIFORM_SQL.format(col=key_sql)).cast("double")

    # Burst-structured event_ts (mirrors player_events): session_start uniform in
    # [window_start, window_end - BURST_SPAN_S] keyed on the session index; the event
    # scatters across the next BURST_SPAN_S seconds via the per-event "ts" salt.
    u_sess = _uniform_session()
    u_ts = _uniform("ts")

    # gem_spend: u_gsflag < GEM_SPEND_PROB -> u_gs * GEM_SPEND_MAX, else 0.0
    u_gsflag = _uniform("gsflag")
    u_gs = _uniform("gs")

    # gem_balance: uniform in [0, GEM_BALANCE_CAP). Salt "gb" — no collision with
    # existing salts (ts, gsflag, gs, os, mw). Matches player_events.
    u_gb = _uniform("gb")

    # offer_shown: u_os < OFFER_SHOWN_PROB -> 1, else 0
    u_os = _uniform("os")

    # match_won: u_mw < MATCH_WON_PROB -> 1, else 0; match_lost = 1 - match_won
    u_mw = _uniform("mw")

    result = (
        events_df.withColumn(
            "event_ts",
            (
                sf.lit(window_start)
                + u_sess * sf.lit(span - BURST_SPAN_S)
                + u_ts * sf.lit(BURST_SPAN_S)
            ).cast("double"),
        )
        .withColumn(
            "gem_spend",
            sf.when(u_gsflag < sf.lit(GEM_SPEND_PROB), u_gs * sf.lit(GEM_SPEND_MAX))
            .otherwise(sf.lit(0.0))
            .cast("double"),
        )
        .withColumn(
            "gem_balance",
            (u_gb * sf.lit(GEM_BALANCE_CAP)).cast("double"),
        )
        .withColumn(
            "offer_shown",
            sf.when(u_os < sf.lit(OFFER_SHOWN_PROB), sf.lit(1)).otherwise(sf.lit(0)),
        )
        .withColumn(
            "match_won",
            sf.when(u_mw < sf.lit(MATCH_WON_PROB), sf.lit(1)).otherwise(sf.lit(0)),
        )
        .withColumn("match_lost", sf.lit(1) - sf.col("match_won"))
        .select(
            "player_id",
            "event_index",
            "event_ts",
            "gem_spend",
            "gem_balance",
            "offer_shown",
            "match_won",
            "match_lost",
        )
    )

    return result
