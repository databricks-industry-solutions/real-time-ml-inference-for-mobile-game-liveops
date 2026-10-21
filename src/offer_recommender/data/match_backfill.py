"""Historical match-event backfill for the managed Streaming Feature View.

The 5 rolling-window streaming features (`gems_spent_15m`, `avg_gem_balance_15m`, ...) are
computed by a managed Streaming Feature View (SFV) over the Kafka `match_events`
topic. A managed stream reads from the LATEST offset, so it ignores any messages
already on the topic and starts with an empty backward window -- training would
have to wait ~15 min for the RollingWindow history to fill.

To give TRAINING real historical values, `jobs/setup_stream.py` builds this frame of
backdated match events and appends it DIRECTLY into the SFV ingestion table
`match_events` (`record_source='backfill'`), then preprocess's PIT recompute reads
`value.event_ts` (a TIMESTAMP) from that history and yields non-zero rolling-window
aggregates at every impression's timestamp. We append directly rather than via the
managed `StreamBackfillSource` MERGE because that MERGE is async + opaque (gated on a
`stream_record_timestamp` overlap check, no queryable progress surface, can take many
minutes); a direct blind append lands synchronously and is observable in the same
task. (`match_events` is an append-only Delta table; the live SFV stream and this
append both blind-append with no conflict.)

**Row-level correspondence (Task 7)**: the backfill is built from
`sessions.emit_events_spark`, the same deterministic session model that
`generate.py` uses when computing the purchase labels. Because both sides hash
via ``md5_uniform_str(f"{player_id}:{i}:{salt}:{seed}")``, the backfill events
are bit-for-bit identical to the ones the label windowed over: label-feature
correspondence is exact, not approximate. Event density is intrinsic to the session
model (~80 events/player before segment scaling).

**Window-bounds alignment**: row-level label-feature correspondence only holds when
``build_backfill_frame`` receives the EXACT ``window_start`` / ``window_end``
that ``generate.py`` passed to ``window_counts_batch`` when computing the labels.
``setup_stream.py`` reads these from the ``session_window_bounds`` Delta table
written by ``generate_data`` (``config.TABLE_SESSION_WINDOW_BOUNDS``). Deriving
them via ``min/max(impressions.event_ts)`` shifts every timestamp by 15-172 s,
causing 13-19% of rows to disagree on rolling-window counts (measured).  When
``setup_stream`` reads the persisted bounds, disagreement is 0%.

Column-for-column, the frame matches the live SFV match-event ingestion table
`match_events` (the append fails on any schema mismatch):

    key                     string
    value                   struct<player_id:string, event_ts:timestamp,
                              event_type:string, gem_spend:double,
                              gem_balance:double, offer_shown:double,
                              match_won:double, match_lost:double>
    record_source           string   (= "backfill")
    stream_record_timestamp timestamp
    kafka_topic             string
    kafka_partition         int
    kafka_offset            bigint

`stream_record_timestamp` is spread `end .. end + 2h` (a leftover of the abandoned
managed-MERGE overlap heuristic; harmless here -- the RollingWindow recompute reads
`value.event_ts`, not this column).

The `value` field distributions mirror the live producer
(jobs/produce_events.py) so train/serve parity
holds. The frame is DETERMINISTIC given ``seed`` + ``end_ts``: all per-row random
columns use ``md5_uniform_str`` keyed on ``player_id``, ``event_index``, a column
salt, and ``seed`` -- no ``sf.rand`` (which is partition-order-dependent).

Used by jobs/setup_stream.py, which appends the frame into `match_events` before
preprocess's streaming PIT join.
"""

from __future__ import annotations

import time

from offer_recommender.data.hashing import SPARK_MD5_UNIFORM_SQL
from offer_recommender.data.sessions import _segment_max_events
from offer_recommender.logging_config import get_logger
from offer_recommender.schema import (
    ENTITY_KEY,
    EVENT_TIMESTAMP,
)

logger = get_logger(__name__)

# stream_record_timestamp spread (seconds): end .. end + 2h, so the >=1h MERGE
# overlap with the just-created live stream clears immediately.
_STREAM_TS_SPREAD_S = 2 * 3600

# Salt for the stream_record_timestamp draw (NOT a session signal): an independent
# per-row md5 draw keyed on f"{player_id}:{event_index}:{salt}:{seed}" so the frame is
# FULLY DETERMINISTIC across Spark runs (no sf.rand, which is partition-order-dependent).
# All match-event VALUE signals come straight from emit_events_spark (the session
# model), so the SFV aggregates at training match window_counts_batch for the label.
_SALT_STREAM_TS = "bfst"

# kafka_offset stride: must exceed the busiest player's event_count so
# player_index * stride + event_index never collides across players. Derived from
# the session constants (next power of ten above the max event count) so it tracks
# any future SESSIONS_PER_SEGMENT / EVENTS_PER_SESSION change automatically.
_OFFSET_STRIDE = 10 ** len(str(_segment_max_events()))


def build_backfill_frame(
    spark,
    players_df,
    window_start: float,
    window_end: float,
    topic: str,
    end_ts: float | None = None,
    seed: int = 17,
):
    """Build the backdated match-event frame matching the SFV ingestion schema.

    Built from ``sessions.emit_events_spark`` so every emitted event is
    bit-for-bit identical to the session timeline used by ``generate.py`` when
    computing the purchase labels. Row-level label-feature correspondence is
    therefore exact: the backfill events ARE the session events the label counted.

    ``players_df`` must have columns ``player_id`` (string) and ``segment``
    (string). Event density is intrinsic to the session model (~80 events/player
    before segment scaling; bounded ``[MIN_EVENTS, MAX_EVENTS]``).

    All match-outcome signals (gem_spend, gem_balance, offer_shown, match_won,
    match_lost) come directly from ``emit_events_spark`` so the SFV-computed rolling
    aggregates at training match ``window_counts_batch`` bit-for-bit.
    ``kafka_offset`` is derived from player_index * _OFFSET_STRIDE + event_index:
    unique, stable and deterministic, no global sort required.

    Args:
        spark: active SparkSession.
        players_df: DataFrame with at least ``player_id`` (string) and
            ``segment`` (string) columns.
        window_start: observation-window start as unix seconds; must match the
            ``ext_window_start_s`` used in ``generate.py`` for the same run.
        window_end: observation-window end as unix seconds; must match
            ``ext_window_end_s`` (= spine end) in ``generate.py``.
        topic: Kafka topic name recorded in ``kafka_topic``.
        end_ts: upper bound for the ``stream_record_timestamp`` spread (unix
            seconds); ``None`` -> current time (production). The REAL backdated
            spread lives in ``value.event_ts``; this is only the MERGE overlap
            engineering field.
        seed: base integer seed; forwarded to ``emit_events_spark`` so every
            signal hash key matches the numpy ``player_events`` call exactly.

    Returns:
        A Spark DataFrame with exactly the SFV ingestion-table schema.
    """
    from pyspark.sql import functions as sf

    from offer_recommender.data.sessions import emit_events_spark

    end = float(end_ts) if end_ts is not None else time.time()

    # Step 1: explode each player into (player_id, event_index, signals) via the
    # shared session model. event_ts comes out as a unix-seconds double; we cast
    # to TIMESTAMP below to match the ingestion schema (T15 gotcha).
    events = emit_events_spark(
        spark, players_df, window_start=window_start, window_end=window_end, seed=seed
    )

    # Step 2: build the deterministic stream_record_timestamp draw (all match
    # signals come from emit_events_spark; only this MERGE-overlap field is drawn
    # here).  Uses md5_uniform keyed on
    # f"{player_id}:{event_index}:{salt}:{seed}" — byte-identical to the numpy
    # key pattern and fully reproducible across Spark runs (no sf.rand).
    seed_str = str(seed)

    def _md5_col(salt: str):
        """Return a [0,1) Spark Column via md5 on the canonical per-row key."""
        key_expr = (
            f"concat_ws(':', player_id, CAST(event_index AS STRING),"
            f" '{salt}', '{seed_str}')"
        )
        return sf.expr(SPARK_MD5_UNIFORM_SQL.format(col=key_expr)).cast("double")

    # Match-outcome signals come DIRECTLY from emit_events_spark, so the SFV-computed
    # rolling aggregates at training match window_counts_batch exactly.
    # event_ts: double (unix seconds) -> TIMESTAMP (T15 requirement).
    event_ts_col = sf.to_timestamp(sf.from_unixtime(sf.col("event_ts").cast("double")))
    gem_spend_col = sf.col("gem_spend").cast("double")
    gem_balance_col = sf.col("gem_balance").cast("double")
    offer_shown_col = sf.col("offer_shown").cast("double")
    match_won_col = sf.col("match_won").cast("double")
    match_lost_col = sf.col("match_lost").cast("double")

    # Assemble the value struct in the match-event ingestion-schema order (mirrors
    # store._match_payload_json_schema: match-outcome signals only; request-time
    # passthrough rides the separate logins topic, not here).
    value_fields = [
        sf.col("player_id").alias(ENTITY_KEY),
        event_ts_col.alias(EVENT_TIMESTAMP),
        sf.lit("match").alias("event_type"),
        gem_spend_col.alias("gem_spend"),
        gem_balance_col.alias("gem_balance"),
        offer_shown_col.alias("offer_shown"),
        match_won_col.alias("match_won"),
        match_lost_col.alias("match_lost"),
    ]

    # stream_record_timestamp: spreads end .. end+2h so the >=1h overlap with the
    # just-created live stream fires immediately (MERGE overlap engineering).
    # Uses md5_uniform for full determinism (no sf.rand).
    stream_record_timestamp = sf.timestamp_seconds(
        sf.lit(end) + _md5_col(_SALT_STREAM_TS) * sf.lit(float(_STREAM_TS_SPREAD_S))
    )

    # kafka_offset: collision-free deterministic bigint, no global sort.
    # player_id is always "player_NNNNNN" (6-digit zero-padded index); event_index
    # < max event_count (burstiest player's sessions * EVENTS_PER_SESSION, e.g.
    # round(20 * 3) * 100 = 6000). RIGHT(player_id, 6) extracts the numeric index; multiplying
    # by _OFFSET_STRIDE (> max event_count) and adding event_index gives a unique
    # offset per row without a partition-spanning sort (a studio-scale hazard).
    player_index_col = sf.expr("CAST(RIGHT(player_id, 6) AS LONG)")

    return events.withColumn(
        "_kafka_offset",
        (
            player_index_col * sf.lit(_OFFSET_STRIDE).cast("long")
            + sf.col("event_index").cast("long")
        ).cast("long"),
    ).select(
        sf.col("player_id").alias("key"),
        sf.struct(*value_fields).alias("value"),
        sf.lit("backfill").alias("record_source"),
        stream_record_timestamp.alias("stream_record_timestamp"),
        sf.lit(topic).alias("kafka_topic"),
        sf.lit(0).cast("int").alias("kafka_partition"),
        sf.col("_kafka_offset").alias("kafka_offset"),
    )
