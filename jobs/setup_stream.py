"""Pipeline task: create the Kafka match SFV + append the backfill DIRECTLY.

Runs as a task in ``offer_recommender_pipeline`` (generate_data -> setup_stream ->
preprocess). Two steps, both synchronous and observable in this one task:

1. **Create the Streaming Feature View** via the FS API
   (``create_match_stream(recreate=True)`` with NO ``backfill_table``). This starts
   the managed ingestion pipeline reading the Kafka ``match_events`` topic into
   the ``match_events`` Delta table (``record_source='stream'``).

2. **Append the backfill rows DIRECTLY into ``match_events``** ourselves (a plain
   Spark append, ``record_source='backfill'``), instead of the managed
   ``StreamBackfillSource`` MERGE. The managed MERGE is async + opaque (no queryable
   surface, gated on a ``stream_record_timestamp`` overlap check, can take many
   minutes) -- a direct append lands the rows in the same task, so preprocess (which
   PIT-joins the streaming Features from ``match_events``) always sees the history.
   ``match_events`` is an append-only Delta table; the live SFV stream and this
   append both blind-append with no conflict.

The backfill is built from ``sessions.emit_events_spark`` using the EXACT window
bounds ``generate.py`` used for the impression spine, read from the
``session_window_bounds`` Delta table written by ``generate_data``. This gives 0%
row-level disagreement between the backfill events and the purchase labels (see
``match_backfill.build_backfill_frame``).

NOTE: the direct append feeds TRAINING (preprocess recomputes RollingWindows from
``match_events`` at PIT). It does NOT prepopulate the online store -- managed
streaming materialization computes from the point it is materialized FORWARD off the
live Kafka stream only (materialization backfill is not supported in Public Preview).

Args (spark_python_task parameters): [catalog, schema, kafka_bootstrap, session_end_ts]
"""

from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession

from offer_recommender.config import (
    MATCH_TOPIC,
    TABLE_IMPRESSIONS,
    TABLE_MATCH_INGEST,
    TABLE_PLAYERS,
    TABLE_SESSION_WINDOW_BOUNDS,
    configure_from_argv,
    get_config,
    get_kafka_bootstrap,
    get_session_end_ts,
)
from offer_recommender.data.match_backfill import build_backfill_frame
from offer_recommender.features.store import create_match_stream, get_client
from offer_recommender.logging_config import get_logger

logger = get_logger(__name__)


def _read_window_bounds(
    spark,
    bounds_table: str,
    impressions_table: str,
    pinned_end_ts: float | None = None,
) -> tuple[float, float, int]:
    """Read the exact window bounds persisted by ``generate_data``.

    Reads ``window_start_s``, ``window_end_s``, and ``seed`` from the
    ``session_window_bounds`` Delta table written by ``write_delta``.  These are
    the EXACT integer-second floats that ``generate_frames`` passed to
    ``window_counts_batch`` when computing the purchase labels, so the backfill
    built with these bounds has 0% row-level disagreement with the labels.

    Fallback hierarchy (only reached if ``session_window_bounds`` is absent):

    1. If ``pinned_end_ts`` is provided (from ``RECO_SESSION_END_TS`` / DAB var),
       derive bounds from it -- same formula ``generate_frames`` uses -- so that a
       clean workspace (no prior ``generate_data`` run) does NOT crash and produces
       bounds consistent with the pin.
    2. Otherwise attempt a min/max derivation from the impressions table.  This
       introduces a 15-172 s window_start shift (13-19% row disagreement on
       rolling-window counts) and is kept ONLY as a last-resort compatibility path.
       Raises ``RuntimeError`` if impressions is also absent (clean workspace) so
       the error is actionable rather than an opaque table-not-found crash.

    Args:
        spark: active SparkSession.
        bounds_table: fully-qualified ``session_window_bounds`` table name.
        impressions_table: fully-qualified impressions table (fallback only).
        pinned_end_ts: optional pinned window-end in unix seconds (from
            ``RECO_SESSION_END_TS``); used as the clean-workspace fallback.

    Returns:
        ``(window_start, window_end, seed)`` -- exact unix-second floats and seed.
    """
    from pyspark.sql import functions as sf

    _BACKFILL_EXTENSION_DAYS = 15

    try:
        # One-row metadata table; .first() reads a single row (no full-table collect).
        row = spark.read.table(bounds_table).first()
        if row is None:
            raise ValueError("session_window_bounds is empty")
        window_start = float(row["window_start_s"])
        window_end = float(row["window_end_s"])
        seed = int(row["seed"])
        logger.info(
            "Read persisted window bounds: window_start=%.0f, window_end=%.0f, seed=%d.",
            window_start,
            window_end,
            seed,
        )
        return window_start, window_end, seed
    except Exception as exc:
        logger.warning(
            "session_window_bounds table not found (%s). Attempting fallback.",
            str(exc)[:120],
        )

    if pinned_end_ts is not None:
        # Derive bounds using the same formula as generate_frames so the backfill
        # is consistent with whatever generate_data will produce on this install.
        # span_days=14 matches GenConfig.span_days default; seed default = 17.
        _span_days = 14
        _seed = int(os.environ.get("RECO_SEED", "17"))
        window_end = float(int(pinned_end_ts))
        window_start = window_end - (_span_days + _BACKFILL_EXTENSION_DAYS) * 86400.0
        logger.warning(
            "Clean-workspace fallback via pinned end_ts=%.0f: "
            "window_start=%.0f, window_end=%.0f, seed=%d. "
            "Run generate_data first to persist exact bounds for 0%% disagreement.",
            pinned_end_ts,
            window_start,
            window_end,
            _seed,
        )
        return window_start, window_end, _seed

    # Last resort: derive from impressions min/max (degrades to 13-19% disagreement).
    # Guard against a clean workspace where impressions also does not exist.
    try:
        imp_row = (
            spark.read.table(impressions_table)
            .agg(
                sf.min("event_ts").alias("min_ts"),
                sf.max("event_ts").alias("max_ts"),
            )
            .collect()[0]
        )
    except Exception as imp_exc:
        raise RuntimeError(
            f"Cannot derive window bounds: neither '{bounds_table}' nor "
            f"'{impressions_table}' exists on this workspace. "
            "Run generate_data first (or set the session_end_ts DAB var so "
            "setup_stream can derive bounds independently on a clean workspace)."
        ) from imp_exc

    spine_start_s = float(imp_row["min_ts"].timestamp())
    spine_end_s = float(imp_row["max_ts"].timestamp())
    window_start = spine_start_s - _BACKFILL_EXTENSION_DAYS * 86400.0
    window_end = spine_end_s
    seed = 17
    logger.warning(
        "Fallback window from impressions min/max: spine [%.0f, %.0f], "
        "ext window_start %.0f (-%dd), window_end %.0f, seed=%d. "
        "This shifts the backfill window by 15-172 s relative to the label, "
        "causing ~13-19%% row disagreement on rolling-window counts.",
        spine_start_s,
        spine_end_s,
        window_start,
        _BACKFILL_EXTENSION_DAYS,
        window_end,
        seed,
    )
    return window_start, window_end, seed


def main() -> None:
    """Create the match SFV, then append the backfill directly into match_events.

    Argv layout: [catalog, schema, kafka_bootstrap, session_end_ts (unix secs)].
    """
    spark = SparkSession.builder.getOrCreate()
    argv = sys.argv[1:]
    configure_from_argv(argv)  # consumes [catalog, schema]
    # 3rd param = kafka bootstrap (required) (mirrors preprocess/publish_features arg style).
    if len(argv) >= 3 and argv[2]:
        os.environ.setdefault("RECO_KAFKA_BOOTSTRAP", argv[2])
    # 4th param = pinned session-window end timestamp (unix seconds).  Sets
    # RECO_SESSION_END_TS so get_session_end_ts() returns the same value that
    # generate_data uses, making window bounds order-independent.
    if len(argv) >= 4 and argv[3]:
        os.environ.setdefault("RECO_SESSION_END_TS", argv[3])
    # 5th param = FS reset flag (mirrors preprocess/publish_features). Controls
    # whether we DELETE+recreate the live stream. Default false so a DAILY pipeline
    # run REUSES the existing stream (never interrupting live Kafka ingestion nor
    # re-arming the online-materialization warm-up clock). Only an explicit rebuild
    # (--var fs_reset=true) recreates it.
    reset = (len(argv) >= 5 and argv[4].lower() == "true") or os.environ.get(
        "RECO_FS_RESET", "false"
    ).lower() == "true"
    # 6th param = optional UC Kafka connection name (Event Hubs path). When set,
    # create_match_stream authenticates the SFV via StreamConnectionConfig instead of
    # DirectMtlsConfig. A pre-set env still wins (local overrides).
    if len(argv) >= 6 and argv[5]:
        os.environ.setdefault("RECO_KAFKA_UC_CONNECTION", argv[5])
    uc = get_config()
    kafka_bootstrap = get_kafka_bootstrap()
    if not kafka_bootstrap:
        raise RuntimeError(
            "RECO_KAFKA_BOOTSTRAP required: setup_stream creates the Kafka SFV "
            "stream. Deploy with --var kafka_bootstrap=<broker host:port>."
        )
    fe = get_client()

    # Pinned end_ts for the clean-workspace fallback in _read_window_bounds.
    _pinned_end_ts = get_session_end_ts().timestamp()

    # Read players table to get player_id + segment for the session model.
    players_df = spark.read.table(uc.table(TABLE_PLAYERS)).select(
        "player_id", "segment"
    )
    n_players = players_df.count()

    # Read exact window bounds from the metadata table written by generate_data.
    # The seed is also read from the metadata so it always matches the label run.
    window_start, window_end, seed = _read_window_bounds(
        spark,
        uc.table(TABLE_SESSION_WINDOW_BOUNDS),
        uc.table(TABLE_IMPRESSIONS),
        pinned_end_ts=_pinned_end_ts,
    )

    # Step 1: ensure the SFV exists. recreate=reset: a daily run (reset=false)
    # REUSES the live stream as-is (create_match_stream ensures-exists, never
    # deletes) so live Kafka ingestion and the online-materialization warm-up clock
    # are untouched; only an explicit rebuild (fs_reset=true) DELETE+recreates it.
    # NO backfill_source either way -- we append the backfill manually below.
    ingest = create_match_stream(
        fe,
        uc,
        kafka_bootstrap,
        spark=spark,
        backfill_table=None,
        recreate=reset,
    )
    logger.info(
        "Match stream ensured (recreate=%s, no managed backfill) -> %s", reset, ingest
    )

    # Step 2: append the backfill DIRECTLY into the ingestion table ourselves.
    # A plain blind append (match_events is appendOnly Delta; the live SFV stream
    # also blind-appends, no conflict). Synchronous + observable: the rows are in
    # match_events when this task returns, so preprocess's PIT recompute sees them.
    #
    # Idempotent guard: appendOnly forbids DELETE, and this task runs every pipeline
    # run, so we only append when NO backfill rows are already present. The frame is
    # deterministic given the pinned window bounds + seed, so an existing backfill
    # (from a prior run, if create_stream did not reset the table) is identical --
    # skipping it avoids duplicate-count skew in the PIT recompute.
    existing_backfill = (
        spark.read.table(uc.table(TABLE_MATCH_INGEST))
        .where("record_source = 'backfill'")
        .count()
    )
    if existing_backfill > 0:
        logger.info(
            "Backfill already present in %s (%d rows); skipping append (idempotent).",
            ingest,
            existing_backfill,
        )
        return

    frame = build_backfill_frame(
        spark,
        players_df=players_df,
        window_start=window_start,
        window_end=window_end,
        topic=MATCH_TOPIC,
        seed=seed,
    )
    frame.write.format("delta").mode("append").saveAsTable(uc.table(TABLE_MATCH_INGEST))
    appended = (
        spark.read.table(uc.table(TABLE_MATCH_INGEST))
        .where("record_source = 'backfill'")
        .count()
    )
    logger.info(
        "Appended backfill directly into %s: %d backfill rows now present (%d players).",
        ingest,
        appended,
        n_players,
    )


if __name__ == "__main__":
    main()
