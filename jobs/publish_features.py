"""Job task: build feature-source tables, register Feature Views, materialize.

New Feature Views framework (feature-engineering-client >= 0.16.0). Writes the
offline and real-time feature SOURCE Delta tables (plain Delta, from players and
the impression log), registers a ColumnSelection Feature per feature column, and
materializes both groups to the Lakebase online store. Both groups are
ColumnSelection, so both use a `TableTrigger` (re-sync on every source commit);
ColumnSelection cannot take a CronSchedule.

The real-time source is backfilled with history here so the batch pipeline is
self-sufficient for training; the streaming job keeps the same table fresh for
serving. No more legacy create_table / publish_table / FeatureFunction.
"""

from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf

from offer_recommender.config import (
    ONLINE_STORE_NAME,
    TABLE_IMPRESSIONS,
    TABLE_OFFLINE_FEATURES,
    TABLE_PLAYERS,
    configure_from_argv,
    get_config,
    get_kafka_bootstrap,
)
from offer_recommender.features.store import (
    OFFLINE_FEATURE_COLUMNS,
    OFFLINE_SERVING_PREFIX,
    STREAMING_SERVING_PREFIX,
    build_lookup_features,
    build_streaming_features,
    create_match_stream,
    ensure_feature_serving,
    ensure_online_store,
    get_client,
    materialize_online,
    write_offline_source,
)
from offer_recommender.logging_config import get_logger
from offer_recommender.schema import (
    EVENT_TIMESTAMP,
)

logger = get_logger(__name__)


def main() -> None:
    """Build feature source tables, register Features and materialize online."""
    spark = SparkSession.builder.getOrCreate()
    argv = sys.argv[1:]
    configure_from_argv(argv)
    # 4th param carries the optional Kafka bootstrap (DAB var.kafka_bootstrap). Set
    # the env var so get_kafka_bootstrap() sees it; a pre-set env still wins (local
    # runs / overrides). Empty -> synthetic stream path.
    if len(argv) >= 4 and argv[3]:
        os.environ.setdefault("RECO_KAFKA_BOOTSTRAP", argv[3])
    uc = get_config()
    fe = get_client()
    logger.info(
        "publish_features starting: catalog=%s schema=%s",
        uc.catalog,
        uc.schema,
    )

    # The streaming feature group is a managed Streaming Feature View over the
    # Kafka login stream; the broker is mandatory. Fail fast BEFORE the idempotent
    # source-table writes so a misconfigured run wastes no Spark work.
    kafka_bootstrap = get_kafka_bootstrap()
    if not kafka_bootstrap:
        raise RuntimeError(
            "RECO_KAFKA_BOOTSTRAP is required: the streaming feature group is now a "
            "managed Streaming Feature View over the Kafka login stream. Deploy with "
            "--var kafka_bootstrap=<broker host:port>."
        )

    logger.info("Writing offline + real-time feature SOURCE tables...")
    impressions = spark.read.table(uc.table(TABLE_IMPRESSIONS))

    # Offline features are slow-moving player state; stamp them at the START of the
    # observation window (the earliest impression) so the point-in-time training
    # join (feature.event_ts <= spine.event_ts) sees them for every spine row.
    # Stamping current_timestamp() future-dates them past the impression log, which
    # silently NULLs every offline feature out of training while serving reads real
    # values -- a train/serve skew nothing errors on.
    floor_ts = impressions.agg(sf.min(EVENT_TIMESTAMP)).first()[0]
    if floor_ts is None:
        raise ValueError(
            f"{uc.table(TABLE_IMPRESSIONS)} is empty; run generate_data first"
        )
    players = spark.read.table(uc.table(TABLE_PLAYERS)).withColumn(
        EVENT_TIMESTAMP, sf.lit(floor_ts).cast("timestamp")
    )
    # Offline group carries the slow-moving numerics AND the player-attribute
    # categoricals (country, segment): serving looks them ALL up by player_id, so
    # they must live in the offline source, not just on the training spine.
    write_offline_source(
        uc, players.select("player_id", EVENT_TIMESTAMP, *OFFLINE_FEATURE_COLUMNS)
    )
    logger.info(
        "Offline source %s written (features: %s), stamped at %s.",
        uc.table(TABLE_OFFLINE_FEATURES),
        list(OFFLINE_FEATURE_COLUMNS),
        floor_ts,
    )

    # The SFV StreamSource must exist before build_streaming_features registers
    # features against it (the broker was already validated at the top of main).
    ingest = create_match_stream(
        fe, uc, kafka_bootstrap, spark=spark, recreate=False, require_exists=True
    )
    logger.info("Kafka login stream active, ingesting -> %s", ingest)

    # Register Features and materialize to the Lakebase online store. The 3rd task
    # parameter (or RECO_FS_RESET) forces a teardown+rebuild of existing
    # materializations (needed after a feature SET change, e.g. adding
    # country/segment); a normal daily run leaves it false so an unchanged group
    # safely skips re-materialization.
    argv = sys.argv[1:]
    reset = (len(argv) >= 3 and argv[2].lower() == "true") or os.environ.get(
        "RECO_FS_RESET", "false"
    ).lower() == "true"
    logger.info(
        "Ensuring Lakebase online store %s (reset=%s)...", ONLINE_STORE_NAME, reset
    )
    ensure_online_store(fe)
    # Only the OFFLINE (slow-moving) features are looked up from the online store,
    # so only they are materialized online. Real-time features are request-time
    # passthrough (the app sends the login's values in the request, ms-fresh) and
    # the offer passthrough features are likewise request-time; both are registered
    # by build_training_set (register_model), not materialized here.
    offline_features = build_lookup_features(fe, uc)
    logger.info(
        "Registered %d offline lookup Features; materializing online...",
        len(offline_features),
    )
    materialize_online(fe, uc, offline_features, OFFLINE_SERVING_PREFIX, reset=reset)
    logger.info(
        "Offline group materialized online (prefix %s).", OFFLINE_SERVING_PREFIX
    )

    streaming_features = build_streaming_features(fe, uc, reset=reset)
    logger.info(
        "Registered %d streaming Features; materializing online...",
        len(streaming_features),
    )
    materialize_online(
        fe,
        uc,
        streaming_features,
        STREAMING_SERVING_PREFIX,
        reset=reset,
        streaming=True,
    )
    logger.info(
        "Streaming group materialized online (prefix %s).", STREAMING_SERVING_PREFIX
    )

    # Feature Serving endpoint: serves the online feature VALUES by player_id over
    # REST so the app drill-down reads them without touching Lakebase/tables.
    endpoint = ensure_feature_serving(fe, uc, reset=reset, spark=spark)
    logger.info("Feature Serving endpoint ready: %s", endpoint)


if __name__ == "__main__":
    main()
