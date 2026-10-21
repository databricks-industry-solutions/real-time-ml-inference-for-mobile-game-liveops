"""Job task: fit the SparkML preprocessing pipeline and materialise features.

Fits the pipeline over the impression log (this is the at-scale fit that
justifies serverless Spark), writes the transformed training table (with the
numeric vector expanded to an array for Ray Data), and persists the fitted
TransformParams JSON to the UC Volume so training can bundle it into the model.
"""

from __future__ import annotations

import os
import sys
import time

from pyspark.sql import SparkSession

from offer_recommender.config import (
    TABLE_IMPRESSIONS,
    TABLE_TRAINING,
    configure_from_argv,
    get_config,
    get_kafka_bootstrap,
)
from offer_recommender.features.store import (
    build_streaming_features,
    build_training_set_streaming,
    create_match_stream,
    get_client,
)
from offer_recommender.logging_config import get_logger
from offer_recommender.preprocessing.pipeline import (
    add_feature_arrays,
    build_pipeline,
    extract_params,
)
from offer_recommender.schema import WINDOWS

logger = get_logger(__name__)


def main() -> None:
    """Fit preprocessing, write the transformed table and the params artifact."""
    spark = SparkSession.builder.getOrCreate()
    argv = sys.argv[1:]
    configure_from_argv(argv)
    # Param layout mirrors publish_features EXACTLY: [catalog, schema, fs_reset,
    # kafka_bootstrap]. 3rd param = FS reset flag; 4th = Kafka bootstrap
    # (DAB var.kafka_bootstrap). Set the env var so get_kafka_bootstrap() sees it; a
    # pre-set env still wins (local runs / overrides).
    if len(argv) >= 4 and argv[3]:
        os.environ.setdefault("RECO_KAFKA_BOOTSTRAP", argv[3])
    # On a reset run create_match_stream DELETE+recreates the login stream, orphaning
    # the previously-registered streaming Features (their StreamSource full_name
    # dangles -> _from_sdk ValueError on the next get_feature). Derive reset the SAME
    # way publish_features does so build_streaming_features(reset=True) clears them.
    reset = (len(argv) >= 3 and argv[2].lower() == "true") or os.environ.get(
        "RECO_FS_RESET", "false"
    ).lower() == "true"
    uc = get_config()

    # The 5 streaming rolling-window features are NOT on the impression log any
    # more; they are a managed Streaming Feature View (AggregationFunction +
    # RollingWindow over the Kafka login stream). preprocess PIT-joins them into
    # the training table so SparkML fits on their REAL point-in-time values
    # (recomputed from the stream ingestion history) -- training and serving read
    # the same source, no impression-backfill, no train/serve skew. The broker is
    # therefore mandatory; fail fast BEFORE any Spark work on a misconfigured run.
    kafka_bootstrap = get_kafka_bootstrap()
    if not kafka_bootstrap:
        raise RuntimeError(
            "RECO_KAFKA_BOOTSTRAP is required: the streaming feature group is now a "
            "managed Streaming Feature View over the Kafka login stream, and "
            "preprocess PIT-joins it into the training table. Deploy with "
            "--var kafka_bootstrap=<broker host:port>."
        )

    # Disable MLflow autologging before the SparkML fit. A preprocessing job does
    # not need its fit auto-tracked (register_model logs the model explicitly), and
    # autolog wrapping .fit() (MLWorkloadsInstrumentation) is pure overhead here.
    import mlflow

    mlflow.autolog(disable=True)
    mlflow.pyspark.ml.autolog(disable=True)

    impressions = spark.read.table(uc.table(TABLE_IMPRESSIONS))
    n_input = impressions.count()
    logger.info(
        "Preprocess input: %s (%d rows). Enriching with streaming features...",
        uc.table(TABLE_IMPRESSIONS),
        n_input,
    )

    fe = get_client()
    # Consume the login stream created by the upstream setup_stream pipeline task
    # (generate_data -> setup_stream -> preprocess). recreate=False = ensure-exists +
    # reuse AS-IS (never delete/recreate); raises if the stream is absent. setup_stream
    # appended the backfill DIRECTLY into match_events (a synchronous Spark append, not
    # the async managed MERGE), so the PIT join below finds real historical events.
    ingest = create_match_stream(
        fe, uc, kafka_bootstrap, spark=spark, recreate=False, require_exists=True
    )
    logger.info("Consuming pre-warmed login stream -> %s", ingest)

    # Register the streaming Features (idempotent) so create_training_set can PIT-
    # join them onto the spine by player_id + event_ts.
    build_streaming_features(fe, uc, reset=reset)
    impressions = build_training_set_streaming(fe, uc, impressions).load_df()
    # A backward RollingWindow with no ingested events in-window PIT-joins as NULL.
    # On a freshly-created stream this is nearly every impression (the spine spans a
    # recent multi-day window; the stream has ~no history yet), and VectorAssembler's
    # default handleInvalid="error" would raise on the NULLs (log1p(NULL)->NULL). Fill
    # all 6 SFV rolling-window features with 0.0: an empty backward window genuinely
    # means zero (0 logins/gems/sessions/distinct offers, avg over no rows -> 0).
    # match_events_15m is filled even though not a model input (keeps the training_set
    # column non-NULL). This is train/serve-parity-safe -- the online store returns the
    # same 0.0 for a player with an empty window -- and once the stream has real history
    # both sides read the same non-zero values. NON-streaming NULLs are left to surface
    # as real bugs.
    impressions = impressions.fillna(0.0, subset=list(WINDOWS))
    logger.info(
        "Enriched impressions with streaming SFV features. Fitting SparkML pipeline..."
    )

    # The fitted PipelineModel is tiny (SQLTransformer statement + ~60 StringIndexer
    # labels + 13 mean/std pairs), but on serverless Spark Connect the fitted model
    # is held in a SESSION-scoped ML cache whose per-model size check is
    # non-deterministic and warm-pool-dependent: identical code measured 256-271 MiB
    # across runs and intermittently tripped CONNECT_ML.MODEL_SIZE_OVERFLOW (256 MiB
    # cap). The cap is a per-workspace SAFEr flag (maxModelSize), NOT settable from
    # here, and this workspace is serverless-only (no classic escape). The one
    # in-code mitigation Databricks documents for the ML-cache pressure is to DROP
    # the Python reference to the fitted model as soon as it is no longer needed,
    # so it is freed from the session cache and does not accumulate across the
    # warm-pool session reused by successive runs -- hence the explicit
    # `del model; gc.collect()` after the transform+write below. autolog is disabled
    # above for the same reason (it would retain the model in the cache too).
    fit_start = time.perf_counter()
    model = build_pipeline().fit(impressions)
    params = extract_params(model)
    logger.info(
        "Fit complete in %.1fs: %d categorical vocabs (sizes %s), "
        "%d numeric mean/std pairs.",
        time.perf_counter() - fit_start,
        len(params.vocabs),
        {k: len(v) for k, v in params.vocabs.items()},
        len(params.mean),
    )

    # overwriteSchema=true so a feature/column change to add_feature_arrays fully
    # replaces the table schema (plain overwrite keeps the old columns). Training
    # reads this table with ray.data.read_databricks_tables (a SELECT on the SQL
    # warehouse), so the intermediate VectorUDT columns are already dropped in
    # add_feature_arrays and no Delta-protocol gymnastics are needed here.
    write_start = time.perf_counter()
    transformed = add_feature_arrays(model.transform(impressions))
    transformed.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        uc.table(TABLE_TRAINING)
    )
    # Also stage the training_set as PARQUET on the UC Volume: the GPU training node
    # has NO SparkSession and reads its data via `ray.data.read_parquet` off the
    # FUSE-mounted Volume (no SQL warehouse / host / token). The Delta table above is
    # kept because register_model reads it for the held-out `val` split. All columns
    # are plain types here (add_feature_arrays dropped the VectorUDT cols), so parquet
    # is clean for Ray.
    parquet_path = f"{uc.volume_path}/training_set_parquet"
    transformed.write.mode("overwrite").parquet(parquet_path)
    logger.info("Staged training parquet for Ray at %s", parquet_path)
    logger.info(
        "Wrote %s in %.1fs (transformed %d input rows).",
        uc.table(TABLE_TRAINING),
        time.perf_counter() - write_start,
        n_input,
    )

    # Free the fitted model from the session-scoped Spark Connect ML cache now that
    # the transform is written and params are extracted (see the fit comment above).
    # Both refs must go, then gc, so a reused warm-pool session starts clean for the
    # next run instead of accumulating models toward the cap.
    import gc

    del model, transformed
    gc.collect()

    # Persist the fitted params to the UC Volume for the training job to bundle.
    params_path = f"{uc.volume_path}/transform_params.json"
    dbutils = _get_dbutils(spark)
    dbutils.fs.put(params_path, params.to_json(), overwrite=True)
    logger.info("Wrote transform params -> %s. Preprocess complete.", params_path)


def _get_dbutils(spark):
    """Return dbutils in a Databricks job context."""
    from pyspark.dbutils import DBUtils

    return DBUtils(spark)


if __name__ == "__main__":
    main()
