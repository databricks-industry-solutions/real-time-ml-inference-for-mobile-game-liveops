"""Job task: register the trained offer ranker with feature metadata.

Runs on serverless Spark (NOT GPU) because `fe.log_model(training_set=...)`
needs a SparkSession and the FeatureEngineeringClient, neither of which exists
in the GPU training subprocess. Reads the weights the GPU task wrote to the UC
Volume, rebuilds the net, builds the PIT-correct training set, and logs the
pyfunc so serving auto-resolves feature lookups by player_id.
"""

from __future__ import annotations

import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf

from offer_recommender.config import (
    TABLE_IMPRESSIONS,
    TABLE_TRAINING,
    configure_from_argv,
    get_config,
)
from offer_recommender.features.store import build_training_set, get_client
from offer_recommender.logging_config import get_logger
from offer_recommender.model.net import OfferRankerNet
from offer_recommender.preprocessing.pipeline import (
    SPLIT_COLUMN,
    FeatureColumns,
    TransformParams,
)
from offer_recommender.schema import (
    ENTITY_KEY,
    EVENT_TIMESTAMP,
    OFFER_CATEGORICAL_FEATURES,
    OFFER_NUMERICAL_FEATURES,
    REALTIME_NUMERICAL_FEATURES,
    TARGET_PURCHASED,
    TARGET_REVENUE_D7,
)
from offer_recommender.serving.deploy import log_and_register
from offer_recommender.training.core import TrainConfig
from offer_recommender.training.evaluate import HeldoutMetrics, heldout_metrics

logger = get_logger(__name__)


def main() -> None:
    """Rebuild the net from the Volume checkpoint and register it to UC."""
    import torch

    spark = SparkSession.builder.getOrCreate()
    configure_from_argv(sys.argv[1:])
    uc = get_config()

    params = TransformParams.from_json(
        _read_text(spark, f"{uc.volume_path}/transform_params.json")
    )
    net = OfferRankerNet(params, embedding_dim=TrainConfig().embedding_dim)
    state = torch.load(
        f"{uc.volume_path}/model_state.pt", map_location="cpu", weights_only=True
    )
    net.load_state_dict(state)

    # Feature-store-aware registration: the spine carries the entity key,
    # timestamp, targets and the OFFER-side columns (numeric + categorical) that
    # are sent in the request at serving (passthrough, not looked up). The
    # player-side features (offline numerics + country/segment, and the realtime
    # group) are joined by the embedded FeatureSpec at train and serve time, so
    # they are deliberately NOT selected here.
    fe = get_client()
    spine = spark.read.table(uc.table(TABLE_IMPRESSIONS)).select(
        ENTITY_KEY,
        EVENT_TIMESTAMP,
        *OFFER_NUMERICAL_FEATURES,
        *OFFER_CATEGORICAL_FEATURES,
        # Real-time numerics are request-time passthrough (the app sends the
        # login's values at serving), so the training set must source them from the
        # spine -- their point-in-time values recorded on each impression -- exactly
        # like the offer columns. Only the OFFLINE features are online lookups.
        *REALTIME_NUMERICAL_FEATURES,
        # Streaming aggregates (1h/3h/6h/24h rolling windows) are NOT selected from
        # impressions -- generate.py no longer emits them. They are sourced by
        # build_training_set's create_training_set (via build_features ->
        # build_streaming_features, the managed SFV), so the logged FeatureSpec
        # still covers them and they are auto-resolved by player_id at serving.
        TARGET_REVENUE_D7,
        TARGET_PURCHASED,
    )
    training_set = build_training_set(fe, uc, spine)

    # Score the held-out val split (preprocess tagged ~10% of players 'val'; the
    # trainer never saw these rows) for honest generalization metrics. The val rows
    # already carry the SparkML-transformed features + labels in training_set, so
    # scoring re-uses those directly (no re-transform, no feature-store lookup).
    metrics = _score_holdout(spark, uc, net)
    if metrics is not None:
        logger.info("HELDOUT_METRICS=%s", metrics.as_mlflow_metrics())

    registered, version = log_and_register(
        fe, uc, net, params, training_set, metrics=metrics
    )
    logger.info("Registered %s v%s and promoted to @champion", registered, version)


def _score_holdout(spark, uc, net) -> HeldoutMetrics | None:
    """Run the net on the held-out val rows and return metrics (None if empty).

    Reads the transformed val rows straight from training_set (features already
    scaled/indexed by preprocess), so this measures the SAME net serving will run.
    """
    import numpy as np
    import torch

    cols = FeatureColumns.default()
    val = (
        spark.read.table(uc.table(TABLE_TRAINING))
        .where(sf.col(SPLIT_COLUMN) == "val")
        .select(cols.numeric, *cols.index, cols.revenue, cols.purchase)
    ).toPandas()
    if val.empty:
        logger.warning("No held-out val rows found; skipping metrics.")
        return None

    numeric = np.stack(val[cols.numeric].to_numpy()).astype(np.float32)
    categorical = val[list(cols.index)].to_numpy(dtype=np.int64)
    net.eval()
    with torch.no_grad():
        out = net(torch.from_numpy(numeric), torch.from_numpy(categorical))
    return heldout_metrics(
        revenue_pred_dollars=np.expm1(out["revenue_d7"].numpy()),
        purchase_prob=torch.sigmoid(out["purchase_logit"]).numpy(),
        revenue_true_dollars=val[cols.revenue].to_numpy(dtype=np.float64),
        purchase_true=val[cols.purchase].to_numpy(dtype=np.float64),
    )


def _read_text(spark, path: str) -> str:
    """Read a small text file from the UC Volume in a job context."""
    from pyspark.dbutils import DBUtils

    return DBUtils(spark).fs.head(path, 1_000_000)


if __name__ == "__main__":
    main()
