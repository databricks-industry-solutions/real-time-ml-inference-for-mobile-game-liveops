"""The critical correctness guard: SparkML transform == in-model numpy twin.

If these fail, the model served (which uses the numpy/torch twin, no Spark)
produces different features than the model trained on the SparkML output. That
is training/serving skew, and it must never merge silently.

Also guards the embedding out-of-vocab path: an unseen category must route to
the StringIndexer keep index (len(vocab)) and must be a valid embedding index
(embeddings are sized len(vocab)+1), so serving on a never-before-seen offer or
country does not crash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from offer_recommender.data.generate import GenConfig, generate_frames
from offer_recommender.model.net import OfferRankerNet
from offer_recommender.model.transform import transform
from offer_recommender.preprocessing.pipeline import (
    NUMERIC_ARRAY_COL,
    add_feature_arrays,
    build_pipeline,
    extract_params,
    index_col,
)
from offer_recommender.schema import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    STREAMING_NUMERICAL_FEATURES,
)


def _training_frame() -> pd.DataFrame:
    """Return a wide impression frame (players joined to offers) for fitting.

    Streaming feature columns (owned by the SFV at train/serve time) are absent
    from generate_frames; inject them at 0.0 so the SparkML pipeline and numpy
    transform twin both see all NUMERICAL_FEATURES.
    """
    _, _, impressions = generate_frames(
        GenConfig(n_players=400, n_offers=5, impressions_per_player=6, seed=3)
    )
    for col in STREAMING_NUMERICAL_FEATURES:
        impressions[col] = 0.0
    return impressions


def test_numeric_and_categorical_parity(spark):
    """SparkML output equals the numpy twin element-wise on a holdout."""
    pdf = _training_frame()
    sdf = spark.createDataFrame(pdf)

    model = build_pipeline().fit(sdf)
    params = extract_params(model)

    transformed = add_feature_arrays(model.transform(sdf))
    # Bring Spark results back in a stable order for row-wise comparison.
    keep = [NUMERIC_ARRAY_COL] + [index_col(f) for f in CATEGORICAL_FEATURES]
    spark_pd = transformed.select("player_id", "offer_id", *keep).toPandas()

    # Recreate the SAME row order in pandas by joining on the keys.
    merged = pdf.merge(
        spark_pd, on=["player_id", "offer_id"], how="inner", suffixes=("", "_spark")
    )
    # A (player, offer) pair can repeat across impressions; dedupe to unique
    # feature rows so the join is unambiguous for comparison.
    merged = merged.drop_duplicates(subset=["player_id", "offer_id"]).reset_index(
        drop=True
    )

    numeric_twin, categorical_twin = transform(merged, params)

    spark_numeric = np.stack(merged[NUMERIC_ARRAY_COL].to_numpy())
    assert spark_numeric.shape == numeric_twin.shape
    # StandardScaler and numpy both use sample std; agree to float tolerance.
    np.testing.assert_allclose(spark_numeric, numeric_twin, rtol=1e-5, atol=1e-5)
    # The scaled numeric block must have one column per declared numeric feature.
    assert numeric_twin.shape[1] == len(NUMERICAL_FEATURES)

    for i, feature in enumerate(CATEGORICAL_FEATURES):
        spark_idx = merged[index_col(feature)].to_numpy(dtype=np.int64)
        np.testing.assert_array_equal(spark_idx, categorical_twin[:, i])


def test_unseen_category_routes_to_oov_and_does_not_crash(spark):
    """An unseen categorical value maps to the keep index and scores cleanly."""
    pdf = _training_frame()
    model = build_pipeline().fit(spark.createDataFrame(pdf))
    params = extract_params(model)

    # A row with categories that never appeared in training.
    unseen = pdf.iloc[[0]].copy()
    unseen["country"] = "ATLANTIS"
    unseen["offer_id"] = "offer_999"
    unseen["offer_type"] = "mythical"
    unseen["price_tier"] = "tier_9"
    unseen["segment"] = "kraken"

    _, categorical_twin = transform(unseen, params)
    sizes = params.embedding_sizes()
    for i, feature in enumerate(CATEGORICAL_FEATURES):
        oov_index = len(params.vocabs[feature])
        assert categorical_twin[0, i] == oov_index
        # The OOV index must be a valid embedding row (sized len+1).
        assert oov_index < sizes[feature]

    numeric_twin, _ = transform(unseen, params)
    net = OfferRankerNet(params).eval()
    with torch.no_grad():
        out = net(
            torch.tensor(numeric_twin),
            torch.tensor(categorical_twin),
        )
    # No index error, finite outputs, revenue non-negative.
    assert torch.isfinite(out["revenue_d7"]).all()
    assert (out["revenue_d7"] >= 0).all()


def test_transform_params_roundtrip():
    """TransformParams survives JSON round-trip (MLflow artifact contract)."""
    from offer_recommender.preprocessing.pipeline import TransformParams

    params = TransformParams(
        vocabs={"country": ["US", "FI"]},
        mean=[1.0, 2.0],
        std=[0.5, 1.5],
        numerical_features=list(NUMERICAL_FEATURES),
        categorical_features=["country"],
        log1p_features=["lifetime_spend"],
    )
    restored = TransformParams.from_json(params.to_json())
    assert restored == params
    assert restored.embedding_sizes() == {"country": 3}
