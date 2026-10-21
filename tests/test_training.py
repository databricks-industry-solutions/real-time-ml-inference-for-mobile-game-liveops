"""Training + ranking-quality guards (local, no Ray/Spark cluster needed).

Fits the SparkML pipeline once (local Spark), trains the net with the shared
core loop, and asserts the recommender captures a meaningful fraction of the
oracle-over-random revenue lift. This is the "does it actually learn" guard;
it pins the demo's headline metric so a regression in the model or the data
generator is caught.
"""

from __future__ import annotations

import numpy as np

from offer_recommender.data.generate import GenConfig, generate_frames
from offer_recommender.model.transform import transform
from offer_recommender.preprocessing.pipeline import (
    build_pipeline,
    extract_params,
)
from offer_recommender.schema import (
    REALTIME_NUMERICAL_FEATURES,
    STREAMING_NUMERICAL_FEATURES,
    TARGET_PURCHASED,
    TARGET_REVENUE_D7,
)
from offer_recommender.training.core import TrainConfig, make_dataset, train_model
from offer_recommender.training.evaluate import EvalConfig, evaluate_ranking


def test_model_beats_random_and_captures_oracle_lift(spark):
    """Trained ranker captures a solid share of oracle-over-random lift."""
    players, offers, impressions = generate_frames(
        GenConfig(n_players=2000, n_offers=30, impressions_per_player=8, seed=5)
    )
    # Streaming columns are absent from generate_frames (owned by SFV at train/serve
    # time). Inject at 0.0 so the SparkML pipeline and numpy twin both see all features.
    for col in STREAMING_NUMERICAL_FEATURES:
        impressions[col] = 0.0
    params = extract_params(build_pipeline().fit(spark.createDataFrame(impressions)))

    numeric, categorical = transform(impressions, params)
    dataset = make_dataset(
        numeric,
        categorical,
        impressions[TARGET_REVENUE_D7].to_numpy(np.float32),
        impressions[TARGET_PURCHASED].to_numpy(np.float32),
    )
    net = train_model(params, dataset, TrainConfig(epochs=30, seed=0))

    realtime_defaults = {
        f: float(impressions[f].median()) for f in REALTIME_NUMERICAL_FEATURES
    }
    result = evaluate_ranking(
        net,
        params,
        players,
        offers,
        EvalConfig(realtime_defaults=realtime_defaults, n_players=300),
    )

    # Model must clearly beat a random offer pick...
    assert result.model_revenue > result.random_revenue * 1.3
    # ...and capture a meaningful share of the achievable oracle lift.
    assert result.oracle_lift_captured > 0.4


def test_rank_offers_is_nonnegative_dollars(spark):
    """rank_offers returns non-negative dollar revenue (expm1 of log head)."""
    _, _, impressions = generate_frames(
        GenConfig(n_players=200, n_offers=15, impressions_per_player=4, seed=1)
    )
    for col in STREAMING_NUMERICAL_FEATURES:
        impressions[col] = 0.0
    params = extract_params(build_pipeline().fit(spark.createDataFrame(impressions)))
    numeric, categorical = transform(impressions, params)
    dataset = make_dataset(
        numeric,
        categorical,
        impressions[TARGET_REVENUE_D7].to_numpy(np.float32),
        impressions[TARGET_PURCHASED].to_numpy(np.float32),
    )
    net = train_model(params, dataset, TrainConfig(epochs=3, seed=0))
    from offer_recommender.training.core import rank_offers

    predicted = rank_offers(net, numeric, categorical)
    assert (predicted >= 0).all()
    assert np.isfinite(predicted).all()


def test_heldout_metrics_perfect_and_directional():
    """heldout_metrics: zero error on perfect predictions; sane on imperfect."""
    from offer_recommender.training.evaluate import heldout_metrics

    rev_true = np.array([0.0, 5.0, 20.0, 100.0])
    pur_true = np.array([0.0, 1.0, 1.0, 1.0])

    # Perfect predictions -> zero regression error, AUC 1, tiny logloss.
    perfect = heldout_metrics(
        revenue_pred_dollars=rev_true,
        purchase_prob=np.array([0.01, 0.99, 0.99, 0.99]),
        revenue_true_dollars=rev_true,
        purchase_true=pur_true,
    )
    assert perfect.n_rows == 4
    assert perfect.revenue_rmse_dollars == 0.0
    assert perfect.revenue_rmse_log == 0.0
    assert perfect.purchase_auc == 1.0
    assert perfect.purchase_logloss < 0.05

    # Worse revenue predictions -> strictly larger RMSE than perfect.
    worse = heldout_metrics(
        revenue_pred_dollars=rev_true + 10.0,
        purchase_prob=np.array([0.5, 0.5, 0.5, 0.5]),
        revenue_true_dollars=rev_true,
        purchase_true=pur_true,
    )
    assert worse.revenue_rmse_dollars > perfect.revenue_rmse_dollars
    assert abs(worse.purchase_auc - 0.5) < 1e-9  # constant score -> no-skill


def test_heldout_auc_single_class_is_half():
    """AUC falls back to 0.5 when the held-out labels are one class only."""
    from offer_recommender.training.evaluate import heldout_metrics

    m = heldout_metrics(
        revenue_pred_dollars=np.array([1.0, 2.0]),
        purchase_prob=np.array([0.3, 0.7]),
        revenue_true_dollars=np.array([1.0, 2.0]),
        purchase_true=np.array([1.0, 1.0]),
    )
    assert m.purchase_auc == 0.5


def test_heldout_classification_metrics_perfect_and_recall_weighted():
    """New threshold metrics: 1.0 on perfect preds; F2 > F1 when recall > precision."""
    from offer_recommender.training.evaluate import heldout_metrics

    rev = np.array([1.0, 2.0, 3.0, 4.0])
    # Perfect: probs on the correct side of 0.5 for every label.
    perfect = heldout_metrics(
        revenue_pred_dollars=rev,
        purchase_prob=np.array([0.1, 0.9, 0.1, 0.9]),
        revenue_true_dollars=rev,
        purchase_true=np.array([0.0, 1.0, 0.0, 1.0]),
    )
    for v in (
        perfect.purchase_accuracy,
        perfect.purchase_precision,
        perfect.purchase_recall,
        perfect.purchase_f1,
        perfect.purchase_f2,
    ):
        assert v == 1.0

    # Recall 1.0, precision 0.5 (one false positive): F2 weights recall -> F2 > F1.
    # preds all-positive vs labels [0,1,1,1]: tp=3, fp=1, fn=0.
    skewed = heldout_metrics(
        revenue_pred_dollars=rev,
        purchase_prob=np.array([0.9, 0.9, 0.9, 0.9]),
        revenue_true_dollars=rev,
        purchase_true=np.array([0.0, 1.0, 1.0, 1.0]),
    )
    assert skewed.purchase_recall == 1.0
    assert abs(skewed.purchase_precision - 0.75) < 1e-9
    assert skewed.purchase_f2 > skewed.purchase_f1


def test_split_column_is_deterministic_and_player_level(spark):
    """add_feature_arrays writes a stable player-level train/val split."""
    from offer_recommender.preprocessing.pipeline import (
        SPLIT_COLUMN,
        add_feature_arrays,
        build_pipeline,
    )
    from offer_recommender.schema import ENTITY_KEY

    _, _, impressions = generate_frames(
        GenConfig(n_players=300, n_offers=20, impressions_per_player=6, seed=7)
    )
    for col in STREAMING_NUMERICAL_FEATURES:
        impressions[col] = 0.0
    sdf = spark.createDataFrame(impressions)
    model = build_pipeline().fit(sdf)
    out = (
        add_feature_arrays(model.transform(sdf))
        .select(ENTITY_KEY, SPLIT_COLUMN)
        .toPandas()
    )

    # Only the two expected labels, and both present (300 players -> ~10% val).
    assert set(out[SPLIT_COLUMN].unique()) <= {"train", "val"}
    assert out[SPLIT_COLUMN].nunique() == 2
    # A player never straddles splits (no leakage): each player_id -> one split.
    per_player = out.groupby(ENTITY_KEY)[SPLIT_COLUMN].nunique()
    assert (per_player == 1).all()
