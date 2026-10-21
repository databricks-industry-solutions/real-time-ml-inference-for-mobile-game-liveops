"""Recommender evaluation: how much offer revenue does the ranking capture?

Global MSE on a zero-inflated dollar target understates ranking quality, so the
demo's headline metric is oracle-relative: for each player, score the full offer
slate, take the model's top-1, and compare the captured true expected revenue
against a random pick (floor) and the oracle best offer (ceiling).

`oracle_lift_captured` in [0, 1] is the fraction of the achievable
oracle-over-random revenue lift the model captures. Because we own the data
generator, the "true" expected revenue per (player, offer) is computable
exactly, which makes this an honest measure rather than a proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from offer_recommender.data import generate as gen
from offer_recommender.model.net import OfferRankerNet
from offer_recommender.model.transform import transform
from offer_recommender.preprocessing.pipeline import TransformParams
from offer_recommender.schema import (
    REALTIME_NUMERICAL_FEATURES,
    STREAMING_NUMERICAL_FEATURES,
)
from offer_recommender.training.core import rank_offers


@dataclass(frozen=True)
class HeldoutMetrics:
    """Held-out head metrics for the registered model (logged to MLflow).

    Revenue is scored in BOTH spaces: log1p (the space the head is trained and
    MSE-optimised in) and dollars (expm1, what the business reads). Purchase is
    the auxiliary binary head.
    """

    n_rows: int
    revenue_rmse_log: float
    revenue_rmse_dollars: float
    revenue_mae_dollars: float
    purchase_auc: float
    purchase_logloss: float
    # Threshold-based classification metrics for the purchase head (at
    # PURCHASE_THRESHOLD). AUC/logloss are threshold-free; these describe the
    # head's quality at a concrete decision cutoff. F2 weights recall over
    # precision (2x) -- the business-relevant tradeoff when a missed buyer costs
    # more than an unnecessary offer.
    purchase_accuracy: float
    purchase_precision: float
    purchase_recall: float
    purchase_f1: float
    purchase_f2: float

    def as_mlflow_metrics(self) -> dict[str, float]:
        """Flatten to a metric_name -> value dict for mlflow.log_metrics."""
        return {
            "val_n_rows": float(self.n_rows),
            "val_revenue_rmse_log": self.revenue_rmse_log,
            "val_revenue_rmse_dollars": self.revenue_rmse_dollars,
            "val_revenue_mae_dollars": self.revenue_mae_dollars,
            "val_purchase_auc": self.purchase_auc,
            "val_purchase_logloss": self.purchase_logloss,
            "val_purchase_accuracy": self.purchase_accuracy,
            "val_purchase_precision": self.purchase_precision,
            "val_purchase_recall": self.purchase_recall,
            "val_purchase_f1": self.purchase_f1,
            "val_purchase_f2": self.purchase_f2,
        }


# Decision threshold for the binary purchase head's threshold-based metrics.
PURCHASE_THRESHOLD: float = 0.5


def heldout_metrics(
    revenue_pred_dollars: np.ndarray,
    purchase_prob: np.ndarray,
    revenue_true_dollars: np.ndarray,
    purchase_true: np.ndarray,
) -> HeldoutMetrics:
    """Compute held-out regression + classification metrics.

    Args:
        revenue_pred_dollars: model revenue prediction in dollars (expm1 of head).
        purchase_prob: predicted purchase probability (sigmoid of the logit).
        revenue_true_dollars: ground-truth d7 revenue in dollars.
        purchase_true: ground-truth purchase label (0/1).
    """
    # sklearn.metrics is the Databricks-documented path for classification metrics
    # (fbeta_score gives F2 natively). Imported lazily so importing this module
    # (e.g. for the ranking eval) does not require sklearn.
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        fbeta_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    rev_pred = np.asarray(revenue_pred_dollars, dtype=np.float64)
    rev_true = np.asarray(revenue_true_dollars, dtype=np.float64)
    pur_prob = np.clip(np.asarray(purchase_prob, dtype=np.float64), 1e-7, 1 - 1e-7)
    pur_true = np.asarray(purchase_true, dtype=np.int64)
    pur_pred = (pur_prob >= PURCHASE_THRESHOLD).astype(np.int64)

    rev_err_dollars = rev_pred - rev_true
    rev_err_log = np.log1p(np.maximum(rev_pred, 0.0)) - np.log1p(
        np.maximum(rev_true, 0.0)
    )
    # roc_auc/log_loss need both classes present; guard the single-class val split.
    both_classes = pur_true.min() != pur_true.max()
    auc = float(roc_auc_score(pur_true, pur_prob)) if both_classes else 0.5
    logloss = float(log_loss(pur_true, pur_prob, labels=[0, 1]))
    return HeldoutMetrics(
        n_rows=len(rev_true),
        revenue_rmse_log=float(np.sqrt(np.mean(rev_err_log**2))),
        revenue_rmse_dollars=float(np.sqrt(np.mean(rev_err_dollars**2))),
        revenue_mae_dollars=float(np.mean(np.abs(rev_err_dollars))),
        purchase_auc=auc,
        purchase_logloss=logloss,
        purchase_accuracy=float(accuracy_score(pur_true, pur_pred)),
        purchase_precision=float(precision_score(pur_true, pur_pred, zero_division=0)),
        purchase_recall=float(recall_score(pur_true, pur_pred, zero_division=0)),
        purchase_f1=float(f1_score(pur_true, pur_pred, zero_division=0)),
        purchase_f2=float(fbeta_score(pur_true, pur_pred, beta=2, zero_division=0)),
    )


@dataclass(frozen=True)
class RankingEval:
    """Result of the oracle-relative ranking evaluation."""

    model_revenue: float
    oracle_revenue: float
    random_revenue: float

    @property
    def oracle_lift_captured(self) -> float:
        """Fraction of oracle-over-random lift the model captures, in [0, 1]."""
        denom = self.oracle_revenue - self.random_revenue
        if denom <= 0:
            return 0.0
        return (self.model_revenue - self.random_revenue) / denom


@dataclass(frozen=True)
class EvalConfig:
    """Knobs for the ranking evaluation.

    realtime_defaults: the real-time feature values to hold fixed while scoring
    each player's slate (at serving these arrive from the stream).
    streaming_defaults: streaming rolling-window feature values to hold fixed
    while scoring; at serving these come from the Streaming Feature View. If not
    provided, all streaming features default to 0.0.
    """

    realtime_defaults: dict[str, float]
    streaming_defaults: dict[str, float] | None = None
    n_players: int = 400
    seed: int = 1


def _true_expected_revenue(slate: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Compute the generator's true expected d7 revenue per candidate offer."""
    s = slate.copy()
    s["_propensity"] = s["segment"].map(gen._SEGMENT_PROPENSITY)
    prob = gen._purchase_propensity(s, rng)
    propensity = s["_propensity"].to_numpy()
    gems = s["offer_gem_amount"].to_numpy()
    match_count = s["match_count"].to_numpy()
    downstream_mean = (
        propensity * 6.0
        + np.log1p(gems) * propensity * 1.5
        + np.clip(match_count / 10.0, 0, 2) * 3.0
    )
    return prob * (s["offer_price_usd"].to_numpy() + downstream_mean)


def evaluate_ranking(
    net: OfferRankerNet,
    params: TransformParams,
    players: pd.DataFrame,
    offers: pd.DataFrame,
    config: EvalConfig,
) -> RankingEval:
    """Evaluate top-1 offer ranking against random and oracle baselines.

    Each sampled player is scored against the full offer catalogue with the
    config's real-time feature values (at serving these come from the stream).
    """
    rng = np.random.default_rng(config.seed)
    sample = players.sample(
        min(config.n_players, len(players)), random_state=config.seed
    )
    model_rev, oracle_rev, random_rev = [], [], []
    for _, player in sample.iterrows():
        slate = (
            offers.assign(_k=1)
            .merge(pd.DataFrame([player]).assign(_k=1), on="_k")
            .drop(columns="_k")
        )
        for feature in REALTIME_NUMERICAL_FEATURES:
            slate[feature] = config.realtime_defaults[feature]
        # streaming_defaults=None -> the 5 streaming aggregate columns are set to 0.0
        # on the eval slate. Oracle revenue comes from _purchase_propensity, which does
        # NOT read streaming features, so oracle/random/model all see the same constant
        # and the RELATIVE ranking metric stays valid. Limitation: a model that leans
        # on streaming features to differentiate offers may show lower lift here than in
        # production (a false negative, never a false positive). Pass realistic medians
        # via streaming_defaults for a tighter eval.
        streaming = config.streaming_defaults or {}
        for feature in STREAMING_NUMERICAL_FEATURES:
            slate[feature] = streaming.get(feature, 0.0)
        true_rev = _true_expected_revenue(slate, rng)
        numeric, categorical = transform(slate, params)
        predicted = rank_offers(net, numeric, categorical)
        model_rev.append(true_rev[int(predicted.argmax())])
        oracle_rev.append(float(true_rev.max()))
        random_rev.append(true_rev[rng.integers(len(true_rev))])
    return RankingEval(
        model_revenue=float(np.mean(model_rev)),
        oracle_revenue=float(np.mean(oracle_rev)),
        random_revenue=float(np.mean(random_rev)),
    )
