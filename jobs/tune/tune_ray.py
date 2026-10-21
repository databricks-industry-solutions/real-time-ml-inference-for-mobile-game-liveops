"""Ray Tune HPO showcase for the offer ranker on the Databricks AI Runtime.

A narrative/showcase workload (NOT part of the daily pipeline): the ranker is tiny
and does not need HPO, but this demonstrates the sanctioned Ray Tune pattern on the
AI Runtime -- ASHA early-stopping over a small search space, one GPU per trial, N
concurrent trials across the provisioned nodes, best config logged to MLflow.

Runs as an `ai_runtime_task` under the same `ray_bootstrap.sh` as training (node 0 runs
this driver after the Ray cluster forms). It reads the SAME training_set parquet off
the UC Volume that training uses, and reuses the pure-PyTorch `core.train_model` +
`evaluate.heldout_metrics` as the per-trial train+eval so the trial fn is unit-
testable without Ray/GPU.
"""

from __future__ import annotations

import os

from offer_recommender.logging_config import get_logger
from offer_recommender.preprocessing.pipeline import (
    SPLIT_COLUMN,
    FeatureColumns,
    TransformParams,
)

logger = get_logger(__name__)

# Search space for the sweep. Kept small + honest: the two knobs that actually move
# this net. loguniform lr, a few embedding widths. ASHA prunes the weak ones early.
TUNE_PARAM_SPACE_SPEC: dict[str, tuple] = {
    "lr": ("loguniform", 1e-4, 1e-2),
    "embedding_dim": ("choice", (8, 16, 32)),
}
TUNE_METRIC = "val_revenue_rmse_log"
TUNE_NUM_SAMPLES = 8
TUNE_MAX_EPOCHS = 20
TUNE_GRACE_PERIOD = 4


def _param_space() -> dict:
    """Build the Ray Tune param space from TUNE_PARAM_SPACE_SPEC (import ray lazily)."""
    from ray import tune

    return {
        "lr": tune.loguniform(*TUNE_PARAM_SPACE_SPEC["lr"][1:]),
        "embedding_dim": tune.choice(TUNE_PARAM_SPACE_SPEC["embedding_dim"][1]),
    }


def _frames(pdf, columns: FeatureColumns):
    """Split a pandas frame into (numeric, categorical, revenue, purchased) arrays.

    `columns.numeric` is a single array-of-double column (numeric_arr); the index
    columns are the StringIndexer outputs. Pure numpy -- no Ray.
    """
    import numpy as np

    numeric = np.stack(pdf[columns.numeric].to_numpy()).astype(np.float32)
    categorical = np.stack([pdf[c].to_numpy() for c in columns.index], axis=1).astype(
        np.int64
    )
    revenue = pdf[columns.revenue].to_numpy(dtype=np.float64)
    purchased = pdf[columns.purchase].to_numpy(dtype=np.float64)
    return numeric, categorical, revenue, purchased


def train_and_eval(
    config: dict,
    train: tuple,
    val: tuple,
    params: TransformParams,
    columns: FeatureColumns,
) -> dict:
    """One trial: train on `train`, return held-out metrics on `val`. Pure torch/numpy.

    Reuses `core.train_model` (the same loop training uses) + `evaluate.heldout_metrics`,
    so a trial is exactly a small training run scored on the val split. No Ray here, so
    this is unit-testable directly.
    """
    import numpy as np
    import torch

    from offer_recommender.training.core import (
        TrainConfig,
        make_dataset,
        rank_offers,
        train_model,
    )
    from offer_recommender.training.evaluate import heldout_metrics

    t_num, t_cat, t_rev, t_pur = train
    cfg = TrainConfig(
        epochs=int(config.get("epochs", TUNE_MAX_EPOCHS)),
        lr=float(config["lr"]),
        embedding_dim=int(config["embedding_dim"]),
        seed=0,
    )
    net = train_model(params, make_dataset(t_num, t_cat, t_rev, t_pur), cfg)

    v_num, v_cat, v_rev, v_pur = val
    revenue_pred = rank_offers(net, v_num, v_cat)
    net.eval()
    with torch.no_grad():
        out = net(
            torch.tensor(v_num, dtype=torch.float32),
            torch.tensor(v_cat, dtype=torch.long),
        )
    purchase_prob = torch.sigmoid(out["purchase_logit"]).numpy()
    metrics = heldout_metrics(
        revenue_pred_dollars=revenue_pred,
        purchase_prob=purchase_prob,
        revenue_true_dollars=v_rev,
        purchase_true=np.asarray(v_pur, dtype=np.float64),
    )
    return {TUNE_METRIC: metrics.revenue_rmse_log}


def _trial(config: dict, train: tuple, val: tuple, params, columns) -> None:
    """Ray Tune trainable: run one trial and report the metric.

    Uses `ray.tune.report` (NOT `ray.train.report`): inside a function passed to
    Ray Tune, `ray.train.report` is deprecated and raised as an error on this Ray
    (v2 Train API), erroring every trial.
    """
    from ray import tune

    tune.report(train_and_eval(config, train, val, params, columns))


def main() -> None:
    """Read the Volume parquet, sweep with ASHA, log the best config to MLflow."""
    import ray
    import ray.data  # ray.data is a submodule; `import ray` alone does not load it
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    from offer_recommender.config import get_config

    uc = get_config()
    columns = FeatureColumns.default()
    with open(f"{uc.volume_path}/transform_params.json") as f:
        params = TransformParams.from_json(f.read())

    from offer_recommender.training.ray_multinode import attach_cluster

    attach_cluster()
    # file_extensions skips Spark's 0-byte marker files (_SUCCESS/_committed_*/
    # _started_*) that would otherwise fail parquet parsing.
    pdf = ray.data.read_parquet(
        f"{uc.volume_path}/training_set_parquet", file_extensions=["parquet"]
    ).to_pandas()
    train = _frames(pdf[pdf[SPLIT_COLUMN] == "train"], columns)
    val = _frames(pdf[pdf[SPLIT_COLUMN] == "val"], columns)
    logger.info("TUNE_DATA train=%d val=%d", len(train[0]), len(val[0]))

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                _trial, train=train, val=val, params=params, columns=columns
            ),
            resources={"gpu": 1},
        ),
        param_space=_param_space(),
        tune_config=tune.TuneConfig(
            metric=TUNE_METRIC,
            mode="min",
            scheduler=ASHAScheduler(
                max_t=TUNE_MAX_EPOCHS,
                grace_period=TUNE_GRACE_PERIOD,
                reduction_factor=2,
            ),
            num_samples=TUNE_NUM_SAMPLES,
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric=TUNE_METRIC, mode="min")
    logger.info(
        "TUNE_BEST config=%s metric=%s", best.config, best.metrics.get(TUNE_METRIC)
    )

    run_id = os.environ.get("MLFLOW_RUN_ID")
    if run_id:
        try:
            import mlflow

            with mlflow.start_run(run_id=run_id):
                mlflow.log_params({f"best_{k}": v for k, v in best.config.items()})
                mlflow.log_metric(f"best_{TUNE_METRIC}", best.metrics[TUNE_METRIC])
                for i, r in enumerate(results):
                    if r.metrics and TUNE_METRIC in r.metrics:
                        mlflow.log_metric(
                            f"trial_{TUNE_METRIC}", r.metrics[TUNE_METRIC], step=i
                        )
        except Exception as e:  # observability must never fail the sweep
            logger.warning("MLflow logging failed: %s", e)


if __name__ == "__main__":
    main()
