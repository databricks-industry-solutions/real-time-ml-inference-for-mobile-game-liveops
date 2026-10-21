"""Ray Train (plain-PyTorch `TorchTrainer`) on the Databricks AI Runtime.

Training runs on serverless GPU as an `ai_runtime_task` (see `jobs/train_multinode/`):
a bootstrap script starts `ray start --head` on node 0 and joins the worker nodes
into ONE Ray cluster, then the node-0 driver attaches (`ray_multinode.attach_cluster`)
and fits the `TorchTrainer` below with one DDP worker per GPU. The task runs in the
Databricks AI environment (Ray + PyTorch preinstalled), so there is no per-run pip
install.

Data crosses no Spark boundary: the GPU node has NO SparkSession, so training reads
the `training_set` that `preprocess` stages as PARQUET on the UC Volume via
`ray.data.read_parquet` (the Volume is FUSE-mounted on the node; parallel direct
read, no SQL warehouse, no host/token plumbing). Registration (`fe.log_model`)
needs Spark + the FE client and runs in a separate task; this module only trains
and checkpoints to the UC Volume.

Honest GPU framing (see the design spec): the model is small; the reason to run
distributed is data-parallel throughput over a large impression log to hit the
daily-retrain window, not model FLOPs. `num_workers` defaults to 1 GPU and scales
with `gpus`; the same code runs on 1xA10 or an 8xH100 node. AIR is Public Preview.
"""

from __future__ import annotations

from dataclasses import dataclass

from offer_recommender.preprocessing.pipeline import FeatureColumns, TransformParams
from offer_recommender.training.core import TrainConfig


@dataclass(frozen=True)
class AIRScaleConfig:
    """Distributed-training scale knob.

    gpus: total GPUs across the provisioned node(s); one Ray worker per GPU. Drives
    `ScalingConfig(num_workers=gpus, use_gpu=True)`. Default 1 (cheap A10 / single
    H100); the multi-node driver sets it to the total GPU count AIR provisioned. The
    training code is unchanged regardless of the value.
    """

    gpus: int = 1


@dataclass(frozen=True)
class DataConfig:
    """Where Ray Data reads the training data from (no Spark on the GPU node).

    `preprocess` writes the transformed `training_set` as parquet to
    `parquet_path` on the UC Volume (in addition to the Delta table, which
    register_model still reads for the held-out val split). The GPU node has the
    Volume FUSE-mounted, so `ray.data.read_parquet(parquet_path)` is a direct,
    parallel read with no SQL warehouse and no host/token -- the training rows are
    all plain typed columns (the intermediate VectorUDT columns are dropped by
    `add_feature_arrays` before the write).
    """

    parquet_path: str


def _train_loop_per_worker(loop_config: dict) -> None:
    """Ray Train per-worker entry point (runs on each GPU worker).

    Reads its shard of the training data from the Ray Data context, wraps the
    model with DDP via `prepare_model`, trains, and reports the model on the
    last epoch. Imports Ray inside the function so this module imports (and unit
    tests) without Ray installed.
    """
    import os
    import tempfile

    import ray.train
    import ray.train.torch
    import torch

    from offer_recommender.model.net import OfferRankerNet, multitask_loss

    params: TransformParams = loop_config["params"]
    cfg: TrainConfig = loop_config["train_config"]
    columns: FeatureColumns = loop_config["columns"]

    torch.manual_seed(cfg.seed)
    net = ray.train.torch.prepare_model(
        OfferRankerNet(params, embedding_dim=cfg.embedding_dim)
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    # The air workload's MLflow env vars open an active run; log directly (no
    # start_run) from RANK 0 ONLY so the run isn't spammed by every worker. Logging
    # is observability, not correctness: never let an MLflow hiccup fail training.
    is_rank_0 = ray.train.get_context().get_world_rank() == 0

    def _mlflow_log(fn_name: str, *args, **kwargs) -> None:
        if not is_rank_0:
            return
        try:
            import mlflow

            getattr(mlflow, fn_name)(*args, **kwargs)
        except Exception:
            # Observability must never break training.
            pass

    _mlflow_log(
        "log_params",
        {
            "lr": cfg.lr,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "embedding_dim": cfg.embedding_dim,
            "purchase_weight": cfg.purchase_weight,
            "seed": cfg.seed,
            "num_workers": ray.train.get_context().get_world_size(),
        },
    )

    shard = ray.train.get_dataset_shard("train")
    for epoch in range(cfg.epochs):
        net.train()
        loss = None
        for batch in shard.iter_torch_batches(batch_size=cfg.batch_size):
            numeric = batch[columns.numeric].float()
            categorical = torch.stack([batch[c].long() for c in columns.index], dim=1)
            revenue_log = torch.log1p(batch[columns.revenue].float())
            purchased = batch[columns.purchase].float()
            optimizer.zero_grad()
            outputs = net(numeric, categorical)
            loss = multitask_loss(
                outputs, revenue_log, purchased, purchase_weight=cfg.purchase_weight
            )
            loss.backward()
            optimizer.step()

        # Checkpoint model weights on the final epoch from rank 0 only, so the
        # registration task can rebuild the net. Under DDP the wrapped module is
        # at net.module; unwrap before saving the state dict.
        metrics = {"loss": float(loss.detach())} if loss is not None else {}
        if loss is not None:
            _mlflow_log("log_metric", "loss", float(loss.detach()), step=epoch)
        if epoch == cfg.epochs - 1 and ray.train.get_context().get_world_rank() == 0:
            state = getattr(net, "module", net).state_dict()
            with tempfile.TemporaryDirectory() as tmp:
                torch.save(state, os.path.join(tmp, "model_state.pt"))
                ray.train.report(
                    metrics, checkpoint=ray.train.Checkpoint.from_directory(tmp)
                )
        else:
            ray.train.report(metrics)


def read_training_dataset(data: DataConfig, columns: FeatureColumns):
    """Read the training split from the Volume parquet into a materialized Ray Dataset.

    `ray.data.read_parquet` reads the FUSE-mounted UC Volume directly (no warehouse,
    no Spark). Filter to the training split (`preprocess` wrote a deterministic
    player-level `split` column; the held-out `val` rows are scored at registration
    and must NOT be trained on), project to the trainer's columns, and `materialize()`
    to pin the dataset in the cluster object store for multi-epoch training.
    """
    import ray.data

    from offer_recommender.preprocessing.pipeline import SPLIT_COLUMN

    selected = list(columns.selected())
    return (
        # file_extensions filters out Spark's 0-byte marker files (_SUCCESS,
        # _committed_*, _started_*) in the output dir; without it read_parquet tries
        # to parse them as parquet and raises ArrowInvalid ("size is 0 bytes").
        ray.data.read_parquet(data.parquet_path, file_extensions=["parquet"])
        .filter(lambda row: row[SPLIT_COLUMN] == "train")
        .select_columns(selected)
        .materialize()
    )


def _fit_ray_train(
    data: DataConfig,
    params: TransformParams,
    columns: FeatureColumns,
    scale: AIRScaleConfig,
    cfg: TrainConfig,
) -> tuple[dict, bytes]:
    """Read the Volume parquet into Ray Data, fit the TorchTrainer, return (metrics,
    model_state_bytes).

    Runs on the Ray head the bootstrap started (the caller already
    `ray.init(address="auto")`ed). The trained weights are read out as bytes HERE,
    while the checkpoint is still live on the cluster, because the Ray
    Result/Checkpoint object does not survive being pickled back across the
    subprocess boundary.
    """
    import datetime
    from pathlib import Path

    from ray.train import CheckpointConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    dataset = read_training_dataset(data, columns)

    trainer = TorchTrainer(
        train_loop_per_worker=_train_loop_per_worker,
        train_loop_config={"params": params, "train_config": cfg, "columns": columns},
        scaling_config=ScalingConfig(num_workers=scale.gpus, use_gpu=True),
        datasets={"train": dataset},
        run_config=RunConfig(
            # Unique per run: a fixed name under the persistent storage_path makes Ray
            # restore the previous run's snapshot (a stale resume, and a crash when
            # the snapshot came from another Ray version). Every run starts fresh.
            # ponytail: one small run dir per training on the Volume; prune if it grows.
            name=datetime.datetime.now(datetime.UTC).strftime(
                "offer-reco-%Y%m%dT%H%M%S"
            ),
            checkpoint_config=CheckpointConfig(num_to_keep=1),
            # MULTI-NODE requires a shared storage_path (UC Volume) all nodes can
            # reach; without it Ray defaults to a head-node-local dir the workers
            # cannot read -> the TrainController fails to persist the checkpoint.
            # Single-node leaves it None (Ray's local default, head == worker).
            storage_path=cfg.storage_path,
        ),
    )
    result = trainer.fit()
    with result.checkpoint.as_directory() as ckpt_dir:
        model_bytes = Path(ckpt_dir, "model_state.pt").read_bytes()
    return dict(result.metrics or {}), model_bytes
