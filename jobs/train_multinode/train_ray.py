"""Node-0 driver for multi-node Ray Train on the Databricks AI Runtime.

Invoked by `ray_bootstrap.sh` ONLY on node 0 (NODE_RANK=0), AFTER
`ray start --head` and the worker nodes' `ray start --address=...` have formed one
Ray cluster spanning all provisioned nodes. This process attaches to that cluster
and fits the SAME `TorchTrainer` the single-node pipeline uses, with one DDP worker
per GPU across every node.

All coordinates come from environment variables exported by `ray_bootstrap.sh`,
so nothing is hardcoded and no secret is committed:

  RECO_CATALOG / RECO_SCHEMA   UC location (deploy-time bundle variables, from the
                               task's hyperparameters.yaml) -> Volume path
  RECO_EPOCHS                  training epochs (same source)
  RECO_TOTAL_GPUS              NUM_NODES x LOCAL_WORLD_SIZE (set by AI Runtime)

The training data is read as parquet from the UC Volume (staged by preprocess),
so no SQL warehouse / host / token is needed on the GPU node.

The trained weights are written to `model_state.pt` on the SAME UC Volume path
the single-node path uses, so `jobs/register_model.py` reads them unchanged.
"""

from __future__ import annotations

import os

from offer_recommender.config import get_config
from offer_recommender.logging_config import get_logger
from offer_recommender.preprocessing.pipeline import FeatureColumns, TransformParams
from offer_recommender.training.core import TrainConfig
from offer_recommender.training.ray_air import DataConfig
from offer_recommender.training.ray_multinode import train_multinode

logger = get_logger(__name__)


def _require(name: str) -> str:
    """Return the env var `name` or raise a clear error if missing."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set; the multi-node driver expects ray_bootstrap.sh "
            "to export it."
        )
    return value


def main() -> None:
    """Fit the ranker across all provisioned GPU nodes; checkpoint to the Volume."""
    uc = get_config()

    # Transform params were written to the Volume by preprocess; Ray workers bundle
    # them into the model. Plain file read (no Spark needed on the AI Runtime node).
    with open(f"{uc.volume_path}/transform_params.json") as f:
        params = TransformParams.from_json(f.read())

    # preprocess stages the training_set as parquet on the Volume; the GPU node
    # has the Volume FUSE-mounted, so Ray reads it directly (no warehouse/token).
    data = DataConfig(parquet_path=f"{uc.volume_path}/training_set_parquet")
    total_gpus = int(_require("RECO_TOTAL_GPUS"))
    # Multi-node: checkpoint to a UC Volume subdir (shared filesystem reachable by
    # EVERY GPU node). Without a shared storage_path Ray Train defaults to a
    # head-node-local dir the worker nodes cannot read, and the TrainController
    # fails to persist -- a placement-dependent flake that fails the task and
    # forces a retry. The Volume is already mounted here (we read
    # transform_params.json and write model_state.pt to it).
    train_config = TrainConfig(
        epochs=int(os.environ.get("RECO_EPOCHS", "40")),
        storage_path=f"{uc.volume_path}/ray_train",
    )

    metrics, model_bytes = train_multinode(
        data=data,
        params=params,
        columns=FeatureColumns.default(),
        total_gpus=total_gpus,
        train_config=train_config,
    )
    logger.info("TRAIN_METRICS=%s", metrics)

    # Same Volume path + filename the single-node path writes, so register_model
    # picks up the checkpoint identically regardless of which trainer produced it.
    ckpt_path = f"{uc.volume_path}/model_state.pt"
    with open(ckpt_path, "wb") as f:
        f.write(model_bytes)
    logger.info("CHECKPOINT_WRITTEN=%s", ckpt_path)


if __name__ == "__main__":
    main()
