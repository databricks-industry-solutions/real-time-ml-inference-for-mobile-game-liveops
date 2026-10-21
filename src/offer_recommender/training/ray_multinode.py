"""Multi-node Ray Train entry point for the Databricks AI Runtime (`air` CLI).

All Ray training on the AI Runtime uses the sanctioned `air` CLI + workload-YAML +
`ray_bootstrap.sh` pattern (single OR multi node -- a small run is just
`num_accelerators: 1`). AIR provisions N nodes from `compute.num_accelerators` /
`accelerator_type` (e.g. 16 accelerators of GPU_8xH100 -> 2 nodes x 8 GPUs) and
sets per-node rendezvous env vars (NODE_RANK, MASTER_ADDR, RAY_HEAD_PORT,
GPUS_PER_NODE, LOCAL_WORLD_SIZE). The bootstrap starts `ray start --head` on node 0
and `ray start --address=...` on the rest, forming ONE Ray cluster spanning the
nodes; node 0 then runs the driver below.

The honest justification for spanning nodes is data-parallel throughput over a
large impression log to hit a daily-retrain window, NOT model FLOPs -- the net is
tiny. This module is the node-0 driver's core: it attaches to the already-running
Ray cluster (`ray.init(address="auto")`), waits until every node's GPUs have
joined, then fits the `TorchTrainer` from `ray_air._fit_ray_train` (same per-worker
loop, reads the training_set PARQUET off the UC Volume, same checkpoint->bytes
contract) with `num_workers` set to the total GPU count so one DDP worker lands on
each GPU across all nodes.

AIR is Public Preview; treat multi-node as a scaling-narrative upgrade, not a GA
requirement.
"""

from __future__ import annotations

import time

from offer_recommender.logging_config import get_logger
from offer_recommender.preprocessing.pipeline import FeatureColumns, TransformParams
from offer_recommender.training.core import TrainConfig
from offer_recommender.training.ray_air import (
    AIRScaleConfig,
    DataConfig,
    _fit_ray_train,
)

logger = get_logger(__name__)


def attach_cluster() -> None:
    """Attach to the Ray cluster `ray_bootstrap.sh` started, shipping this package.

    Ray worker processes on an `ai_runtime_task` do not inherit the driver's
    PYTHONPATH, so the `offer_recommender` package travels in the job's
    `runtime_env` instead (otherwise the TrainController fails to unpickle it).
    No-op if Ray is already initialized.
    """
    import ray

    import offer_recommender

    if not ray.is_initialized():
        ray.init(address="auto", runtime_env={"py_modules": [offer_recommender]})


def _wait_for_gpus(expected_gpus: int, timeout_s: float, poll_s: float = 5.0) -> int:
    """Block until the Ray cluster reports `expected_gpus` GPUs (or timeout).

    Node 0's bootstrap runs the driver immediately after `ray start --head`, but
    worker nodes join asynchronously via `ray start --address=...`. Building the
    `TorchTrainer` before all workers have joined would under-size `num_workers`
    (fewer DDP ranks than GPUs) or fail to place workers. Polling
    `ray.cluster_resources()["GPU"]` until it reaches the count AIR provisioned
    makes the launch deterministic. On timeout we proceed with whatever joined so
    a partial cluster still trains rather than hanging forever.

    Args:
        expected_gpus: total GPUs AIR provisioned (== compute.num_accelerators).
        timeout_s: max seconds to wait for all workers to register.
        poll_s: seconds between cluster-resource polls.

    Returns:
        The GPU count to use for `num_workers` (min of available and expected).
    """
    import ray

    deadline = time.monotonic() + timeout_s
    available = 0
    while time.monotonic() < deadline:
        available = int(ray.cluster_resources().get("GPU", 0))
        if available >= expected_gpus:
            logger.info("RAY_CLUSTER_GPUS_READY=%d/%d", available, expected_gpus)
            return expected_gpus
        logger.info("WAITING_FOR_GPUS=%d/%d", available, expected_gpus)
        time.sleep(poll_s)
    logger.warning(
        "GPU_WAIT_TIMEOUT available=%d expected=%d; training on what joined",
        available,
        expected_gpus,
    )
    return max(available, 1)


def train_multinode(
    data: DataConfig,
    params: TransformParams,
    columns: FeatureColumns,
    total_gpus: int,
    train_config: TrainConfig | None = None,
    gpu_wait_timeout_s: float = 900.0,
) -> tuple[dict, bytes]:
    """Attach to the multi-node Ray cluster and fit the ranker; return (metrics,
    weights).

    Runs on node 0 ONLY (the bootstrap script gates the driver on NODE_RANK=0).
    The bootstrap already started the Ray head and joined every worker node into
    ONE cluster, so this attaches with `attach_cluster()` and sets
    `num_workers` to the total GPU count so DDP spans all nodes. The trained
    weights are read out as bytes here (the Ray Result/Checkpoint object does not
    survive being returned across process boundaries).

    Args:
        data: UC Volume parquet path for the Ray Data read (no Spark/warehouse).
        params: fitted transform params (bundled into the model at registration).
        columns: the SparkML->Ray column contract (numeric / index / targets).
        total_gpus: GPUs AIR provisioned across all nodes (== num_accelerators);
            one DDP worker per GPU.
        train_config: hyperparameters (defaults to `TrainConfig()`).
        gpu_wait_timeout_s: max seconds to wait for all worker nodes to join.

    Returns:
        (metrics, model_state_bytes); register_model reads the same `model_state.pt`.
    """

    cfg = train_config or TrainConfig()

    # Attach to the head the bootstrap started; do NOT start a new local cluster.
    attach_cluster()

    num_workers = _wait_for_gpus(total_gpus, gpu_wait_timeout_s)

    # `gpus` drives ScalingConfig(num_workers=..., use_gpu=True) inside _fit_ray_train.
    scale = AIRScaleConfig(gpus=num_workers)
    logger.info(
        "MULTINODE_FIT num_workers=%d parquet=%s", num_workers, data.parquet_path
    )
    return _fit_ray_train(data, params, columns, scale, cfg)
