"""Framework-agnostic training core for the offer ranker.

`train_model` is a plain-PyTorch loop that both the local runner and each Ray
Train worker call. It is deliberately free of Ray/Spark imports so it can be
unit-tested and reasoned about on its own; the Ray Train wrapper in
`offer_recommender.training.ray_air` wraps the model with `prepare_model`, shards the
data with Ray Data, and calls the same per-batch step.

Revenue is modelled in log1p space (the target is heavy-tailed dollars). Since
ranking is invariant under the monotonic log1p/expm1 map, the recommender is
unaffected; serving applies expm1 to report dollar revenue. The purchase head
is an auxiliary BCE signal that regularises the shared trunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from offer_recommender.model.net import OfferRankerNet, multitask_loss


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for a training run."""

    epochs: int = 40
    batch_size: int = 256
    lr: float = 2e-3
    embedding_dim: int = 16
    purchase_weight: float = 0.5
    seed: int = 0
    # Ray Train checkpoint storage path. MUST be a shared filesystem/cloud URI
    # reachable by EVERY node for MULTI-NODE runs (a UC Volume path or s3://) --
    # a head-node-local path is not visible to worker nodes and the TrainController
    # fails to persist ("storage_path should be a cloud storage URI or a shared
    # filesystem"). None (single-node @ray_launch) uses Ray's default local path,
    # which is fine because head and worker are the same node.
    storage_path: str | None = None


def make_dataset(
    numeric: np.ndarray,
    categorical: np.ndarray,
    revenue: np.ndarray,
    purchased: np.ndarray,
) -> TensorDataset:
    """Build a TensorDataset; revenue is log1p-transformed here."""
    return TensorDataset(
        torch.tensor(numeric, dtype=torch.float32),
        torch.tensor(categorical, dtype=torch.long),
        torch.tensor(np.log1p(revenue), dtype=torch.float32),
        torch.tensor(purchased, dtype=torch.float32),
    )


def train_step(
    net: OfferRankerNet,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, ...],
    purchase_weight: float,
) -> float:
    """Run one optimisation step; return the scalar loss."""
    numeric, categorical, revenue_log, purchased = batch
    optimizer.zero_grad()
    outputs = net(numeric, categorical)
    loss = multitask_loss(
        outputs, revenue_log, purchased, purchase_weight=purchase_weight
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def train_model(
    params,
    dataset: TensorDataset,
    config: TrainConfig | None = None,
) -> OfferRankerNet:
    """Train an OfferRankerNet on a local (single-process) dataset.

    Used for local runs and tests. The Ray Train worker path reuses
    `train_step` and `make_dataset` but drives the loop with a distributed
    sampler; see `offer_recommender.training.ray_air`.
    """
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)
    net = OfferRankerNet(params, embedding_dim=cfg.embedding_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    net.train()
    for _ in range(cfg.epochs):
        for batch in loader:
            train_step(net, optimizer, batch, cfg.purchase_weight)
    return net


def rank_offers(
    net: OfferRankerNet, numeric: np.ndarray, categorical: np.ndarray
) -> np.ndarray:
    """Return predicted dollar revenue per row (expm1 of the log-space head).

    Ranking by this value equals ranking by the raw head output (monotonic);
    expm1 is applied so downstream code and the app show dollars.
    """
    net.eval()
    with torch.no_grad():
        out = net(
            torch.tensor(numeric, dtype=torch.float32),
            torch.tensor(categorical, dtype=torch.long),
        )
    return np.expm1(out["revenue_d7"].numpy())
