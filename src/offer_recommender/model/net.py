"""Multi-head offer-ranking network.

Embeddings for each categorical (sized len(vocab)+1 for the StringIndexer
OOV/keep slot) are concatenated with the scaled numeric block and fed through a
shared MLP trunk. Two heads:

- revenue_d7 (primary, regression): expected offer revenue at a d7 horizon,
  which drives the ranking. Softplus keeps it non-negative.
- purchase (auxiliary, binary): purchase-probability logit.

The fitted `TransformParams` are held on the module so a serving wrapper can
reproduce the transform without Spark. The net operates on already-transformed
tensors; `offer_recommender.model.transform` produces those from a raw feature frame.
"""

from __future__ import annotations

import torch
from torch import nn

from offer_recommender.preprocessing.pipeline import TransformParams


class OfferRankerNet(nn.Module):
    """Embedding + numeric MLP with revenue and purchase heads."""

    def __init__(
        self,
        params: TransformParams,
        embedding_dim: int = 16,
        trunk_dims: tuple[int, ...] = (128, 64),
    ) -> None:
        """Build the network from the fitted transform params.

        Args:
            params: fitted vocab + scaler stats; determines embedding sizes and
                the numeric input width.
            embedding_dim: width of each categorical embedding.
            trunk_dims: hidden layer widths of the shared MLP.
        """
        super().__init__()
        self._categorical_features = list(params.categorical_features)
        sizes = params.embedding_sizes()
        # ModuleList (ordered, index-based) rather than ModuleDict: TorchScript
        # cannot index a ModuleDict by a non-literal, and the column order is
        # already fixed by params.categorical_features. Index i here aligns with
        # column i of the categorical input tensor.
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(sizes[feature], embedding_dim)
                for feature in self._categorical_features
            ]
        )
        n_numeric = len(params.numerical_features)
        trunk_in = embedding_dim * len(self._categorical_features) + n_numeric

        layers: list[nn.Module] = []
        prev = trunk_in
        for width in trunk_dims:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(0.1)]
            prev = width
        self.trunk = nn.Sequential(*layers)
        self.revenue_head = nn.Linear(prev, 1)
        self.purchase_head = nn.Linear(prev, 1)

    def forward(
        self, numeric: torch.Tensor, categorical: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass.

        Args:
            numeric: (batch, n_numeric) scaled float tensor.
            categorical: (batch, n_categorical) long index tensor, column order
                matching `params.categorical_features`.

        Returns:
            dict with `revenue_d7` (non-negative) and `purchase_logit`.
        """
        # Enumerate the ModuleList (TorchScript-friendly); column i of the
        # categorical tensor maps to embedding i.
        parts = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        parts.append(numeric)
        hidden = self.trunk(torch.cat(parts, dim=1))
        return {
            "revenue_d7": torch.nn.functional.softplus(
                self.revenue_head(hidden)
            ).squeeze(-1),
            "purchase_logit": self.purchase_head(hidden).squeeze(-1),
        }


def script_and_save(net: OfferRankerNet, path: str) -> str:
    """TorchScript the trained net and save it as a serving artifact.

    TorchScript (not a pickled nn.Module) so the serving container loads it with
    `torch.jit.load` and no project code on the path. Returns the saved path.
    """
    scripted = torch.jit.script(net.eval())
    scripted.save(path)
    return path


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    revenue_target: torch.Tensor,
    purchase_target: torch.Tensor,
    purchase_weight: float = 0.5,
) -> torch.Tensor:
    """Revenue MSE plus weighted auxiliary purchase BCE.

    The revenue head is primary (drives ranking) and carries weight 1; purchase
    is a lighter auxiliary signal (`purchase_weight`) that regularises the
    shared trunk.
    """
    revenue_loss = torch.nn.functional.mse_loss(outputs["revenue_d7"], revenue_target)
    purchase_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["purchase_logit"], purchase_target
    )
    return revenue_loss + purchase_weight * purchase_loss
