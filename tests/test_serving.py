"""Serving-artifact guards: the net must TorchScript, save, reload and match.

Model Serving loads the net with `torch.jit.load` (no project code on the
container path), so scripting must not silently break. This caught a real
ModuleDict-indexing / list-unpack scripting failure during the build.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from offer_recommender.model.net import OfferRankerNet, script_and_save
from offer_recommender.preprocessing.pipeline import TransformParams
from offer_recommender.schema import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def _toy_params() -> TransformParams:
    """Small fitted-params stand-in covering every categorical feature."""
    return TransformParams(
        vocabs={f: [f"{f}_a", f"{f}_b", f"{f}_c"] for f in CATEGORICAL_FEATURES},
        mean=[0.0] * len(NUMERICAL_FEATURES),
        std=[1.0] * len(NUMERICAL_FEATURES),
        numerical_features=list(NUMERICAL_FEATURES),
        categorical_features=list(CATEGORICAL_FEATURES),
        log1p_features=[],
    )


def test_net_scripts_saves_and_matches_eager():
    """Scripted + reloaded net matches the eager net element-wise."""
    params = _toy_params()
    net = OfferRankerNet(params).eval()
    rng = np.random.default_rng(0)
    numeric = torch.tensor(
        rng.standard_normal((8, len(NUMERICAL_FEATURES))), dtype=torch.float32
    )
    categorical = torch.tensor(
        rng.integers(0, 3, (8, len(CATEGORICAL_FEATURES))), dtype=torch.long
    )
    with torch.no_grad():
        eager = net(numeric, categorical)

    path = str(Path(tempfile.mkdtemp()) / "model.ts")
    script_and_save(net, path)
    reloaded = torch.jit.load(path).eval()
    with torch.no_grad():
        out = reloaded(numeric, categorical)

    assert torch.allclose(eager["revenue_d7"], out["revenue_d7"], atol=1e-5)
    assert torch.allclose(eager["purchase_logit"], out["purchase_logit"], atol=1e-5)
