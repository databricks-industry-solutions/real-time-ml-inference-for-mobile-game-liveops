"""Fast unit tests for the Ray Tune showcase driver (no Ray / no GPU).

Covers the search-space contract and the pure `_frames` numpy split. The per-trial
train+eval reuses `core.train_model` + `evaluate.heldout_metrics` (already tested),
and the Tune wrapper is exercised by the GPU showcase run, so those are not re-tested
here.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd

from offer_recommender.preprocessing.pipeline import FeatureColumns


def _load_tune_ray():
    """Import jobs/tune/tune_ray.py (jobs/ is not an importable package)."""
    path = (
        pathlib.Path(__file__).resolve().parent.parent / "jobs" / "tune" / "tune_ray.py"
    )
    spec = importlib.util.spec_from_file_location("tune_ray", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_param_space_spec_is_the_two_honest_knobs():
    m = _load_tune_ray()
    assert set(m.TUNE_PARAM_SPACE_SPEC) == {"lr", "embedding_dim"}
    assert m.TUNE_PARAM_SPACE_SPEC["lr"][0] == "loguniform"
    assert m.TUNE_PARAM_SPACE_SPEC["embedding_dim"][0] == "choice"
    assert m.TUNE_METRIC == "val_revenue_rmse_log"
    # ASHA needs a grace period strictly below max_t to prune.
    assert 0 < m.TUNE_GRACE_PERIOD < m.TUNE_MAX_EPOCHS


def test_frames_projects_arrays_with_correct_shapes_and_dtypes():
    m = _load_tune_ray()
    cols = FeatureColumns.default()
    n, d = 4, 3
    data = {cols.numeric: [list(np.random.rand(d)) for _ in range(n)]}
    for c in cols.index:
        data[c] = np.arange(n)
    data[cols.revenue] = np.random.rand(n) * 10.0
    data[cols.purchase] = (np.arange(n) % 2).astype(float)
    pdf = pd.DataFrame(data)

    numeric, categorical, revenue, purchased = m._frames(pdf, cols)
    assert numeric.shape == (n, d)
    assert numeric.dtype == np.float32
    assert categorical.shape == (n, len(cols.index))
    assert categorical.dtype == np.int64
    assert revenue.shape == (n,)
    assert purchased.shape == (n,)
