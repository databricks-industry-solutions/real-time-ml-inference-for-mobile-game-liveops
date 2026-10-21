"""Numpy/torch twin of the SparkML preprocessing pipeline.

Reproduces `offer_recommender.preprocessing.pipeline` exactly from the fitted
`TransformParams`, so the serving container needs no SparkSession. The parity
test asserts this agrees element-wise with SparkML on a holdout; any drift here
is training/serving skew and must fail that test.

Transform order matches the Spark pipeline: log1p on the configured features,
then standard-scale the numeric block; categoricals map to their vocab index
(unseen -> the OOV/keep index = len(vocab)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from offer_recommender.preprocessing.pipeline import TransformParams
from offer_recommender.schema import STREAMING_NUMERICAL_FEATURES

# Streaming SFV features are looked up from the online store at serving. A player
# with no in-window events (or a freshly-materialized online table that has not
# emitted yet) has NO online row, so the FS lookup returns NULL -> NaN. Training
# fills exactly these columns with 0.0 BEFORE the SparkML fit (jobs/preprocess.py:
# an empty backward RollingWindow genuinely means zero activity), so the scaler
# mean/std were computed on 0-filled values. Serving must fill the same columns
# the same way or the NaN propagates through log1p/scale into a NaN prediction.
# Non-streaming NULLs are intentionally left to surface as real bugs.
_STREAMING_FILL = set(STREAMING_NUMERICAL_FEATURES)


def transform_numeric(df: pd.DataFrame, params: TransformParams) -> np.ndarray:
    """Apply log1p + standard-scaling to the numeric block.

    Returns an (n_rows, n_numeric) float32 array aligned to
    `params.numerical_features`.
    """
    log_set = set(params.log1p_features)
    cols = []
    for feature in params.numerical_features:
        values = df[feature].to_numpy(dtype=np.float64)
        if feature in _STREAMING_FILL:
            values = np.nan_to_num(values, nan=0.0)
        if feature in log_set:
            values = np.log1p(values)
        cols.append(values)
    matrix = np.stack(cols, axis=1)
    mean = np.asarray(params.mean, dtype=np.float64)
    std = np.asarray(params.std, dtype=np.float64)
    # StandardScaler leaves zero-variance features unscaled (std stored as 0);
    # guard against divide-by-zero the same way (subtract mean, keep scale 1).
    safe_std = np.where(std == 0.0, 1.0, std)
    scaled = (matrix - mean) / safe_std
    return scaled.astype(np.float32)


def transform_categorical(df: pd.DataFrame, params: TransformParams) -> np.ndarray:
    """Map categorical columns to their StringIndexer indices.

    Unseen categories map to the OOV/keep index (len(vocab)), matching
    StringIndexer handleInvalid='keep'.
    """
    cols = []
    for feature in params.categorical_features:
        vocab = params.vocabs[feature]
        lookup = {label: i for i, label in enumerate(vocab)}
        oov = len(vocab)
        idx = df[feature].map(lambda v, lookup=lookup, oov=oov: lookup.get(v, oov))
        cols.append(idx.to_numpy(dtype=np.int64))
    return np.stack(cols, axis=1)


def transform(
    df: pd.DataFrame, params: TransformParams
) -> tuple[np.ndarray, np.ndarray]:
    """Return (numeric_array, categorical_index_array) for a batch."""
    return transform_numeric(df, params), transform_categorical(df, params)
