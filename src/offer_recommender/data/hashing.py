"""Cross-engine deterministic uniform hashing.

md5-based so numpy (hashlib) and Spark SQL (md5/conv) produce BIT-IDENTICAL
uniforms for the same key -- verified 0.0 max diff. This is the primitive that
lets the impression-label side (numpy, generate.py) and the Kafka-stream side
(Spark, match_backfill.py) agree on the SAME session events without any runtime
data handoff. Do NOT swap md5 for xxhash / Spark hash() -- those diverge across
engines (integer width + ANSI overflow).
"""

from __future__ import annotations

import hashlib

import numpy as np

_DIVISOR = float(16**8)  # 4294967296.0; scales an 8-hex-digit int into [0,1)
SPARK_MD5_UNIFORM_SQL = "conv(substring(md5({col}),1,8),16,10)/4294967296.0"


def md5_uniform_str(key: str) -> float:
    """Return a deterministic uniform in [0,1) for one key (matches Spark)."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / _DIVISOR


def md5_uniform_np(keys: np.ndarray) -> np.ndarray:
    """Vectorized md5_uniform over an array of string keys."""
    return np.array([md5_uniform_str(str(k)) for k in keys], dtype=np.float64)


def md5_seed(key: str) -> int:
    """Deterministic 64-bit integer seed from a key (for np.random.default_rng).

    Uses the full 128-bit md5 truncated to 64 bits. Lets every player derive an
    INDEPENDENT numpy Generator (`default_rng(md5_seed(f"{player_id}:{seed}"))`), so
    per-player generation is reproducible AND shardable across Spark executors with
    no cross-player RNG state -- the basis for distributing generate_data off-driver.
    """
    return int(hashlib.md5(key.encode()).hexdigest()[:16], 16)
