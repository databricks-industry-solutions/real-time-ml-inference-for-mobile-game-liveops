"""Rolling-window serving-observability metrics for the Serving Ops dashboard.

Records, server-side (where the calls actually happen), two things:
- INGESTION: timestamps of consumed login events -> events/sec over a window.
- SERVING OBSERVABILITY: per-call latency of the ranking endpoint (offer-ranker-rt)
  and, separately, the feature-serving endpoint (offer-features-fs). These are the
  TWO real latencies the app can measure client-of-the-endpoint side; we do NOT
  split inference-vs-lookup INSIDE the ranking call (that happens server-side in one
  round-trip and is reported as one honest end-to-end number).

Thread-safe (the Kafka consumer thread + request threads both write). Fixed-size
deques bound memory. Percentiles computed on read. No external deps.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque

_WINDOW_S = 60.0  # rolling window for rates + percentiles
# Safety backstop only (NOT a sampler): bounds memory if snapshot() is never
# called to prune. Set well above _WINDOW_S x any realistic call rate so the
# window is never truncated by count -- pruning is by AGE (see _prune), so every
# call within the window is always kept (no downsampling).
_SAFETY_MAX = 500_000

_lock = threading.Lock()
_ingest_ts: deque[float] = deque(maxlen=_SAFETY_MAX)  # consumed-login timestamps
_ranking: deque[tuple[float, float]] = deque(maxlen=_SAFETY_MAX)  # (ts, latency_ms)
_feature: deque[tuple[float, float]] = deque(maxlen=_SAFETY_MAX)  # (ts, latency_ms)
_started = time.time()
_inferences = 0  # cumulative count (not windowed)


def _prune(dq: deque, cutoff: float, keyed: bool) -> None:
    """Drop entries older than `cutoff` from the left (deques are append-right, so
    the oldest is leftmost). `keyed` -> entries are (ts, value); else bare ts."""
    while dq and (dq[0][0] if keyed else dq[0]) < cutoff:
        dq.popleft()


def record_ingest() -> None:
    """Mark one login event consumed from Kafka."""
    with _lock:
        _ingest_ts.append(time.time())


def record_ranking(latency_ms: float) -> None:
    """Record one ranking-endpoint (offer-ranker-rt) round-trip latency."""
    global _inferences
    with _lock:
        _ranking.append((time.time(), latency_ms))
        _inferences += 1


def record_feature(latency_ms: float) -> None:
    """Record one feature-serving-endpoint (offer-features-fs) round-trip latency."""
    with _lock:
        _feature.append((time.time(), latency_ms))


def _percentiles(values: list[float]) -> dict:
    """p50 / p95 / p99 / max over `values` (ms), or zeros if empty."""
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0}
    s = sorted(values)
    n = len(s)

    def pct(p: float) -> float:
        # nearest-rank percentile; clamp index to the last element.
        i = min(n - 1, math.ceil((p / 100.0) * n) - 1)
        return round(s[max(0, i)], 1)

    return {"p50": pct(50), "p95": pct(95), "p99": pct(99), "count": n}


def snapshot() -> dict:
    """Return the current metrics snapshot for /api/metrics."""
    now = time.time()
    cutoff = now - _WINDOW_S
    with _lock:
        # Prune by AGE so the deques hold exactly the window (bounded by rate x
        # window, never truncated by count -> no sampling), then read all of it.
        _prune(_ingest_ts, cutoff, keyed=False)
        _prune(_ranking, cutoff, keyed=True)
        _prune(_feature, cutoff, keyed=True)
        recent_ingest = list(_ingest_ts)
        rank_lat = [ms for (_, ms) in _ranking]
        feat_lat = [ms for (_, ms) in _feature]
        total_inf = _inferences
        uptime = now - _started
    win = _WINDOW_S
    return {
        "window_s": win,
        "uptime_s": round(uptime, 1),
        "app_compute": _app_compute(),
        "ingestion": {
            "events_per_sec": round(len(recent_ingest) / win, 1),
            "window_events": len(recent_ingest),
        },
        "ranking_endpoint": {
            "name": "offer-ranker-rt",
            "calls_per_sec": round(len(rank_lat) / win, 1),
            "total_inferences": total_inf,
            "latency_ms": _percentiles(rank_lat),
        },
        "feature_endpoint": {
            "name": "offer-features-fs",
            "calls_per_sec": round(len(feat_lat) / win, 1),
            "latency_ms": _percentiles(feat_lat),
        },
    }


def _app_compute() -> dict:
    """Return the APP container's own CPU saturation (not the serving endpoint's).

    Databricks Apps expose no container-CPU metrics API, so measure it in-process:
    os.getloadavg() (1-min run-queue length) normalized by CPU count is the honest
    saturation signal -- load_per_cpu >= ~1.0 means the app's vCPUs are the
    bottleneck (the scorer workers are CPU/GIL-bound, not waiting on the endpoint).
    No extra dependency (os.getloadavg is stdlib; unavailable on Windows -> None).
    """
    ncpu = os.cpu_count() or 1
    try:
        load1 = os.getloadavg()[0]
        return {
            "ncpu": ncpu,
            "load1": round(load1, 2),
            "load_per_cpu": round(load1 / ncpu, 2),
        }
    except (OSError, AttributeError):
        return {"ncpu": ncpu, "load1": None, "load_per_cpu": None}
