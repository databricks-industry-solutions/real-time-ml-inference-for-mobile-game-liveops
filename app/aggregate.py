"""Global, server-side aggregate of scored login decisions.

ONE window for ALL viewers: the board's totals/leaderboard/segment-mix/recent-feed
live here, not in each browser. A single background scorer (app._scorer_loop)
records every scored event into this module; the browser just polls /api/board and
renders the snapshot. So a refresh (or a second viewer) sees the SAME running
aggregate, never a "new session" -- the accumulation persists for the app process's
lifetime, independent of who is connected.

Thread-safe (the scorer runs in a worker thread; requests read concurrently).
Single-replica assumption: the Databricks App runs one uvicorn process, so this
in-process singleton IS the global state. (Multi-replica would need shared storage.)
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque

# Newest-first ring of scored decisions to drill into. At ~1100 logins/sec this ring
# spans ~22s of traffic. The board feed scans the full ring to find the ~1% bursty
# players (is_bursty==True) whose 30s match-event burst populates their SFV windows;
# without a full scan, only ~2 bursty candidates appear in a 200-event slice.
_RECENT_MAX = 24000
_RATE_WINDOW_S = 60.0  # window for the LIVE inferences/min rate
_RATE_MAX = 40000  # cap on rate-timestamp ring (>> 300/s x 60s), memory backstop
_PROB_BUCKET_S = 1.0  # one median-purchase-prob point per second (matches the poll)
_PROB_SERIES_MAX = 300  # ~5 min of 1s points on the chart

_lock = threading.Lock()
_captured = 0.0  # summed expected revenue of chosen offers
_count = 0  # total decisions scored
_offers: dict[str, dict] = {}  # offer_id -> {offer_id, count, revenue}
_segments: dict[str, int] = {}  # segment -> count
_latest: dict | None = None  # most recent scored event (the "now serving")
_recent: deque[dict] = deque(maxlen=_RECENT_MAX)
_first_ts: float | None = None  # when the first event landed (for uptime)
# Timestamps of recent scored events -> a LIVE (last-60s) rate, not a lifetime
# average. The login rate wanders (100-300/s band), so a cumulative avg would lag
# reality; a windowed rate reflects current throughput.
_rate_ts: deque[float] = deque(maxlen=_RATE_MAX)
# Per-_PROB_BUCKET_S time series (shared bucket window, so both are time-aligned):
# - _prob_series: median chosen-offer purchase probability. (end_ts, median).
# - _share_series: per-offer share of recommendations in the bucket. (end_ts, {offer_id: frac}).
_prob_series: deque[tuple[float, float]] = deque(maxlen=_PROB_SERIES_MAX)
_share_series: deque[tuple[float, dict]] = deque(maxlen=_PROB_SERIES_MAX)
_bucket_start: float | None = None
_prob_bucket_vals: list[float] = []
_share_bucket_counts: dict[str, int] = {}


def _roll_buckets(now: float) -> None:
    """Finalize any elapsed buckets into the prob + offer-share series (they share the
    bucket window so points line up on the x-axis). Caller holds the lock. Appends one
    point per fully-elapsed bucket that had samples."""
    global _bucket_start, _prob_bucket_vals, _share_bucket_counts
    if _bucket_start is None:
        _bucket_start = now
        return
    while now >= _bucket_start + _PROB_BUCKET_S:
        end = _bucket_start + _PROB_BUCKET_S
        if _prob_bucket_vals:
            _prob_series.append((end, statistics.median(_prob_bucket_vals)))
        if _share_bucket_counts:
            total = sum(_share_bucket_counts.values())
            _share_series.append(
                (end, {k: v / total for k, v in _share_bucket_counts.items()})
            )
        _prob_bucket_vals = []
        _share_bucket_counts = {}
        _bucket_start = end


def record(event: dict) -> None:
    """Fold one scored decision into the global aggregate.

    Mirrors the old client-side reducer: error events are ignored; the
    `chosen_offer.expected_offer_revenue` truthiness guard both keeps $0/null-revenue
    offers off the leaderboard and prevents NaN in `captured`.
    """
    global _captured, _count, _latest, _first_ts
    if event.get("error"):
        return
    with _lock:
        now = time.time()
        if _first_ts is None:
            _first_ts = now
        _count += 1
        _rate_ts.append(now)
        event["scored_at"] = now  # wall-clock the login was scored (feed recency gate)
        _latest = event
        _recent.appendleft(event)
        co = event.get("chosen_offer")
        # Feed the per-bucket time series (median purchase prob + offer share) from
        # the chosen offer; both share the bucket window so their points align.
        _roll_buckets(now)
        if co and co.get("purchase_probability") is not None:
            _prob_bucket_vals.append(co["purchase_probability"])
        if co and co.get("offer_id"):
            oid = co["offer_id"]
            _share_bucket_counts[oid] = _share_bucket_counts.get(oid, 0) + 1
        if co and co.get("expected_offer_revenue"):
            rev = co["expected_offer_revenue"]
            _captured += rev
            row = _offers.get(co["offer_id"]) or {
                "offer_id": co["offer_id"],
                "count": 0,
                "revenue": 0.0,
            }
            row["count"] += 1
            row["revenue"] += rev
            _offers[co["offer_id"]] = row
        seg = event.get("segment")
        if seg:
            _segments[seg] = _segments.get(seg, 0) + 1


def snapshot(recent_n: int = 60) -> dict:
    """Return the current global board state for /api/board.

    `rate` is a LIVE inferences/MINUTE over the last _RATE_WINDOW_S (not a lifetime
    average), so it tracks the wandering login rate. Until uptime reaches the full
    window we annualize over the actual elapsed time so the early reading isn't
    understated. `recent` is newest-first, capped at `recent_n`.
    """
    now = time.time()
    cutoff = now - _RATE_WINDOW_S
    with _lock:
        while _rate_ts and _rate_ts[0] < cutoff:
            _rate_ts.popleft()
        _roll_buckets(now)  # keep the series current even if traffic is sparse
        elapsed = min(_RATE_WINDOW_S, now - _first_ts) if _first_ts else 0.0
        rate = (len(_rate_ts) / elapsed) * 60.0 if elapsed > 1.0 else 0.0
        return {
            "count": _count,
            "rate": round(rate),
            "offers": {k: dict(v) for k, v in _offers.items()},
            "segments": dict(_segments),
            "latest": _latest,
            "recent": list(_recent)[:recent_n],
            # Median chosen-offer purchase probability over time: [{t, median}],
            # oldest->newest. t is epoch seconds (bucket end); the UI formats clock labels.
            "purchase_prob_series": [
                {"t": round(t, 1), "median": round(m, 4)} for (t, m) in _prob_series
            ],
            # Per-offer recommendation share over time: [{t, shares:{offer_id: frac}}].
            # Shares within a bucket sum to 1.0 (each login recommends exactly one offer).
            "offer_share_series": [
                {"t": round(t, 1), "shares": {k: round(v, 4) for k, v in sh.items()}}
                for (t, sh) in _share_series
            ],
            "uptime_s": round(now - _first_ts, 1) if _first_ts else 0.0,
        }


def find_recent(player_id: str) -> dict | None:
    """Return this player's most recent REAL scored decision from the recent ring,
    or None if they haven't been scored in the retained window.

    The drill-down uses this so it shows the ACTUAL prediction + the real login
    realtime values that produced it -- never a re-score with invented inputs
    (the realtime passthrough features exist only in the live login).
    """
    with _lock:
        for event in _recent:  # newest-first
            if event.get("player_id") == player_id:
                return event
    return None
