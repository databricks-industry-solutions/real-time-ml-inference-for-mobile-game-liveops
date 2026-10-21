"""Client for the offer-ranker Model Serving endpoint.

The app sends a player_id plus the candidate-offer slate; the endpoint resolves
the player's offline + real-time features from the online store (via the model's
embedded FeatureSpec) and returns expected revenue per candidate. Auth uses the
app's service principal via the Databricks SDK (no tokens in code).
"""

from __future__ import annotations

import contextlib
import datetime
import logging
import os
import time

import metrics

logger = logging.getLogger(__name__)

ENDPOINT_NAME = os.environ.get("SERVING_ENDPOINT_NAME", "offer-ranker-rt")
FEATURE_ENDPOINT_NAME = os.environ.get("FEATURE_ENDPOINT_NAME", "offer-features-fs")

_CLIENT = None


def _client():
    """Return a cached authenticated WorkspaceClient (app service principal).

    Built once and reused: constructing a WorkspaceClient re-runs auth resolution
    (profile read + token mint/refresh) each time, which dominated the per-request
    latency when it was done on every call.

    Sized for the scorer's concurrent fan-out. The SDK's HTTP layer defaults
    `max_connections_per_pool` to None -> requests' adapter default of 10, WITH
    `pool_block=True`: so >10 concurrent serving calls BLOCK on the connection pool,
    capping inference at ~10 / round-trip (~155/sec at 65ms) no matter how many scorer
    workers run. Raising the pool above the worker count (_SCORER_WORKERS=80 in app.py)
    lets every worker's call go out in parallel, so throughput tracks the ingestion
    rate instead of the connection pool.

    TWO query paths on this one client (intentional):
    - rank_offers -> `serving_endpoints_data_plane.query`: offer-ranker-rt is
      ROUTE-OPTIMIZED (bypasses the ~200 QPS/workspace model-serving-proxy cap for the
      high-QPS board), which REQUIRES the data-plane query (endpoint-scoped OAuth +
      distinct URL; the SDK handles it transparently for the app SP).
    - fetch_features -> `serving_endpoints.query`: offer-features-fs is STANDARD (the
      drill-down is a few req/sec, far under the proxy cap), so the normal query path.
    """
    global _CLIENT
    if _CLIENT is None:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.config import Config

        # 128 > _SCORER_WORKERS (80) + drill-down/feature/metrics calls, so no query
        # ever waits on the connection pool. One pool (single endpoint host).
        cfg = Config(max_connection_pools=8, max_connections_per_pool=128)
        _CLIENT = WorkspaceClient(config=cfg)
    return _CLIENT


# Real-time feature columns the endpoint accepts as request-time passthrough
# (mirrors schema.REALTIME_NUMERICAL_FEATURES; duplicated so the app has no ML
# deps). Sent from the consumed login event so the prediction reflects the
# login's live state without an online-store lookup.
_REALTIME_FEATURE_KEYS = (
    "gem_balance",
    "minutes_since_last_purchase",
    "match_count",
)


def _build_records(
    player_id: str, candidate_offers: list[dict], realtime: dict | None
) -> list[dict]:
    """Build the dataframe_records payload: one row per candidate offer, each
    carrying player_id + event_ts + the login's real-time passthrough values +
    offer cols.

    `event_ts` is in the model's serving signature (optional there, but MLflow
    schema enforcement rejects the frame when the column is absent), so we pass the
    login's event_ts through. The FS auto-lookup serves LATEST online values, not
    point-in-time, so the exact timestamp does not change the looked-up features; it
    only satisfies the signature. Falls back to now() if the login lacks it."""
    rt = (
        {k: realtime[k] for k in _REALTIME_FEATURE_KEYS if k in realtime}
        if realtime
        else {}
    )
    # The model signature wants event_ts as a tz-NAIVE datetime64[ns]. The login
    # carries it as an ISO/'Z' string (producer emits "yyyy-MM-dd'T'HH:mm:ss'Z'"), and
    # passing that straight through makes MLflow's serving conversion fail with
    # "cannot supply both a tz and a timezone-naive dtype". Normalize to
    # "YYYY-MM-DD HH:MM:SS" (drop the 'T'/'Z'/fractional). It only satisfies the
    # signature -- the FS auto-lookup serves LATEST, so the exact value is irrelevant.
    _raw_ts = (realtime or {}).get("event_ts")
    event_ts = (
        str(_raw_ts).replace("T", " ")[:19]
        if _raw_ts
        else datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
    )
    return [
        {"player_id": player_id, "event_ts": event_ts, **rt, **offer}
        for offer in candidate_offers
    ]


def _apply_predictions(candidate_offers: list[dict], predictions) -> list[dict]:
    """Annotate offers with the endpoint's predictions, sorted by revenue desc."""
    scored = []
    for offer, pred in zip(candidate_offers, predictions, strict=False):
        if isinstance(pred, dict):
            revenue = pred.get("expected_offer_revenue")
            purchase_prob = pred.get("purchase_probability")
        else:
            revenue, purchase_prob = pred, None
        scored.append(
            {
                **offer,
                "expected_offer_revenue": revenue,
                "purchase_probability": purchase_prob,
            }
        )
    scored.sort(key=lambda o: o["expected_offer_revenue"] or 0.0, reverse=True)
    return scored


# --- Async inference path (the high-QPS scorer) ------------------------------
# The SDK is blocking (requests-based): scoring the login stream through
# serving_endpoints.query means one blocked OS thread per in-flight call, and past
# ~80 threads the GIL thrash starves the asyncio event loop (kafka "blocking the
# event loop" warnings) -- throughput plateaus ~280/sec. So the SCORER fires its
# inference POSTs with httpx.AsyncClient as real coroutines on the loop; the SDK is
# used ONLY to resolve the route-optimized data-plane URL + mint the endpoint-scoped
# OAuth token (both cached below). Route-opt REQUIRES that data-plane URL + a token
# with authorization_details scoped to the endpoint; a plain workspace token is
# rejected, so we reuse the SDK's own token source rather than hand-roll it.
_ASYNC_CLIENT = None
# (url, token, expires_at_epoch) for the route-opt data-plane target. The
# endpoint-scoped token is short-lived, so cache it and refresh a bit before expiry.
_DP_TARGET: tuple[str, str, float] | None = None
_DP_REFRESH_SKEW_S = 60.0


def _async_client():
    """Cached httpx.AsyncClient sized for the scorer's concurrent fan-out.

    One long-lived client (per httpx guidance: never create a client per request --
    it defeats connection pooling). Connection-pool limit set above the scorer's
    concurrency so no coroutine waits on a connection.

    http2=True is the key knob for sustained concurrency: over HTTP/1.1, N concurrent
    requests need N separate TCP+TLS connections, and each TLS handshake is synchronous
    crypto on the event-loop thread -- at a few hundred new handshakes/sec that
    serializes the loop (coroutines reach the endpoint in ~15ms but aren't rescheduled
    to read the response for ~500ms -- the observed client-vs-platform latency gap).
    Route-optimized serving endpoints speak HTTP/2, which MULTIPLEXES many concurrent
    requests over a few long-lived connections, so there is no per-request handshake
    to block the loop. Requires the h2 package (httpx[http2] in requirements.txt).
    """
    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is None:
        import httpx

        limits = httpx.Limits(max_connections=512, max_keepalive_connections=512)
        _ASYNC_CLIENT = httpx.AsyncClient(
            http2=True, limits=limits, timeout=httpx.Timeout(30.0, connect=10.0)
        )
    return _ASYNC_CLIENT


def _data_plane_target() -> tuple[str, str]:
    """Return (invocations_url, bearer_token) for the route-optimized endpoint.

    Resolves via the SDK's data-plane machinery (the same path
    serving_endpoints_data_plane.query uses): _data_plane_info_query gives the
    data-plane URL + authorization_details, and the SDK's data-plane token source
    mints the endpoint-scoped OAuth token. Cached with a refresh skew so the token
    is renewed before it expires -- the resolve is a control-plane round-trip we do
    NOT want on every inference call.
    """
    global _DP_TARGET
    now = time.time()
    if _DP_TARGET is not None and now < _DP_TARGET[2] - _DP_REFRESH_SKEW_S:
        return _DP_TARGET[0], _DP_TARGET[1]
    dp = _client().serving_endpoints_data_plane
    info = dp._data_plane_info_query(name=ENDPOINT_NAME)
    tok = dp._dpts.token(info.endpoint_url, info.authorization_details)
    # tok.expiry is a datetime; fall back to a 5-min TTL if unavailable.
    expires_at = now + 300.0
    expiry = getattr(tok, "expiry", None)
    if expiry is not None:
        with contextlib.suppress(AttributeError, TypeError):
            expires_at = expiry.timestamp()
    _DP_TARGET = (info.endpoint_url, f"{tok.token_type} {tok.access_token}", expires_at)
    return _DP_TARGET[0], _DP_TARGET[1]


async def rank_offers_async(
    player_id: str, candidate_offers: list[dict], realtime: dict | None = None
) -> tuple[list[dict], float]:
    """Async twin of rank_offers for the high-QPS scorer (coroutine, not a thread).

    Same contract as rank_offers -- returns (sorted_offers, query_ms) and degrades
    to unscored offers on any error -- but issues the inference POST via
    httpx.AsyncClient so hundreds of calls overlap on the event loop without a thread
    each. Targets the route-optimized data-plane URL with the endpoint-scoped token.
    """
    records = _build_records(player_id, candidate_offers, realtime)
    started = time.perf_counter()
    try:
        # STANDARD serving invocations URL, NOT the route-optimized data-plane path:
        # offer-ranker-rt is NOT route-optimized on this workspace, so the data-plane
        # query is rejected ("Resource does not support direct Data Plane access").
        # The standard proxy path works on any endpoint; cfg.authenticate() mints and
        # refreshes the app-SP bearer. httpx still issues these as coroutines so the
        # loop is never thread-blocked (the proxy caps ~200 QPS/workspace -- ample for
        # the board once the producer rate is kept under that).
        cfg = _client().config
        url = f"{cfg.host.rstrip('/')}/serving-endpoints/{ENDPOINT_NAME}/invocations"
        resp = await _async_client().post(
            url,
            json={"dataframe_records": records},
            headers=cfg.authenticate(),
        )
        if resp.status_code >= 400:
            logger.warning(
                "Async serving %s body: %s", resp.status_code, resp.text[:600]
            )
        resp.raise_for_status()
        predictions = resp.json().get("predictions", [])
    except Exception as e:
        logger.warning("Async serving query failed: %s", e)
        return [{**o, "expected_offer_revenue": None} for o in candidate_offers], 0.0
    query_ms = round((time.perf_counter() - started) * 1000, 1)
    metrics.record_ranking(query_ms)
    return _apply_predictions(candidate_offers, predictions), query_ms


def rank_offers(
    player_id: str, candidate_offers: list[dict], realtime: dict | None = None
) -> tuple[list[dict], float]:
    """Score candidate offers for a player; return (sorted_offers, query_ms).

    Each candidate dict carries the offer-side columns plus the entity key /
    timestamp the endpoint needs. `realtime` (the login event's real-time feature
    values) is merged into every record as request-time passthrough, so the
    prediction reflects the login's live state (millisecond-fresh) instead of a
    seconds-stale online-store lookup. Returns the candidates annotated with
    `expected_offer_revenue` (highest first) and the endpoint round-trip latency
    in ms, measured around the `.query()` call ONLY. On any serving error, returns
    the candidates unscored so the UI degrades gracefully.
    """
    records = _build_records(player_id, candidate_offers, realtime)
    started = time.perf_counter()
    try:
        response = _client().serving_endpoints.query(
            name=ENDPOINT_NAME, dataframe_records=records
        )
        predictions = response.predictions
    except Exception as e:
        logger.warning("Serving query failed: %s", e)
        return [{**o, "expected_offer_revenue": None} for o in candidate_offers], 0.0
    query_ms = round((time.perf_counter() - started) * 1000, 1)
    metrics.record_ranking(query_ms)
    return _apply_predictions(candidate_offers, predictions), query_ms


def _histogram_quantile(
    buckets: list[tuple[float, float]], total: float, q: float
) -> float:
    """Linear-interpolated quantile from cumulative Prometheus histogram buckets.

    `buckets` is [(upper_bound_ms, cumulative_count)] sorted ascending (the last
    bound is +inf). Mirrors Prometheus histogram_quantile: find the bucket the
    rank falls in and interpolate within its [lower, upper] bound.
    """
    if total <= 0 or not buckets:
        return 0.0
    rank = q * total
    prev_bound, prev_count = 0.0, 0.0
    for bound, cum in buckets:
        if cum >= rank:
            if bound == float("inf"):
                return round(prev_bound, 1)
            span = cum - prev_count
            frac = (rank - prev_count) / span if span > 0 else 0.0
            return round(prev_bound + (bound - prev_bound) * frac, 1)
        prev_bound, prev_count = bound, cum
    return round(prev_bound, 1)


def _parse_prometheus(text: str) -> dict:
    """Parse the serving-endpoint Prometheus feed into server-side ops metrics.

    Extracts the per-minute request_latency_ms histogram (-> p50/p90/p99), request
    count (-> QPS over the 60s window), 4xx/5xx error counts, and cpu/mem gauges.
    Returns zeros for anything absent (a cold or never-queried endpoint).
    """
    buckets: list[tuple[float, float]] = []
    scalars: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            name = line[: line.index("{")] if "{" in line else line.split()[0]
            value = (
                float(line.rsplit(maxsplit=2)[-2])
                if line.count(" ") >= 2
                else float(line.split()[-1])
            )
        except (ValueError, IndexError):
            continue
        if name == "request_latency_ms_bucket":
            le = line[line.index('le="') + 4 : line.index('"', line.index('le="') + 4)]
            buckets.append((float("inf") if le == "+Inf" else float(le), value))
        else:
            scalars[name] = value
    buckets.sort()
    total = scalars.get("request_latency_ms_count", 0.0)
    return {
        "latency_ms": {
            "p50": _histogram_quantile(buckets, total, 0.50),
            "p90": _histogram_quantile(buckets, total, 0.90),
            "p99": _histogram_quantile(buckets, total, 0.99),
            "count": int(total),
        },
        "requests_per_sec": round(scalars.get("request_count_total", 0.0) / 60.0, 1),
        "error_count": int(
            scalars.get("request_4xx_count_total", 0.0)
            + scalars.get("request_5xx_count_total", 0.0)
        ),
        "cpu_pct": round(scalars.get("cpu_usage_percentage", 0.0), 1),
        "mem_pct": round(scalars.get("mem_usage_percentage", 0.0), 1),
    }


# Prometheus gauges reset per minute; cache scrapes briefly so a 2s dashboard
# poll doesn't hammer the metrics API. (ts, payload) per endpoint.
_PLATFORM_CACHE: dict[str, tuple[float, dict]] = {}
_PLATFORM_TTL_S = 12.0


def platform_metrics(endpoint: str) -> dict | None:
    """Scrape an endpoint's server-side Prometheus metrics (platform-measured).

    This is the latency Databricks measures through the full backend route (the
    ranking endpoint's total includes its internal online feature lookup, which is
    NOT exposed as a separate sub-metric). Richer and more authoritative than the
    app's client round-trip. Cached ~12s. Returns None on any failure.
    """
    now = time.time()
    cached = _PLATFORM_CACHE.get(endpoint)
    if cached and now - cached[0] < _PLATFORM_TTL_S:
        return cached[1]
    try:
        resp = _client().api_client.do(
            "GET",
            f"/api/2.0/serving-endpoints/{endpoint}/metrics",
            headers={"Accept": "text/plain; version=0.0.4; charset=utf-8"},
            raw=True,
        )
        raw = resp["contents"].read() if isinstance(resp, dict) else resp
        text = raw.decode() if isinstance(raw, bytes) else str(raw)
    except Exception as e:
        logger.warning("Prometheus scrape failed for %s: %s", endpoint, e)
        return None
    parsed = _parse_prometheus(text)
    _PLATFORM_CACHE[endpoint] = (now, parsed)
    return parsed


def fetch_features(player_id: str) -> dict | None:
    """Fetch a player's online feature values from the Feature Serving endpoint.

    Queries `offer-features-fs` by primary key over REST via the same SDK client
    used for model serving (no extra deps): the feature store serves the looked-up
    values, so the app never touches Lakebase/tables directly. Returns a flat
    `{feature_name: value}` dict, or None if the endpoint is unreachable (caller
    falls back to synthetic values so the app still renders locally).
    """
    started = time.perf_counter()
    try:
        resp = _client().serving_endpoints.query(
            name=FEATURE_ENDPOINT_NAME,
            dataframe_records=[{"player_id": player_id}],
        )
    except Exception as e:
        logger.warning("Feature Serving query failed for %s: %s", player_id, e)
        return None
    metrics.record_feature(round((time.perf_counter() - started) * 1000, 1))
    # A Feature Serving endpoint returns the looked-up values under `outputs`
    # (a model endpoint uses `predictions`); accept either.
    rows = getattr(resp, "outputs", None) or getattr(resp, "predictions", None)
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return rows[0]
    logger.warning("Feature Serving returned unexpected shape for %s", player_id)
    return None
