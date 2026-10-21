"""Nordfall offer recommender: FastAPI backend for the live inference console.

Thin routing layer over three surfaces:

- GET /api/stream: SSE feed of simulated shop-open events. For each event the
  backend calls the real-time serving endpoint to rank the candidate offers and
  streams the decision (chosen top-1, per-offer scores, latency).
- GET /api/player/{player_id}: drill-down showing the feature vector split into
  offline vs real-time features and the ranked offers.
- GET /api/run: the run panel (model version, serving/online-store status).

Business logic lives in serving_client (endpoint calls), catalog (offer slate)
and lakebase (online-store status). The React SPA is served from static/.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aggregate
import catalog
import login_consumer
import metrics
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from serving_client import (
    ENDPOINT_NAME,
    FEATURE_ENDPOINT_NAME,
    platform_metrics,
    rank_offers,
    rank_offers_async,
)
from starlette.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# httpx logs EVERY request at INFO ("HTTP Request: POST ... 200 OK"). The scorer fires
# hundreds/sec, which floods the app log + burns stdout I/O on the hot path. Silence it
# to WARNING (errors still surface); the /api/metrics latency/throughput is the real
# signal, not per-request log lines.
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI(title="Nordfall Offer Recommender")

# Start the Kafka login consumer (no-op unless RECO_KAFKA_BOOTSTRAP is set). When
# active, the live board advances on real streamed logins; otherwise it falls back
# to random players so the demo still runs with no external broker.
login_consumer.start()

# Default-loop executor threads. Since the async refactor almost nothing runs here:
# the inference is an httpx coroutine (loop, no thread), and Kafka is drained by ONE
# dedicated bridge thread (see _drain_kafka), not one-thread-per-worker. The executor
# now only serves the drill-down's sync feature lookup + the metrics scrape (a handful
# of concurrent request handlers). A small pool suffices; it is NOT a throughput lever
# (the scorer COROUTINES are -- see _SCORER_WORKERS).
_SERVING_POOL_SIZE = 32


@app.on_event("startup")
async def _size_thread_pool() -> None:
    """Enlarge the default-loop executor so the scorer coroutines' blocking Kafka
    queue-gets (next_login) + the sync drill-down lookups all get a thread promptly
    (the running loop, hence the executor, only exists inside the app lifecycle)."""
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=_SERVING_POOL_SIZE))
    logger.info("Serving thread pool sized to %d workers", _SERVING_POOL_SIZE)


STATIC_DIR = Path(__file__).parent / "static"

# Non-feature keys the login carries for routing/context. Stripped from the retained
# `realtime` snapshot so the drill-down shows ONLY the model's passthrough features
# (player_id is already the drill-down header; event_ts/event_type are not features).
_NON_FEATURE_KEYS = ("player_id", "event_ts", "event_type", "is_bursty")


def _feature_snapshot(realtime: dict | None) -> dict | None:
    """Return the login's realtime values with non-feature keys stripped, or None."""
    if not realtime:
        return None
    snap = {k: v for k, v in realtime.items() if k not in _NON_FEATURE_KEYS}
    return snap or None


def _rank_event(player_id: str, realtime: dict | None = None) -> dict:
    """Rank the offer slate for one player and build a feed event.

    `realtime` carries the login event's real-time feature values (gem_balance,
    match_count, ...); they are passed through into the serving request
    (request-time passthrough, millisecond-fresh) rather than looked up from the
    online store. None -> the endpoint gets no real-time overrides (drill-down/
    off-stream path).

    The `realtime` values used are RETAINED on the event so the drill-down can show
    the REAL point-in-time session snapshot that produced this scored decision --
    those values live only in the login (RequestSource passthrough, never
    materialized online), so this is the only honest source for them.
    """
    slate = catalog.candidate_slate(player_id)
    scored, latency_ms = rank_offers(player_id, slate, realtime=realtime)
    chosen = scored[0] if scored else None
    return {
        "player_id": player_id,
        "segment": catalog.player_segment(player_id),
        "chosen_offer": chosen,
        "ranked_offers": scored[:5],
        "latency_ms": latency_ms,
        "is_bursty": bool(realtime.get("is_bursty")) if realtime else False,
        "realtime": _feature_snapshot(realtime),
    }


async def _rank_event_async(player_id: str, realtime: dict | None = None) -> dict:
    """Async twin of _rank_event for the scorer: awaits rank_offers_async so the
    inference call is a coroutine on the event loop, not a blocked thread. The slate
    build + segment lookup are cheap in-memory ops, safe to run inline in the loop."""
    slate = catalog.candidate_slate(player_id)
    scored, latency_ms = await rank_offers_async(player_id, slate, realtime=realtime)
    chosen = scored[0] if scored else None
    return {
        "player_id": player_id,
        "segment": catalog.player_segment(player_id),
        "chosen_offer": chosen,
        "ranked_offers": scored[:5],
        "latency_ms": latency_ms,
        "is_bursty": bool(realtime.get("is_bursty")) if realtime else False,
        "realtime": _feature_snapshot(realtime),
    }


# Number of PERSISTENT scorer worker COROUTINES (not threads). Each awaits its
# inference call via httpx/HTTP2 (rank_offers_async), so N workers overlap N in-flight
# calls on the single event loop. SIZING (measured on LARGE, 2026-08-01): throughput
# plateaued at ~450/sec at BOTH 150 and 400 workers -- only LATENCY changed (150 ->
# 310ms, 400 -> 720ms). That is a single-EVENT-LOOP ceiling (one Python thread juggling
# all in-flight httpx calls; server-side is 14ms, so the ~450/sec + growing latency is
# loop-scheduling overhead, NOT CPU (load ~0.35 on 8 cores) or the endpoint). So MORE
# workers = pure in-flight QUEUEING = worse latency, same throughput. Keep it near the
# knee: ~64 workers x the ~150ms round-trip ~= the loop's ~450/sec ceiling with MINIMAL
# queueing, so latency stays low. Genuinely exceeding ~450/sec needs multiple event
# loops (multi-process uvicorn), which needs the global aggregate moved to shared
# storage first. uvloop (below) raises the single-loop ceiling ~2-4x as a cheaper lever.
_SCORER_WORKERS = 64


async def _score_and_record(login: dict) -> None:
    """Score one login (async inference) and record it into the global aggregate."""
    event = await _rank_event_async(login["player_id"], login)
    event["source"] = "login-stream"
    aggregate.record(event)


# Bounded hand-off queue between the single Kafka bridge (one thread) and the N
# scorer coroutines. Bounded so a scoring slowdown applies BACKPRESSURE (the bridge
# blocks on put) instead of letting an unbounded backlog build -- an earlier design
# where every coroutine did its own blocking next_login parked N threads and the
# backlog grew without bound (latency ran away to multi-second while CPU sat idle).
_LOGIN_QUEUE_MAX = 2000
_login_aqueue: asyncio.Queue[dict] | None = None


def _enqueue_drop_oldest(aq: asyncio.Queue, login: dict) -> None:
    """Runs ON the event loop: enqueue a login, dropping the oldest if the queue is
    full. Never raises (a bare put_nowait would raise QueueFull, and asyncio logs the
    exception with the queue's FULL repr -- thousands of dicts -- which floods the log
    and stalls the app). Drop-oldest = bounded memory + newest-events-win, no flood."""
    if aq.full():
        with contextlib.suppress(asyncio.QueueEmpty):
            aq.get_nowait()
    # racing producer may have refilled it; drop this one rather than raise/log payload.
    with contextlib.suppress(asyncio.QueueFull):
        aq.put_nowait(login)


def _drain_kafka(loop: asyncio.AbstractEventLoop, aq: asyncio.Queue) -> None:
    """Blocking bridge (ONE background thread): pull logins from the consumer and hand
    them to the asyncio queue the scorer coroutines await. This is the only thread
    parked on the Kafka get; the scorers never touch a thread.

    Drains in BATCHES (next_batch), not one-at-a-time: a per-login next_login +
    call_soon_threadsafe hop capped the FEED at ~190/sec (one loop-hop per item),
    starving the scorers regardless of how many coroutines waited. next_batch grabs
    everything already queued in one blocking call, and we schedule the whole batch
    with a SINGLE call_soon_threadsafe, so the feed rate is bounded by the login
    stream, not by per-item loop-hop overhead. Enqueue drops oldest on overflow (the
    board wants fresh events), so a scoring slowdown sheds backlog without a flood.
    """

    def _enqueue_batch(batch: list[dict]) -> None:
        for login in batch:
            _enqueue_drop_oldest(aq, login)

    while True:
        batch = login_consumer.next_batch(500, 30.0)
        if batch:
            loop.call_soon_threadsafe(_enqueue_batch, batch)


async def _scorer_worker(aq: asyncio.Queue) -> None:
    """One scorer coroutine: await a login from the async queue, score it (awaited
    httpx inference -- no thread), record it, forever. N of these overlap N in-flight
    inference calls cheaply on the one event loop. A slow call stalls only itself;
    the shared bounded queue balances work across all workers. Never dies."""
    while True:
        try:
            login = await aq.get()
            await _score_and_record(login)
        except Exception as e:
            logger.warning("scorer worker iteration failed: %s", e)
            await asyncio.sleep(1)


_scorer_tasks: list[asyncio.Task] = []


@app.on_event("startup")
async def _start_scorer() -> None:
    """Start the Kafka->asyncio bridge (one thread) + the pool of scorer coroutines.

    Decoupled producer/consumer: one thread drains Kafka into a bounded asyncio queue;
    N coroutines await that queue and score via async httpx. Keeps module refs so the
    tasks are not garbage-collected mid-run.
    """
    global _scorer_tasks, _login_aqueue
    loop = asyncio.get_running_loop()
    _login_aqueue = asyncio.Queue(maxsize=_LOGIN_QUEUE_MAX)
    # The single blocking Kafka bridge runs in a thread (not the loop).
    loop.run_in_executor(None, _drain_kafka, loop, _login_aqueue)
    _scorer_tasks = [
        asyncio.create_task(_scorer_worker(_login_aqueue))
        for _ in range(_SCORER_WORKERS)
    ]
    logger.info(
        "Global scorer started: 1 kafka bridge + %d async scorer coroutines",
        _SCORER_WORKERS,
    )


# At 1% bursty fraction, a 200-event slice yields ~2 bursty candidates on average
# and ~13.5% of polls yield zero. Scan the full ring so we always have tens of
# bursty candidates to show before applying the display cap.
_FEED_SCAN = aggregate._RECENT_MAX


@app.get("/api/board")
async def board(recent: int = 60) -> JSONResponse:
    """Global board snapshot: session-free aggregate shared by all viewers.

    Returns the running totals / offer leaderboard / segment mix / latest decision /
    recent-decision sample accumulated server-side since the app started, so a refresh
    (or a second viewer) sees the SAME window, never a fresh 'session'.

    The recent-decision feed shows only the ~1% BURSTY players (is_bursty flag on the
    login event). Only bursty logins emit a dense match-event burst, so is_bursty==True
    is exactly the set whose online SFV windows are populated and blip in the drill-down.
    We scan the full _FEED_SCAN ring (not just the newest 200) so the 1% bursty
    fraction always yields enough candidates before the display cap is applied. A cheap
    flag read -- no per-login feature-endpoint lookup. Off-workspace/synthetic logins
    carry no is_bursty flag, so if NOTHING is bursty we fall back to the unfiltered
    feed rather than a blank board.
    """
    snap = await asyncio.to_thread(aggregate.snapshot, _FEED_SCAN)
    candidates = snap["recent"]
    bursty = [e for e in candidates if e.get("is_bursty")][:recent]
    snap["recent"] = bursty if bursty else candidates[:recent]
    return JSONResponse(snap)


@app.get("/api/player/{player_id}")
async def player_detail(player_id: str) -> JSONResponse:
    """Player drill-down: the player's REAL scored decision + feature split.

    The drill-down is always opened from the recent-logins feed, so the player was
    already scored off a real Kafka login (with real request-time realtime values).
    We surface THAT retained decision -- ranked offers + the live session snapshot
    that produced it -- rather than re-scoring: the realtime passthrough features
    live only in the login (never materialized online), so re-scoring here would
    mean inventing them. Offline + streaming (SFV) features ARE recoverable by
    player_id from offer-features-fs, so those are looked up live and merged.
    """
    event, features = await asyncio.gather(
        asyncio.to_thread(aggregate.find_recent, player_id),
        asyncio.to_thread(catalog.feature_split, player_id),
    )
    if event is None:
        # Not in the retained scored window: show only the recoverable features.
        detail: dict = {
            "player_id": player_id,
            "segment": catalog.player_segment(player_id),
        }
    else:
        detail = dict(event)
    # The retained event's `realtime` is ALREADY the stripped feature snapshot (see
    # _feature_snapshot at attach time), so `detail` inherits it via dict(event).
    detail["features"] = features
    return JSONResponse(detail)


@app.get("/api/run")
async def run_panel() -> JSONResponse:
    """Run panel: model + serving endpoint status."""
    return JSONResponse(
        {
            "endpoint": ENDPOINT_NAME,
            "model": catalog.model_status(),
        }
    )


@app.get("/api/provenance")
async def provenance() -> JSONResponse:
    """Training provenance strip: the AIR+Ray headline from the MLflow run."""
    return JSONResponse(await asyncio.to_thread(catalog.provenance))


@app.get("/api/metrics")
async def serving_metrics() -> JSONResponse:
    """Serving Ops dashboard: Kafka ingestion rate + endpoint observability.

    Two sources, merged:
    - App-measured (metrics.snapshot): Kafka login consume rate and the client-side
      round-trip latency the app itself observes per call.
    - Platform-measured (serving_client.platform_metrics): the endpoints' own
      Prometheus feed -- the server-side request_latency_ms histogram (p50/p90/p99),
      QPS, error count, and cpu/mem, measured by Databricks through the full backend
      route. The ranking endpoint's server-side latency INCLUDES its internal online
      feature lookup (Databricks exposes no separate lookup sub-metric); the
      feature-serving endpoint's own histogram is the standalone feature-lookup
      latency, populated when a drill-down queries it.
    """
    snap = metrics.snapshot()
    ranking_platform, feature_platform = await asyncio.gather(
        asyncio.to_thread(platform_metrics, ENDPOINT_NAME),
        asyncio.to_thread(platform_metrics, FEATURE_ENDPOINT_NAME),
    )
    snap["ranking_endpoint"]["platform"] = ranking_platform
    snap["feature_endpoint"]["platform"] = feature_platform
    return JSONResponse(snap)


class _NoCacheStatic(StaticFiles):
    """Serve static files with no-cache so a redeployed app.jsx/index.html is never
    stale. The no-build path serves app.jsx as a plain static file; browsers (and the
    Apps CDN) cache it aggressively, so code changes silently didn't take without a hard
    reload. `no-cache` forces ETag/Last-Modified revalidation on every load (a 304 when
    unchanged, so it's cheap) instead of blind cache reuse.
    """

    def file_response(self, *args, **kwargs) -> Response:
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app.mount("/", _NoCacheStatic(directory=str(STATIC_DIR), html=True), name="static")
