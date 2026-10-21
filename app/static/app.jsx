/*
 * Nordfall Offer Recommender: live inference console.
 *
 * The board is a GLOBAL, server-side aggregate: the backend scorer (app.py) ranks
 * every streamed login into one shared window (aggregate.py), and this UI just POLLS
 * /api/board once a second and renders the snapshot. There is no per-browser
 * "session" -- a refresh (or a second viewer) sees the same running totals. Shows
 * live totals (revenue/count/rate), an offer leaderboard (served count + revenue,
 * click to filter), a segment mix, the "now serving" board, and a recent-login
 * sample you click to drill into a player's offline vs real-time feature split.
 *
 * Classic React (createElement via Babel 7); no build step, all CDN.
 */
const { useState, useEffect, useRef, useCallback, useLayoutEffect } = React;

// Pure helpers (money, aggregation, chart math, ...) live in pure.js so they are
// unit-testable under `node --test` (see pure.test.js); loaded as window.Pure by
// the plain <script> before this file.
const {
  money, visibleRecent, sparkPoints, multiSparkChart, probChart, shareChart, tweenValue,
} = window.Pure;

// Categorical palette for the offer-share lines (cycled; brand-neutral, distinct hues).
const SHARE_PALETTE = [
  "#46e3a0", "#5cc8ff", "#f4b740", "#b28dff", "#ff8f6b", "#59d2fe",
  "#e879a6", "#8ce563", "#ffd166", "#7aa5ff", "#4de1c1", "#ff6b9d",
];

const REDUCED_MOTION =
  window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Tween a number toward `target` so the displayed value counts up/down instead
// of snapping — the "dancing numbers". requestAnimationFrame, ~450ms ease-out.
function useTweenedNumber(target) {
  const [display, setDisplay] = useState(target || 0);
  // The LIVE displayed value, updated every frame. Animating from this (not the
  // last COMPLETED target) means an interrupted or null-target tween resumes from
  // whatever is actually on screen, so a serving-failure event (revenue -> null,
  // display -> 0) doesn't make the next tween snap 0 -> stale-prev -> new.
  const displayRef = useRef(target || 0);
  const startRef = useRef(null);
  const rafRef = useRef(null);

  useEffect(() => {
    if (REDUCED_MOTION || target === null || target === undefined) {
      const v = target || 0;
      displayRef.current = v;
      setDisplay(v);
      return;
    }
    const from = displayRef.current;
    const dur = 450;
    startRef.current = null;
    const step = (ts) => {
      if (startRef.current === null) startRef.current = ts;
      const t = Math.min(1, (ts - startRef.current) / dur);
      const val = tweenValue(from, target, t);
      displayRef.current = val;
      setDisplay(val);
      if (t < 1) rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafRef.current);
  }, [target]);

  return display;
}

// Big count-up figure that eases toward its target (used in Serving Ops).
function BigNumber({ value, prefix, decimals }) {
  const shown = useTweenedNumber(value);
  const d = decimals === undefined ? 2 : decimals;
  return (
    <span className="big-num">{(prefix || "") + Number(shown).toFixed(d)}</span>
  );
}

// Hero: median purchase probability across all scored logins, over time. A single
// live line on a labeled % y-axis with clock labels on x. `latest` supplies the
// `series` is [{t, median}] from /api/board.
function PurchaseProbChart({ series }) {
  const w = 1000;
  const h = 260;
  const chart = probChart(series, w, h, { padL: 44, padT: 14, padB: 26 });
  const latestPct =
    series && series.length ? (series[series.length - 1].median * 100).toFixed(1) : null;
  return (
    <div className="card prob-board">
      <div className="aurora" aria-hidden="true"></div>
      <div className="pb-head">
        <div>
          <h2>Purchase probability · live</h2>
          <div className="pb-sub">Median modeled buy-likelihood of the served offer, across all logins</div>
        </div>
        {latestPct != null && (
          <div className="pb-now">
            <span className="pb-now-val">{latestPct}%</span>
            <span className="pb-now-lbl">current median</span>
          </div>
        )}
      </div>
      <span className="pb-ylabel">purchase probability (%)</span>
      {!chart ? (
        <div className="prob-empty">Aggregating live logins…</div>
      ) : (
        <svg className="prob-svg" viewBox={"0 0 " + w + " " + h} preserveAspectRatio="none"
          role="img" aria-label="median purchase probability over time, percent">
          {/* y grid + % labels */}
          {chart.yTicks.map((t, i) => (
            <g key={"y" + i}>
              <line x1={chart.padL} y1={t.y} x2={w} y2={t.y}
                stroke="var(--border)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
              <text x={chart.padL - 8} y={t.y + 4} textAnchor="end"
                fill="var(--muted)" fontSize="13" fontFamily="'JetBrains Mono', monospace">{t.label}</text>
            </g>
          ))}
          {/* x clock labels */}
          {chart.xTicks.map((t, i) => (
            <text key={"x" + i} x={t.x} y={h - 6}
              textAnchor={i === 0 ? "start" : i === chart.xTicks.length - 1 ? "end" : "middle"}
              fill="var(--muted)" fontSize="12" fontFamily="'JetBrains Mono', monospace">{t.label}</text>
          ))}
          <polygon points={chart.area} fill="var(--accent)" fillOpacity="0.08" />
          <polyline points={chart.line} fill="none" stroke="var(--accent)"
            strokeWidth="2.5" vectorEffect="non-scaling-stroke" strokeLinejoin="round" strokeLinecap="round" />
        </svg>
      )}
      <div className="pb-axis-label">time →</div>
    </div>
  );
}

// Side chart: per-offer share of recommendations over time. One line per offer_id,
// fixed 0-100% y-axis (lines sum to ~100% at each x). `series` is
// [{t, shares:{offer_id: frac}}] from /api/board.
function OfferShareChart({ series }) {
  const w = 1000;
  const h = 260;
  const chart = shareChart(series, w, h, { padL: 44, padT: 14, padB: 26, palette: SHARE_PALETTE });
  return (
    <div className="card prob-board">
      <div className="aurora" aria-hidden="true"></div>
      <div className="pb-head">
        <div>
          <h2>Offer recommendation mix · live</h2>
          <div className="pb-sub">Share of logins recommended each offer (sums to 100%)</div>
        </div>
      </div>
      <span className="pb-ylabel">recommendation share (%)</span>
      {!chart ? (
        <div className="prob-empty">Aggregating live logins…</div>
      ) : (
        <svg className="prob-svg" viewBox={"0 0 " + w + " " + h} preserveAspectRatio="none"
          role="img" aria-label="per-offer recommendation share over time, percent">
          {chart.yTicks.map((t, i) => (
            <g key={"y" + i}>
              <line x1={chart.padL} y1={t.y} x2={w} y2={t.y}
                stroke="var(--border)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
              <text x={chart.padL - 8} y={t.y + 4} textAnchor="end"
                fill="var(--muted)" fontSize="13" fontFamily="'JetBrains Mono', monospace">{t.label}</text>
            </g>
          ))}
          {chart.xTicks.map((t, i) => (
            <text key={"x" + i} x={t.x} y={h - 6}
              textAnchor={i === 0 ? "start" : i === chart.xTicks.length - 1 ? "end" : "middle"}
              fill="var(--muted)" fontSize="12" fontFamily="'JetBrains Mono', monospace">{t.label}</text>
          ))}
          {chart.lines.map((ln) => (
            <polyline key={ln.offer_id} points={ln.pts} fill="none" stroke={ln.color}
              strokeWidth="1.8" vectorEffect="non-scaling-stroke" strokeLinejoin="round" strokeLinecap="round" />
          ))}
        </svg>
      )}
      {chart && (
        <div className="share-legend">
          {chart.lines.map((ln) => (
            <span className="share-leg" key={ln.offer_id}>
              <span className="share-swatch" style={{ background: ln.color }}></span>
              {ln.offer_id}
            </span>
          ))}
        </div>
      )}
      <div className="pb-axis-label">time →</div>
    </div>
  );
}

function RunPanel({ run }) {
  if (!run) return null;
  return (
    <div className="run-panel">
      <span><span className="dot up"></span>Endpoint <b>{run.endpoint}</b></span>
      <span>Model <b>{run.model.version}</b> ({run.model.mode})</span>
    </div>
  );
}

function Feed({ events, onSelect, selectedId, filter }) {
  return (
    <div className="card">
      <h2>
        Recent logins <span className="agg-hint">sampled · click to drill in</span>
        {filter && <span className="feed-filter">offer {filter} ✕</span>}
      </h2>
      <div className="feed">
        {events.length === 0 && <div className="detail"><div className="empty">{filter ? "No " + filter + " in the recent sample (the count above is the full running total)…" : "Aggregating live logins…"}</div></div>}
        {events.map((e, i) => (
          <div key={(e.player_id || "") + i}
            className={"feed-row" + (e.player_id === selectedId ? " sel" : "")}
            role="button" tabIndex={0}
            aria-label={"Drill into " + (e.player_id || "player")}
            onClick={() => onSelect(e)}
            onKeyDown={(ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); onSelect(e); } }}>
            <div className="top">
              <span className="pid">{e.player_id}</span>
              <span className="seg">{e.segment}</span>
            </div>
            <div className="chosen">
              {e.chosen_offer ? (
                <React.Fragment>
                  <span className="oid">{e.chosen_offer.offer_id}</span>
                  <span className="rev">{money(e.chosen_offer.expected_offer_revenue)}</span>
                </React.Fragment>
              ) : <span className="lat">no eligible offer</span>}
              <span className="lat" style={{ marginLeft: "auto" }}>{e.latency_ms} ms</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function FeatureGroup({ kind, label, source, feats }) {
  // Freshness cue (streaming group): flash a value when it CHANGES, and show
  // "updated Ns ago" so the viewer sees the 0.5s refresh is live even when a
  // per-player 15-min window holds steady between that player's own logins.
  const live = kind === "streaming";
  const prevRef = useRef({});
  const changedRef = useRef({}); // key -> timestamp of last change (for the flash key)
  const [lastChange, setLastChange] = useState(null);
  const [, forceTick] = useState(0);

  useEffect(() => {
    if (!live) return;
    let anyChanged = false;
    for (const [k, v] of window.Pure.featureEntries(feats)) {
      if (k in prevRef.current && prevRef.current[k] !== v) {
        changedRef.current[k] = Date.now();
        anyChanged = true;
      }
    }
    prevRef.current = { ...feats };
    if (anyChanged) setLastChange(Date.now());
  }, [feats, live]);

  // Re-render every second so "updated Ns ago" counts up (only while streaming shown).
  useEffect(() => {
    if (!live) return;
    const id = setInterval(() => forceTick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, [live]);

  const agoLabel = () => {
    if (lastChange == null) return "waiting for an update…";
    const s = Math.round((Date.now() - lastChange) / 1000);
    return s <= 1 ? "updated just now" : `updated ${s}s ago`;
  };

  return (
    <div className={"feat-group " + kind}>
      <div className="label">
        {label}
        {source && <span className="feat-src">{source}</span>}
        {live && <span className="feat-fresh">{agoLabel()}</span>}
      </div>
      <div className="feat-grid">
        {window.Pure.featureEntries(feats).map(([k, v]) => {
          // A changed value flashes: the animation restarts because the key changes
          // (React remounts the span) whenever this key's last-change timestamp bumps.
          const changedAt = live ? changedRef.current[k] : undefined;
          const recentlyChanged = changedAt && Date.now() - changedAt < 1000;
          return (
            <div className="feat" key={k}>
              <span className="k">{k}</span>
              <span className={"v" + (recentlyChanged ? " changed" : "")} key={changedAt || 0}>{v}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Detail({ detail }) {
  if (!detail) {
    return <div className="card detail"><h2>Player drill-down</h2><div className="empty">Select a player from the feed.</div></div>;
  }
  if (detail.error) {
    return <div className="card detail"><h2>Player drill-down</h2><div className="empty">{detail.error}</div></div>;
  }
  return (
    <div className="card detail">
      <h2>Player drill-down — {detail.player_id}</h2>
      {detail.features && (
        <React.Fragment>
          {/* SFV streaming rolling-window aggregates FIRST: these are the live
              online-store values that tick every 0.5s as the streaming feature view
              advances, so they lead the drill-down. */}
          {detail.features.streaming && Object.keys(detail.features.streaming).length > 0 && (
            <FeatureGroup kind="streaming" label="● Last-15-min trends"
              source={detail.features.streaming_source} feats={detail.features.streaming} />
          )}
          {/* Live session snapshot = the RequestSource passthrough values from the
              REAL login that was scored (detail.realtime, retained on the event).
              Not materialized online, so only shown when this player is in the
              recent-scored window. */}
          {detail.realtime && Object.keys(detail.realtime).length > 0 && (
            <FeatureGroup kind="realtime" label="● Live session snapshot"
              source="point-in-time · from the scored login" feats={detail.realtime} />
          )}
          {detail.features.offline && Object.keys(detail.features.offline).length > 0 && (
            <FeatureGroup kind="offline" label="● Offline features"
              source={detail.features.offline_source} feats={detail.features.offline} />
          )}
        </React.Fragment>
      )}
      {detail.chosen_offer && detail.chosen_offer.purchase_probability != null && (
        <div className="buy-prob">
          <span className="buy-prob-label">Purchase probability
            <span className="buy-prob-sub">chosen offer {detail.chosen_offer.offer_id}</span>
          </span>
          <span className="buy-prob-val">{(detail.chosen_offer.purchase_probability * 100).toFixed(1)}%</span>
        </div>
      )}
      <h2 style={{ marginTop: "6px" }}>Ranked offers</h2>
      <div className="offer-list">
        {(detail.ranked_offers || []).map((o, i) => (
          <div key={i} className={"offer" + (i === 0 ? " top" : "")}>
            <span className="oid">{o.offer_id}</span>
            <span>
              {o.purchase_probability != null && <span className="buy-pct">{(o.purchase_probability * 100).toFixed(0)}% buy</span>}
              <span className="rev">{money(o.expected_offer_revenue)}</span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}


// Serving Ops dashboard: server-side telemetry from /api/metrics. Two panels --
// event INGESTION (Kafka consume rate) and SERVING observability (the two real
// endpoint latencies) -- plus a clearly-labeled platform-capability reference so
// our real (small) live numbers are never conflated with the DAIS ceiling.
function StatCard({ label, value, unit, accent }) {
  return (
    <div className="ops-stat">
      <div className="ops-stat-label">{label}</div>
      <div className={"ops-stat-value" + (accent ? " " + accent : "")}>
        {value}<span className="ops-stat-unit">{unit}</span>
      </div>
    </div>
  );
}

// Percentile lines for the latency chart, each its own colour.
const _LAT_SERIES = [
  { key: "p50", color: "var(--accent)" },
  { key: "p95", color: "var(--realtime)" },
  { key: "p99", color: "var(--amber)" },
];

// Multi-line latency chart: p50/p95/p99 over time on ONE shared ms y-scale, with a
// labelled y-axis (gridlines + tick values) so the magnitudes are readable. `hists`
// is {p50:[...], p95:[...], p99:[...]} accumulated from the poll. No chart lib.
function LatencyChart({ hists, height }) {
  const h = height || 200;
  const w = 620; // viewBox width; stretches to container via preserveAspectRatio=none
  const chart = multiSparkChart(
    _LAT_SERIES.map((s) => ({ key: s.key, values: hists[s.key] || [] })),
    w, h, { padL: 34, padT: 10, padB: 8 }
  );
  if (!chart) {
    return <div className="lat-chart-empty" style={{ height: h }}>waiting for inference traffic…</div>;
  }
  const colorOf = (key) => (_LAT_SERIES.find((s) => s.key === key) || {}).color;
  return (
    <svg className="lat-chart" viewBox={"0 0 " + w + " " + h} preserveAspectRatio="none"
      role="img" aria-label="model inference latency percentiles over time, in milliseconds">
      {/* y grid + tick labels (ms) */}
      {chart.ticks.map((t, i) => (
        <g key={i}>
          <line x1={chart.padL} y1={t.y} x2={w} y2={t.y}
            stroke="var(--border)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
          <text x={chart.padL - 6} y={t.y + 3} textAnchor="end"
            fill="var(--muted)" fontSize="11" fontFamily="'JetBrains Mono', monospace">{t.value}</text>
        </g>
      ))}
      {/* percentile lines */}
      {chart.lines.map((ln) => (
        <polyline key={ln.key} points={ln.pts} fill="none" stroke={colorOf(ln.key)}
          strokeWidth="2" vectorEffect="non-scaling-stroke"
          strokeLinejoin="round" strokeLinecap="round" />
      ))}
    </svg>
  );
}

const _HIST_LEN = 90; // ~3 min of chart history at the 2s poll

// Presentational: all state (metrics + chart history) lives in App so it survives
// tab switches (this component unmounts when you leave the Ops tab). `latHist` is
// the accumulated {p50,p95,p99} percentile series for the latency chart.
function ServingOps({ m, latHist }) {
  const ing = (m && m.ingestion) || { events_per_sec: 0, window_events: 0 };
  const ep = (m && m.ranking_endpoint) || null;
  const l = (ep && ep.latency_ms) || { p50: 0, p95: 0, p99: 0, count: 0 };
  const idle = !l.count;
  // Server-side latency measured by the serving platform (Prometheus histogram),
  // WITHOUT the app->workspace network hop the round-trip chart includes. This is the
  // figure the <10ms p99 overhead claim refers to. Minute-bucketed + laggy -> reads 0
  // between traffic; only show it when the histogram has data.
  const plat = (ep && ep.platform && ep.platform.latency_ms) || null;
  const platReady = plat && (plat.p50 || plat.p90 || plat.p99);
  return (
    <React.Fragment>
      <div className="ops-row two">
        <div className="card ops-panel ingest">
          <h2>Event ingestion <span className="agg-hint">Kafka login stream</span></h2>
          <div className="ops-big"><BigNumber value={ing.events_per_sec} decimals={0} /><span className="ops-big-unit">logins / sec</span></div>
          <div className="ops-sub">{ing.window_events.toLocaleString()} in the last {m ? Math.round(m.window_s) : 60}s · uptime {m ? Math.round(m.uptime_s) : 0}s</div>
        </div>
        <div className="card ops-panel">
          <h2>Model inference <span className="agg-hint">offer-ranker-rt · round-trip latency</span></h2>
          <div className="ops-endpoint">{ep && ep.name}</div>
          <div className="lat-chart-wrap">
            <span className="lat-yaxis-label">latency (ms)</span>
            <LatencyChart hists={latHist} />
          </div>
          <div className="lat-legend">
            {_LAT_SERIES.map((s) => (
              <span className="lat-leg" key={s.key}>
                <span className="lat-swatch" style={{ background: s.color }}></span>
                {s.key} <b>{l[s.key]}ms</b>
              </span>
            ))}
          </div>
          {platReady && (
            <div className="lat-legend lat-legend-server">
              <span className="lat-leg-label">server-side</span>
              <span className="lat-leg">p50 <b>{Math.round(plat.p50)}ms</b></span>
              <span className="lat-leg">p90 <b>{Math.round(plat.p90)}ms</b></span>
              <span className="lat-leg">p99 <b>{Math.round(plat.p99)}ms</b></span>
            </div>
          )}
          {idle && <div className="ops-sub">no inference calls yet</div>}
        </div>
      </div>
      <div className="ops-hint-line">
        The chart + legend are the app's live <b>round-trip</b> latency (p50/p95/p99, measured
        per call) — it includes the <b>app→workspace network hop + offer-ranker-rt's internal
        online feature lookup</b>, so ~200-300ms here is dominated by network + FS lookup, not
        model inference. The <b>server-side</b> row is the platform's own Prometheus histogram
        (no network hop): that is where the <b>&lt;10ms p99 serving overhead</b> shows up. No
        separate lookup sub-metric is exposed via the metrics API (the endpoint's monitoring
        page shows the Model Inference / Feature Lookup / Business Logic split).
        <br /><b>AI Gateway</b> inference tables are enabled on offer-ranker-rt: every
        request/response is logged to Unity Catalog for audit.
      </div>
      <div className="card ops-note">
        <h2>Platform capability <span className="agg-hint">reference, not this demo's live load</span></h2>
        <p className="ops-cap">
          The same Databricks Model Serving engine reaches <b>300K+ QPS at &lt;10ms p99 latency
          overhead</b> across CPU &amp; GPU models — no knobs to tune, adapting to each model and
          its traffic automatically. That &lt;10ms figure is the <b>server-side</b> overhead
          (the row above), not the round-trip a remote client sees (which adds network + the
          online feature lookup). Online features are served from the <b>Lakebase</b> online
          store; real-time features here ride the request (millisecond-fresh). The figures above
          are THIS demo's modest live traffic; the 300K QPS figure is the platform ceiling this
          architecture scales to.
        </p>
      </div>
    </React.Fragment>
  );
}

function App() {
  // The board is a GLOBAL, server-side aggregate (aggregate.py): one window shared by
  // every viewer, accumulated by the backend scorer independent of any browser. The
  // client just POLLS /api/board and renders the snapshot -- so a refresh (or a second
  // viewer) sees the SAME running totals, never a fresh "session". No client-side
  // accumulation, no per-connection SSE state.
  const [agg, setAgg] = useState({
    count: 0, rate: 0, offers: {}, segments: {}, latest: null,
  });
  const [recent, setRecent] = useState([]);
  const [detail, setDetail] = useState(null);
  const [run, setRun] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  // Serving Ops telemetry + chart history. Held here (not in ServingOps) so it
  // KEEPS ACCUMULATING and survives leaving/returning to the Ops tab, which
  // unmounts ServingOps. Polling runs whenever the Ops tab is active.
  const [ops, setOps] = useState(null);
  // Accumulated p50/p95/p99 series for the Model-inference latency chart.
  const [latHist, setLatHist] = useState({ p50: [], p95: [], p99: [] });
  // Median purchase-probability time series (server-side), for the board's line chart.
  const [probSeries, setProbSeries] = useState([]);
  // Per-offer recommendation-share time series, for the side-by-side mix chart.
  const [shareSeries, setShareSeries] = useState([]);
  const [tab, setTab] = useState("board");
  const [connDown, setConnDown] = useState(false);

  useEffect(() => {
    fetch("/api/run").then(r => r.json()).then(setRun).catch(() => {});
    // Poll the global board snapshot every second. The server holds the full window;
    // we ask for a recent sample large enough that offer-filtering finds matches (the
    // Feed renders only the first ~40).
    let alive = true;
    let emptyPolls = 0;
    const poll = () => fetch("/api/board?recent=200").then(r => r.json()).then(d => {
      if (!alive) return;
      setAgg({
        count: d.count, rate: d.rate,
        offers: d.offers || {}, segments: d.segments || {}, latest: d.latest,
      });
      setRecent(d.recent || []);
      setProbSeries(d.purchase_prob_series || []);
      setShareSeries(d.offer_share_series || []);
      // "Idle" if the global count isn't advancing (no new logins being scored).
      emptyPolls = d.count === (poll._lastCount ?? -1) ? emptyPolls + 1 : 0;
      poll._lastCount = d.count;
      setConnDown(d.count === 0 || emptyPolls >= 8);
    }).catch(() => { if (alive) setConnDown(true); });
    poll();
    const id = setInterval(poll, 1000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  // Poll /api/metrics ALWAYS (from app load), not just while the Ops tab is open,
  // so the latency chart is already populated with history the moment you step into
  // the tab -- no empty-then-fill lag. History persists in App state regardless of
  // which tab is showing.
  useEffect(() => {
    let alive = true;
    const poll = () => fetch("/api/metrics").then(r => r.json()).then(d => {
      if (!alive) return;
      setOps(d);
      // Chart the CLIENT-side p50/p95/p99 (the app's own round-trip): always live,
      // unlike the minute-bucketed + laggy platform feed which reads 0 after a
      // rollover. Only record when there were real calls, so an idle window doesn't
      // draw false flat-0 lines or pin the chart's scale at 0.
      const re = d.ranking_endpoint;
      if (re && re.latency_ms.count > 0) {
        const lm = re.latency_ms;
        setLatHist(h => ({
          p50: [...h.p50, lm.p50].slice(-_HIST_LEN),
          p95: [...h.p95, lm.p95].slice(-_HIST_LEN),
          p99: [...h.p99, lm.p99].slice(-_HIST_LEN),
        }));
      }
    }).catch(() => {});
    poll();
    const id = setInterval(poll, 2000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const select = useCallback((event) => {
    // `event` is the FULL scored decision from the feed (chosen_offer, ranked_offers,
    // realtime) -- the real values scored off this player's live login. Seed the
    // drill-down from it directly so ranked offers + the session snapshot are ALWAYS
    // the ones the user clicked, never lost to a ring-eviction race on a re-fetch.
    // The /api/player fetch only ENRICHES with the offline + streaming online lookups.
    const pid = event.player_id;
    setSelectedId(pid);
    setDetail(event); // show the clicked decision immediately (no blank flash)
    fetch("/api/player/" + encodeURIComponent(pid))
      .then(r => r.json())
      .then(data => setDetail(prev => {
        if (!prev || prev.player_id !== pid) return prev;
        // Keep the clicked event's ranked_offers/chosen_offer/realtime; add the
        // freshly looked-up offline + streaming feature groups.
        return { ...prev, features: data.features, realtime: prev.realtime || data.realtime };
      }))
      .catch(() => {}); // keep the seeded event on fetch failure
  }, []);

  // While a drill-down is open, poll /api/player/<id> every 0.5s and patch in the
  // fresh `streaming` (SFV) group so those values tick as the SFV online store
  // advances. ONLY runs while a player is shown (guarded on selectedId; the effect
  // is torn down when the panel closes -> zero requests when nothing is selected).
  // Offline and realtime stay as first loaded (they change on daily/request cadence).
  // Guard: if a request is still in-flight, skip the next tick (no overlapping calls).
  useEffect(() => {
    if (!selectedId) return;
    let alive = true;
    let inFlight = false;
    const poll = () => {
      if (inFlight) return;
      inFlight = true;
      fetch("/api/player/" + encodeURIComponent(selectedId))
        .then(r => r.json())
        .then(data => {
          if (!alive) return;
          if (data && data.features && data.features.streaming) {
            setDetail(prev => {
              // Guard: discard if the panel was closed or switched to another player.
              if (!prev || prev.player_id !== selectedId) return prev;
              return {
                ...prev,
                features: { ...prev.features, streaming: data.features.streaming },
              };
            });
          }
        })
        .catch(() => {})
        .finally(() => { inFlight = false; });
    };
    const id = setInterval(poll, 500);
    return () => { alive = false; clearInterval(id); };
  }, [selectedId]);

  const shownRecent = visibleRecent(recent, null, 40);

  return (
    <div className="container">
      <div className="header">
        <div className="brand">
          <h1>Nordfall Offer Recommender</h1>
          <span className="sub">Harvest Haven · live login-driven ranking</span>
          {connDown && <span className="conn-down" title="No live events; showing last known data">stream idle</span>}
        </div>
        <RunPanel run={run} />
      </div>
      <div className="tabs" role="tablist">
        <button role="tab" aria-selected={tab === "board"} className={"tab" + (tab === "board" ? " active" : "")} onClick={() => setTab("board")}>Live board</button>
        <button role="tab" aria-selected={tab === "ops"} className={"tab" + (tab === "ops" ? " active" : "")} onClick={() => setTab("ops")}>Serving Ops</button>
      </div>
      {tab === "board" ? (
        <React.Fragment>
          <div className="chart-row">
            <PurchaseProbChart series={probSeries} />
            <OfferShareChart series={shareSeries} />
          </div>
          <div className="layout">
            <Feed events={shownRecent} onSelect={select} selectedId={selectedId} filter={null} />
            <Detail detail={detail} />
          </div>
        </React.Fragment>
      ) : (
        <ServingOps m={ops} latHist={latHist} />
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
