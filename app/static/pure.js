/*
 * Pure, framework-free helpers extracted from app.jsx.
 *
 * WHY THIS FILE EXISTS: app.jsx runs only through in-browser Babel (no build
 * step), so its logic was untestable. These functions hold the risky arithmetic
 * (NaN/zero-division guards, null-revenue handling, rate floor, buffer caps) and
 * are pure -- no React, no DOM -- so `node --test` can exercise them directly.
 *
 * DUAL EXPORT (bottom): a classic <script src="pure.js"> defines window.Pure for
 * the browser; `module.exports` lets `require()` load it in node (the repo has no
 * package.json, so .js is CommonJS). Load pure.js BEFORE the text/babel app.jsx.
 */

// Format a dollar value; "—" for null/undefined/non-finite (NaN would otherwise
// render "$NaN"). 0 is a valid value -> "$0.00" (do NOT collapse with `!v`).
function money(v) {
  if (v === null || v === undefined) return "—";
  const n = Number(v);
  return Number.isFinite(n) ? "$" + n.toFixed(2) : "—";
}

// Safe [key, value] pairs for a feature group. A drill-down feature group can be
// undefined/null on a race (the /api/player enrich sets `features` but a sub-group
// like `offline` may be absent, or an empty split) -- `Object.entries(undefined)`
// THROWS, which unmounts the whole React tree (blank page on spam-click). Coerce a
// missing/non-object group to [] so FeatureGroup renders nothing instead of crashing.
function featureEntries(feats) {
  if (!feats || typeof feats !== "object") return [];
  return Object.entries(feats);
}

// Fold one streamed decision into the session aggregate (mutates + returns acc).
// `now` is passed in so the function stays pure (no Date.now() inside). Guards:
// error events don't count; the `&& expected_offer_revenue` check both skips
// $0/null-revenue offers from the leaderboard AND prevents `captured += null`
// (NaN) when a serving failure propagates a null revenue.
function accumulate(acc, ev, now) {
  if (ev.error) return acc;
  if (acc.firstTs == null) acc.firstTs = now;
  acc.count += 1;
  acc.latest = ev;
  const co = ev.chosen_offer;
  if (co && co.expected_offer_revenue) {
    acc.captured += co.expected_offer_revenue;
    const o = acc.offers[co.offer_id] || {
      offer_id: co.offer_id,
      count: 0,
      revenue: 0,
    };
    o.count += 1;
    o.revenue += co.expected_offer_revenue;
    acc.offers[co.offer_id] = o;
  }
  if (ev.segment) acc.segments[ev.segment] = (acc.segments[ev.segment] || 0) + 1;
  return acc;
}

// Decisions/minute since the first event. The 0.02-minute (1.2s) floor stops the
// very first sub-second flush from reporting an absurd extrapolated rate.
function computeRate(count, firstTs, now) {
  const mins = firstTs ? (now - firstTs) / 60000 : 0;
  return mins > 0.02 ? count / mins : 0;
}

// Merge a flush batch into the recent-decisions buffer: newest first, keep the
// last 12 of the batch, cap the buffer. Buffer is larger than the render slice so
// offer-filtering still finds matches at high stream rates.
function mergeRecent(prev, sinceFlush, cap) {
  cap = cap || 300;
  return [...sinceFlush.slice(-12).reverse(), ...prev].slice(0, cap);
}

// The recent rows actually rendered: optionally filtered to one offer, then
// capped. The `chosen_offer &&` guard drops null-chosen (no-eligible-offer) rows
// from a filtered view.
function visibleRecent(recent, offerFilter, cap) {
  cap = cap || 40;
  const filtered = offerFilter
    ? recent.filter(
        (e) => e.chosen_offer && e.chosen_offer.offer_id === offerFilter
      )
    : recent;
  return filtered.slice(0, cap);
}

// Sparkline polyline/polygon point strings, scaled to the series' own min/max.
// null (draw the placeholder instead) for <2 points -- i/(n-1) would be 0/0=NaN
// at n=1. `hi-lo || 1` guards a flat series (all-equal -> span 0).
function sparkPoints(series, w, h) {
  if (!series || series.length < 2) return null;
  const lo = Math.min(...series);
  const hi = Math.max(...series);
  const span = hi - lo || 1;
  const n = series.length;
  const pts = series
    .map((v, i) => {
      const x = (i / (n - 1)) * w;
      const y = h - 3 - ((v - lo) / span) * (h - 6); // 3px padding top/bottom
      return x.toFixed(1) + "," + y.toFixed(1);
    })
    .join(" ");
  const area = "0," + h + " " + pts + " " + w + "," + h;
  return { pts, area };
}

// "Nice" step for an axis so ticks land on 1/2/5 x 10^k. `rough` is the raw
// value-per-tick; returns a rounded step >= rough.
function niceStep(rough) {
  if (!(rough > 0)) return 1;
  const pow = Math.pow(10, Math.floor(Math.log10(rough)));
  const frac = rough / pow;
  const nice = frac <= 1 ? 1 : frac <= 2 ? 2 : frac <= 5 ? 5 : 10;
  return nice * pow;
}

// Multi-series line chart sharing ONE y-scale (for p50/p95/p99 together). Returns
// per-series polyline point strings + y-axis tick marks (value + pixel y), so all
// lines are directly comparable and the axis is labeled in real units.
//   series: [{key, values:[...]}]  -> longest length sets the x-count
//   w,h: viewBox size; padL: left gutter for axis labels; padT/padB: top/bottom pad
// y-scale runs 0 -> niceMax (a nice tick above the global max) so the axis reads
// cleanly and bars are comparable across polls; returns null if no series has >=2 pts.
function multiSparkChart(series, w, h, opts) {
  opts = opts || {};
  const padL = opts.padL != null ? opts.padL : 30;
  const padT = opts.padT != null ? opts.padT : 8;
  const padB = opts.padB != null ? opts.padB : 6;
  const live = (series || []).filter((s) => s.values && s.values.length >= 2);
  if (!live.length) return null;
  const n = Math.max(...live.map((s) => s.values.length));
  const globalMax = Math.max(...live.flatMap((s) => s.values), 0);
  const targetTicks = 4;
  const step = niceStep((globalMax || 1) / targetTicks);
  const niceMax = Math.max(step, Math.ceil(globalMax / step) * step);
  const plotW = w - padL;
  const plotH = h - padT - padB;
  const xOf = (i, len) => padL + (len === 1 ? 0 : (i / (len - 1)) * plotW);
  const yOf = (v) => padT + plotH - (v / niceMax) * plotH;
  const lines = live.map((s) => ({
    key: s.key,
    pts: s.values.map((v, i) => xOf(i, s.values.length).toFixed(1) + "," + yOf(v).toFixed(1)).join(" "),
  }));
  const ticks = [];
  for (let v = 0; v <= niceMax + 1e-9; v += step) {
    ticks.push({ value: Math.round(v), y: +yOf(v).toFixed(1) });
  }
  return { lines, ticks, niceMax, padL, plotW, w, h };
}

// Format an epoch-seconds timestamp as a clock label HH:MM:SS (local time).
function clockLabel(epochSecs) {
  const d = new Date(epochSecs * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
}

// Single-series TIME chart for the median purchase-probability line. `series` is
// [{t: epochSecs, median: 0..1}] oldest->newest. Y-axis is a FIXED 0-100% scale
// (auto-zooms to a nice ceiling above the data max so a low, flat signal is still
// readable), labeled in %; x-axis carries a few evenly-spaced clock labels.
// Returns null for <2 points (the placeholder renders instead).
//   w,h: viewBox size; padL/padB gutters for the labelled axes.
function probChart(series, w, h, opts) {
  opts = opts || {};
  const padL = opts.padL != null ? opts.padL : 40;
  const padT = opts.padT != null ? opts.padT : 10;
  const padB = opts.padB != null ? opts.padB : 22;
  const pts = (series || []).filter((p) => p && typeof p.median === "number");
  if (pts.length < 2) return null;
  const dataMaxPct = Math.max(...pts.map((p) => p.median)) * 100;
  // Nice y-ceiling in %, with a 5% floor so a near-zero signal still has headroom and
  // the tick labels stay whole percents (not "0%,1%"); capped at 100%.
  const step = niceStep(Math.max(dataMaxPct, 5) / 4);
  const yMaxPct = Math.min(
    100, Math.max(5, step, Math.ceil(dataMaxPct / step) * step)
  );
  const plotW = w - padL;
  const plotH = h - padT - padB;
  const n = pts.length;
  const xOf = (i) => padL + (n === 1 ? 0 : (i / (n - 1)) * plotW);
  const yOf = (pct) => padT + plotH - (pct / yMaxPct) * plotH;
  const line = pts
    .map((p, i) => xOf(i).toFixed(1) + "," + yOf(p.median * 100).toFixed(1))
    .join(" ");
  const area = xOf(0).toFixed(1) + "," + (padT + plotH) + " " + line + " " +
    xOf(n - 1).toFixed(1) + "," + (padT + plotH);
  const yTicks = [];
  for (let v = 0; v <= yMaxPct + 1e-9; v += step) {
    yTicks.push({ label: Math.round(v) + "%", y: +yOf(v).toFixed(1) });
  }
  // ~6 x-axis clock labels evenly spaced across the series (finer time ticks).
  const xTicks = [];
  const nLabels = Math.min(opts.xLabels || 6, n);
  for (let k = 0; k < nLabels; k++) {
    const i = Math.round((k / (nLabels - 1)) * (n - 1));
    xTicks.push({ label: clockLabel(pts[i].t), x: +xOf(i).toFixed(1) });
  }
  const baselineY = padT + plotH;
  return { line, area, yTicks, xTicks, yMaxPct, padL, baselineY, w, h };
}

// Multi-line offer-share chart: one line per offer_id over time, on a FIXED 0-100%
// y-axis (shares within a bucket sum to 100%). `series` is
// [{t: epochSecs, shares: {offer_id: frac}}] oldest->newest. Returns per-offer
// polylines (missing offer in a bucket => 0%), % y-ticks, time x-ticks. null for
// <2 points. `palette` (array of colors) is cycled for stable per-offer coloring.
function shareChart(series, w, h, opts) {
  opts = opts || {};
  const padL = opts.padL != null ? opts.padL : 44;
  const padT = opts.padT != null ? opts.padT : 14;
  const padB = opts.padB != null ? opts.padB : 26;
  const palette = opts.palette || ["#46e3a0"];
  const pts = (series || []).filter((p) => p && p.shares);
  if (pts.length < 2) return null;
  // Stable offer ordering: by total share across the window (biggest first), so
  // colors don't reshuffle as offers come and go.
  const totals = {};
  for (const p of pts) {
    for (const k in p.shares) totals[k] = (totals[k] || 0) + p.shares[k];
  }
  const offers = Object.keys(totals).sort((a, b) => totals[b] - totals[a]);
  const yMaxPct = 100;
  const plotW = w - padL;
  const plotH = h - padT - padB;
  const n = pts.length;
  const xOf = (i) => padL + (n === 1 ? 0 : (i / (n - 1)) * plotW);
  const yOf = (pct) => padT + plotH - (pct / yMaxPct) * plotH;
  const lines = offers.map((oid, idx) => ({
    offer_id: oid,
    color: palette[idx % palette.length],
    pts: pts
      .map((p, i) => xOf(i).toFixed(1) + "," + yOf((p.shares[oid] || 0) * 100).toFixed(1))
      .join(" "),
  }));
  const yTicks = [];
  for (let v = 0; v <= 100 + 1e-9; v += 25) {
    yTicks.push({ label: v + "%", y: +yOf(v).toFixed(1) });
  }
  const xTicks = [];
  const nLabels = Math.min(opts.xLabels || 6, n);
  for (let k = 0; k < nLabels; k++) {
    const i = Math.round((k / (nLabels - 1)) * (n - 1));
    xTicks.push({ label: clockLabel(pts[i].t), x: +xOf(i).toFixed(1) });
  }
  return { lines, yTicks, xTicks, padL, w, h };
}

// Offer leaderboard rows: highest revenue first, top `limit`.
function topOffers(offersMap, limit) {
  limit = limit || 8;
  return Object.values(offersMap)
    .sort((a, b) => b.revenue - a.revenue)
    .slice(0, limit);
}

// Bar width percent with a 0 floor when there's no max (empty/all-zero).
function barPct(value, max) {
  return max > 0 ? (value / max) * 100 : 0;
}

// Segment mix rows: [segment, count, percent], highest count first. `|| 1` guards
// divide-by-zero on an empty map. Percentages round independently, so they need
// not sum to 100 (three equal segments -> 33/33/33) -- intentional, not a bug.
function segmentRows(segmentsMap) {
  const rows = Object.entries(segmentsMap).sort((a, b) => b[1] - a[1]);
  const total = rows.reduce((s, [, n]) => s + n, 0) || 1;
  return rows.map(([seg, n]) => [seg, n, Math.round((n / total) * 100)]);
}

// Non-mutating sort of the live-board offers by revenue (null coerced to 0 so
// unscored offers sink without NaN).
function sortByRevenue(offers) {
  return [...offers].sort(
    (a, b) => (b.expected_offer_revenue || 0) - (a.expected_offer_revenue || 0)
  );
}

function maxRevenue(sorted) {
  return sorted.length ? sorted[0].expected_offer_revenue || 0 : 0;
}

// Live-board bar percent with a 3% minimum so every bar is visible; 3 when
// unscored (null) or there's no max (distinct from barPct's 0 floor).
function offerBarPct(rev, max) {
  return max > 0 && rev != null ? Math.max(3, (rev / max) * 100) : 3;
}

// Tween kernel (the rAF hook in app.jsx calls these; kept pure + testable).
function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

function tweenValue(from, to, t) {
  return from + (to - from) * easeOutCubic(t);
}

const Pure = {
  money,
  featureEntries,
  accumulate,
  computeRate,
  mergeRecent,
  visibleRecent,
  sparkPoints,
  niceStep,
  multiSparkChart,
  clockLabel,
  probChart,
  shareChart,
  topOffers,
  barPct,
  segmentRows,
  sortByRevenue,
  maxRevenue,
  offerBarPct,
  easeOutCubic,
  tweenValue,
};

if (typeof module !== "undefined" && module.exports) module.exports = Pure;
if (typeof window !== "undefined") window.Pure = Pure;
