/*
 * Unit tests for the pure helpers in pure.js. Zero deps: node's built-in test
 * runner + assert. Run: `node --test app/static/`.
 *
 * Covers the risky guards the review flagged: NaN prevention, null-revenue
 * poisoning, the rate floor, buffer caps, and divide-by-zero floors.
 */
const test = require("node:test");
const assert = require("node:assert/strict");
const P = require("./pure.js");

test("money: null/undefined/non-finite -> em dash; 0 is valid", () => {
  assert.equal(P.money(null), "—");
  assert.equal(P.money(undefined), "—");
  assert.equal(P.money(0), "$0.00"); // must NOT collapse with `!v`
  assert.equal(P.money(-1.5), "$-1.50");
  assert.equal(P.money("12.5"), "$12.50");
  assert.equal(P.money(1.005), "$1.00"); // float rounding, documented
  assert.equal(P.money(NaN), "—");
  assert.equal(P.money("abc"), "—");
});

test("featureEntries: null/undefined/non-object -> [] (no crash); object -> entries", () => {
  // The bug: Object.entries(undefined) THROWS, unmounting the React tree (blank
  // page on spam-click when a drill-down feature group is momentarily absent).
  assert.deepEqual(P.featureEntries(undefined), []);
  assert.deepEqual(P.featureEntries(null), []);
  assert.deepEqual(P.featureEntries({}), []);
  assert.deepEqual(P.featureEntries("nope"), []); // non-object guarded
  assert.deepEqual(P.featureEntries(42), []);
  assert.deepEqual(P.featureEntries({ match_wins_15m: 3, gems_spent_15m: 0 }), [
    ["match_wins_15m", 3],
    ["gems_spent_15m", 0],
  ]);
});

test("accumulate: two events on one offer sum count + revenue", () => {
  const acc = { count: 0, captured: 0, offers: {}, segments: {}, firstTs: null };
  P.accumulate(acc, { chosen_offer: { offer_id: "o0", expected_offer_revenue: 1.5 } }, 100);
  P.accumulate(acc, { chosen_offer: { offer_id: "o0", expected_offer_revenue: 2.5 } }, 200);
  assert.equal(acc.count, 2);
  assert.equal(acc.captured, 4.0);
  assert.deepEqual(acc.offers.o0, { offer_id: "o0", count: 2, revenue: 4.0 });
  assert.equal(acc.firstTs, 100); // set once, on the first event
});

test("accumulate: error event does not count", () => {
  const acc = { count: 5, captured: 9, offers: {}, segments: {}, firstTs: 1 };
  P.accumulate(acc, { error: "boom" }, 300);
  assert.equal(acc.count, 5);
  assert.equal(acc.captured, 9);
});

test("accumulate: null revenue counts but never poisons captured with NaN", () => {
  const acc = { count: 0, captured: 0, offers: {}, segments: {}, firstTs: null };
  P.accumulate(acc, { chosen_offer: { offer_id: "o1", expected_offer_revenue: null } }, 100);
  assert.equal(acc.count, 1);
  assert.equal(acc.captured, 0);
  assert.ok(Number.isFinite(acc.captured));
  assert.deepEqual(acc.offers, {}); // no revenue -> no leaderboard entry
});

test("accumulate: $0 chosen offer counts but stays off the leaderboard (locks the && guard)", () => {
  const acc = { count: 0, captured: 0, offers: {}, segments: {}, firstTs: null };
  P.accumulate(acc, { chosen_offer: { offer_id: "o2", expected_offer_revenue: 0 } }, 100);
  assert.equal(acc.count, 1);
  assert.deepEqual(acc.offers, {});
});

test("accumulate: missing segment leaves segments untouched", () => {
  const acc = { count: 0, captured: 0, offers: {}, segments: {}, firstTs: null };
  P.accumulate(acc, { chosen_offer: null }, 100);
  assert.deepEqual(acc.segments, {});
  assert.equal(acc.count, 1); // null chosen still counts as a decision
});

test("computeRate: normal + sub-floor", () => {
  assert.equal(P.computeRate(10, 1000, 1000 + 6000), 100); // 10 in 0.1 min
  assert.equal(P.computeRate(1, 1000, 1100), 0); // 0.0017 min, below 0.02 floor
  assert.equal(P.computeRate(5, null, 9999), 0); // no firstTs
});

test("mergeRecent: newest first, capped", () => {
  const prev = [{ id: "old" }];
  const batch = Array.from({ length: 5 }, (_, i) => ({ id: i }));
  const merged = P.mergeRecent(prev, batch, 300);
  assert.equal(merged[0].id, 4); // last batch item is newest -> index 0
  assert.equal(merged[merged.length - 1].id, "old");
  // only the last 12 of a batch are kept per call
  const big = Array.from({ length: 400 }, (_, i) => ({ id: i }));
  assert.equal(P.mergeRecent([], big, 300).length, 12);
  // cap bites once prev accumulates: 12 new + 295 old -> capped to 300
  const prevFull = Array.from({ length: 295 }, () => ({ id: "p" }));
  assert.equal(P.mergeRecent(prevFull, big, 300).length, 300);
});

test("visibleRecent: filter drops null-chosen, then caps at 40", () => {
  const list = [
    { chosen_offer: { offer_id: "x" } },
    { chosen_offer: null },
    { chosen_offer: { offer_id: "y" } },
  ];
  assert.equal(P.visibleRecent(list, null, 40).length, 3);
  const onlyX = P.visibleRecent(list, "x", 40);
  assert.equal(onlyX.length, 1);
  assert.equal(onlyX[0].chosen_offer.offer_id, "x");
  const big = Array.from({ length: 60 }, () => ({ chosen_offer: { offer_id: "z" } }));
  assert.equal(P.visibleRecent(big, null, 40).length, 40);
});

test("sparkPoints: <2 points -> null (no NaN)", () => {
  assert.equal(P.sparkPoints(null, 100, 48), null);
  assert.equal(P.sparkPoints([5], 100, 48), null);
});

test("sparkPoints: flat series uses the || 1 span guard", () => {
  const { pts } = P.sparkPoints([1, 1, 1], 100, 48);
  assert.ok(!pts.includes("NaN"));
  // all y at the baseline h-3 = 45.0
  pts.split(" ").forEach((p) => assert.equal(p.split(",")[1], "45.0"));
});

test("sparkPoints: rising series spans the width, no NaN", () => {
  const { pts, area } = P.sparkPoints([0, 10], 100, 48);
  assert.ok(pts.startsWith("0.0,"));
  assert.ok(pts.includes("100.0,"));
  assert.ok(!pts.includes("NaN") && !area.includes("NaN"));
});

test("niceStep: rounds to 1/2/5 x 10^k", () => {
  assert.equal(P.niceStep(1), 1);
  assert.equal(P.niceStep(1.5), 2);
  assert.equal(P.niceStep(3), 5);
  assert.equal(P.niceStep(7), 10);
  assert.equal(P.niceStep(23), 50);
  assert.equal(P.niceStep(0), 1); // guard
});

test("multiSparkChart: null when no series has >=2 points", () => {
  assert.equal(P.multiSparkChart([], 300, 120), null);
  assert.equal(P.multiSparkChart([{ key: "p50", values: [5] }], 300, 120), null);
});

test("multiSparkChart: shared y-scale, labeled ticks, no NaN", () => {
  const c = P.multiSparkChart(
    [
      { key: "p50", values: [10, 20, 15] },
      { key: "p99", values: [40, 60, 55] },
    ],
    300, 120, { padL: 30 }
  );
  assert.equal(c.lines.length, 2);
  // niceMax is a nice tick >= global max (60)
  assert.ok(c.niceMax >= 60);
  // ticks start at 0, ascend, are integers (ms)
  assert.equal(c.ticks[0].value, 0);
  assert.ok(c.ticks.every((t) => Number.isInteger(t.value)));
  assert.ok(c.ticks[c.ticks.length - 1].value >= 60);
  // top tick (niceMax) sits at the top pad (y=8), value 0 at the bottom
  const top = c.ticks[c.ticks.length - 1];
  assert.ok(Math.abs(top.y - 8) < 0.5);
  // no NaN anywhere in the point strings
  assert.ok(c.lines.every((ln) => !ln.pts.includes("NaN")));
  // x starts at the left gutter (padL) and reaches the right edge (w)
  assert.ok(c.lines[0].pts.startsWith("30.0,"));
  assert.ok(c.lines[0].pts.includes("300.0,"));
});

test("multiSparkChart: single-point-per-series after filter still handles flat", () => {
  const c = P.multiSparkChart([{ key: "p50", values: [5, 5, 5] }], 200, 100);
  assert.ok(c && c.niceMax >= 5 && !c.lines[0].pts.includes("NaN"));
});

test("clockLabel: HH:MM:SS zero-padded", () => {
  // 1970-01-01 00:00:05 UTC -> depends on TZ, so just assert shape
  const s = P.clockLabel(1785000000);
  assert.match(s, /^\d{2}:\d{2}:\d{2}$/);
});

test("probChart: null for <2 points", () => {
  assert.equal(P.probChart([], 400, 160), null);
  assert.equal(P.probChart([{ t: 1, median: 0.3 }], 400, 160), null);
});

test("probChart: labeled % y-axis + time x-axis, no NaN", () => {
  const series = [
    { t: 1785000000, median: 0.10 },
    { t: 1785000003, median: 0.22 },
    { t: 1785000006, median: 0.18 },
    { t: 1785000009, median: 0.31 },
  ];
  const c = P.probChart(series, 400, 160, { padL: 40, padB: 22 });
  // y ceiling is a nice value >= data max (31%) and <= 100
  assert.ok(c.yMaxPct >= 31 && c.yMaxPct <= 100);
  // y ticks are % strings, ascending from 0%
  assert.equal(c.yTicks[0].label, "0%");
  assert.ok(c.yTicks.every((t) => /^\d+%$/.test(t.label)));
  // x ticks are clock labels, first at left gutter (padL) and last at right edge
  assert.ok(c.xTicks.length >= 2);
  assert.match(c.xTicks[0].label, /^\d{2}:\d{2}:\d{2}$/);
  assert.equal(c.xTicks[0].x, 40);
  assert.equal(c.xTicks[c.xTicks.length - 1].x, 400);
  // line + area have no NaN
  assert.ok(!c.line.includes("NaN") && !c.area.includes("NaN"));
});

test("probChart: near-zero flat signal gets a 5% floor + whole-% ticks", () => {
  const c = P.probChart([{ t: 1, median: 0.002 }, { t: 4, median: 0.003 }], 400, 160);
  assert.ok(c.yMaxPct >= 5 && c.yMaxPct <= 100 && !c.line.includes("NaN"));
  assert.ok(c.yTicks.every((t) => /^\d+%$/.test(t.label)));  // no fractional %
});

test("shareChart: null for <2 points", () => {
  assert.equal(P.shareChart([], 400, 160), null);
  assert.equal(P.shareChart([{ t: 1, shares: { o1: 1 } }], 400, 160), null);
});

test("shareChart: one line per offer, fixed 0-100% axis, stable order, no NaN", () => {
  const series = [
    { t: 1785000000, shares: { o1: 0.6, o2: 0.4 } },
    { t: 1785000001, shares: { o1: 0.5, o2: 0.3, o3: 0.2 } },
    { t: 1785000002, shares: { o1: 0.7, o2: 0.3 } },  // o3 missing -> 0%
  ];
  const c = P.shareChart(series, 400, 160, { palette: ["#a", "#b", "#c"] });
  // one line per distinct offer, ordered by total share desc (o1>o2>o3)
  assert.deepEqual(c.lines.map((l) => l.offer_id), ["o1", "o2", "o3"]);
  // fixed 0..100% axis with 25% ticks
  assert.deepEqual(c.yTicks.map((t) => t.label), ["0%", "25%", "50%", "75%", "100%"]);
  // colors assigned stably from palette
  assert.equal(c.lines[0].color, "#a");
  // no NaN in any line (missing offer coerced to 0)
  assert.ok(c.lines.every((l) => !l.pts.includes("NaN")));
  // x ticks are clock labels; first at padL, last at right edge
  assert.equal(c.xTicks[0].x, 44);
  assert.equal(c.xTicks[c.xTicks.length - 1].x, 400);
});

test("topOffers: empty, order, slice", () => {
  assert.deepEqual(P.topOffers({}), []);
  const rows = P.topOffers({ a: { offer_id: "a", revenue: 1 }, b: { offer_id: "b", revenue: 9 } });
  assert.deepEqual(rows.map((r) => r.offer_id), ["b", "a"]);
  const ten = {};
  for (let i = 0; i < 10; i++) ten["o" + i] = { offer_id: "o" + i, revenue: i };
  assert.equal(P.topOffers(ten).length, 8);
});

test("barPct: divide-by-zero floors to 0", () => {
  assert.equal(P.barPct(5, 10), 50);
  assert.equal(P.barPct(5, 0), 0);
  assert.equal(P.barPct(0, 0), 0);
});

test("segmentRows: empty, order, independent rounding artifact", () => {
  assert.deepEqual(P.segmentRows({}), []);
  assert.deepEqual(P.segmentRows({ whale: 3, minnow: 1 }), [
    ["whale", 3, 75],
    ["minnow", 1, 25],
  ]);
  const three = P.segmentRows({ a: 1, b: 1, c: 1 });
  assert.deepEqual(three.map((r) => r[2]), [33, 33, 33]);
  assert.equal(three.reduce((s, r) => s + r[2], 0), 99); // pins the artifact
});

test("sortByRevenue: non-mutating, null sinks to bottom", () => {
  const input = [
    { expected_offer_revenue: 1 },
    { expected_offer_revenue: null },
    { expected_offer_revenue: 5 },
  ];
  const sorted = P.sortByRevenue(input);
  assert.deepEqual(sorted.map((o) => o.expected_offer_revenue), [5, 1, null]);
  assert.equal(input[0].expected_offer_revenue, 1); // input untouched
});

test("maxRevenue: empty + null-first", () => {
  assert.equal(P.maxRevenue([]), 0);
  assert.equal(P.maxRevenue([{ expected_offer_revenue: null }]), 0);
  assert.equal(P.maxRevenue([{ expected_offer_revenue: 7 }]), 7);
});

test("offerBarPct: 3% floor, null -> 3", () => {
  assert.equal(P.offerBarPct(null, 10), 3);
  assert.equal(P.offerBarPct(0, 10), 3);
  assert.equal(P.offerBarPct(5, 10), 50);
  assert.equal(P.offerBarPct(5, 0), 3);
});

test("tween kernel: endpoints exact, midpoint eased", () => {
  assert.equal(P.easeOutCubic(0), 0);
  assert.equal(P.easeOutCubic(1), 1);
  assert.ok(Math.abs(P.easeOutCubic(0.5) - 0.875) < 1e-9);
  assert.equal(P.tweenValue(0, 10, 0), 0);
  assert.equal(P.tweenValue(0, 10, 1), 10);
  assert.equal(P.tweenValue(0, 10, 0.5), 8.75);
});
