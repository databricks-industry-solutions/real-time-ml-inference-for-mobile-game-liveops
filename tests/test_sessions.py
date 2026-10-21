"""Tests for data.sessions — the shared per-player session timeline (numpy core).

These tests verify:
1. window_counts_batch returns exactly the keys in schema.STREAMING_NUMERICAL_FEATURES.
2. The invariant: window_counts_batch == aggregate of player_events in-window (exactly),
   for all 5 streaming features over a genuinely non-empty rolling window.
3. Higher-rate segments produce more events than lower-rate ones.
"""

import numpy as np

from offer_recommender import schema
from offer_recommender.data import sessions


def test_window_counts_keys_match_schema():
    pid = np.array(["player_000001"])
    seg = np.array(["whale"])
    at = np.array([1_000_000.0])
    out = sessions.window_counts_batch(pid, seg, at, 0.0, 2_000_000.0, seed=17)
    # window_counts_batch returns all schema.WINDOWS keys (6), which is a superset of
    # STREAMING_NUMERICAL_FEATURES (5 model inputs); match_events_15m is the extra one.
    assert set(out) == set(schema.WINDOWS)
    assert set(schema.STREAMING_NUMERICAL_FEATURES).issubset(set(out))
    for v in out.values():
        assert v.shape == (1,) and np.isfinite(v).all()


def test_window_counts_matches_bruteforce_over_events():
    """window_counts_batch must match direct player_events reductions for all 6 WINDOWS.

    Uses a 1-hour observation span with a whale segment (~160 events) so that all
    15-min rolling windows near at_ts=max(event times) are reliably non-empty (~40
    events).  Validates the match-outcome set (R2):
      gems_spent_15m, avg_gem_balance_15m, offers_seen_15m,
      match_wins_15m, match_losses_15m, match_events_15m.

    Window boundary convention mirrors window_counts_batch: (at - W, at] — strict
    lower bound, inclusive upper.
    """
    pid, seg, seed = "player_000007", "whale", 17
    # 1-hour span: ~160 whale events in 3600 s -> ~40 per 15-min window.
    ws, we = 0.0, 3600.0
    ev = sessions.player_events(pid, seg, ws, we, seed)
    at = float(ev["t"].max())

    W_15m = 15 * 60.0  # 900 s — all features use 15-min windows

    # Window mask — strict lower bound, inclusive upper (mirrors window_counts_batch).
    mask_15m = (ev["t"] > at - W_15m) & (ev["t"] <= at)
    n_15m = int(mask_15m.sum())

    # Brute-force expected values using the same reductions as store._operator:
    #   gems_spent_15m      : sum of gem_spend over the 15-min window.
    #   avg_gem_balance_15m : mean of gem_balance in 15-min window; 0.0 if empty.
    #   offers_seen_15m     : sum of offer_shown (0/1) over the 15-min window.
    #   match_wins_15m      : sum of match_won (0/1) over the 15-min window.
    #   match_losses_15m    : sum of match_lost (0/1) over the 15-min window.
    #   match_events_15m    : row count (every event is a login) in the 15-min window.
    exp_gems_spent_15m = float((ev["gem_spend"] * mask_15m).sum())
    exp_avg_gem_balance_15m = (
        float(ev["gem_balance"][mask_15m].mean()) if n_15m > 0 else 0.0
    )
    exp_offers_seen_15m = float((ev["offer_shown"] * mask_15m).sum())
    exp_match_wins_15m = float((ev["match_won"] * mask_15m).sum())
    exp_match_losses_15m = float((ev["match_lost"] * mask_15m).sum())
    exp_match_events_15m = float(n_15m)

    # Sanity: window must be non-empty so assertions are substantive.
    assert mask_15m.any(), "15-min window is empty — choose a different at_ts or span"

    # At least one feature must be positive (non-trivial signal routing check).
    assert (
        exp_gems_spent_15m > 0
        or exp_avg_gem_balance_15m > 0
        or exp_offers_seen_15m > 0
        or exp_match_wins_15m > 0
        or exp_match_losses_15m > 0
    ), "All features are zero — window may be degenerate"

    got = sessions.window_counts_batch(
        np.array([pid]), np.array([seg]), np.array([at]), ws, we, seed
    )

    # Float-valued aggregates: tight tolerance.
    np.testing.assert_allclose(got["gems_spent_15m"][0], exp_gems_spent_15m, rtol=1e-9)
    np.testing.assert_allclose(
        got["avg_gem_balance_15m"][0], exp_avg_gem_balance_15m, rtol=1e-9
    )
    np.testing.assert_allclose(
        got["offers_seen_15m"][0], exp_offers_seen_15m, rtol=1e-9
    )
    np.testing.assert_allclose(got["match_wins_15m"][0], exp_match_wins_15m, rtol=1e-9)
    np.testing.assert_allclose(
        got["match_losses_15m"][0], exp_match_losses_15m, rtol=1e-9
    )
    np.testing.assert_allclose(
        got["match_events_15m"][0], exp_match_events_15m, rtol=1e-9
    )


def test_higher_rate_segment_has_more_events():
    pids = np.array([f"player_{i:06d}" for i in range(400)])
    whales = sessions.event_count(pids, np.array(["whale"] * 400), 17)
    lapsed = sessions.event_count(pids, np.array(["lapsed"] * 400), 17)
    assert whales.mean() > lapsed.mean()
