"""Contract tests for the feature-set split after the match-model rework."""

from __future__ import annotations

from offer_recommender import schema


def test_streaming_model_inputs_exclude_match_events():
    """STREAMING_NUMERICAL_FEATURES (model inputs) is the 5-set without match_events_15m."""
    assert schema.STREAMING_NUMERICAL_FEATURES == (
        "gems_spent_15m",
        "avg_gem_balance_15m",
        "offers_seen_15m",
        "match_wins_15m",
        "match_losses_15m",
    )
    assert "match_events_15m" not in schema.STREAMING_NUMERICAL_FEATURES


def test_windows_is_superset_with_match_events():
    """WINDOWS (materialized SFV set) is the 6-set including match_events_15m."""
    assert set(schema.WINDOWS) == set(schema.STREAMING_NUMERICAL_FEATURES) | {
        "match_events_15m"
    }
    assert schema.WINDOWS["match_events_15m"] == ("count", "login", 900.0)
    assert schema.WINDOWS["match_wins_15m"] == ("sum", "match_won", 900.0)
    assert schema.WINDOWS["match_losses_15m"] == ("sum", "match_lost", 900.0)


def test_realtime_passthrough_has_match_count():
    """Passthrough swaps session_depth for match_count."""
    assert schema.REALTIME_NUMERICAL_FEATURES == (
        "gem_balance",
        "minutes_since_last_purchase",
        "match_count",
    )


def test_log1p_is_subset_of_numerical():
    """LOG1P must be a subset of the model input set (transform only touches inputs)."""
    assert set(schema.LOG1P_FEATURES) <= set(schema.NUMERICAL_FEATURES)
    assert "match_wins_15m" in schema.LOG1P_FEATURES
    assert "match_losses_15m" in schema.LOG1P_FEATURES
    assert "match_events_15m" not in schema.LOG1P_FEATURES
