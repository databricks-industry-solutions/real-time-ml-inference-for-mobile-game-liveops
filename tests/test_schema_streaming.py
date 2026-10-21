from offer_recommender import schema


def test_streaming_features_declared_and_appended():
    expected = (
        "gems_spent_15m",
        "avg_gem_balance_15m",
        "offers_seen_15m",
        "match_wins_15m",
        "match_losses_15m",
    )
    assert expected == schema.STREAMING_NUMERICAL_FEATURES
    # appended LAST so the trained numeric-block ordering is deterministic
    assert schema.NUMERICAL_FEATURES[-5:] == expected
    # no duplicates across the numeric groups
    assert len(schema.NUMERICAL_FEATURES) == len(set(schema.NUMERICAL_FEATURES))
    # gems_spent_15m is money-like and must be log1p-transformed
    assert "gems_spent_15m" in schema.LOG1P_FEATURES


def test_streaming_windows_are_short_for_demo_viable_warmup():
    # A managed streaming online aggregation is empty until its window elapses,
    # so the largest window bounds serving warm-up. Keep them <=15 min.
    for _op, _signal, wsec in schema.WINDOWS.values():
        assert wsec <= 15 * 60


def test_windows_cover_streaming_features_and_map_to_schema():
    from offer_recommender.schema import STREAMING_NUMERICAL_FEATURES, WINDOWS

    # WINDOWS has one extra entry (match_events_15m) that is materialized and served
    # but is NOT a model input (collinear with wins + losses). STREAMING_NUMERICAL_FEATURES
    # is a strict subset.
    assert set(STREAMING_NUMERICAL_FEATURES).issubset(set(WINDOWS))
    assert "match_events_15m" in WINDOWS
    assert "match_events_15m" not in STREAMING_NUMERICAL_FEATURES
    # each entry is (operator, signal, window_seconds>0)
    for op, signal, wsec in WINDOWS.values():
        assert op in {"count", "sum", "avg"}
        assert isinstance(signal, str) and signal
        assert wsec > 0
