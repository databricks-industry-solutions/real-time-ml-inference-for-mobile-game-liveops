"""Unit tests for the streaming rolling-window feature wiring in features.store.

These are the cheap, client-free checks: the column contract matches the schema
module, and the `ensure_feature_serving` retry/error behavior is correct without
a live FeatureEngineeringClient. The materialize/create_feature calls themselves
need a live client and are covered by the pipeline runs, not here.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from offer_recommender import schema
from offer_recommender.features import store


def test_streaming_columns_match_windows() -> None:
    """The store materializes every WINDOWS feature (superset of the model inputs)."""
    assert tuple(schema.WINDOWS) == store.STREAMING_FEATURE_COLUMNS


def test_operator_inputs_are_value_prefixed() -> None:
    """All _operator inputs must carry the `value.` prefix (StreamSource payload).

    Over a Kafka StreamSource the payload is under the `value` struct; bare column
    refs raise InvalidParameterValue at create_feature time. The prefix is
    registration-only and does not propagate to training-set joins or serving
    lookups, which stay bare.
    """
    from databricks.feature_engineering.entities import Count

    # "login" virtual signal -> Count over the entity key
    count_op = store._operator("count", "login")
    assert isinstance(count_op, Count)
    assert count_op._inputs[0].startswith("value."), (
        f"Count input must be value.-prefixed, got {count_op._inputs[0]!r}"
    )

    # Real signal -> Sum/Avg over value.<signal>
    for op_name, signal in [
        ("sum", "gem_spend"),
        ("sum", "offer_shown"),
        ("sum", "match_won"),
        ("avg", "gem_balance"),
    ]:
        op = store._operator(op_name, signal)
        assert op._inputs[0] == f"value.{signal}", (
            f"_operator({op_name!r}, {signal!r})._inputs[0] = {op._inputs[0]!r}; "
            f"expected 'value.{signal}'"
        )

    # Unknown operator must raise immediately.
    with pytest.raises(ValueError, match="Unknown streaming"):
        store._operator("median", "gem_spend")


class _FakeFE:
    """Minimal FeatureEngineeringClient stub for ensure_feature_serving tests.

    Endpoint absent at first (get raises); create raises an async-teardown
    "in progress" error `create_fail_times` times before succeeding.
    """

    def __init__(self, create_fail_times: int) -> None:
        self._create_fail_times = create_fail_times
        self.create_attempts = 0

    def delete_feature_serving_endpoint(self, name):
        pass

    def delete_feature_spec(self, name):
        pass

    def create_feature_spec(self, name, features):
        pass

    def get_feature_serving_endpoint(self, name):
        raise RuntimeError("RESOURCE_DOES_NOT_EXIST")  # not found -> create path

    def create_feature_serving_endpoint(self, name, config):
        self.create_attempts += 1
        if self.create_attempts <= self._create_fail_times:
            raise RuntimeError("Endpoint update in progress")


def _patch_ensure(monkeypatch) -> None:
    """Stub the spec-feature builders, the inline FE SDK import, and the sleep."""
    monkeypatch.setattr(store, "build_lookup_features", lambda fe, uc: [])
    monkeypatch.setattr(store, "build_streaming_features", lambda fe, uc: [])
    monkeypatch.setattr(store.time, "sleep", lambda _s: None)
    # ensure_feature_serving imports these inline from the (uninstalled-in-dev) FE SDK.
    fake_mod = ModuleType(
        "databricks.feature_engineering.entities.feature_serving_endpoint"
    )
    fake_mod.EndpointCoreConfig = lambda **kw: None
    fake_mod.ServedEntity = lambda **kw: None
    for name in (
        "databricks",
        "databricks.feature_engineering",
        "databricks.feature_engineering.entities",
    ):
        sys.modules.setdefault(name, ModuleType(name))
    sys.modules["databricks.feature_engineering.entities.feature_serving_endpoint"] = (
        fake_mod
    )


def test_ensure_feature_serving_retries_async_teardown(monkeypatch) -> None:
    """reset recreate retries past a transient async-teardown error, then succeeds."""
    _patch_ensure(monkeypatch)
    fe = _FakeFE(create_fail_times=3)
    uc = SimpleNamespace(catalog="c", schema="s")

    result = store.ensure_feature_serving(fe, uc, reset=True)

    assert result == store.ONLINE_FS_ENDPOINT
    assert fe.create_attempts == 4  # 3 transient failures + 1 success


def test_build_streaming_features_uses_sfv_aggregation(monkeypatch) -> None:
    """build_streaming_features delegates unconditionally to _streaming_agg_features."""
    monkeypatch.setattr(
        store,
        "_streaming_agg_features",
        lambda fe, uc, reset=False: ["agg-feature"] * 5,
    )
    result = store.build_streaming_features(fe=object(), uc=object())
    assert result == ["agg-feature"] * 5


class _FakeStreamingFE:
    """Records delete_feature / create_feature calls for the reset test.

    create_feature succeeds fresh (returns a sentinel); the AlreadyExists ->
    get_feature fallback is never exercised here (each col is created once).
    list_materialized_features returns a canned list so the reset path deletes
    materializations before the feature.
    """

    def __init__(self, materialized_per_feature: int = 1) -> None:
        self._materialized_per_feature = materialized_per_feature
        self.deleted_features: list[str] = []
        self.deleted_materializations: list[str] = []
        self.created_features: list[str] = []

    def list_materialized_features(self, feature_name):
        return [f"{feature_name}#mf{i}" for i in range(self._materialized_per_feature)]

    def delete_materialized_feature(self, materialized_feature):
        self.deleted_materializations.append(materialized_feature)

    def delete_feature(self, full_name):
        self.deleted_features.append(full_name)

    def create_feature(self, name, **kwargs):
        self.created_features.append(name)
        return SimpleNamespace(name=name, full_name=f"c.s.{name}")


def test_build_streaming_features_reset_deletes_stale_features() -> None:
    """reset=True deletes each streaming feature (materializations first) then creates."""
    fe = _FakeStreamingFE(materialized_per_feature=2)
    uc = SimpleNamespace(catalog="c", schema="s")
    expected = [f"c.s.{col}" for col in store.STREAMING_FEATURE_COLUMNS]

    result = store.build_streaming_features(fe, uc, reset=True)

    assert fe.deleted_features == expected
    # Materializations deleted BEFORE delete_feature (delete_feature refuses while
    # a feature has materializations): 2 per feature, all cleared.
    assert len(fe.deleted_materializations) == 2 * len(store.STREAMING_FEATURE_COLUMNS)
    # After deletion each feature is re-created fresh against the new stream.
    assert fe.created_features == list(store.STREAMING_FEATURE_COLUMNS)
    assert len(result) == len(store.STREAMING_FEATURE_COLUMNS)


def test_build_streaming_features_no_reset_keeps_features() -> None:
    """reset=False (daily re-run) never deletes features; just registers/fetches."""
    fe = _FakeStreamingFE()
    uc = SimpleNamespace(catalog="c", schema="s")

    store.build_streaming_features(fe, uc, reset=False)

    assert fe.deleted_features == []
    assert fe.deleted_materializations == []
    assert fe.created_features == list(store.STREAMING_FEATURE_COLUMNS)


def test_ensure_feature_serving_raises_if_never_recreatable(monkeypatch) -> None:
    """A create that never settles raises (job fails loudly, not green-but-broken)."""
    _patch_ensure(monkeypatch)
    monkeypatch.setattr(store, "_FS_RECREATE_RETRIES", 3)
    fe = _FakeFE(create_fail_times=99)
    uc = SimpleNamespace(catalog="c", schema="s")

    with pytest.raises(RuntimeError, match="still not recreatable"):
        store.ensure_feature_serving(fe, uc, reset=True)
    assert fe.create_attempts == 3


class _FakeShowSpark:
    """Stub of spark.sql('SHOW TABLES ... LIKE prefix*') returning scripted table
    rows per poll, so _wait_online_materialized can be driven deterministically."""

    def __init__(self, per_poll: list[list[str]]) -> None:
        self._per_poll = per_poll
        self._i = 0

    def sql(self, _q):
        names = self._per_poll[min(self._i, len(self._per_poll) - 1)]
        self._i += 1
        return SimpleNamespace(collect=lambda: [{"tableName": n} for n in names])


def test_wait_online_materialized_returns_when_tables_appear(monkeypatch) -> None:
    """The gate returns once a non-_latest_view synced table exists for the prefix."""
    monkeypatch.setattr(store.time, "sleep", lambda _s: None)
    uc = SimpleNamespace(catalog="c", schema="s")
    # poll 1: only the latest_view (not ready); poll 2: the synced data table appears.
    spark = _FakeShowSpark([["p_latest_view"], ["p_latest_view", "p_ab12cd"]])
    store._wait_online_materialized(spark, uc, ["p"], attempts=5, delay_s=0)


def test_wait_online_materialized_times_out(monkeypatch) -> None:
    """Only ever seeing the _latest_view (no data table) times out loudly."""
    monkeypatch.setattr(store.time, "sleep", lambda _s: None)
    uc = SimpleNamespace(catalog="c", schema="s")
    spark = _FakeShowSpark([["p_latest_view"]])
    with pytest.raises(TimeoutError, match="never appeared"):
        store._wait_online_materialized(spark, uc, ["p"], attempts=3, delay_s=0)


def test_wait_online_materialized_noop_without_spark() -> None:
    """No spark (unit path) -> gate is a no-op, never raises."""
    store._wait_online_materialized(
        None, SimpleNamespace(catalog="c", schema="s"), ["p"]
    )


class _ExistingEndpointFE:
    """Stub whose get_feature_serving_endpoint returns an endpoint in a given state,
    recording whether delete + create were called."""

    def __init__(self, state: dict) -> None:
        self._state = state
        self.deleted = False
        self.created = False

    def create_feature_spec(self, name, features):
        pass

    def get_feature_serving_endpoint(self, name):
        return SimpleNamespace(name=name, state=self._state)

    def delete_feature_serving_endpoint(self, name):
        self.deleted = True

    def create_feature_serving_endpoint(self, name, config):
        self.created = True


def test_ensure_feature_serving_short_circuits_when_healthy(monkeypatch) -> None:
    """A healthy existing endpoint is left untouched (no delete, no recreate)."""
    _patch_ensure(monkeypatch)
    fe = _ExistingEndpointFE(state={"config_update": "NOT_UPDATING", "ready": "READY"})
    store.ensure_feature_serving(fe, SimpleNamespace(catalog="c", schema="s"))
    assert not fe.deleted and not fe.created


def test_ensure_feature_serving_recreates_broken_endpoint(monkeypatch) -> None:
    """A stale UPDATE_FAILED endpoint is deleted + recreated (not green-but-broken)."""
    _patch_ensure(monkeypatch)
    fe = _ExistingEndpointFE(
        state={"config_update": "UPDATE_FAILED", "ready": "NOT_READY"}
    )
    store.ensure_feature_serving(fe, SimpleNamespace(catalog="c", schema="s"))
    assert fe.deleted and fe.created


def test_create_match_stream_recreate_false_reuses_existing() -> None:
    """recreate=False reuses an existing stream without delete/create."""
    calls = {"deleted": False, "created": False}

    class _FE:
        def get_stream(self, *, name):
            return object()

        def delete_stream(self, *, name):
            calls["deleted"] = True

        def create_stream(self, **kw):
            calls["created"] = True

    uc = SimpleNamespace(
        catalog="c",
        schema="s",
        table=lambda n: f"c.s.{n}",
        volume_path="/Volumes/c/s/v",
    )
    out = store.create_match_stream(_FE(), uc, "b:9093", recreate=False)
    assert out == "c.s.match_events"
    assert calls["deleted"] is False and calls["created"] is False


def test_create_match_stream_require_exists_missing_raises() -> None:
    """recreate=False + require_exists=True (the CONSUMER path, e.g. preprocess)
    raises if the stream does not exist -- a consumer must not silently create it."""

    class _FE:
        def get_stream(self, *, name):
            raise Exception("RESOURCE_DOES_NOT_EXIST")

    uc = SimpleNamespace(
        catalog="c",
        schema="s",
        table=lambda n: f"c.s.{n}",
        volume_path="/Volumes/c/s/v",
    )
    with pytest.raises(RuntimeError, match="setup_stream"):
        store.create_match_stream(
            _FE(), uc, "b:9093", recreate=False, require_exists=True
        )


def test_create_match_stream_creates_if_absent_when_not_required() -> None:
    """recreate=False + require_exists=False (setup_stream first-run path): a missing
    stream is CREATED, not an error. Falls through to create_stream."""

    class _FE:
        def __init__(self) -> None:
            self.created = False

        def get_stream(self, *, name):
            raise Exception("RESOURCE_DOES_NOT_EXIST")

        def delete_stream(self, *, name):
            pass

        def create_stream(self, **kwargs):
            self.created = True

    uc = SimpleNamespace(
        catalog="c",
        schema="s",
        table=lambda n: f"c.s.{n}",
        volume_path="/Volumes/c/s/v",
    )
    fe = _FE()
    # spark=None -> skips the _wait_for_table poll; returns the ingest table name.
    result = store.create_match_stream(fe, uc, "b:9093", recreate=False)
    assert fe.created and result == "c.s.match_events"


class _FakeRequestFE:
    """Records delete_feature / create_feature calls for _request_features tests.

    delete_feature is best-effort (no error). create_feature returns a sentinel.
    get_feature must never be called (the old AlreadyExists fallback is gone).
    """

    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.created: list[dict] = []

    def delete_feature(self, *, full_name: str) -> None:
        self.deleted.append(full_name)

    def create_feature(self, *, name: str, **kwargs) -> SimpleNamespace:
        self.created.append({"name": name, "kwargs": kwargs})
        return SimpleNamespace(name=name, full_name=f"c.s.{name}")

    def get_feature(self, **kwargs) -> None:
        raise AssertionError(
            "get_feature must not be called; delete-then-create replaced AlreadyExists fallback"
        )


def _make_fake_fe_sdk(monkeypatch) -> None:
    """Stub the FE SDK imports used by _request_features."""
    fake_entities = ModuleType("databricks.feature_engineering.entities")
    # Provide minimal stand-ins so the import block inside _request_features resolves.
    fake_entities.ColumnSelection = lambda col: SimpleNamespace(col=col)
    fake_entities.FieldDefinition = lambda **kw: SimpleNamespace(**kw)
    fake_entities.RequestSource = lambda **kw: SimpleNamespace(**kw)

    class _ScalarDataType:
        DOUBLE = "DOUBLE"
        STRING = "STRING"

    fake_entities.ScalarDataType = _ScalarDataType

    for mod_name in (
        "databricks",
        "databricks.feature_engineering",
    ):
        sys.modules.setdefault(mod_name, ModuleType(mod_name))
    sys.modules["databricks.feature_engineering.entities"] = fake_entities


def test_request_features_delete_then_create(monkeypatch) -> None:
    """_request_features deletes each col first, then creates WITHOUT entity/timeseries.

    The delete-then-create pattern clears any stale 0.16.0-contract registration
    (which stored entity+timeseries on a RequestSource, unreadable by 0.16.1) before
    re-registering fresh. get_feature must never be called.
    """
    _make_fake_fe_sdk(monkeypatch)
    fe = _FakeRequestFE()
    uc = SimpleNamespace(catalog="c", schema="s")

    from offer_recommender.schema import (
        OFFER_CATEGORICAL_FEATURES,
        OFFER_NUMERICAL_FEATURES,
        REALTIME_NUMERICAL_FEATURES,
    )

    result = store._request_features(fe, uc)

    expected_cols = (
        list(OFFER_NUMERICAL_FEATURES)
        + list(OFFER_CATEGORICAL_FEATURES)
        + list(REALTIME_NUMERICAL_FEATURES)
    )
    # Every col was deleted (best-effort) before create.
    assert fe.deleted == [f"c.s.{c}" for c in expected_cols]
    # Every col was created exactly once.
    assert [r["name"] for r in fe.created] == expected_cols
    # entity and timeseries_column must NOT appear in any create call.
    for call in fe.created:
        assert "entity" not in call["kwargs"], (
            f"entity leaked into create_feature for {call['name']!r}"
        )
        assert "timeseries_column" not in call["kwargs"], (
            f"timeseries_column leaked into create_feature for {call['name']!r}"
        )
    assert len(result) == len(expected_cols)


def test_request_features_delete_not_found_is_silent(monkeypatch) -> None:
    """A not-found delete (first-ever run) must not abort the loop."""
    _make_fake_fe_sdk(monkeypatch)

    class _RaisingDeleteFE(_FakeRequestFE):
        def delete_feature(self, *, full_name: str) -> None:
            raise RuntimeError("RESOURCE_DOES_NOT_EXIST")

    fe = _RaisingDeleteFE()
    uc = SimpleNamespace(catalog="c", schema="s")

    from offer_recommender.schema import (
        OFFER_CATEGORICAL_FEATURES,
        OFFER_NUMERICAL_FEATURES,
        REALTIME_NUMERICAL_FEATURES,
    )

    result = store._request_features(fe, uc)

    expected_len = (
        len(OFFER_NUMERICAL_FEATURES)
        + len(OFFER_CATEGORICAL_FEATURES)
        + len(REALTIME_NUMERICAL_FEATURES)
    )
    # create_feature was still called for every col despite delete errors.
    assert len(fe.created) == expected_len
    assert len(result) == expected_len


class _OnlineStoreFE:
    """Stub for ensure_online_store: get returns a store (or None), records the
    create call. update_online_store must NEVER be called (resizing a live store
    rotates Lakebase credentials and breaks the deployed serving endpoints' auth)."""

    def __init__(self, existing_capacity):
        self._existing = (
            None
            if existing_capacity is None
            else SimpleNamespace(capacity=existing_capacity)
        )
        self.created = None

    def get_online_store(self, *, name):
        return self._existing

    def create_online_store(self, *, name, capacity):
        self.created = capacity
        return SimpleNamespace(capacity=capacity)

    def update_online_store(self, *, name, capacity):
        raise AssertionError(
            "ensure_online_store must never resize a live store: a resize rotates "
            "Lakebase creds and wedges the serving endpoints' auth -> 'No available "
            "connection in 2s'."
        )


def test_ensure_online_store_creates_at_default_capacity() -> None:
    """Absent store -> created at the default capacity, no resize."""
    fe = _OnlineStoreFE(existing_capacity=None)
    store.ensure_online_store(fe)
    assert fe.created == store.ONLINE_STORE_CAPACITY


def test_ensure_online_store_leaves_existing_untouched() -> None:
    """Existing store (any size) is returned AS-IS: never created, never resized."""
    fe = _OnlineStoreFE(existing_capacity="CU_1")
    result = store.ensure_online_store(fe)
    assert fe.created is None
    assert result.capacity == "CU_1"
