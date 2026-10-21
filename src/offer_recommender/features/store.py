"""Feature store wiring on the new Feature Views framework (Batch + realtime).

Uses databricks-feature-engineering == 0.16.1 (Public Preview Feature Views), NOT
the legacy create_table / FeatureLookup / publish_table API. Two feature groups
keyed by `player_id`, both defined as `Feature` objects and both fed to serving
through the Lakebase-backed online store:

- OFFLINE (slow-moving): a Batch Feature View over the offline player-feature
  Delta table. Lifetime spend, tenure, historical LTV, etc.
- REAL-TIME: a Feature View over the real-time player-feature Delta table that
  the synthetic stream (or, in production, a Kafka Streaming Feature View) keeps
  fresh. Events-this-session, gem balance, session depth, etc.
- STREAMING AGGREGATES: 6 rolling-window aggregates (gems_spent_15m, avg_gem_balance_15m,
  match_events_15m, ...) looked up online by player_id. Managed Streaming Feature View:
  each column is an AggregationFunction + RollingWindow over the Kafka match StreamSource
  (see `_streaming_agg_features`). Requires the match stream (create_match_stream)
  to exist and the producer to emit the source signals.

The OFFLINE and REAL-TIME groups are `ColumnSelection` (latest-value): the model
trains on the SparkML-transformed impression log where those features are recorded
as point-in-time values, so serving must emit the same values. They materialize as
an always-current VIEW and so cannot take a `CronSchedule` (the client rejects it);
each uses a `TableTrigger` that re-syncs to the online store on every commit to the
source table.

The STREAMING group is a managed Streaming Feature View: each column is an
`AggregationFunction` + `RollingWindow` over the Kafka match `StreamSource`
(mirroring `schema.WINDOWS`). Training values are recomputed point-in-time from the
stream ingestion history; a freshly-created stream reads from LATEST offset, so a
`StreamBackfillSource` Delta table (see `data/match_backfill.py`) is MERGEd into the
ingestion table to give training real history without a ~24h warm-up. This path
requires the Kafka broker and the producer emitting the source signals.

The model is logged with `fe.log_model(training_set=...)` so the serving endpoint
auto-resolves all groups by `player_id` at inference.

The legacy on-demand `session_urgency` FeatureFunction is removed: it was never a
model input (absent from NUMERICAL_FEATURES/CATEGORICAL_FEATURES) and the new
framework has no on-demand-UDF equivalent.
"""

from __future__ import annotations

import contextlib
import time

from offer_recommender.config import (
    APP_NAME,
    FEATURE_SPEC_NAME,
    KAFKA_CERTS_SUBDIR,
    KAFKA_JKS_PASSWORD_KEY,
    KAFKA_KEYSTORE_FILE,
    KAFKA_SECRET_SCOPE,
    KAFKA_TRUSTSTORE_FILE,
    MATCH_STREAM_NAME,
    MATCH_TOPIC,
    ONLINE_FS_ENDPOINT,
    ONLINE_STORE_CAPACITY,
    ONLINE_STORE_NAME,
    TABLE_MATCH_INGEST,
    TABLE_OFFLINE_FEATURES,
    UCConfig,
    get_kafka_uc_connection,
)
from offer_recommender.logging_config import get_logger
from offer_recommender.schema import (
    ENTITY_KEY,
    EVENT_TIMESTAMP,
    OFFER_CATEGORICAL_FEATURES,
    OFFER_NUMERICAL_FEATURES,
    OFFLINE_CATEGORICAL_FEATURES,
    OFFLINE_NUMERICAL_FEATURES,
    REALTIME_NUMERICAL_FEATURES,
    TARGET_PURCHASED,
    TARGET_REVENUE_D7,
    WINDOWS,
)

logger = get_logger(__name__)

# The offline feature group carries both the slow-moving numeric player features
# and the player-attribute categoricals (country, segment); serving looks up all
# of them by player_id. Keep numerics first so the group's split into
# numeric/categorical Features stays aligned with the model's input order.
OFFLINE_FEATURE_COLUMNS = OFFLINE_NUMERICAL_FEATURES + OFFLINE_CATEGORICAL_FEATURES

# Streaming rolling-window aggregates materialized online + looked up by player_id.
# This is the FULL WINDOWS set (6), a superset of the model's 5 STREAMING_NUMERICAL_
# FEATURES: match_events_15m is materialized + served + shown in the app but is not a
# model input (== wins + losses). Decoupled so the SFV serves it without the net
# consuming it.
STREAMING_FEATURE_COLUMNS = tuple(WINDOWS.keys())

# Materialized-table name prefixes for the online store sync (one per group).
OFFLINE_SERVING_PREFIX = "offer_features_offline_serving"
STREAMING_SERVING_PREFIX = "offer_features_streaming_serving"

# On a reset, the deleted feature-serving endpoint tears down asynchronously; the
# recreate is retried until that settles (~5 min ceiling) so the job never returns
# SUCCESS with the endpoint still gone.
_FS_RECREATE_RETRIES = 20
_FS_RECREATE_BACKOFF_S = 15


def get_client():
    """Return a FeatureEngineeringClient."""
    from databricks.feature_engineering import FeatureEngineeringClient

    return FeatureEngineeringClient()


def ensure_online_store(fe, capacity: str | None = None):
    """Create the Lakebase-backed online store if absent; otherwise leave it AS-IS.

    Create-only by design. `get_online_store` returns None when absent (does NOT
    raise); on the first install we create at ONLINE_STORE_CAPACITY. On every
    subsequent run we return the existing store UNTOUCHED and never call
    `update_online_store`.

    WHY no reconcile/resize: resizing a live online store is an admin/failover
    operation that rotates the Lakebase instance's credentials, which invalidates the
    OAuth token / password the already-deployed serving endpoints baked in at deploy
    time. The endpoints then fail Lakebase auth ("password authentication failed" /
    KeyError 'OAUTH_TOKEN') and every FS lookup times out with "No available
    connection in 2s" because the client pool can never fill. A resize on the daily
    hot path would silently break serving until the endpoints are redeployed. If the
    store genuinely needs a bigger size, change RECO_ONLINE_STORE_CAPACITY and resize
    out-of-band, THEN redeploy the serving endpoints so they re-mint credentials.

    Managed here via the FS API (not the DAB): the store is FS-registered by this
    client, so a DAB database_instance resource would collide with it.
    """
    capacity = capacity or ONLINE_STORE_CAPACITY
    existing = fe.get_online_store(name=ONLINE_STORE_NAME)
    if existing is None:
        logger.info("Creating online store %s at %s.", ONLINE_STORE_NAME, capacity)
        return fe.create_online_store(name=ONLINE_STORE_NAME, capacity=capacity)
    return existing


def _delta_source(uc: UCConfig, table_key: str, transformation_sql: str | None = None):
    """Return a DeltaTableSource for one of this project's UC feature tables.

    `transformation_sql` (optional) projects/flattens the table inline; the
    materialization still fires on the named table's commits (used on the Kafka
    path to flatten the SFV ingestion table's `value` STRUCT to flat columns).
    """
    from databricks.feature_engineering.entities import DeltaTableSource

    kwargs = {
        "catalog_name": uc.catalog,
        "schema_name": uc.schema,
        "table_name": table_key,
    }
    if transformation_sql:
        kwargs["transformation_sql"] = transformation_sql
    return DeltaTableSource(**kwargs)


def write_offline_source(uc: UCConfig, players_df) -> str:
    """Overwrite the offline player-feature Delta table (plain Delta, no FE API).

    The Feature Views framework reads plain Delta tables as sources, so this is a
    normal `saveAsTable` (not `fe.create_table`). `players_df` carries
    `player_id`, `event_ts` and the offline numerical features. CDF is enabled so
    a `TableTrigger` view could re-sync on commit if ever wanted. Returns the
    table name.
    """
    table = uc.table(TABLE_OFFLINE_FEATURES)
    (
        players_df.select(ENTITY_KEY, EVENT_TIMESTAMP, *OFFLINE_FEATURE_COLUMNS)
        .write.mode("overwrite")
        .option("delta.enableChangeDataFeed", "true")
        .option("overwriteSchema", "true")
        .saveAsTable(table)
    )
    return table


def _column_features(
    fe, uc: UCConfig, table_key: str, columns, transformation_sql: str | None = None
) -> list:
    """Create+register a latest-value Feature per column over a source table.

    ColumnSelection preserves the point-in-time values the model trained on
    (train/serve parity). `create_feature` returns the registered Feature
    (carrying catalog+schema); idempotent on re-run via AlreadyExists -> fetch.
    `transformation_sql` flattens the source inline (Kafka-path login table).
    """
    from databricks.feature_engineering.entities import ColumnSelection
    from databricks.sdk.errors import AlreadyExists

    source = _delta_source(uc, table_key, transformation_sql)
    features = []
    for col in columns:
        try:
            feature = fe.create_feature(
                name=col,
                source=source,
                entity=[ENTITY_KEY],
                timeseries_column=EVENT_TIMESTAMP,
                function=ColumnSelection(col),
                catalog_name=uc.catalog,
                schema_name=uc.schema,
            )
        except AlreadyExists:
            # A prior run registered it; fetch by full name (kwarg is full_name).
            feature = fe.get_feature(full_name=f"{uc.catalog}.{uc.schema}.{col}")
        features.append(feature)
    return features


def _request_features(fe, uc: UCConfig) -> list:
    """Declare the request-time offer columns as RequestSource passthrough Features.

    Offer attributes are sent by the app in the request, NOT looked up from the
    online store. In the declarative Feature Views framework a plain spine column
    has a training-only source and is DROPPED at serving (no provider populates it
    into the inner model), which is why serving hit KeyError('offer_price_usd')
    despite the request carrying it. A RequestSource Feature gives the serving
    layer an explicit provider (the request payload), so the value reaches the
    pyfunc under its own name. Offer numerics -> DOUBLE, categoricals -> STRING.

    On FE client 0.16.1+: RequestSource features must NOT specify entity or
    timeseries_column (the server rejects them: "entity and timeseries_column do not
    apply to a RequestSource feature ... Omit both"). Contrast with DeltaTableSource
    / StreamSource features (see _column_features / _streaming_agg_features), which
    MUST still supply them.

    Idempotency: uses delete-then-create, NOT AlreadyExists -> get_feature. A prior
    0.16.0 run may have registered these features WITH entity/timeseries stored in
    UC; the 0.16.1 client rejects those on read-back (Feature.__init__ raises
    "Omit both"), so the fallback `fe.get_feature(...)` crashes. Because request
    features are NOT materialized online, `fe.delete_feature` is never blocked by
    materializations, making delete-first safe. On a first-ever run the delete is a
    no-op (contextlib.suppress silences the not-found).

    Other invariants (proven on earlier runs, still binding):
    (1) An inline Feature() has no catalog/schema, and create_training_set rejects
    it ("Feature does not have a catalog and schema"), so the feature MUST be
    registered via `create_feature`. (2) For a RequestSource the feature `name`
    MUST equal the ColumnSelection column name. (3) Not materialized online
    (request-time passthrough).
    """
    from databricks.feature_engineering.entities import (
        ColumnSelection,
        FieldDefinition,
        RequestSource,
        ScalarDataType,
    )

    # Offer attributes AND the real-time numerics are request-time passthrough:
    # the app sends the just-consumed login's real-time feature values in the
    # serving request (millisecond-fresh), so they are NOT looked up from the
    # online store (which would lag the login by the SFV micro-batch + TableTrigger
    # sync). Training sources these same columns from the impression spine (their
    # point-in-time values), so train/serve stay consistent.
    typed_cols = (
        [(c, ScalarDataType.DOUBLE) for c in OFFER_NUMERICAL_FEATURES]
        + [(c, ScalarDataType.STRING) for c in OFFER_CATEGORICAL_FEATURES]
        + [(c, ScalarDataType.DOUBLE) for c in REALTIME_NUMERICAL_FEATURES]
    )
    features = []
    for col, dtype in typed_cols:
        full_name = f"{uc.catalog}.{uc.schema}.{col}"
        source = RequestSource(schema=[FieldDefinition(name=col, data_type=dtype)])
        # Delete any pre-existing registration before (re-)creating. A RequestSource
        # feature from a prior 0.16.0 run was stored WITH entity/timeseries; the
        # 0.16.1 client rejects those on read-back (get_feature -> Feature.__init__
        # -> "Omit both"). A realtime column (gem_balance etc.) may ALSO
        # carry an ONLINE materialization from the old schema where it was a lookup
        # feature -- and delete_feature REFUSES while a feature has materializations
        # ("has N materialized feature(s)"), so delete the materializations FIRST
        # (same ordering as _reset_streaming_features), then the feature. Otherwise
        # the suppressed delete no-ops and create_feature hits AlreadyExists.
        with contextlib.suppress(Exception):
            for mf in fe.list_materialized_features(feature_name=full_name):
                with contextlib.suppress(Exception):
                    fe.delete_materialized_feature(materialized_feature=mf)
        with contextlib.suppress(Exception):
            fe.delete_feature(full_name=full_name)
        feature = fe.create_feature(
            name=col,  # MUST equal the ColumnSelection column for a RequestSource
            source=source,
            function=ColumnSelection(col),
            catalog_name=uc.catalog,
            schema_name=uc.schema,
            # entity + timeseries_column MUST be omitted for RequestSource (0.16.1+
            # server rejects them). They are required only for DeltaTableSource /
            # StreamSource features (see _column_features / _streaming_agg_features).
        )
        features.append(feature)
    return features


def build_lookup_features(fe, uc: UCConfig) -> list:
    """Register+return the online-lookup Features: the OFFLINE group only.

    Offline (slow-moving) player features are looked up from the online store by
    `player_id` -- they change daily, so a store lookup is fine and cheap. The
    real-time features are NOT online lookups: they are request-time passthrough
    (see `_request_features`), because the app sends the just-consumed login's
    real-time values in the serving request (millisecond-fresh), which a store
    lookup could only serve seconds-stale. So only the offline group is
    materialized online.
    """
    return _column_features(fe, uc, TABLE_OFFLINE_FEATURES, OFFLINE_FEATURE_COLUMNS)


# (op, signal) from aggregates.WINDOWS -> a 0.16.0 Function operator. Mirrors
# aggregates._agg_one EXACTLY so serve==train: numpy "count" over the virtual
# "login" signal is a row COUNT (Count over the always-present entity key), but
# "count" over a real signal (offer_shown flags) is that signal's SUM.
def _operator(op: str, signal: str):
    """Return the FE AggregationFunction operator for one (op, signal) pair.

    Column references MUST be `value.`-prefixed because this function is used
    exclusively over a StreamSource (Kafka), where payload fields live inside the
    message `value` struct. The prefix is stream-registration-only: the
    training-set join and online-serving lookups use the bare leaf name
    (player_id, gem_spend, …) and are unaffected by this prefix.

    Raises ValueError on an unknown operator so a WINDOWS change can't silently
    skew train vs serve.
    """
    from databricks.feature_engineering.entities import (
        Avg,
        Count,
        Sum,
    )

    # StreamSource payload is nested under the `value` struct; all column refs
    # inside an AggregationFunction operator must carry the `value.` prefix.
    val_entity = f"value.{ENTITY_KEY}"
    val_signal = f"value.{signal}"

    if op == "count":
        # "login" is virtual (every event row is a login) -> count ROWS via the
        # always-present entity key. NOTE: for any non-"login" signal this returns a
        # SUM of the signal's value, NOT a row count -- so a "count" entry is only
        # correct over the virtual "login" signal. To count occurrences of a real
        # event, emit a 0/1 indicator and use ("sum", <indicator>) (see offer_shown
        # -> offers_seen_15m, match_won -> match_wins_15m).
        return Count(input=val_entity) if signal == "login" else Sum(input=val_signal)
    if op == "sum":
        return Sum(input=val_signal)
    if op == "avg":
        return Avg(input=val_signal)
    raise ValueError(f"Unknown streaming aggregate operator: {op!r}")


def _reset_streaming_features(fe, uc: UCConfig) -> None:
    """Delete the 6 registered streaming Features (+ their online materializations).

    Called before re-registering on a reset run. `create_match_stream` DELETE+
    recreates the match StreamSource on every reset, but a streaming Feature stays
    registered across runs and stays bound to the NOW-DELETED old stream: its
    `stream_source.full_name` dangles to None, so the next `get_feature` fallback
    (AlreadyExists path) raises `ValueError: SDK StreamSource must include
    'full_name'` on `_from_sdk`. Clearing the stale Features here lets the
    subsequent `create_feature` register each one FRESH against the new stream.

    Order matters: `delete_feature` REFUSES while a feature still has
    materializations ("Cannot delete feature ... has N materialized feature(s)"),
    so each feature's online materializations are deleted FIRST, then the feature.
    Both are best-effort (`contextlib.suppress`): a not-found on a first-ever run is
    fine, and an already-deleted materialization must not abort the loop.
    """
    for col in STREAMING_FEATURE_COLUMNS:
        full_name = f"{uc.catalog}.{uc.schema}.{col}"
        deleted_mfs = 0
        with contextlib.suppress(Exception):
            for mf in fe.list_materialized_features(feature_name=full_name):
                with contextlib.suppress(Exception):
                    fe.delete_materialized_feature(materialized_feature=mf)
                    deleted_mfs += 1
        with contextlib.suppress(Exception):
            fe.delete_feature(full_name=full_name)
            logger.info(
                "Reset: deleted stale streaming feature %s (%d materialization(s)).",
                full_name,
                deleted_mfs,
            )


def _streaming_agg_features(fe, uc: UCConfig, reset: bool = False) -> list:
    """Register the 6 streaming aggregates as LIVE SFV AggregationFunction Features.

    Defines each feature as an `AggregationFunction(operator, RollingWindow)` over
    the login `StreamSource`, using the same operator + window per feature as
    `schema.WINDOWS` so the live serve values match training. RollingWindow is the
    point-in-time, current-value backward-looking window [event_ts - W, event_ts).

    reset=False (default, non-reset + all internal callers): idempotent register via
    AlreadyExists -> `get_feature` fetch, so a same-set re-run reuses the existing
    Features.

    reset=True (the match stream was DELETE+recreated by create_match_stream this
    run): the AlreadyExists path would fetch a STALE Feature bound to the destroyed
    old stream (dangling full_name -> `_from_sdk` ValueError), so first delete the
    stale Features (`_reset_streaming_features`), then `create_feature` registers
    each fresh against the NEW stream.

    Two-job reset flow (both derive reset the same way, from fs_reset / RECO_FS_RESET):
      - preprocess: create_match_stream (delete+recreate stream) ->
        build_streaming_features(reset=True) here (delete stale -> create fresh).
        There is NO materialize_online in preprocess, so nothing follows.
      - publish_features: create_match_stream -> build_streaming_features(reset=True)
        (delete stale -> create fresh) THEN materialize_online(reset=True). The
        latter's _delete_materialized lists the FRESH feature's materializations
        (none, just recreated) and creates the online sync -- no double-delete error.

    PRODUCER SIGNALS: the match Kafka stream must carry `gem_spend` (sum ->
    gems_spent_15m), `gem_balance` (avg -> avg_gem_balance_15m), `offer_shown`
    (0/1 sum -> offers_seen_15m), `match_won` (sum -> match_wins_15m), `match_lost`
    (sum -> match_losses_15m). match_events_15m (count of login events) is the 6th
    materialized feature, not a model input. The producer
    (jobs/produce_events.py) and the backfill frame
    (data/match_backfill.py) both emit the 8-field value struct.
    Feature `name` is the feature column (gems_spent_15m, ...).
    """
    import datetime

    from databricks.feature_engineering.entities import (
        AggregationFunction,
        RollingWindow,
        StreamSource,
    )
    from databricks.sdk.errors import AlreadyExists

    if reset:
        _reset_streaming_features(fe, uc)

    source = StreamSource(full_name=f"{uc.catalog}.{uc.schema}.{MATCH_STREAM_NAME}")
    features = []
    for col in STREAMING_FEATURE_COLUMNS:
        op, signal, wsec = WINDOWS[col]
        function = AggregationFunction(
            operator=_operator(op, signal),
            time_window=RollingWindow(window_duration=datetime.timedelta(seconds=wsec)),
        )
        try:
            # Over a StreamSource the payload is nested under the `value` struct,
            # so entity, timeseries_column AND operator inputs must be prefixed.
            # The prefix is stream-registration-only: create_training_set and
            # online serving lookups still reference the bare leaf name (player_id).
            feature = fe.create_feature(
                name=col,
                source=source,
                function=function,
                entity=[f"value.{ENTITY_KEY}"],
                timeseries_column=f"value.{EVENT_TIMESTAMP}",
                catalog_name=uc.catalog,
                schema_name=uc.schema,
            )
        except AlreadyExists:
            feature = fe.get_feature(full_name=f"{uc.catalog}.{uc.schema}.{col}")
        features.append(feature)
    return features


def build_streaming_features(fe, uc: UCConfig, reset: bool = False) -> list:
    """Register+return the 6 streaming rolling-window aggregate Features.

    Managed Streaming Feature View: each column is an AggregationFunction +
    RollingWindow over the Kafka match StreamSource (see _streaming_agg_features),
    mirroring schema.WINDOWS. Values are recomputed from the stream ingestion
    history at training time (PIT) and materialized online for serving. Requires
    the match stream (create_match_stream) to exist and the producer to emit the
    source signals (gem_spend, gem_balance, offer_shown, match_won, match_lost).

    reset=True (the match stream was DELETE+recreated this run): first delete the
    stale registered Features (bound to the destroyed stream) so they re-register
    fresh against the new stream, avoiding the dangling-full_name `_from_sdk`
    crash. Pass reset from the fs_reset flag; see _streaming_agg_features for the
    materialization-delete ordering across the two-job reset flow.
    """
    return _streaming_agg_features(fe, uc, reset=reset)


def _wait_online_materialized(
    spark, uc: UCConfig, prefixes, attempts: int = 120, delay_s: int = 15
) -> None:
    """Block until each online-materialization synced table exists in UC (bounded).

    `materialize_features` fires an ASYNC managed synced-table pipeline that creates
    the online-store table `<prefix>_<suffix>` (a UC-visible synced Delta table, plus
    a `_latest_view`). `create_feature_serving_endpoint` referencing the FeatureSpec
    fails its async served-entity deploy with `RESOURCE_DOES_NOT_EXIST: Feature table
    ...<prefix>...` if that table has not been created yet -- the recurring
    "Online feature store setup failed" endpoint UPDATE_FAILED. This gate closes that
    race: poll `SHOW TABLES LIKE '<prefix>*'` until every prefix has a synced table,
    so the endpoint is only created once its backing online tables exist.

    No-op when `spark` is None (unit tests) or on any prefix that never appears within
    the budget (raises TimeoutError so the job fails loudly rather than creating an
    endpoint doomed to UPDATE_FAILED).
    """
    if spark is None:
        return

    pending = list(prefixes)
    for i in range(attempts):
        still = []
        for prefix in pending:
            rows = spark.sql(
                f"SHOW TABLES IN {uc.catalog}.{uc.schema} LIKE '{prefix}*'"
            ).collect()
            # A '<prefix>_<suffix>' data table (not just the '<prefix>*_latest_view').
            names = [r["tableName"] for r in rows]
            if any(not n.endswith("_latest_view") for n in names):
                logger.info("Online materialization ready: %s (%s)", prefix, names)
            else:
                still.append(prefix)
        if not still:
            return
        pending = still
        logger.info(
            "Waiting for online materialization of %s (attempt %d/%d)...",
            pending,
            i + 1,
            attempts,
        )
        time.sleep(delay_s)
    raise TimeoutError(
        f"Online materialization tables never appeared for prefixes {pending} "
        f"after {attempts * delay_s}s; the feature-serving endpoint would fail its "
        "deploy (RESOURCE_DOES_NOT_EXIST). The managed synced-table pipeline may be "
        "stuck -- check the Lakeflow synced-table pipeline for these prefixes."
    )


def _grant_app_query(endpoint_name: str) -> None:
    """Grant the Databricks App's service principal CAN_QUERY on a serving endpoint.

    DAB grants this on app deploy (resources/app.yml declares the endpoints as app
    resources), but an fs_reset DELETE+recreate of offer-features-fs makes a NEW
    endpoint object, dropping that grant -> the app drill-down 403s and shows
    synthetic values. Re-apply it here so a pipeline re-run keeps the app working
    without a manual permissions patch. Best-effort + non-fatal: a missing app
    (RECO_APP_NAME empty, or no app deployed yet) or a permissions error must not
    fail the pipeline.
    """
    if not APP_NAME:
        return
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import (
            ServingEndpointAccessControlRequest,
            ServingEndpointPermissionLevel,
        )

        w = WorkspaceClient()
        sp = w.apps.get(name=APP_NAME).service_principal_client_id
        if not sp:
            logger.info("App %s has no service principal id; skipping grant.", APP_NAME)
            return
        ep = w.serving_endpoints.get(name=endpoint_name)
        w.serving_endpoints.update_permissions(
            serving_endpoint_id=ep.id,
            access_control_list=[
                ServingEndpointAccessControlRequest(
                    service_principal_name=sp,
                    permission_level=ServingEndpointPermissionLevel.CAN_QUERY,
                )
            ],
        )
        logger.info(
            "Granted app SP %s CAN_QUERY on %s (re-applied after (re)create).",
            sp,
            endpoint_name,
        )
    except Exception as exc:  # non-fatal: never fail the pipeline on the grant
        logger.warning(
            "Could not re-grant app CAN_QUERY on %s (non-fatal): %s",
            endpoint_name,
            str(exc)[:200],
        )


def ensure_feature_serving(fe, uc: UCConfig, reset: bool = False, spark=None) -> str:
    """Create (idempotently) the FeatureSpec + Feature Serving endpoint that serves
    the online feature VALUES by player_id over REST. Returns the endpoint name.

    The app drill-down GETs a player's offline + real-time feature values from this
    endpoint (no direct Lakebase/table access). It serves the SAME lookup Features
    that back the model, so the displayed values match what the model consumes.
    Prereq: the features are already materialized online (publish_features).

    `spark` (optional): when provided, gate the endpoint create on the online
    materialization synced tables actually existing (see `_wait_online_materialized`)
    -- this closes the race where the endpoint deploy fails with
    RESOURCE_DOES_NOT_EXIST because the async managed sync has not yet created the
    online table. None (unit tests) skips the gate.

    reset=False (daily re-run, unchanged feature set): the FeatureSpec create
    suppresses AlreadyExists and the endpoint is get-or-create, so the daily pipeline
    re-runs without thrashing the endpoint (no downtime).

    reset=True (feature-SET change, e.g. adding the streaming features): the existing
    spec is stale -- create_feature_spec's AlreadyExists suppression would keep it, so
    the endpoint keeps serving the OLD feature set. Delete the endpoint THEN the spec
    (an endpoint referencing a spec blocks the spec delete), then recreate both with
    the CURRENT spec_features. Each delete is guarded so a not-found is non-fatal.
    """
    from databricks.feature_engineering.entities.feature_serving_endpoint import (
        EndpointCoreConfig,
        ServedEntity,
    )

    spec_name = f"{uc.catalog}.{uc.schema}.{FEATURE_SPEC_NAME}"
    # Both online-materialized groups (offline + streaming) are served by player_id
    # so the drill-down surfaces all of them. The streaming group is the managed
    # SFV aggregation (AggregationFunction + RollingWindow over the match stream).
    spec_features = build_lookup_features(fe, uc) + build_streaming_features(fe, uc)

    if reset:
        # Endpoint first: it references the spec, so the spec delete is blocked while
        # the endpoint exists. Both deletes are best-effort (a not-found means it was
        # never created / already gone -- non-fatal, same as the AlreadyExists path).
        try:
            fe.delete_feature_serving_endpoint(name=ONLINE_FS_ENDPOINT)
            logger.info(
                "Deleted feature serving endpoint %s for reset.", ONLINE_FS_ENDPOINT
            )
        except Exception as exc:
            logger.info(
                "No feature serving endpoint to delete (%s): %s",
                ONLINE_FS_ENDPOINT,
                exc,
            )
        try:
            fe.delete_feature_spec(name=spec_name)
            logger.info("Deleted feature spec %s for reset.", spec_name)
        except Exception as exc:
            logger.info("No feature spec to delete (%s): %s", spec_name, exc)

    try:
        fe.create_feature_spec(name=spec_name, features=spec_features)
    except Exception as exc:
        # create_feature_spec raises a RAW Exception wrapping the REST error JSON
        # (MLflow-style: `raise Exception(json.loads(response.text))`), NOT the
        # typed SDK AlreadyExists/ResourceAlreadyExists -- so suppressing those
        # types misses the re-run case. Treat "already exists" as success (the spec
        # is content-stable), re-raise anything else.
        if "RESOURCE_ALREADY_EXISTS" not in str(exc) and "already exists" not in str(
            exc
        ):
            raise

    try:
        existing = fe.get_feature_serving_endpoint(name=ONLINE_FS_ENDPOINT)
    except Exception:
        existing = None  # not found -> create (below)
    if existing is not None:
        # Only short-circuit if the existing endpoint is HEALTHY. A stale endpoint
        # in UPDATE_FAILED (e.g. a prior run raced ahead of the online materialization)
        # would otherwise let the job return green while the endpoint stays broken
        # (the "offer-features-fs Update failed" symptom). state is {config_update,
        # ready}; treat any *_FAILED config_update as broken -> delete + recreate.
        state = getattr(existing, "state", None) or {}
        config_update = str(state.get("config_update", "")).upper()
        if "FAILED" not in config_update:
            _grant_app_query(ONLINE_FS_ENDPOINT)  # keep the app SP grant current
            return ONLINE_FS_ENDPOINT  # healthy / still updating -> leave it
        logger.warning(
            "Existing feature serving endpoint %s is broken (state=%s); "
            "deleting to recreate.",
            ONLINE_FS_ENDPOINT,
            state,
        )
        with contextlib.suppress(Exception):
            fe.delete_feature_serving_endpoint(name=ONLINE_FS_ENDPOINT)

    # Gate on the online-materialization synced tables existing before creating the
    # endpoint -- otherwise its async served-entity deploy fails with
    # RESOURCE_DOES_NOT_EXIST (the recurring "Online feature store setup failed").
    # Both served groups (offline + streaming) must have their online table synced.
    _wait_online_materialized(
        spark, uc, [OFFLINE_SERVING_PREFIX, STREAMING_SERVING_PREFIX]
    )

    def _create() -> None:
        fe.create_feature_serving_endpoint(
            name=ONLINE_FS_ENDPOINT,
            config=EndpointCoreConfig(
                served_entities=ServedEntity(
                    feature_spec_name=spec_name,
                    workload_size="Small",
                    # NOT scale-to-zero: the app drill-down is the only caller, so a
                    # scaled-to-zero endpoint cold-starts (many seconds) on the first
                    # click -- the "drill-down takes forever" symptom. Keep it warm so
                    # the interactive lookup is ~sub-second. (Small workload is cheap.)
                    scale_to_zero_enabled=False,
                )
            ),
        )

    # On a reset the just-deleted endpoint tears down ASYNC, so the recreate can hit
    # "already exists"/"in progress" for a minute. DO NOT swallow that and return
    # SUCCESS -- the job would report green while the drill-down endpoint stays gone
    # (green-but-broken). Retry with backoff until the delete settles; raise if it
    # never does so the job fails loudly. Any other error re-raises immediately.
    for attempt in range(_FS_RECREATE_RETRIES):
        try:
            _create()
            _grant_app_query(ONLINE_FS_ENDPOINT)  # DAB grant is dropped on recreate
            return ONLINE_FS_ENDPOINT
        except Exception as exc:
            msg = str(exc).lower()
            transient = (
                "already exists" in msg
                or "resource_already_exists" in msg
                or "in progress" in msg
            )
            if not transient:
                raise
            if attempt == _FS_RECREATE_RETRIES - 1:
                raise RuntimeError(
                    f"Feature serving endpoint {ONLINE_FS_ENDPOINT} still not "
                    f"recreatable after {_FS_RECREATE_RETRIES} attempts (prior "
                    f"delete not yet settled): {exc}"
                ) from exc
            logger.info(
                "Endpoint %s recreate blocked by async teardown (attempt %d/%d): %s; "
                "retrying in %ds.",
                ONLINE_FS_ENDPOINT,
                attempt + 1,
                _FS_RECREATE_RETRIES,
                exc,
                _FS_RECREATE_BACKOFF_S,
            )
            time.sleep(_FS_RECREATE_BACKOFF_S)
    return ONLINE_FS_ENDPOINT


def build_features(fe, uc: UCConfig) -> list:
    """Register+return ALL Features for the training set: the online-lookup groups
    (offline + streaming aggregates, looked up by player_id), plus the request-time
    passthrough features (offer cols + real-time numerics via RequestSource). Used
    by create_training_set so the served model's FeatureSpec resolves lookups AND
    accepts the request's passthrough columns. Only the lookup features (offline +
    streaming) are materialized online. The streaming group is the managed SFV
    aggregation (AggregationFunction + RollingWindow over the Kafka match stream).
    """
    return (
        build_lookup_features(fe, uc)
        + build_streaming_features(fe, uc)
        + _request_features(fe, uc)
    )


def _online_config(uc: UCConfig, prefix: str):
    """OnlineStoreConfig targeting this project's Lakebase online store."""
    from databricks.feature_engineering.entities import OnlineStoreConfig

    return OnlineStoreConfig(
        catalog_name=uc.catalog,
        schema_name=uc.schema,
        table_name_prefix=prefix,
        online_store_name=ONLINE_STORE_NAME,
    )


def _delete_materialized(fe, features) -> int:
    """Delete any existing online materializations for these features.

    The "already materialized" state is bound to the Feature objects, NOT to the
    online-store table_name_prefix: re-materializing a feature that is already
    materialized raises ResourceAlreadyExists regardless of prefix, and deleting
    the pipeline/tables does not clear it. `delete_materialized_feature`
    (singular) IS available on the 0.16+ client (the plural form is not). For our
    ColumnSelection features the online materialization is deleted directly (no
    offline pair). Returns the number deleted.
    """
    deleted = 0
    for feature in features:
        for mf in fe.list_materialized_features(feature_name=feature.full_name):
            fe.delete_materialized_feature(materialized_feature=mf)
            deleted += 1
    return deleted


def materialize_online(
    fe,
    uc: UCConfig,
    features,
    prefix: str,
    reset: bool = False,
    streaming: bool = False,
) -> None:
    """Materialize a feature group to the Lakebase online store.

    Selects the trigger based on the feature group type:

    - ``streaming=False`` (default): Delta-source ColumnSelection features
      (offline, real-time). Uses ``TableTrigger`` — re-syncs on every commit to
      the source Delta table. ColumnSelection features reject ``CronSchedule``
      (they expose an always-current VIEW) so TableTrigger is the only valid
      batch trigger.

    - ``streaming=True``: Streaming SFV ``AggregationFunction`` features over a
      ``StreamSource`` / ``KafkaSource``. Uses ``StreamingMode`` (Structured
      Streaming continuous pipeline). TableTrigger is invalid for these because
      Kafka has no Delta upstream; passing it raises ``ValueError: TableTrigger
      is not supported for streaming (StreamSource/KafkaSource) features``.

    Idempotent for the daily re-run: ``materialize_features`` raises
    ``ResourceAlreadyExists`` once a group is materialized; the existing pipeline
    keeps syncing, so a same-feature-set re-run safely skips.

    IMPORTANT (proven the hard way): the "already materialized" state is bound to
    the Feature objects, NOT the prefix. If the feature SET changed (a column
    added/removed), a plain re-run keeps the OLD online schema and serving fails
    with "No online materialized feature found for features: ...". Pass
    ``reset=True`` (wired to ``RECO_FS_RESET``) to first delete the existing
    materializations, so this recreates the group with the new column set.
    Deleting pipelines/tables by hand does NOT work; only
    ``delete_materialized_feature`` clears the binding.
    """
    from databricks.feature_engineering.entities import StreamingMode, TableTrigger
    from databricks.sdk.errors import ResourceAlreadyExists

    trigger = StreamingMode() if streaming else TableTrigger()

    if reset:
        n = _delete_materialized(fe, features)
        logger.info(
            "Reset: deleted %d existing materialized feature(s) for '%s'", n, prefix
        )

    try:
        fe.materialize_features(
            features=features,
            online_config=_online_config(uc, prefix),
            trigger=trigger,
        )
    except ResourceAlreadyExists:
        if reset:
            raise  # teardown ran yet state persists -> fail loud, don't serve stale schema
        trigger_name = "StreamingMode" if streaming else "TableTrigger"
        logger.warning(
            "Online materialization for prefix '%s' already exists; skipping "
            "re-create (%s keeps it synced). If you CHANGED the feature "
            "set, re-run publish_features with RECO_FS_RESET=true to rebuild it.",
            prefix,
            trigger_name,
        )


def build_training_set(fe, uc: UCConfig, spine_df):
    """Build the PIT-correct training set from a spine of impressions.

    `spine_df` carries `player_id`, `event_ts`, the offer-side features and the
    two targets. Feature values are joined by entity key + timeseries; the entity
    key and timestamp are excluded so serving requests only the key + candidate
    offers. Uses the new `features=[Feature]` arg (NOT legacy feature_lookups).
    The streaming group is the managed SFV aggregation over the Kafka match stream.
    """
    return fe.create_training_set(
        df=spine_df,
        features=build_features(fe, uc),
        label=[TARGET_REVENUE_D7, TARGET_PURCHASED],
        exclude_columns=[ENTITY_KEY, EVENT_TIMESTAMP],
    )


def build_training_set_streaming(fe, uc: UCConfig, spine_df):
    """PIT-join ONLY the streaming SFV aggregates onto an impressions spine.

    Used by preprocess to enrich the impression log with the 5 rolling-window
    streaming features (recomputed point-in-time from the Kafka login-stream
    ingestion history) BEFORE fitting the SparkML pipeline, so training sees the
    same streaming values the SFV serves online -- no impression-backfill, no
    train/serve skew.

    Unlike `build_training_set`, entity key and timestamp are NOT excluded: the
    caller reads `load_df()` and needs `player_id`/`event_ts` for the train/val
    split and the full model-feature set to fit the pipeline. Only the streaming
    group is joined here; the offline/realtime/offer feature wiring is handled at
    registration time (`build_training_set`).
    """
    return fe.create_training_set(
        df=spine_df,
        features=build_streaming_features(fe, uc),
        label=[TARGET_REVENUE_D7, TARGET_PURCHASED],
        exclude_columns=[],
    )


def _match_payload_json_schema() -> str:
    """JSON Schema (string) for a match (gameplay) event.

    Kept in sync with jobs/produce_events.py's match events and
    data/match_backfill.py's value struct: the entity key + event time + the
    match-event signals the streaming aggregates are built from (gem_spend,
    gem_balance, offer_shown, match_won, match_lost). Numbers are `number` (JSON
    has no int/float split we rely on) so the SFV parser is lenient.
    """
    import json

    props = {
        ENTITY_KEY: {"type": "string"},
        EVENT_TIMESTAMP: {"type": "string", "format": "date-time"},
        "event_type": {"type": "string"},
        "gem_spend": {"type": "number"},
        "gem_balance": {"type": "number"},
        "offer_shown": {"type": "number"},
        "match_won": {"type": "number"},
        "match_lost": {"type": "number"},
    }
    return json.dumps({"type": "object", "properties": props})


def create_match_stream(
    fe,
    uc: UCConfig,
    bootstrap_servers: str,
    spark=None,
    backfill_table: str | None = None,
    recreate: bool = True,
    require_exists: bool = False,
) -> str:
    """Create (idempotently) the Streaming Feature View that ingests Kafka match
    (gameplay) events into a Delta table. Returns the ingestion table name.

    Auth is config-gated (see config.get_kafka_uc_connection): when
    RECO_KAFKA_UC_CONNECTION is set, the SFV authenticates to Azure Event Hubs
    (Kafka protocol) through a UC Kafka connection carrying SASL/PLAIN
    (StreamConnectionConfig) — no certs; otherwise it uses DirectMtlsConfig
    against a self-managed Kafka broker.

    The rolling-window aggregates are built from these match events (session
    signals); the `logins` topic is consumed separately by the app to trigger
    scoring and is NOT an SFV.

    Consumes the `match_events` topic (MATCH_TOPIC) from a self-managed
    Kafka broker via `create_stream` + `DirectMtlsConfig`: the client keystore /
    truststore JKS live on the UC Volume (kafka_certs/), the JKS password in the
    `offer-reco-kafka` secret scope. Hostname verification is disabled because the
    broker cert's SAN is the IP, not a DNS name. The managed ingestion pipeline
    lands each match event in `TABLE_MATCH_INGEST`; publish_features derives the
    real-time feature source from there. Only used when RECO_KAFKA_BOOTSTRAP is
    set; the synthetic stream is the default otherwise.

    `backfill_table` (optional): full name of a Delta table of historical login
    events (see data/match_backfill.py) that the managed ingestion pipeline MERGEs
    into the ingestion table via `StreamBackfillSource`, so the streaming
    rolling-window features have real history at training time instead of an empty
    LATEST-offset window. When None, behaves exactly as today (no backfill).

    deduplication_columns is left None: the backfill events are historical
    (backdated `value.event_ts`, before the just-created stream's LATEST-offset
    start), so they cannot collide with a forward-filled live event -- there is no
    double-count to dedup away, and adding a dedup key would only add MERGE cost.

    `recreate` (default True): if True, delete then recreate the stream (setup
    or reset path). If False, ensure an existing stream exists and reuse it as-is
    (pipeline consumption path); raises RuntimeError if absent. The pipeline path
    (recreate=False) avoids deleting/recreating the stream on every run, which
    would wipe accumulated + backfilled history. Set by the calling job
    (setup job: recreate=True; pipeline training task: recreate=False).
    """
    from databricks.feature_engineering.entities import (
        DirectMtlsConfig,
        DirectSchemas,
        IngestionConfig,
        IngestionDestination,
        KafkaStreamConfig,
        KafkaSubscriptionMode,
        MtlsConfig,
        SchemaConfig,
        SecretScopeReference,
        StreamBackfillSource,
        StreamConnectionConfig,
    )

    stream_name = f"{uc.catalog}.{uc.schema}.{MATCH_STREAM_NAME}"
    ingest_table = uc.table(TABLE_MATCH_INGEST)

    if not recreate:
        # Reuse the existing stream AS-IS if present -- never delete/recreate (that
        # wipes accumulated live history + re-arms the online warm-up clock). If it
        # is ABSENT, create it (first run on a fresh workspace: setup_stream passes
        # recreate=reset, and reset defaults to false, so the very first setup must
        # still create the stream rather than fail). `require_exists=True` (set by
        # the pipeline CONSUMER path, e.g. preprocess) keeps the old fail-fast: a
        # consumer must not silently create the stream.
        try:
            fe.get_stream(name=stream_name)
            logger.info("Reusing existing match stream %s (no recreate).", stream_name)
            return ingest_table
        except Exception as exc:
            if require_exists:
                raise RuntimeError(
                    f"Match stream {stream_name} not found; the setup_stream pipeline "
                    f"task must run first to create it."
                ) from exc
            logger.info(
                "Match stream %s absent; creating it (first-time setup).", stream_name
            )
            # fall through to create below

    # Auth: Event Hubs path (UC Kafka connection, SASL) if configured, else the
    # self-managed broker path (DirectMtlsConfig). The SFV can't take SASL
    # directly in preview, but a UC connection carries it — see config.py.
    uc_conn = get_kafka_uc_connection()
    if uc_conn:
        connection_config = StreamConnectionConfig(uc_connection_name=uc_conn)
        logger.info(
            "Match stream auth via UC Kafka connection %s (Event Hubs SASL).", uc_conn
        )
    else:
        certs = f"{uc.volume_path}/{KAFKA_CERTS_SUBDIR}"
        pwd = SecretScopeReference(scope=KAFKA_SECRET_SCOPE, key=KAFKA_JKS_PASSWORD_KEY)
        connection_config = DirectMtlsConfig(
            bootstrap_servers=bootstrap_servers,
            mtls_config=MtlsConfig(
                keystore_location=f"{certs}/{KAFKA_KEYSTORE_FILE}",
                keystore_password_ref=pwd,
                key_password_ref=pwd,
                truststore_location=f"{certs}/{KAFKA_TRUSTSTORE_FILE}",
                truststore_password_ref=pwd,
                disable_hostname_verification=True,
            ),
        )
        logger.info("Match stream auth via DirectMtlsConfig (self-managed broker).")

    with contextlib.suppress(Exception):
        fe.delete_stream(name=stream_name)
        logger.info("Deleted existing match stream %s before recreate", stream_name)

    backfill_source = None
    if backfill_table:
        backfill_source = StreamBackfillSource(delta_table_name=backfill_table)
        logger.info(
            "Match stream %s will backfill from %s", stream_name, backfill_table
        )
    ingestion_config = IngestionConfig(
        ingestion_destination=IngestionDestination(delta_table_name=ingest_table),
        backfill_source=backfill_source,
    )

    fe.create_stream(
        name=stream_name,
        source_config=KafkaStreamConfig(
            subscription_mode=KafkaSubscriptionMode(subscribe=MATCH_TOPIC),
        ),
        connection_config=connection_config,
        schema_config=DirectSchemas(
            payload_schema=SchemaConfig(json_schema=_match_payload_json_schema()),
        ),
        ingestion_config=ingestion_config,
    )
    logger.info("Created match stream %s -> %s", stream_name, ingest_table)

    # The managed ingestion pipeline creates `ingest_table` asynchronously (JSON
    # payload stored as a `value` STRUCT). We wait for it to be readable so the
    # app's live board (which consumes logins from Kafka directly) has a populated
    # table for the drill-down. The real-time FEATURES are request-passthrough, not
    # a lookup over this table, so no feature binds to it -- it exists for the
    # login feed + drill-down display.
    if spark is not None:
        _wait_for_table(spark, ingest_table)
    return ingest_table


def _wait_for_table(spark, table: str, attempts: int = 60, delay_s: int = 15) -> None:
    """Block until `table` is READABLE with >=1 row (bounded). Raises on timeout.

    The managed SFV ingestion pipeline registers the table in the catalog BEFORE
    its Delta files are committed, so `tableExists()` is True too early -- a read
    then fails with DELTA_PATH_DOES_NOT_EXIST (metadata exists, data path not yet
    written). We therefore gate on a successful COUNT of at least one row, which
    only succeeds once the first micro-batch of logins has actually landed. This is
    what create_feature (downstream) needs: real data at the path, not just a name.
    """
    import time

    for i in range(attempts):
        try:
            if spark.catalog.tableExists(table):
                n = spark.table(table).limit(1).count()
                if n >= 1:
                    logger.info(
                        "Ingestion table %s readable with data after ~%ds",
                        table,
                        i * delay_s,
                    )
                    return
        except Exception as e:
            logger.info("Waiting for %s to materialize (%s)", table, str(e)[:80])
        time.sleep(delay_s)
    raise TimeoutError(
        f"Login ingestion table {table} had no readable rows within "
        f"{attempts * delay_s}s; check the Streaming Feature View / broker / topic."
    )
