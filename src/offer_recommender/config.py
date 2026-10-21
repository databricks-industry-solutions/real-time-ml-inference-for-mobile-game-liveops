"""Central Unity Catalog and resource configuration.

Every stage reads catalog / schema / table names from here, so the dedicated UC
schema is defined in exactly one place. DAB injects overrides via environment
variables per target (dev/prod), so no stage hardcodes a location.

Defaults: catalog `main`, dedicated schema `offer_recommender` (created and owned
by the DAB bundle). Jobs receive the bundle's catalog/schema as task parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class UCConfig:
    """Unity Catalog locations for this project's securables."""

    catalog: str
    schema: str
    volume: str

    def table(self, name: str) -> str:
        """Return the fully-qualified table name `catalog.schema.name`."""
        return f"{self.catalog}.{self.schema}.{name}"

    def model_name(self, name: str) -> str:
        """Return the UC-registered model name `catalog.schema.name`."""
        return f"{self.catalog}.{self.schema}.{name}"

    @property
    def volume_path(self) -> str:
        """Return the UC Volume path for checkpoints / stats artifacts."""
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"


@lru_cache(maxsize=1)
def get_config() -> UCConfig:
    """Return the UC config, overridable via RECO_CATALOG/SCHEMA/VOLUME env vars."""
    return UCConfig(
        catalog=os.environ.get("RECO_CATALOG", "main"),
        schema=os.environ.get("RECO_SCHEMA", "offer_recommender"),
        volume=os.environ.get("RECO_VOLUME", "artifacts"),
    )


def configure_from_argv(argv: list[str]) -> None:
    """Apply DAB-injected `[catalog, schema]` task parameters, if present.

    Databricks passes a job task's `parameters` as argv to the Python file. The
    DAB resolves the schema name per target (dev prefixes it, e.g.
    `dev_<user>_offer_recommender`), so jobs must read the resolved name here
    rather than trust the default. Any pre-set RECO_* env var still wins.
    """
    if len(argv) >= 2:
        os.environ.setdefault("RECO_CATALOG", argv[0])
        os.environ.setdefault("RECO_SCHEMA", argv[1])
    get_config.cache_clear()


# Table names (single source of truth; used by data gen, preprocessing,
# feature store, training and serving).
TABLE_PLAYERS = "players"
TABLE_IMPRESSIONS = "impressions"
TABLE_OFFLINE_FEATURES = "player_features_offline"
TABLE_TRAINING = "training_set"

# Registered model + serving endpoint names.
MODEL_NAME = "offer_ranker"
ENDPOINT_REALTIME = "offer-ranker-rt"

# Online feature store (Lakebase-backed).
ONLINE_STORE_NAME = "offer-online-fs"
# Lakebase capacity. CU_1 comfortably serves this demo's load (one app, a few QPS).
# It is created ONCE at first install and then LEFT ALONE: resizing an online store
# (update_online_store) is an admin/failover op that rotates the instance's
# credentials and invalidates the OAuth token/password the already-deployed serving
# endpoints baked in -> the endpoints then fail Lakebase auth ("password
# authentication failed" / KeyError 'OAUTH_TOKEN') and every FS lookup times out with
# "No available connection in 2s" (the client pool never fills). So we never resize on
# the hot path. Override the CREATE-time size once via RECO_ONLINE_STORE_CAPACITY if a
# heavier workload genuinely needs it; changing it later requires redeploying the
# serving endpoints so they re-mint credentials against the resized instance.
ONLINE_STORE_CAPACITY = os.environ.get("RECO_ONLINE_STORE_CAPACITY", "CU_1")

# Feature Serving: a FeatureSpec + scale-to-zero endpoint that serves the online
# feature VALUES by player_id over REST (the app drill-down reads these, so it
# never touches Lakebase/tables directly). Distinct from the model endpoints.
FEATURE_SPEC_NAME = "offer_features_spec"
ONLINE_FS_ENDPOINT = "offer-features-fs"

# The Databricks App that queries the serving endpoints. resources/app.yml declares
# both endpoints as app `resources` with CAN_QUERY, so DAB grants the app's service
# principal on app deploy. BUT an fs_reset run DELETE+recreates offer-features-fs (a
# new object), which drops the DAB-applied grant -> the app drill-down 403s and falls
# back to synthetic. So ensure_feature_serving re-grants the app SP CAN_QUERY right
# after (re)creating the endpoint (the code that breaks DAB's assumption fixes it).
# Overridable via RECO_APP_NAME; empty disables the re-grant (portable installs with
# no app yet). Default matches resources/app.yml `${var.prefix}-offers-${target}`.
APP_NAME = os.environ.get("RECO_APP_NAME", "offer-reco-offers-classic")

# --- Kafka streams (OPTIONAL, config-gated) ----------------------------------
# TWO decoupled topics when RECO_KAFKA_BOOTSTRAP is set:
#   - `logins` (LOGIN_TOPIC): inference-trigger events, consumed by the app to
#     drive live scoring + the board. Carries the request-time passthrough only.
#   - `match_events` (MATCH_TOPIC): gameplay activity, consumed by the Streaming
#     Feature View (create_stream, mTLS) into TABLE_MATCH_INGEST; publish_features
#     builds the rolling-window aggregates from there. Each login spawns a dense
#     burst of ~100 match events over ~4 min, so per-player 15-min windows are
#     dense (matching the training timeline in data.sessions).
# When UNSET (the portable one-command default), the synthetic in-workspace
# stream (stream_features.py) is used instead. The broker is provisioned
# out-of-band (provisioning/provision_eventhub.sh for Azure Event Hubs).
TABLE_MATCH_INGEST = "match_events"
# Single-row metadata table written by generate_data holding the exact window
# bounds used for the purchase labels.  setup_stream reads this instead of
# deriving from min/max(event_ts), closing the label-feature alignment gap.
TABLE_SESSION_WINDOW_BOUNDS = "session_window_bounds"
MATCH_STREAM_NAME = "offer_match_stream"
LOGIN_TOPIC = "logins"
MATCH_TOPIC = "match_events"
# mTLS keystore/truststore live on the UC Volume; JKS password in a secret scope.
# (Self-managed Kafka broker over mTLS — the alternative to the UC-connection path.)
KAFKA_SECRET_SCOPE = "offer-reco-kafka"
KAFKA_JKS_PASSWORD_KEY = "jks_password"
KAFKA_KEYSTORE_FILE = "keystore.jks"
KAFKA_TRUSTSTORE_FILE = "truststore.jks"
KAFKA_CERTS_SUBDIR = "kafka_certs"

# --- Azure Event Hubs path (SASL/PLAIN via a UC Kafka connection) ------------
# When RECO_KAFKA_UC_CONNECTION is set, the managed SFV authenticates to the
# broker (Azure Event Hubs, Kafka protocol on :9093) through a Unity Catalog
# Kafka connection (StreamConnectionConfig) instead of DirectMtlsConfig — the
# SFV can't take SASL directly in preview, but a UC connection carries it. The
# connection holds SASL/PLAIN (username="$ConnectionString", password=<EH
# connection string>), so there are NO keystore/truststore certs. The app login
# consumer + the producer use the same SASL connection string, read from
# `KAFKA_SECRET_SCOPE` under the key below.
KAFKA_EVENTHUB_CONN_STRING_KEY = "eventhub_connection_string"


# --- Session window end-timestamp pin ----------------------------------------
# Fixed anchor for the impression-spine and backfill window end.  Both
# generate_data AND setup_stream use this value so they compute IDENTICAL
# session-window bounds regardless of wall-clock order or elapsed time between
# runs.  Override once per install with RECO_SESSION_END_TS (unix seconds as a
# float) via the DAB `session_end_ts` var injected as a task parameter.
#
# DEFAULT: 2026-07-30 00:00:00 UTC (= 1785369600.0) — a concrete recent epoch
# chosen at project initialisation.  It never drifts because it is a constant,
# not a now() call.  Bump the constant (or override the DAB var) only when
# regenerating data for a fresh install.
#
# OPERATIONAL RULE: set `session_end_ts` ONCE at deploy time (--var
# session_end_ts=<unix_s>).  generate_data receives it as argv[2]; setup_stream
# as argv[3].  Both call configure_from_argv so os.environ["RECO_SESSION_END_TS"] is
# set before get_session_end_ts() is called.  Never leave it unset on a
# production workspace and rely on the compiled default, which will drift out of
# sync with the live stream after the first fresh install.
_DEFAULT_SESSION_END_TS: float = 1_785_369_600.0  # 2026-07-30 00:00:00 UTC


def get_session_end_ts() -> pd.Timestamp:
    """Return the pinned session-window end timestamp as a UTC-naive Timestamp.

    Reads ``RECO_SESSION_END_TS`` (unix seconds, float) from the environment;
    falls back to ``_DEFAULT_SESSION_END_TS`` when unset.  Never calls
    ``pd.Timestamp.now()`` — this is a pure constant so that generate_data and
    setup_stream compute identical window bounds across multiple runs.

    Returns:
        UTC-naive ``pd.Timestamp`` at the pinned epoch second.
    """
    import pandas as pd  # local import: config has no top-level pandas dep

    raw = os.environ.get("RECO_SESSION_END_TS")
    ts_s = float(raw) if raw else _DEFAULT_SESSION_END_TS
    return pd.Timestamp(ts_s, unit="s")


def get_kafka_bootstrap() -> str | None:
    """Return the Kafka bootstrap servers if the Kafka login path is enabled.

    None means "not configured" -> callers fall back to the synthetic in-workspace
    stream, so the demo installs and runs with one command and no external broker.
    For the Event Hubs path this is `<namespace>.servicebus.windows.net:9093`.
    """
    return os.environ.get("RECO_KAFKA_BOOTSTRAP") or None


def get_kafka_uc_connection() -> str | None:
    """Return the UC Kafka connection name for the Event Hubs SFV path, or None.

    When set (RECO_KAFKA_UC_CONNECTION), `create_match_stream` authenticates the
    managed SFV via `StreamConnectionConfig(uc_connection_name=...)` (SASL/PLAIN
    carried by the UC connection) instead of `DirectMtlsConfig`. None -> the mTLS
    path (a self-managed Kafka broker).
    """
    return os.environ.get("RECO_KAFKA_UC_CONNECTION") or None


def get_warehouse_id() -> str:
    """Return the SQL warehouse id for Ray Data's UC table read.

    The serverless-GPU training subprocess has no SparkSession, so Ray reads the
    training table via `ray.data.read_databricks_tables` over a SQL warehouse.
    Set per target via RECO_WAREHOUSE_ID (DAB passes it as a task parameter).
    """
    warehouse_id = os.environ.get("RECO_WAREHOUSE_ID")
    if not warehouse_id:
        raise RuntimeError(
            "RECO_WAREHOUSE_ID is not set; the GPU training task reads the "
            "training table via a SQL warehouse (ray.data.read_databricks_tables)."
        )
    return warehouse_id
