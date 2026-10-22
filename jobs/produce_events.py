"""Serverless synthetic event producer -> Azure Event Hubs (Kafka sink, SASL/PLAIN).

No broker box to run and no mTLS certs: a managed Kafka endpoint plus a paced
serverless job. Emits the same two causally-related streams
the demo needs:

- `logins` (LOGIN_TOPIC): a login TRIGGERS scoring. Uniform player sampling, the
  request-time passthrough snapshot (gem_balance, minutes_since_last_purchase,
  match_count, is_bursty). NO session signals.
- `match_events` (MATCH_TOPIC): gameplay activity feeding the Streaming Feature View
  rolling-window aggregates. Restricted to BURSTY players (is_bursty, md5 on RECO_SEED,
  mirrors data.sessions) so their 15-min windows are dense (what the app board reads and
  the model trained on); carries the session signals (gem_spend, gem_balance,
  offer_shown, match_won, match_lost).

SERVERLESS CONSTRAINT (why this is a paced batch loop, not a rate->Kafka stream): a
continuous ProcessingTime trigger is rejected on serverless
(INFINITE_STREAMING_TRIGGER_NOT_SUPPORTED) and a rate source under AvailableNow yields
~0 rows at startup. So we PACE BATCH Kafka writes (same Kafka sink, same event
expressions) once per second for RECO_PRODUCER_RUN_SECONDS; the DAB job's `continuous`
trigger relaunches it -> a steady stream of per-second batches.

Config-gated: requires RECO_KAFKA_BOOTSTRAP (Event Hubs namespace :9093) and the SAS
connection string in secret scope `KAFKA_SECRET_SCOPE` / key `KAFKA_EVENTHUB_CONN_STRING_KEY`.

Exact per-login heap coupling (login_producer.py) is not reproduced — the accepted
serverless trade-off; burst DENSITY on the bursty set is preserved.
"""

from __future__ import annotations

import os
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from offer_recommender.config import (
    KAFKA_EVENTHUB_CONN_STRING_KEY,
    KAFKA_SECRET_SCOPE,
    LOGIN_TOPIC,
    MATCH_TOPIC,
    configure_from_argv,
    get_kafka_bootstrap,
)
from offer_recommender.logging_config import get_logger

logger = get_logger(__name__)

N_PLAYERS = int(os.environ.get("RECO_N_PLAYERS", "20000"))
SEED = int(os.environ.get("RECO_SEED", "17"))
LOGIN_RATE = int(os.environ.get("RECO_LOGIN_RATE", "1100"))
MATCH_RATE = int(
    os.environ.get("RECO_MATCH_RATE", "5000")
)  # candidates before bursty filter
BURSTY_FRACTION = float(os.environ.get("RECO_BURSTY_FRACTION", "0.01"))
RUN_SECONDS = int(os.environ.get("RECO_PRODUCER_RUN_SECONDS", "300"))


def _player_id(num_col):
    return F.format_string("player_%06d", num_col.cast("int"))


def _is_bursty(pid_col):
    """Mirror sessions.is_bursty: md5-uniform < BURSTY_FRACTION on the shared key."""
    h = F.conv(
        F.substring(F.md5(F.concat(pid_col, F.lit(f":bursty:{SEED}"))), 1, 8), 16, 10
    )
    return (h / F.lit(4294967296.0)) < F.lit(BURSTY_FRACTION)


def main() -> None:
    spark = SparkSession.builder.getOrCreate()
    argv = sys.argv[1:]
    configure_from_argv(argv)  # consumes [catalog, schema] if passed
    # 3rd param = Event Hubs bootstrap (DAB var.kafka_bootstrap); env still wins.
    if len(argv) >= 3 and argv[2]:
        os.environ.setdefault("RECO_KAFKA_BOOTSTRAP", argv[2])
    # 4th param (optional) = logins/sec. Lets a target throttle the stream below the
    # ~200 QPS standard-serving proxy cap when the app scores via the standard
    # invocations path (non-route-optimized endpoint). Empty -> keep RECO_LOGIN_RATE.
    if len(argv) >= 4 and argv[3]:
        global LOGIN_RATE
        LOGIN_RATE = int(argv[3])
    bootstrap = get_kafka_bootstrap()
    if not bootstrap:
        raise RuntimeError(
            "RECO_KAFKA_BOOTSTRAP required: the producer writes to Event Hubs. "
            "Deploy with --var kafka_bootstrap=<namespace>.servicebus.windows.net:9093."
        )
    conn = dbutils.secrets.get(KAFKA_SECRET_SCOPE, KAFKA_EVENTHUB_CONN_STRING_KEY)  # noqa: F821
    # Event Hubs SASL/PLAIN: the username is the literal "$ConnectionString" and the
    # password is the SAS connection string just read from the secret scope.
    jaas = (
        "kafkashaded.org.apache.kafka.common.security.plain.PlainLoginModule required "
        f'username="$ConnectionString" password="{conn}";'
    )
    kafka_opts = {
        "kafka.bootstrap.servers": bootstrap,
        "kafka.security.protocol": "SASL_SSL",
        "kafka.sasl.mechanism": "PLAIN",
        "kafka.sasl.jaas.config": jaas,
    }
    ts = F.date_format(F.current_timestamp(), "yyyy-MM-dd'T'HH:mm:ss'Z'")

    def login_batch():
        return (
            spark.range(LOGIN_RATE)
            .withColumn("player_id", _player_id(F.rand() * N_PLAYERS))
            .select(
                F.col("player_id").alias("key"),
                F.to_json(
                    F.struct(
                        F.col("player_id"),
                        ts.alias("event_ts"),
                        F.lit("login").alias("event_type"),
                        F.round(F.rand() * 3000).alias("gem_balance"),
                        F.round(F.rand() * 8000, 1).alias(
                            "minutes_since_last_purchase"
                        ),
                        (F.floor(F.rand() * 20) + 1).alias("match_count"),
                        _is_bursty(F.col("player_id")).alias("is_bursty"),
                    )
                ).alias("value"),
            )
        )

    def match_batch():
        return (
            spark.range(MATCH_RATE)
            .withColumn("player_id", _player_id(F.rand() * N_PLAYERS))
            .where(_is_bursty(F.col("player_id")))
            .withColumn("won", (F.rand() < F.lit(0.5)).cast("int"))
            .select(
                F.col("player_id").alias("key"),
                F.to_json(
                    F.struct(
                        F.col("player_id"),
                        ts.alias("event_ts"),
                        F.lit("match").alias("event_type"),
                        F.when(F.rand() < 0.15, F.round(F.rand() * 200, 2))
                        .otherwise(0.0)
                        .alias("gem_spend"),
                        F.round(F.rand() * 3000).alias("gem_balance"),
                        (F.rand() < 0.60).cast("int").alias("offer_shown"),
                        F.col("won").alias("match_won"),
                        (F.lit(1) - F.col("won")).alias("match_lost"),
                    )
                ).alias("value"),
            )
        )

    def send(df, topic):
        df.write.format("kafka").options(**kafka_opts).option("topic", topic).save()

    logger.info(
        "Producing ~%d logins/s -> %s + bursty matches -> %s on %s for %ds",
        LOGIN_RATE,
        LOGIN_TOPIC,
        MATCH_TOPIC,
        bootstrap,
        RUN_SECONDS,
    )
    deadline = time.monotonic() + RUN_SECONDS
    ticks = 0
    while time.monotonic() < deadline:
        start = time.monotonic()
        send(login_batch(), LOGIN_TOPIC)
        send(match_batch(), MATCH_TOPIC)
        ticks += 1
        elapsed = time.monotonic() - start
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
    logger.info("Producer finished after %d ticks (continuous job relaunches).", ticks)


if __name__ == "__main__":
    main()
