"""Kafka login consumer for the live board.

When the Kafka path is enabled, the app is a Kafka consumer (its OWN consumer
group, independent of the Streaming Feature View): each consumed `login` event is
one board tick -> one inference call (the SSE loop drains and scores them in
concurrent batches, so inference throughput tracks the ingestion rate), so the
shown decisions are driven by actual streamed logins rather than a timer over
random players.

mTLS certs are read at startup from the `offer-reco-kafka` secret scope (base64
PEMs) and materialized to /tmp cert files -- the app owns its runtime, so unlike
the managed SFV it CAN write the cert files itself (no UC Volume needed). Secrets,
not app env: multi-KB base64 values get mangled through app.yaml var
interpolation, and secrets are the right home for sensitive material. Falls back
cleanly: if Kafka is off or the client can't be built, `next_login()` returns
None and the board simply waits (no synthetic fallback for now).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import ssl
import tempfile
import threading

import metrics

logger = logging.getLogger(__name__)

_QUEUE: queue.Queue[dict] = queue.Queue(maxsize=2000)
_STARTED = False
_LOCK = threading.Lock()


_SECRET_SCOPE = os.environ.get("RECO_KAFKA_SECRET_SCOPE", "offer-reco-kafka")
# Event Hubs SASL path: SAS connection string stored under this key in the scope
# (mirrors config.KAFKA_EVENTHUB_CONN_STRING_KEY; kept as a literal here so the
# app needs no import of the offer_recommender package).
_EVENTHUB_CONN_STRING_KEY = "eventhub_connection_string"


def _secret(key: str) -> str | None:
    """Read a plaintext secret value from the scope (None if unreadable)."""
    from databricks.sdk import WorkspaceClient

    try:
        api_value = (
            WorkspaceClient().secrets.get_secret(scope=_SECRET_SCOPE, key=key).value
        )
        return base64.b64decode(api_value).decode()
    except Exception as e:
        logger.warning("Kafka secret %s unreadable: %s", key, str(e)[:120])
        return None


def _bootstrap() -> str | None:
    """Kafka bootstrap servers, from the secret scope (preferred) or env.

    The secret scope is the reliable source: app.yaml `${var.*}` interpolation does
    NOT resolve for app env values -- it ships the LITERAL "${var.kafka_bootstrap}"
    string, which is truthy and poisons an env-first lookup. So we take the secret
    first, and only accept the env var if it is set AND not an un-interpolated
    "${...}" placeholder (that path is for genuine local overrides).
    """
    env = os.environ.get("RECO_KAFKA_BOOTSTRAP")
    if env and not env.startswith("${"):
        return env
    return _secret("bootstrap")


def _write_cert_from_secret(key: str, suffix: str) -> str | None:
    """Read a base64 PEM from the secret scope, materialize to a private temp file.

    The mTLS certs are stored base64 in the `offer-reco-kafka` secret scope (not in
    app env: multi-KB values get mangled through app.yaml var interpolation, and
    secrets are the right home for sensitive material). The app service principal
    has READ on the scope. Returns the temp-file path, or None if unreadable.
    """
    from databricks.sdk import WorkspaceClient

    try:
        # secrets.get_secret().value is API-base64 of the STORED string; the stored
        # string is itself our base64 PEM -> two decodes to reach the PEM bytes.
        api_value = (
            WorkspaceClient().secrets.get_secret(scope=_SECRET_SCOPE, key=key).value
        )
        stored_b64 = base64.b64decode(api_value).decode()
        pem = base64.b64decode(stored_b64)
    except Exception as e:
        logger.warning("Kafka cert secret %s unreadable: %s", key, str(e)[:120])
        return None
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="reco_kafka_")
    with os.fdopen(fd, "wb") as f:
        f.write(pem)
    os.chmod(path, 0o600)
    return path


def _consumer_kwargs(bootstrap: str) -> dict | None:
    """Auth kwargs for the KafkaConsumer: Event Hubs SASL if a connection string
    is present in the scope, else self-managed-broker mTLS. None if unconfigured.
    """
    # Event Hubs path: SASL/PLAIN over TLS with the SAS connection string. Server
    # cert is a public CA and the host is a real DNS name, so a default SSL context
    # (hostname verification ON) is correct -- no certs to materialize.
    conn_string = _secret(_EVENTHUB_CONN_STRING_KEY)
    if conn_string:
        logger.info("Kafka login consumer using Event Hubs SASL/PLAIN.")
        return {
            "bootstrap_servers": bootstrap.split(","),
            "security_protocol": "SASL_SSL",
            "sasl_mechanism": "PLAIN",
            "sasl_plain_username": "$ConnectionString",
            "sasl_plain_password": conn_string,
            "ssl_context": ssl.create_default_context(),
            "value_deserializer": lambda b: json.loads(b.decode()),
        }

    # Self-managed broker path: mTLS. Broker cert SAN is the IP, not a DNS name ->
    # disable hostname check (same as the SFV's disable_hostname_verification).
    client_cert = _write_cert_from_secret("client_cert", ".crt")
    client_key = _write_cert_from_secret("client_key", ".key")
    ca_cert = _write_cert_from_secret("ca_cert", ".crt")
    if not (client_cert and client_key and ca_cert):
        logger.warning(
            "Kafka enabled but no SASL conn string and mTLS certs missing; board idle."
        )
        return None
    ssl_ctx = ssl.create_default_context(cafile=ca_cert)
    ssl_ctx.load_cert_chain(certfile=client_cert, keyfile=client_key)
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_REQUIRED
    logger.info("Kafka login consumer using mTLS (self-managed broker).")
    return {
        "bootstrap_servers": bootstrap.split(","),
        "security_protocol": "SSL",
        "ssl_context": ssl_ctx,
        "value_deserializer": lambda b: json.loads(b.decode()),
    }


def _build_consumer():
    """Build a KafkaConsumer (Event Hubs SASL or mTLS), or None if not configured."""
    bootstrap = _bootstrap()
    if not bootstrap:
        return None
    kwargs = _consumer_kwargs(bootstrap)
    if kwargs is None:
        return None

    from kafka import KafkaConsumer, TopicPartition

    topic = os.environ.get("RECO_KAFKA_TOPIC", "logins")
    # NO consumer group: we manually ASSIGN the partition and seek to the end.
    # A group_id needs the group-coordinator + __consumer_offsets round-trip, which
    # stalled against a single-node broker from the app. The board is a single
    # ephemeral consumer that only wants the live tail -- no offset commit, no
    # rebalance -- so a group buys nothing. Manual assign + seek_to_end sidesteps it
    # (works the same against Event Hubs).
    consumer = KafkaConsumer(**kwargs)
    parts = consumer.partitions_for_topic(topic) or {0}
    tps = [TopicPartition(topic, p) for p in parts]
    consumer.assign(tps)
    consumer.seek_to_end()  # live tail only, skip the backlog
    return consumer


def _consume_loop() -> None:
    """Background: pull login events, push player_ids onto the queue."""
    try:
        consumer = _build_consumer()
    except Exception as e:
        logger.warning("Kafka consumer init failed; board stays synthetic: %s", e)
        return
    if consumer is None:
        return
    logger.info("Kafka login consumer started (tailing live logins).")
    consumed = 0
    # Explicit poll() loop, NOT `for msg in consumer`: with a manual assignment the
    # iterator's timeout semantics stalled (yielded nothing); poll() with a real
    # timeout reliably returns the live tail.
    while True:
        batch = consumer.poll(timeout_ms=1000, max_records=50)
        for _tp, msgs in batch.items():
            for msg in msgs:
                event = msg.value or {}
                if not event.get("player_id"):
                    continue
                consumed += 1
                metrics.record_ingest()
                # Heartbeat only: at ~570 logins/sec the old every-100th line was ~6
                # log calls/sec in the hot consumer path (string-format + stdout I/O)
                # on a CPU-bound 2-vCPU app. Log the first one (confirms the consumer
                # is live) then every 10000th (~once every 15-20s) -- enough to see
                # the tail is moving, negligible CPU. The /api/metrics ingestion rate
                # is the real throughput signal, not these lines.
                if consumed == 1 or consumed % 10000 == 0:
                    logger.info(
                        "Consumed login #%d: player=%s (offset %d)",
                        consumed,
                        event["player_id"],
                        msg.offset,
                    )
                _push(event)


def _push(event: dict) -> None:
    """Queue a login event, dropping the oldest if the board is behind."""
    try:
        _QUEUE.put_nowait(event)
    except queue.Full:
        # Board is slower than the stream; drop oldest so we stay near-live.
        try:
            _QUEUE.get_nowait()
            _QUEUE.put_nowait(event)
        except queue.Empty:
            pass


def start() -> None:
    """Start the consumer thread once (idempotent, no-op if Kafka is off).

    Always starts the thread (unless already started); the thread itself resolves
    bootstrap + certs from the secret scope and exits cleanly if Kafka isn't
    configured. We do NOT gate on the env var here because app.yaml `${var.*}`
    interpolation is unreliable -- the real gate is whether _build_consumer can
    resolve a bootstrap + certs from the secret scope.
    """
    global _STARTED
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
    threading.Thread(
        target=_consume_loop, daemon=True, name="kafka-login-consumer"
    ).start()


def next_login(timeout: float = 1.5) -> dict | None:
    """Return the next streamed login EVENT (dict with player_id + real-time
    feature values), or None if none arrive within timeout.

    The full event is returned (not just the key) because the real-time features
    are request-time passthrough: the app sends the login's own values in the
    serving request (millisecond-fresh), rather than looking them up from the
    online store. None -> the board simply waits (no synthetic fallback for now).
    """
    try:
        return _QUEUE.get(timeout=timeout)
    except queue.Empty:
        return None


def next_batch(max_n: int, timeout: float = 30.0) -> list[dict]:
    """Drain up to `max_n` queued login events (blocking up to `timeout` for the
    first, then draining whatever else is already queued without blocking).

    Returns [] if no login arrives within `timeout`. (Not used by the scorer, which
    pulls one login per worker via next_login; kept for ad-hoc/batch callers.)
    """
    first = next_login(timeout)
    if first is None:
        return []
    batch = [first]
    while len(batch) < max_n:
        try:
            batch.append(_QUEUE.get_nowait())
        except queue.Empty:
            break
    return batch
