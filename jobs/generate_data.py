"""Job task: generate synthetic Harvest Haven data into Unity Catalog.

Writes players / offers / impressions Delta tables to the project schema. Scale
via env vars so the DAB can produce a genuinely large impression log in prod
(which is what makes data-parallel GPU training honest) and a small one in dev.

OPERATIONAL RULE: the DAB var ``session_end_ts`` (unix seconds) is passed as
argv[2] so this job and ``setup_stream`` share IDENTICAL window bounds no matter
when each runs.  Set it ONCE per install; never leave it unset on a workspace
where the live stream is already warmed (else bounds drift between runs).
"""

from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession

from offer_recommender.config import (
    TABLE_IMPRESSIONS,
    TABLE_PLAYERS,
    TABLE_SESSION_WINDOW_BOUNDS,
    configure_from_argv,
    get_config,
    get_session_end_ts,
)
from offer_recommender.data.generate import GenConfig, write_delta
from offer_recommender.logging_config import get_logger

logger = get_logger(__name__)


def _data_already_current(spark, uc, config: GenConfig) -> bool:
    """True when the datasets already exist AND match the requested config.

    Regenerating on every pipeline run is wasteful (and, because it re-anchors the
    session window, would drift the label away from an already-warmed stream). We
    skip generation when all output tables are present AND the persisted
    session_window_bounds match the requested seed + window-end -- i.e. "the data
    is already in place and fine". Any mismatch (seed change, new session_end_ts,
    missing table) -> regenerate. Set RECO_FORCE_REGEN=true to always regenerate.
    """
    if os.environ.get("RECO_FORCE_REGEN", "false").lower() == "true":
        logger.info(
            "RECO_FORCE_REGEN=true -> regenerating regardless of existing data."
        )
        return False

    required = (
        TABLE_PLAYERS,
        TABLE_IMPRESSIONS,
        TABLE_SESSION_WINDOW_BOUNDS,
    )
    for name in required:
        if not spark.catalog.tableExists(uc.table(name)):
            logger.info("Table %s absent -> generating.", uc.table(name))
            return False

    try:
        # One-row metadata table; .first() reads a single row (no full-table collect).
        row = spark.read.table(uc.table(TABLE_SESSION_WINDOW_BOUNDS)).first()
        if row is None:
            raise ValueError("session_window_bounds is empty")
    except Exception as exc:  # unreadable/empty bounds -> regenerate to be safe
        logger.info(
            "session_window_bounds unreadable (%s) -> generating.", str(exc)[:80]
        )
        return False

    # window_end_s is the pinned session_end_ts (integer seconds); seed must match.
    want_end = float(int(config.end_ts.timestamp()))
    have_end = float(row["window_end_s"])
    if int(row["seed"]) != config.seed or abs(have_end - want_end) >= 1.0:
        logger.info(
            "Existing data config drift (seed have=%s want=%s; window_end have=%.0f "
            "want=%.0f) -> regenerating.",
            row["seed"],
            config.seed,
            have_end,
            want_end,
        )
        return False

    n_players = spark.read.table(uc.table(TABLE_PLAYERS)).count()
    logger.info(
        "Data already current (seed=%d, window_end=%.0f, %d players); skipping "
        "generation. Set RECO_FORCE_REGEN=true to force.",
        config.seed,
        have_end,
        n_players,
    )
    return True


def main() -> None:
    """Generate and write the datasets to the configured UC schema.

    Argv layout: [catalog, schema, session_end_ts (optional unix seconds)].
    Idempotent: skips generation when the data is already present and matches the
    requested seed + session_end_ts (see _data_already_current).
    """
    spark = SparkSession.builder.getOrCreate()
    argv = sys.argv[1:]
    configure_from_argv(argv)
    # 3rd param = optional pinned session-window end timestamp (unix seconds).
    # Feeds RECO_SESSION_END_TS so get_session_end_ts() returns the same value
    # that setup_stream uses, making window bounds order-independent.
    if len(argv) >= 3 and argv[2]:
        os.environ.setdefault("RECO_SESSION_END_TS", argv[2])
    uc = get_config()
    config = GenConfig(
        n_players=int(os.environ.get("RECO_N_PLAYERS", "20000")),
        n_offers=int(os.environ.get("RECO_N_OFFERS", "5")),
        impressions_per_player=int(os.environ.get("RECO_IMPRESSIONS", "20")),
        seed=int(os.environ.get("RECO_SEED", "17")),
        end_ts=get_session_end_ts(),
    )
    if _data_already_current(spark, uc, config):
        return
    write_delta(spark, uc.catalog, uc.schema, config)


if __name__ == "__main__":
    main()
