# Real-Time ML Inference for Mobile Game LiveOps: project context

A deep-learning gem-offer recommender for a fictional free-to-play studio: SparkML
preprocessing at scale, Ray Train on the serverless-GPU Databricks AI Runtime (AIR),
serving through Model Serving plus a Databricks App backed by the Lakebase online
Feature Store. See README.md for the overview and docs/DEPLOYMENT.md for the runbook.

Private, deployment-specific notes (if any) live in the git-ignored `CLAUDE.local.md`.

## Naming rules

- "Nordfall Games" / "Harvest Haven" are storytelling only (prose, app copy). Never use
  them as a technical identifier, table, schema, resource or env var.
- Technical base name is `offer_recommender` (package, schema) / `offer-` prefix
  (hyphenated resource names). Env-var prefix is `RECO_`.

## Configuration (nothing deployment-specific in git)

- All per-deployment values come from a git-ignored `.env` (template: `env.example`),
  loaded with `set -a && source .env && set +a`. `BUNDLE_VAR_<name>` sets bundle
  variables natively; `EH_*` drives `provisioning/provision_eventhub.sh`.
- `databricks.yml` has only generic `dev` / `prod` targets and no host. Extra named
  targets belong in git-ignored `*.local.yml` files (auto-included).
- Never commit workspace hosts/IDs, catalog names, warehouse IDs, Azure resource names,
  connection strings or profile names. Code defaults stay generic (`main` catalog).

## Stages

1. **Data** (`data/generate.py`, `jobs/generate_data.py`): synthetic players / offers /
   impressions with a learnable latent purchase model. `data/sessions.py` is the one
   per-player session timeline shared bit-for-bit by the numpy label side and the Spark
   backfill (md5-based hashing, `data/hashing.py`; never swap for engine-specific RNGs).
2. **Preprocessing** (`preprocessing/pipeline.py`, `jobs/preprocess.py`): SparkML
   `PipelineModel` (SQLTransformer log1p -> StringIndexer -> assembler -> StandardScaler)
   on serverless, PIT-joined with streaming features (NULL -> 0.0). Exports
   `TransformParams`; `model/transform.py` is the numpy twin used at serving, guarded by
   `tests/test_transform_parity.py`. Also writes `training_set_parquet` to the Volume.
3. **Training** (`training/`, `jobs/train_multinode_launcher.py`, `jobs/train_multinode/`):
   Ray Train `TorchTrainer` (plain PyTorch) via the `air` CLI + workload YAML +
   `ray_bootstrap.sh`, single or multi node. Reads parquet off the UC Volume with
   `ray.data.read_parquet`. Registration is a separate serverless Spark task
   (`jobs/register_model.py`) because `fe.log_model` needs Spark + the FE client.
4. **Serving** (`serving/`, `jobs/deploy_serving.py`): pyfunc runs the numpy transform
   then the TorchScript net; one real-time endpoint `offer-ranker-rt` plus the
   `offer-features-fs` Feature Serving endpoint for the app drill-down.

Pipeline DAG: generate_data -> setup_stream -> preprocess -> {train_multinode,
publish_features} -> register_model -> deploy_serving.

## The model

Multi-head PyTorch net (`model/net.py`): categorical embeddings sized `len(vocab)+1`
(OOV slot) + scaled numeric block -> shared MLP -> `revenue_d7` (primary, log1p space,
drives ranking) and `purchase_logit` (auxiliary BCE). TorchScript needs `ModuleList` and
no `*`-unpack in `torch.cat`.

## Sizing honesty

The model is small and does not need multi-node. The only justification for distributed
GPU is data-parallel throughput over a large impression log. Default is one GPU; frame
multi-node `GPU_8xH100` as the same Ray code scaling to studio-scale data. AIR is Public
Preview (the CLI is Beta): never claim GA. Feature Views are GA since 2026-10-12.

## Platform gotchas (binding)

Serverless / bundles:
- Jobs use serverless environment version 5. SparkML needs environment 4+.
- `requires-python = ">=3.12"` is a floor, not an exact pin (an exact pin gets the wheel
  rejected on serverless).
- Bundle variables are resolved at `bundle deploy`, not `bundle run`.
- Targets use `mode: production`: development-mode schema prefixing produces a quoted
  schema name that the streaming materialization's Lakebase sink rejects.
- Bump the wheel `version` in pyproject on every code change to a job, or serverless may
  install a cached wheel and the change silently does not take effect.
- Serverless caps the Spark ML model cache at 1 GB per session and 100 MB per model,
  and neither limit can be changed; the docs say to free the cache by deleting Python
  references to models. Preprocess's fitted model is tiny, yet we saw
  `MODEL_SIZE_OVERFLOW` intermittently on reused warm sessions. Keep the
  `del model, transformed; gc.collect()` hygiene; re-run on a cold session if it trips.

AI Runtime / Ray:
- Training launches through the Python `air` CLI (`databricks-air`), which the docs now
  mark deprecated in favor of `databricks air` in the Databricks CLI. The notes below are
  observed behavior of the Python CLI.
- The workload uses the named AI env `databricks_ai_v5` (Ray + PyTorch preinstalled). On
  our workspace it only launched after enabling the "Serverless workspace base
  environment support in Jobs" preview, and the CLI rejected `version:` without an inline
  `dependencies:` list (the current docs list neither requirement).
- A relative `code_source.snapshot.root_path` resolves from the workload YAML's directory,
  so the launcher renders an absolute path. Launchers derive paths from `sys.argv[0]`
  (`__file__` is undefined in a serverless spark_python_task).
- `read_parquet` on a Spark output dir needs `file_extensions=["parquet"]` (Spark's
  0-byte marker files otherwise fail as parquet).
- Inside a Ray Tune function use `ray.tune.report`, not `ray.train.report`.
- Workers in `ray_bootstrap.sh` must not retry around `ray start --block`, or they
  re-join a torn-down head and hang the job.

Feature Store (declarative Feature Views, `databricks-feature-engineering==0.16.1`,
exact pin in pyproject + resources/jobs.yml):
- ColumnSelection features use `TableTrigger` (they reject CronSchedule); StreamSource
  aggregation features use `StreamingMode()`.
- `materialize_features` is not idempotent. To reset, delete each feature's
  materializations (`list_materialized_features` -> `delete_materialized_feature`) before
  `delete_feature`. `fs_reset=true` does this (`store.materialize_online(reset=True)`).
- StreamSource features reference payload columns with the `value.` prefix
  (`value.player_id`); the training join and online lookup use bare names. `event_ts`
  must parse as a TIMESTAMP. Streaming RollingWindow supports AVG / COUNT / SUM /
  STDDEV / FIRST / LAST variants, not approx-count-distinct.
- RequestSource passthrough features (offer columns, realtime passthroughs) must be
  declared Features, or serving drops them. On 0.16.1 they take no entity/timeseries.
- The stream backfill (`StreamBackfillSource`) feeds the ingestion table for training
  only. The online store materializes forward from the live stream; a player with no
  in-window events has no online row, so the numpy twin fills streaming NaNs with 0.0
  (matching training).
- On a feature-set or generate-side schema change: deploy with `fs_reset=true` and drop
  `impressions`, `players` and `session_window_bounds` so generate_data regenerates.
- Never resize the online store on the hot path (`ensure_online_store` is create-only):
  a resize rotates credentials and wedges deployed endpoints' Lakebase auth ("No
  available connection in 2s"). Recover by redeploying the endpoint config.

Serving:
- No SparkSession in the container: the transform runs from exported params.
- `log_and_register` ships the package via `code_paths` with explicit `pip_requirements`
  (keeps `databricks-feature-lookup`), registers the model only if absent, adds a version
  and sets `@champion`; deploy serves `@champion`.
- `event_ts` must be tz-naive in the scoring payload.

App (`app/`): FastAPI + CDN React (no npm). One background scorer drains the login stream
into a single in-process aggregate the browser polls (`/api/board`), so the app must stay
single-replica. Deploy app code with `bundle deploy` then `bundle run
offer_recommender_console` (`run` alone serves the last-synced files).

## Conventions

- Logging, not print: `from offer_recommender.logging_config import get_logger`.
- Local dev: `uv venv --python 3.12 && uv pip install -e '.[dev]'`. Spark tests need a
  JDK; `tests/conftest.py` sets `PYSPARK_PYTHON`.
- Tests: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`; JS:
  `node --test app/static/pure.test.js`. Lint: `.venv/bin/ruff check src/ tests/ jobs/ app/`.
