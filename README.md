# Real-Time ML Inference for Mobile Game LiveOps

[![Databricks](https://img.shields.io/badge/Databricks-Solution_Accelerator-FF3621?style=for-the-badge&logo=databricks)](https://databricks.com)
[![Unity Catalog](https://img.shields.io/badge/Unity_Catalog-Enabled-00A1C9?style=for-the-badge)](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)
[![Serverless](https://img.shields.io/badge/Serverless-Compute-00C851?style=for-the-badge)](https://docs.databricks.com/en/compute/serverless.html)

A free-to-play studio ranks in-game gem offers for each player the moment they log
in. This accelerator builds that loop end to end on Databricks: SparkML fits the
feature transforms at scale, **Ray Train** trains a multi-head PyTorch ranker on the
serverless-GPU **Databricks AI Runtime**, the **Feature Store** serves batch and
streaming features from a Lakebase-backed online store and **Model Serving** plus a
**Databricks App** score every login live.

The studio ("Nordfall Games") and its game ("Harvest Haven") are fictional; all data
is synthetic. The Databricks AI Runtime and Streaming Feature Views are in Public
Preview.

## Architecture

1. **Data:** synthetic players, offers and a shop-impression log in Unity Catalog, with
   a learnable purchase model whose label depends on recent in-session activity.
2. **Preprocessing:** a SparkML `PipelineModel` (log1p, StringIndexer, StandardScaler),
   point-in-time joined with the streaming features. Its fitted parameters are exported
   and bundled into the model, so serving needs no Spark.
3. **Training:** Ray Train `TorchTrainer` on the AI Runtime via the `air` CLI and a
   workload YAML. One GPU by default; the same code scales to multi-node `GPU_8xH100`.
4. **Features:** declarative Feature Views. Offline features materialize from Delta;
   rolling-window streaming features (15 minutes) come from a managed Streaming Feature
   View over Kafka. Both sync to the Lakebase online store.
5. **Serving:** the model is logged with its feature spec, so the `offer-ranker-rt`
   endpoint looks up features by `player_id` and the caller sends only the key and the
   candidate offers. A FastAPI + React app consumes the login stream and shows a live
   board of scored decisions with a per-player feature drill-down.

Real-time events come from **Azure Event Hubs** (Kafka protocol). This path is optional:
with no broker configured, the pipeline uses a synthetic in-workspace stream.

## Prerequisites

- A Unity Catalog-enabled Databricks workspace with serverless jobs, Model Serving,
  Databricks Apps and the AI Runtime (see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for
  the required workspace preview).
- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html) and
  [uv](https://docs.astral.sh/uv/).
- Optional, for the real-time path: an Azure subscription and the Azure CLI.

## Configuration

Every deployment-specific value lives in one git-ignored `.env` file; nothing
workspace-specific is committed. `env.example` documents every setting:

```bash
cp env.example .env      # set DATABRICKS_CONFIG_PROFILE and BUNDLE_VAR_catalog at minimum
set -a && source .env && set +a
```

`BUNDLE_VAR_<name>` entries set the bundle variables declared in `databricks.yml`, and
the CLI reads them natively. `EH_*` entries drive the Event Hubs provisioning script. To
keep several deployments side by side, add your own targets to a git-ignored
`*.local.yml` file, which `databricks.yml` includes automatically.

## Deploy

Quick start with the synthetic stream (no external broker):

```bash
set -a && source .env && set +a
databricks bundle deploy
databricks bundle run offer_recommender_pipeline
```

The pipeline generates data, preprocesses, trains on serverless GPU, publishes features,
registers the model and deploys the serving endpoints. The real-time path (Event Hubs,
producer job, live app) has a required deploy order and a few post-deploy steps; follow
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

To remove everything: `databricks bundle destroy` (or `scripts/cleanup.sh`).

## Layout

- `src/offer_recommender/`: the ML package (data, preprocessing, model, training,
  features, serving).
- `jobs/`: job entry points and the AI Runtime workload specs.
- `app/`: the live-inference console (Databricks App).
- `resources/` + `databricks.yml`: the bundle (schema, jobs, app, secret scope).
- `provisioning/`: Event Hubs and UC-connection setup scripts.
- `tests/`: transform parity, out-of-vocabulary, ranking-quality and serving-artifact
  guards.

## Local development

```bash
uv venv --python 3.12 && uv pip install -e '.[dev]'
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q   # Spark tests need a JDK
.venv/bin/ruff check src/ tests/ jobs/ app/
node --test app/static/pure.test.js
```

## Contributing

1. `git clone` this project locally.
2. Use the Databricks CLI to test your changes against a workspace of your choice.
3. Contribute through pull requests, with a second-party review from a capable teammate.

## Third-Party Package Licenses

&copy; 2025 Databricks, Inc. All rights reserved. The source in this project is provided
subject to the Databricks License [https://databricks.com/db-license-source]. All
included or referenced third party libraries are subject to the licenses set forth
below.

| Package | License | Copyright |
|---------|---------|-----------|
| numpy | BSD-3-Clause | NumPy Developers |
| pandas | BSD-3-Clause | pandas Development Team |
| pyspark | Apache-2.0 | The Apache Software Foundation |
| pyarrow | Apache-2.0 | The Apache Software Foundation |
| torch | BSD-3-Clause | PyTorch Contributors |
| ray | Apache-2.0 | The Ray Team |
| scikit-learn | BSD-3-Clause | scikit-learn Developers |
| mlflow | Apache-2.0 | Databricks, Inc. |
| fsspec | BSD-3-Clause | fsspec Developers |
| databricks-sdk | Apache-2.0 | Databricks, Inc. |
| databricks-feature-engineering | Databricks License | Databricks, Inc. |
| databricks-feature-lookup | Databricks License | Databricks, Inc. |
| databricks-air | Databricks License | Databricks, Inc. |
| fastapi | MIT | Sebastián Ramírez |
| uvicorn | BSD-3-Clause | Encode OSS Ltd. |
| uvloop | MIT / Apache-2.0 | MagicStack Inc. |
| httpx | BSD-3-Clause | Encode OSS Ltd. |
| sqlalchemy | MIT | SQLAlchemy Authors |
| psycopg2-binary | LGPL-3.0 with exceptions | Federico Di Gregorio, Daniele Varrazzo |
| kafka-python | Apache-2.0 | Dana Powers, David Arthur |
| React / ReactDOM | MIT | Meta Platforms, Inc. |
| @babel/standalone | MIT | Sebastian McKenzie and contributors |
| Inter, JetBrains Mono (fonts) | SIL OFL-1.1 | The Inter / JetBrains Mono Authors |
| ruff, mypy, pytest (dev only) | MIT | Their respective authors |
