"""Register the offer ranker and deploy Model Serving endpoints.

Registration uses `fe.log_model(training_set=...)` so the served model
auto-resolves offline + real-time feature lookups by `player_id` from the
Lakebase online store: the app sends only the entity key + candidate offers.

One real-time endpoint (shop-open path): `min_provisioned_concurrency > 0` so
there is no scale-to-zero cold start on the interactive path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from offer_recommender.config import (
    ENDPOINT_REALTIME,
    MODEL_NAME,
    UCConfig,
)
from offer_recommender.logging_config import get_logger

logger = get_logger(__name__)

# Prefix for the AI Gateway inference table (payload logging). The full table is
# <catalog>.<schema>.<prefix>_payload; usage tracking lands in system.serving.
INFERENCE_TABLE_PREFIX = "offer_ranker_rt_inference"
from offer_recommender.model.net import OfferRankerNet, script_and_save
from offer_recommender.preprocessing.pipeline import TransformParams
from offer_recommender.serving.pyfunc import (
    MODEL_ARTIFACT,
    TRANSFORM_ARTIFACT,
    OfferRankerPyfunc,
)


def _set_experiment(mlflow) -> None:
    """Set an MLflow experiment under the current user's workspace home.

    Job tasks (non-notebook) have no ambient experiment; set one explicitly so
    `start_run` has a valid target. Path: /Users/<email>/<MODEL_NAME>.
    """
    from databricks.sdk import WorkspaceClient

    user = WorkspaceClient().current_user.me().user_name
    mlflow.set_experiment(f"/Users/{user}/{MODEL_NAME}")


def log_and_register(
    fe,
    uc: UCConfig,
    net: OfferRankerNet,
    params: TransformParams,
    training_set,
    metrics=None,
) -> tuple[str, int]:
    """Log the pyfunc, register it to UC, and promote the new version to
    `@champion`. Returns (registered_name, version).

    `metrics`, if given, is a `HeldoutMetrics` scored on the val split; its values
    are logged to the model's MLflow run so the registered version carries an
    honest held-out RMSE/AUC (not just training loss).

    Bundles the TorchScript net + the fitted transform params as artifacts, and
    attaches the TrainingSet so serving resolves feature lookups automatically.

    Ships the offer_recommender package as SOURCE via `code_paths` and pins
    explicit `pip_requirements`, because otherwise `fe.log_model` auto-infers
    `offer-recommender==0.1.0` (our first-party wheel) into the serving env; the
    Model Serving container then tries to `pip install` it from public indexes,
    finds no distribution, and the container build fails. `code_paths` bundles
    the source instead; the requirements are read from the register-time
    installed versions so serving matches training, and MUST retain
    `databricks-feature-lookup` for the online feature auto-lookup at inference.

    Promotion: the just-registered version is aliased `champion`, the single
    source of truth `deploy_serving` reads, so serving always follows champion.
    """
    import mlflow
    import numpy
    import pandas
    import torch
    from mlflow import MlflowClient

    import offer_recommender

    mlflow.set_registry_uri("databricks-uc")
    # A spark_python_task has no default notebook experiment, so mlflow.start_run
    # would log to experiment id None (RESOURCE_DOES_NOT_EXIST). Pin an explicit
    # workspace experiment path so the run has a home.
    _set_experiment(mlflow)
    workdir = Path(tempfile.mkdtemp())
    model_path = str(workdir / "model.ts")
    params_path = str(workdir / "transform_params.json")
    script_and_save(net, model_path)
    Path(params_path).write_text(params.to_json())

    pkg_source = str(Path(offer_recommender.__file__).parent)
    # torch.__version__ on the GPU register runtime carries a CUDA local-version
    # tag (e.g. "2.13.0+cu130"); PyPI has no such build, so the CPU serving
    # container's `pip install torch==2.13.0+cu130` fails. Strip the local tag
    # (everything after "+") so serving installs the plain CPU wheel.
    torch_version = torch.__version__.split("+", 1)[0]
    pip_requirements = [
        f"mlflow=={mlflow.__version__}",
        f"numpy=={numpy.__version__}",
        f"pandas=={pandas.__version__}",
        f"torch=={torch_version}",
        "cloudpickle",
        "databricks-feature-lookup==1.*",
    ]

    # infer_input_example=True derives the request signature from the training
    # set (fe.log_model does NOT reliably forward a plain input_example on this
    # code path). The offer-side columns (offer_id/offer_type/price_tier + offer
    # numerics) are now in the training-set spine as passthrough inputs, so the
    # FeatureSpec keeps them in the signature instead of dropping them as "extra
    # inputs" (which previously made the in-model transform KeyError). The
    # player-side features stay FS-lookups, absent from the request.
    registered = uc.model_name(MODEL_NAME)
    # Log the model WITHOUT registered_model_name: fe.log_model's built-in
    # registration does a CREATE-registered-model call, which the UC metastore's
    # registered-model quota rejects (QUOTA_EXCEEDED) once the workspace is at its
    # cap -- even though offer_ranker already exists and we only want a new VERSION.
    # So we log unregistered, then add a version to the (get-or-created) model
    # explicitly; a version-add is NOT quota-gated (only new-model creation is).
    with mlflow.start_run(run_name="offer_ranker") as run:
        if metrics is not None:
            mlflow.log_metrics(metrics.as_mlflow_metrics())
        fe.log_model(
            model=OfferRankerPyfunc(),
            artifact_path="model",
            flavor=mlflow.pyfunc,
            training_set=training_set,
            artifacts={
                MODEL_ARTIFACT: model_path,
                TRANSFORM_ARTIFACT: params_path,
            },
            code_paths=[pkg_source],
            pip_requirements=pip_requirements,
            infer_input_example=True,
        )
        run_id = run.info.run_id

    client = MlflowClient(registry_uri="databricks-uc")
    # Get-or-create the registered model. The create is the ONLY quota-gated call;
    # it fires only on a truly fresh workspace where offer_ranker doesn't exist yet.
    try:
        client.get_registered_model(registered)
    except Exception:
        client.create_registered_model(registered)
    # Add a version pointing at the just-logged model artifact (runs:/<id>/model).
    mv = client.create_model_version(
        name=registered, source=f"runs:/{run_id}/model", run_id=run_id
    )
    version = int(mv.version)
    client.set_registered_model_alias(registered, "champion", str(version))
    # Also stamp the held-out metrics as tags on the registered version, so the
    # deployed model's generalization score is visible directly on the UC registry
    # page (not only inside the linked MLflow run). Values are strings on tags.
    if metrics is not None:
        for name, value in metrics.as_mlflow_metrics().items():
            client.set_model_version_tag(registered, str(version), name, f"{value:.6g}")
    return registered, version


def _served_entity(registered_model: str, version: str) -> dict:
    """Build a served-entity spec for a registered model version."""
    return {"entity_name": registered_model, "entity_version": version}


def _deploy_endpoint(name: str, entity: dict, route_optimized: bool = False) -> None:
    """Create the endpoint, or update its config if it already exists.

    Idempotent for the daily pipeline re-run. The mlflow deployments client
    raises a raw `requests.HTTPError` (400 RESOURCE_ALREADY_EXISTS) on a repeat
    `create_endpoint`, NOT `databricks.sdk.errors.ResourceAlreadyExists`, so
    branch on a prior existence check via `list_endpoints` rather than catch by
    type (which would let the create path fail on re-run).

    `route_optimized` can ONLY be set at CREATE (the API rejects toggling it on an
    existing endpoint), so it is passed on the create path only; on the daily update
    of an already-route-optimized endpoint it is already baked in. Route optimization
    bypasses the model-serving-proxy (which caps NON-route-optimized serving at ~200
    QPS PER WORKSPACE, shared across all endpoints) via a direct data-plane path,
    lifting CPU serving to ~40K QPS. Route-optimized endpoints must be queried via the
    SDK `serving_endpoints_data_plane.query` (endpoint-scoped OAuth + distinct URL);
    see app/serving_client.py.
    """
    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("databricks")
    config: dict[str, object] = {"served_entities": [entity]}
    exists = any(e["name"] == name for e in client.list_endpoints())
    if exists:
        client.update_endpoint_config(endpoint=name, config=config)
    else:
        if route_optimized:
            config["route_optimized"] = True
        client.create_endpoint(name=name, config=config)


def _enable_ai_gateway(endpoint: str, uc: UCConfig) -> None:
    """Enable AI Gateway on the endpoint: inference-table payload logging.

    Governance/observability for the serving endpoint (a DAIS "Real-Time ML at
    Scale" capability). The inference table logs each request/response to
    `<catalog>.<schema>.<prefix>_payload` for auditing (what was scored, and -- with
    feature tracing -- which features were looked up). Idempotent (PUT); non-fatal so
    a gateway hiccup never fails the pipeline -- the endpoint itself is already
    deployed.

    NOTE: usage tracking (the `system.serving` feed) is deliberately NOT enabled.
    ENDPOINT_REALTIME is route-optimized (to beat the 200 QPS/workspace proxy cap),
    and the serving API REJECTS usage tracking on a route-optimized endpoint
    ("Usage tracking is currently not supported when route optimization is enabled")
    -- usage tracking is measured at the model-serving-proxy that route-opt bypasses.
    Inference-table payload logging IS supported on route-opt, so we keep that (the
    more useful of the two for the demo) and omit usage tracking.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service import serving as sv

    try:
        WorkspaceClient().serving_endpoints.put_ai_gateway(
            endpoint,
            inference_table_config=sv.AiGatewayInferenceTableConfig(
                enabled=True,
                catalog_name=uc.catalog,
                schema_name=uc.schema,
                table_name_prefix=INFERENCE_TABLE_PREFIX,
            ),
        )
        logger.info(
            "AI Gateway enabled on %s (inference table %s.%s.%s_payload; usage "
            "tracking omitted -- unsupported on route-optimized endpoints)",
            endpoint,
            uc.catalog,
            uc.schema,
            INFERENCE_TABLE_PREFIX,
        )
    except Exception as e:
        logger.warning("AI Gateway enable failed on %s (non-fatal): %s", endpoint, e)


def deploy_realtime(registered_model: str, version: str, uc: UCConfig) -> None:
    """Deploy the interactive endpoint with provisioned (non-zero) concurrency,
    then enable AI Gateway (usage tracking + inference tables) on it."""
    entity = _served_entity(registered_model, version)
    # Provisioned concurrency and workload_size are mutually exclusive (the API
    # rejects both). Provisioned concurrency is what gives the no-cold-start
    # interactive path, so set it and omit workload_size.
    # AUTOSCALING BAND (not a hardcoded fixed capacity). Provisioned concurrency
    # autoscales BETWEEN min and max on real traffic signals: horizontal scale-up
    # decisions every ~5s over the trailing 20s (10->10K QPS in <60s), scale-down
    # every ~5min (conservative), vertical per-replica adjust every ~30s. Setting
    # min==max would PIN it and defeat autoscaling, so we set a band.
    #   min=4: the smallest valid warm floor (concurrency is a multiple of 4) -- keeps
    #     one pod warm so the interactive board never cold-starts (min>0 is the
    #     no-cold-start contract), while using almost nothing when the demo is idle.
    #   max=128: a generous ceiling. SIZING (measured 2026-08-01): the endpoint 429s
    #     (bare retry-after, served-model queue backpressure) any in-flight OVER the
    #     current concurrency. By Little's law in_flight = QPS x time_in_system; server
    #     processing is ~2.5ms p50 / ~5ms p99 (tiny model + Lakebase point read), but the
    #     app's in-region round-trip is ~33ms, so 1000 QPS needs ~33 in flight. 128 is
    #     ~4x that -- ample burst headroom -- and the autoscaler only provisions up to
    #     what traffic demands, so the high max costs nothing at low load. Server p50/p99
    #     held dead-flat (2.5/5.0ms) from 300 to 1000 target QPS in testing. The app
    #     scorer's batch + thread pool (app.py) are sized so the board can issue this.
    # ROUTE-OPTIMIZED (route_optimized=True below): non-route-optimized serving caps at
    # ~200 QPS PER WORKSPACE (a model-serving-proxy limit, shared across ALL endpoints,
    # enforced above the container -- so no amount of provisioned concurrency beats it;
    # measured: the standard endpoint held a dead-flat 200/sec while the login stream ran
    # ~557/sec). Route optimization bypasses that proxy via a direct data-plane path,
    # lifting CPU serving to ~40K QPS, so the endpoint actually tracks the stream. The app
    # queries it via serving_endpoints_data_plane.query (route-opt is OAuth-only + a
    # distinct data-plane URL; the SDK handles both transparently for the app SP).
    entity |= {
        "scale_to_zero_enabled": False,
        "min_provisioned_concurrency": 4,
        "max_provisioned_concurrency": 128,
    }
    _deploy_endpoint(ENDPOINT_REALTIME, entity, route_optimized=True)
    _enable_ai_gateway(ENDPOINT_REALTIME, uc)
