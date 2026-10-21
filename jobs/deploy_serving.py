"""Job task: deploy the `@champion` model version to Model Serving.

The `champion` alias is set by register_model as the promotion step; this task
resolves it (single source of truth) and deploys that exact version to the
real-time endpoint (provisioned concurrency, no cold start). Run after the
training job has registered a model.
"""

from __future__ import annotations

import sys

from offer_recommender.config import MODEL_NAME, configure_from_argv, get_config
from offer_recommender.logging_config import get_logger
from offer_recommender.serving.deploy import deploy_realtime

logger = get_logger(__name__)


def main() -> None:
    """Resolve the champion version and deploy both endpoints."""
    from mlflow import MlflowClient

    configure_from_argv(sys.argv[1:])
    uc = get_config()
    registered = uc.model_name(MODEL_NAME)

    # Serve whatever register_model promoted to @champion, so the alias is the
    # one source of truth (no version drift between register and deploy).
    client = MlflowClient(registry_uri="databricks-uc")
    champion = client.get_model_version_by_alias(registered, "champion")

    deploy_realtime(registered, champion.version, uc)
    logger.info("Deployed %s v%s (@champion) to serving.", registered, champion.version)


if __name__ == "__main__":
    main()
