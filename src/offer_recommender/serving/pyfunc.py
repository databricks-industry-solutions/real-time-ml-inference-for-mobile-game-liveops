"""MLflow pyfunc wrapper for serving the offer ranker.

Runs the numpy/torch transform twin (from the bundled TransformParams) then the
scripted net. It deliberately does NOT touch SparkML: the Model Serving
container has no SparkSession, so the fitted transform is reproduced from the
exported params (the parity test guarantees they match). Feature lookups are
resolved by the endpoint via `fe.log_model(training_set=...)`, so at inference
the request carries the entity key + candidate-offer columns and the offline /
real-time features arrive already joined.

Output is expected dollar revenue per candidate row (expm1 of the log-space
head); the caller ranks and applies the cooldown gate.
"""

from __future__ import annotations

import mlflow
import pandas as pd

TRANSFORM_ARTIFACT = "transform_params"
MODEL_ARTIFACT = "scripted_model"


class OfferRankerPyfunc(mlflow.pyfunc.PythonModel):
    """Pyfunc that applies the in-model transform then scores the net."""

    def load_context(self, context) -> None:
        """Load the scripted net and the fitted transform params (no Spark)."""
        import torch

        from offer_recommender.preprocessing.pipeline import TransformParams

        with open(context.artifacts[TRANSFORM_ARTIFACT]) as f:
            self._params = TransformParams.from_json(f.read())
        self._net = torch.jit.load(context.artifacts[MODEL_ARTIFACT]).eval()

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Score a batch of (player features + candidate offer) rows.

        Returns a frame with, per input row:
        - `expected_offer_revenue` (dollars): the primary ranking signal (expm1 of
          the revenue head, which trains in log1p space).
        - `purchase_probability` (0-1): sigmoid of the auxiliary purchase head --
          the modeled likelihood the player buys THIS offer.
        """
        import numpy as np
        import torch

        from offer_recommender.model.transform import transform

        numeric, categorical = transform(model_input, self._params)
        with torch.no_grad():
            out = self._net(
                torch.tensor(numeric, dtype=torch.float32),
                torch.tensor(categorical, dtype=torch.long),
            )
        revenue = np.expm1(out["revenue_d7"].numpy())
        purchase_prob = torch.sigmoid(out["purchase_logit"]).numpy()
        return pd.DataFrame(
            {
                "expected_offer_revenue": revenue,
                "purchase_probability": purchase_prob,
            }
        )
