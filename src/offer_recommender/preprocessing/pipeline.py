"""SparkML preprocessing for the offer recommender (hilbert-style fit at scale).

Builds and fits a Spark ML `Pipeline` that mirrors hilbert's sample -> stats ->
transform pattern, all stages inside the fitted PipelineModel:

- log1p on skewed money/count features (SQLTransformer)
- StringIndexer per categorical (integer indices for embeddings, NOT one-hot)
- StandardScaler over the assembled numeric block

The fitted statistics (categorical vocabularies and per-feature mean/std) are
extracted into a `TransformParams` artifact that is bundled into the PyTorch
model, so the serving container reproduces the exact transform with no
SparkSession. `offer_recommender.model.transform` is the numpy/torch twin of this
pipeline, and the parity test asserts they agree element-wise.

Key handoff detail: `add_feature_arrays` expands the scaled Spark Vector (a
VectorUDT that Ray Data / torch cannot read) into a plain array column via
`vector_to_array`, and keeps categorical index columns separate from the
numeric vector so embeddings and the MLP get distinct inputs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from offer_recommender.schema import (
    CATEGORICAL_FEATURES,
    LOG1P_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_PURCHASED,
    TARGET_REVENUE_D7,
)

# Column names produced by the pipeline (kept as constants so the model twin
# and the tests reference the same strings).
NUMERIC_VECTOR_COL = "numeric_scaled"
NUMERIC_ARRAY_COL = "numeric_arr"
INDEX_SUFFIX = "_idx"

# Held-out split column written to training_set. Assigned deterministically by a
# hash of player_id (so all of a player's rows land in the same split -- features
# are player-keyed, so straddling would leak). Training reads split='train'; the
# registration step scores split='val' for honest held-out metrics.
SPLIT_COLUMN = "split"
VAL_SPLIT_PERCENT = 10  # ~10% of players held out for validation


def _log1p_output(feature: str) -> str:
    """Return the post-log1p column name for a feature."""
    return f"{feature}_log" if feature in LOG1P_FEATURES else feature


def index_col(feature: str) -> str:
    """Return the StringIndexer output column for a categorical feature."""
    return f"{feature}{INDEX_SUFFIX}"


@dataclass(frozen=True)
class FeatureColumns:
    """The column contract handed from SparkML preprocessing to Ray training.

    Bundles the four column references (scaled-numeric array, categorical index
    columns, and the two target columns) that always travel together, so they
    are defined once and cannot drift between the training reader, the Ray
    worker loop and the evaluation. Built from the schema so it stays in sync.
    """

    numeric: str
    index: tuple[str, ...]
    revenue: str
    purchase: str

    @classmethod
    def default(cls) -> FeatureColumns:
        """Return the standard contract derived from the schema."""
        return cls(
            numeric=NUMERIC_ARRAY_COL,
            index=tuple(index_col(f) for f in CATEGORICAL_FEATURES),
            revenue=TARGET_REVENUE_D7,
            purchase=TARGET_PURCHASED,
        )

    def selected(self) -> list[str]:
        """All columns to select from the transformed table for training."""
        return [self.numeric, self.revenue, self.purchase, *self.index]


def _log1p_statement() -> str:
    """SQL for the log1p stage: append a `<feat>_log` column per skewed feature."""
    exprs = ", ".join(f"log1p({f}) AS {_log1p_output(f)}" for f in LOG1P_FEATURES)
    return f"SELECT *, {exprs} FROM __THIS__"


def build_pipeline():
    """Build the unfitted SparkML preprocessing Pipeline.

    Ordered stages, all inside the fitted PipelineModel so the transform is
    self-contained (the "SparkML preprocessing at scale" showcase):
      SQLTransformer (log1p on skewed money/count features) -> StringIndexer
      (all categoricals) -> VectorAssembler (numeric block, reading the
      `<feat>_log` columns) -> StandardScaler.

    A prior version moved log1p out to a plain withColumn helper to dodge a
    serverless CONNECT_ML.MODEL_SIZE_OVERFLOW; that was a MISDIAGNOSIS. The real
    cause is warm-pool session ML-cache accumulation (non-deterministic size
    across reruns), NOT the SQLTransformer: an A/B showed 297 MB WITH the
    SQLTransformer vs 358 MB WITHOUT it, so the stage does not bloat the model.
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import (
        SQLTransformer,
        StandardScaler,
        StringIndexer,
        VectorAssembler,
    )

    numeric_inputs = [_log1p_output(f) for f in NUMERICAL_FEATURES]
    stages = [
        SQLTransformer(statement=_log1p_statement()),
        StringIndexer(
            inputCols=list(CATEGORICAL_FEATURES),
            outputCols=[index_col(f) for f in CATEGORICAL_FEATURES],
            handleInvalid="keep",  # unseen -> index numLabels (the OOV slot)
            stringOrderType="frequencyDesc",
        ),
        VectorAssembler(inputCols=numeric_inputs, outputCol="numeric_vec"),
        StandardScaler(
            inputCol="numeric_vec",
            outputCol=NUMERIC_VECTOR_COL,
            withMean=True,
            withStd=True,
        ),
    ]
    return Pipeline(stages=stages)


def add_feature_arrays(df):
    """Expand the scaled Vector into a plain array column for Ray Data / torch.

    VectorUDT is unreadable by external readers; `vector_to_array` yields
    `array<double>`. Categorical index columns are left as separate long
    columns (cast from StringIndexer's double output).
    """
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as sf

    from offer_recommender.schema import ENTITY_KEY

    out = df.withColumn(NUMERIC_ARRAY_COL, vector_to_array(sf.col(NUMERIC_VECTOR_COL)))
    for feature in CATEGORICAL_FEATURES:
        col = index_col(feature)
        out = out.withColumn(col, sf.col(col).cast("long"))
    # Deterministic player-level train/val split: crc32 is stable across runs, so
    # the same player always lands in the same split (features are player-keyed, so
    # splitting by row would leak a player across train and val). pmod guards the
    # sign of crc32's output. Bucketing by player_id (not row) prevents leakage.
    out = out.withColumn(
        SPLIT_COLUMN,
        sf.when(
            sf.pmod(sf.crc32(sf.col(ENTITY_KEY).cast("string")), sf.lit(100))
            < sf.lit(VAL_SPLIT_PERCENT),
            sf.lit("val"),
        ).otherwise(sf.lit("train")),
    )
    # Drop the intermediate VectorUDT columns: training reads only NUMERIC_ARRAY_COL
    # (array<double>). Keeping the table all-plain-types leaves no VectorUDT for any
    # non-Spark reader to choke on (torch/Ray, or a future Delta-native reader).
    return out.drop("numeric_vec", NUMERIC_VECTOR_COL)


@dataclass(frozen=True)
class TransformParams:
    """Fitted transform statistics bundled into the PyTorch model.

    - vocabs: categorical feature -> ordered label list (index = position;
      the OOV/keep index is len(labels)).
    - mean/std: per-feature scaler stats, aligned to NUMERICAL_FEATURES with
      log1p already applied to LOG1P_FEATURES upstream.
    - numerical_features / categorical_features / log1p_features: the exact
      column contract, persisted so serving cannot drift from training.
    """

    vocabs: dict[str, list[str]]
    mean: list[float]
    std: list[float]
    numerical_features: list[str]
    categorical_features: list[str]
    log1p_features: list[str]

    def embedding_sizes(self) -> dict[str, int]:
        """Return num_embeddings per categorical: len(vocab) + 1 for the OOV slot.

        Matches StringIndexer handleInvalid='keep', where an unseen category
        maps to index len(labels). Sizing to len+1 prevents an out-of-range
        embedding lookup (a CUDA device-side assert) on the first unseen value.
        """
        return {f: len(v) + 1 for f, v in self.vocabs.items()}

    def to_json(self) -> str:
        """Serialise to JSON for logging as an MLflow artifact."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> TransformParams:
        """Load from a JSON artifact."""
        return cls(**json.loads(text))


def extract_params(pipeline_model) -> TransformParams:
    """Pull fitted vocabularies and scaler stats out of a fitted PipelineModel."""
    from pyspark.ml.feature import StandardScalerModel, StringIndexerModel

    indexer = next(
        s for s in pipeline_model.stages if isinstance(s, StringIndexerModel)
    )
    scaler = next(
        s for s in pipeline_model.stages if isinstance(s, StandardScalerModel)
    )
    # labelsArray is ordered by input column; align to CATEGORICAL_FEATURES.
    vocabs = {
        feature: list(labels)
        for feature, labels in zip(
            CATEGORICAL_FEATURES, indexer.labelsArray, strict=True
        )
    }
    return TransformParams(
        vocabs=vocabs,
        mean=[float(x) for x in scaler.mean],
        std=[float(x) for x in scaler.std],
        numerical_features=list(NUMERICAL_FEATURES),
        categorical_features=list(CATEGORICAL_FEATURES),
        log1p_features=list(LOG1P_FEATURES),
    )
