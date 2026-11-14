"""Define Pydantic schemas and domain bounds for crop recommendation.

This module declares the canonical `FEATURE_ORDER`, `DOMAIN_BOUNDS`, and
strictly validated data contracts shared across artifact loading, inference,
and UI components.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


FEATURE_ORDER: tuple[str, ...] = (
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
)

DOMAIN_BOUNDS: dict[str, tuple[float, float]] = {
    "N": (0.0, 500.0),
    "P": (0.0, 500.0),
    "K": (0.0, 500.0),
    "temperature": (-10.0, 60.0),
    "humidity": (0.0, 100.0),
    "ph": (0.0, 14.0),
    "rainfall": (0.0, 1000.0),
}


class CropInput(BaseModel):
    """Validated soil and climate input parameters for single-row inference.

    Attributes:
        N: Soil nitrogen concentration (0.0 to 500.0).
        P: Soil phosphorus concentration (0.0 to 500.0).
        K: Soil potassium concentration (0.0 to 500.0).
        temperature: Ambient temperature in degrees Celsius (-10.0 to 60.0).
        humidity: Relative humidity percentage (0.0 to 100.0).
        ph: Soil pH level (0.0 to 14.0).
        rainfall: Rainfall volume in millimeters (0.0 to 1000.0).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    N: float = Field(..., ge=0.0, le=500.0, allow_inf_nan=False)
    P: float = Field(..., ge=0.0, le=500.0, allow_inf_nan=False)
    K: float = Field(..., ge=0.0, le=500.0, allow_inf_nan=False)
    temperature: float = Field(..., ge=-10.0, le=60.0, allow_inf_nan=False)
    humidity: float = Field(..., ge=0.0, le=100.0, allow_inf_nan=False)
    ph: float = Field(..., ge=0.0, le=14.0, allow_inf_nan=False)
    rainfall: float = Field(..., ge=0.0, le=1000.0, allow_inf_nan=False)

    def to_feature_row(self) -> dict[str, float]:
        """Convert validated fields into an ordered feature dictionary.

        Returns:
            Dictionary mapping each feature name in `FEATURE_ORDER` to float.
        """
        return {name: float(getattr(self, name)) for name in FEATURE_ORDER}


class FeatureRange(BaseModel):
    """Training summary statistics for a single numeric feature.

    Attributes:
        min: Minimum observed value in the training split.
        max: Maximum observed value in the training split.
        mean: Arithmetic mean in the training split.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    min: float
    max: float
    mean: float


class ModelMetrics(BaseModel):
    """Cross-validation and held-out test metrics exported by training.

    Attributes:
        cv_mean: Mean cross-validation F1 macro score.
        cv_std: Standard deviation of cross-validation F1 macro scores.
        test_accuracy: Held-out test accuracy score.
        test_f1_macro: Held-out test F1 macro score.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    cv_mean: float
    cv_std: float
    test_accuracy: float
    test_f1_macro: float


class ModelMetadata(BaseModel):
    """Validated metadata contract exported alongside `model_pipeline.joblib`.

    Attributes:
        model_name: Name of the selected estimator class.
        artifact_version: Semantic version string of the exported artifact.
        trained_at: ISO-8601 UTC timestamp of model training.
        features: Ordered list of input feature names expected by the model.
        classes: Supported crop class labels.
        metrics: Verified training and evaluation metrics.
        feature_ranges: Per-feature summary statistics from training data.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    model_name: str
    artifact_version: str
    trained_at: str
    features: list[str]
    classes: list[str]
    metrics: ModelMetrics
    feature_ranges: dict[str, FeatureRange]


class PredictionItem(BaseModel):
    """Single candidate crop paired with its predicted posterior probability.

    Attributes:
        crop: Predicted crop label.
        probability: Posterior probability in the closed interval [0.0, 1.0].
    """

    model_config = ConfigDict(frozen=True)

    crop: str
    probability: float = Field(..., ge=0.0, le=1.0)


class PredictionResult(BaseModel):
    """Structured single-sample inference result with latency telemetry.

    Attributes:
        recommended_crop: Top-ranked crop label.
        confidence: Predicted probability of the recommended crop.
        top_predictions: Top-K candidate crops sorted by descending probability.
        out_of_range_features: Features outside observed training min/max bounds.
        latency_ms: Inference execution duration in milliseconds.
        status: Result status indicator, fixed to `"ok"`.
    """

    model_config = ConfigDict(frozen=True)

    recommended_crop: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_predictions: list[PredictionItem]
    out_of_range_features: list[str] = Field(default_factory=list)
    latency_ms: float = Field(..., ge=0.0)
    status: Literal["ok"] = "ok"
