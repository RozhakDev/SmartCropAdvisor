"""Provide single-sample, batch inference, and CSV sanitization services.

This module exposes `CropPredictor` and `BatchValidationError` to execute
validated predictions against a loaded `ArtifactBundle` and neutralize formula
injection prefixes on exported CSV bytes.
"""

import time

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.schemas.prediction import (
    DOMAIN_BOUNDS,
    FEATURE_ORDER,
    CropInput,
    PredictionItem,
    PredictionResult,
)
from src.services.artifact_loader import ArtifactBundle


class BatchValidationError(ValueError):
    """Raised when an uploaded CSV fails schema, row-limit, or value checks."""


_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


class CropPredictor:
    """Stateless inference service backed by a validated `ArtifactBundle`."""

    def __init__(self, bundle: ArtifactBundle, config: AppConfig | None = None) -> None:
        """Initialize the predictor with a validated artifact bundle.

        Args:
            bundle: Loaded pipeline and metadata bundle.
            config: Optional application configuration; defaults to `AppConfig.from_env()`.
        """
        self._bundle = bundle
        self._config = config or AppConfig.from_env()
        self._features = list(bundle.metadata.features)

    def detect_out_of_range(self, row: dict[str, float]) -> list[str]:
        """Identify input features falling outside training min/max bounds.

        Args:
            row: Mapping of feature names to numeric input values.

        Returns:
            List of feature names whose values fall below the training minimum
            or exceed the training maximum.
        """
        flagged: list[str] = []
        ranges = self._bundle.metadata.feature_ranges
        for name, value in row.items():
            bounds = ranges.get(name)
            if bounds is not None and (value < bounds.min or value > bounds.max):
                flagged.append(name)
        return flagged

    def predict_single(self, payload: CropInput, top_k: int = 3) -> PredictionResult:
        """Run single-row inference and return top-K candidates with latency.

        Args:
            payload: Validated soil and climate input parameters.
            top_k: Number of highest-probability candidate crops to return.

        Returns:
            Structured `PredictionResult` with recommended crop, confidence,
            ranked candidates, out-of-range warnings, and latency in ms.

        Raises:
            ValueError: If `top_k` is less than 1.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        row = payload.to_feature_row()
        frame = pd.DataFrame([row], columns=self._features)

        start = time.perf_counter()
        probs = self._bundle.pipeline.predict_proba(frame)[0]
        classes = list(self._bundle.pipeline.classes_)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        ranked_indices = np.argsort(probs)[::-1][:top_k]
        top_items = [
            PredictionItem(crop=str(classes[idx]), probability=float(probs[idx]))
            for idx in ranked_indices
        ]
        winner = top_items[0]

        return PredictionResult(
            recommended_crop=winner.crop,
            confidence=winner.probability,
            top_predictions=top_items,
            out_of_range_features=self.detect_out_of_range(row),
            latency_ms=round(elapsed_ms, 3),
        )

    def validate_batch_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate uploaded DataFrame columns, row cap, finiteness, and bounds.

        Args:
            df: Raw DataFrame parsed from an uploaded CSV file.

        Returns:
            Cleaned copy of `df` restricted to `FEATURE_ORDER` cast to float.

        Raises:
            BatchValidationError: If the frame is empty, exceeds `max_batch_rows`,
                lacks required columns, or contains non-numeric, non-finite, or
                out-of-domain values.
        """
        if df.empty:
            raise BatchValidationError("CSV file contains zero rows.")
        if len(df) > self._config.max_batch_rows:
            raise BatchValidationError(
                f"Batch exceeds maximum allowed rows ({self._config.max_batch_rows})."
            )

        missing = [col for col in FEATURE_ORDER if col not in df.columns]
        if missing:
            raise BatchValidationError(
                f"Missing required columns: {', '.join(missing)}"
            )

        validated = df[list(FEATURE_ORDER)].copy()
        for col in FEATURE_ORDER:
            coerced = pd.to_numeric(validated[col], errors="coerce")
            if coerced.isna().any():
                raise BatchValidationError(
                    f"Column '{col}' contains non-numeric or missing values."
                )
            if not np.isfinite(coerced.to_numpy()).all():
                raise BatchValidationError(
                    f"Column '{col}' contains infinite values."
                )
            low, high = DOMAIN_BOUNDS[col]
            if ((coerced < low) | (coerced > high)).any():
                raise BatchValidationError(
                    f"Column '{col}' has values outside allowed domain [{low}, {high}]."
                )
            validated[col] = coerced.astype(float)

        return validated

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and score a batch DataFrame of crop samples.

        Args:
            df: Raw DataFrame containing columns matching `FEATURE_ORDER`.

        Returns:
            Validated feature DataFrame augmented with `recommended_crop` and
            `confidence` (rounded to 4 decimal places).

        Raises:
            BatchValidationError: If batch validation fails.
        """
        clean_features = self.validate_batch_frame(df)
        feature_frame = clean_features[self._features]

        preds = self._bundle.pipeline.predict(feature_frame)
        probs = self._bundle.pipeline.predict_proba(feature_frame).max(axis=1)

        output = clean_features.copy()
        output["recommended_crop"] = [str(label) for label in preds]
        output["confidence"] = np.round(probs.astype(float), 4)
        return output

    @staticmethod
    def sanitize_csv_export(df: pd.DataFrame) -> bytes:
        """Neutralize spreadsheet formula-injection prefixes before CSV export.

        Prefixes string cells starting with `=`, `+`, `-`, `@`, `\\t`, or `\\r`
        with a leading single quote.

        Args:
            df: Prediction DataFrame to serialize.

        Returns:
            UTF-8 encoded CSV payload safe for spreadsheet download.
        """
        safe_df = df.copy()
        for col in safe_df.select_dtypes(include=["object", "string"]).columns:
            safe_df[col] = safe_df[col].astype(str).map(
                lambda val: f"'{val}" if val.startswith(_FORMULA_PREFIXES) else val
            )
        return safe_df.to_csv(index=False).encode("utf-8")
