"""Verify `CropPredictor` single inference, batch validation, and CSV export.

This module tests top-K ranking, out-of-range detection, DataFrame schema and
row-limit enforcement, and formula-injection sanitization.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from src.config import AppConfig
from src.schemas.prediction import FEATURE_ORDER, CropInput
from src.services.artifact_loader import ArtifactBundle
from src.services.predictor import BatchValidationError, CropPredictor


class TestSinglePrediction:
    """Test suite for single-sample inference contracts and range warnings."""

    def test_returns_valid_contract_and_latency(
        self,
        loaded_predictor: CropPredictor,
        sample_crop_input: CropInput,
    ) -> None:
        """Verify `predict_single` returns expected crop, top-K count, and latency.

        Args:
            loaded_predictor: Session-scoped predictor fixture.
            sample_crop_input: Valid rice-profile input fixture.
        """
        result = loaded_predictor.predict_single(sample_crop_input, top_k=3)
        assert result.recommended_crop == "rice"
        assert len(result.top_predictions) == 3
        assert 0.0 < result.confidence <= 1.0
        assert result.latency_ms < 500.0
        assert result.out_of_range_features == []

    def test_rejects_non_positive_top_k(
        self,
        loaded_predictor: CropPredictor,
        sample_crop_input: CropInput,
    ) -> None:
        """Verify `top_k < 1` raises `ValueError`.

        Args:
            loaded_predictor: Session-scoped predictor fixture.
            sample_crop_input: Valid input fixture.
        """
        with pytest.raises(ValueError):
            loaded_predictor.predict_single(sample_crop_input, top_k=0)

    def test_clamps_top_k_exceeding_class_count(
        self,
        loaded_predictor: CropPredictor,
        loaded_bundle: ArtifactBundle,
        sample_crop_input: CropInput,
    ) -> None:
        """Verify `top_k` exceeding class count returns all available classes.

        Args:
            loaded_predictor: Session-scoped predictor fixture.
            loaded_bundle: Session-scoped artifact bundle fixture.
            sample_crop_input: Valid input fixture.
        """
        result = loaded_predictor.predict_single(sample_crop_input, top_k=999)
        assert len(result.top_predictions) == len(loaded_bundle.metadata.classes)

    def test_flags_features_above_and_below_training_range(
        self,
        loaded_predictor: CropPredictor,
        sample_crop_input: CropInput,
    ) -> None:
        """Verify features outside training min/max are reported in warnings.

        Args:
            loaded_predictor: Session-scoped predictor fixture.
            sample_crop_input: Valid baseline input fixture.
        """
        data = sample_crop_input.model_dump()
        data["N"] = 450.0
        data["temperature"] = -10.0
        result = loaded_predictor.predict_single(CropInput(**data), top_k=2)
        assert "N" in result.out_of_range_features
        assert "temperature" in result.out_of_range_features


class TestBatchValidation:
    """Test suite for batch DataFrame schema validation, limits, and scoring."""

    def test_scores_valid_batch_frame(
        self,
        loaded_predictor: CropPredictor,
        sample_batch_df: pd.DataFrame,
    ) -> None:
        """Verify valid batch input returns expected labels and high confidence.

        Args:
            loaded_predictor: Session-scoped predictor fixture.
            sample_batch_df: Valid two-row batch DataFrame fixture.
        """
        scored = loaded_predictor.predict_batch(sample_batch_df)
        assert list(scored["recommended_crop"]) == ["rice", "apple"]
        assert (scored["confidence"] > 0.5).all()

    def test_rejects_missing_columns(self, loaded_predictor: CropPredictor) -> None:
        """Verify DataFrames missing required features raise `BatchValidationError`.

        Args:
            loaded_predictor: Session-scoped predictor fixture.
        """
        bad_df = pd.DataFrame([{"N": 90.0, "P": 42.0}])
        with pytest.raises(BatchValidationError):
            loaded_predictor.predict_batch(bad_df)

    def test_rejects_empty_frame(self, loaded_predictor: CropPredictor) -> None:
        """Verify zero-row DataFrames raise `BatchValidationError`.

        Args:
            loaded_predictor: Session-scoped predictor fixture.
        """
        empty_df = pd.DataFrame(columns=list(FEATURE_ORDER))
        with pytest.raises(BatchValidationError):
            loaded_predictor.predict_batch(empty_df)

    @pytest.mark.parametrize(
        "bad_row",
        [
            ["abc", 42.0, 43.0, 20.88, 82.0, 6.5, 202.94],
            [np.nan, 42.0, 43.0, 20.88, 82.0, 6.5, 202.94],
            [np.inf, 42.0, 43.0, 20.88, 82.0, 6.5, 202.94],
            [90.0, 42.0, 43.0, 20.88, 82.0, 20.0, 202.94],
            [-1.0, 42.0, 43.0, 20.88, 82.0, 6.5, 202.94],
        ],
    )
    def test_rejects_invalid_cell_values(
        self,
        loaded_predictor: CropPredictor,
        bad_row: list[object],
    ) -> None:
        """Verify non-numeric, NaN, infinite, and out-of-domain cells fail.

        Args:
            loaded_predictor: Session-scoped predictor fixture.
            bad_row: Single invalid row matching `FEATURE_ORDER`.
        """
        bad_df = pd.DataFrame([bad_row], columns=list(FEATURE_ORDER))
        with pytest.raises(BatchValidationError):
            loaded_predictor.predict_batch(bad_df)

    def test_enforces_max_batch_rows(
        self,
        loaded_bundle: ArtifactBundle,
        app_config: AppConfig,
        sample_batch_df: pd.DataFrame,
        build_temp_config: Callable[..., AppConfig],
    ) -> None:
        """Verify batches exceeding `max_batch_rows` raise `BatchValidationError`.

        Args:
            loaded_bundle: Session-scoped artifact bundle fixture.
            app_config: Session-scoped configuration fixture.
            sample_batch_df: Two-row batch DataFrame fixture.
            build_temp_config: Factory creating capped `AppConfig`.
        """
        capped_cfg = build_temp_config(app_config.artifacts_dir, max_batch_rows=1)
        predictor = CropPredictor(bundle=loaded_bundle, config=capped_cfg)
        with pytest.raises(BatchValidationError):
            predictor.predict_batch(sample_batch_df)


class TestCsvSanitization:
    """Test suite for spreadsheet formula-injection neutralization."""

    @pytest.mark.parametrize("payload", ["=CMD()", "+1+1", "-2+3", "@SUM(A1)", "\tcmd", "\rcmd"])
    def test_prefixes_dangerous_formula_characters(self, payload: str) -> None:
        """Verify dangerous leading characters are escaped with a single quote.

        Args:
            payload: Potentially unsafe spreadsheet formula string.
        """
        df = pd.DataFrame([{"recommended_crop": payload}])
        exported = CropPredictor.sanitize_csv_export(df).decode("utf-8")
        assert f"'{payload}" in exported

    def test_leaves_safe_text_untouched(self) -> None:
        """Verify ordinary crop labels are exported without leading quotes."""
        df = pd.DataFrame([{"recommended_crop": "rice"}])
        exported = CropPredictor.sanitize_csv_export(df).decode("utf-8")
        assert "rice" in exported
        assert "'rice" not in exported
