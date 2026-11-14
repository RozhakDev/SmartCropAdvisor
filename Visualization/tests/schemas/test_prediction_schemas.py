"""Verify Pydantic input, metadata, and prediction schema contracts.

This module tests `CropInput` domain bounds, non-finite value rejection,
extra-field prohibition, and metadata/prediction model validation.
"""

import math

from pydantic import ValidationError
import pytest

from src.schemas.prediction import (
    DOMAIN_BOUNDS,
    FEATURE_ORDER,
    CropInput,
    FeatureRange,
    ModelMetadata,
    ModelMetrics,
    PredictionItem,
    PredictionResult,
)


class TestCropInput:
    """Test suite for `CropInput` validation and ordering contracts."""

    def test_valid_payload_preserves_feature_order(self, sample_crop_input: CropInput) -> None:
        """Verify `to_feature_row` preserves canonical `FEATURE_ORDER`.

        Args:
            sample_crop_input: Valid fixture payload.
        """
        row = sample_crop_input.to_feature_row()
        assert list(row.keys()) == list(FEATURE_ORDER)
        assert row["ph"] == 6.5

    def test_accepts_exact_domain_boundaries(self) -> None:
        """Verify exact minimum and maximum `DOMAIN_BOUNDS` values pass."""
        min_payload = CropInput(**{name: DOMAIN_BOUNDS[name][0] for name in FEATURE_ORDER})
        max_payload = CropInput(**{name: DOMAIN_BOUNDS[name][1] for name in FEATURE_ORDER})
        assert min_payload.N == DOMAIN_BOUNDS["N"][0]
        assert max_payload.rainfall == DOMAIN_BOUNDS["rainfall"][1]

    @pytest.mark.parametrize(
        ("field_name", "bad_value"),
        [
            ("N", -0.1),
            ("N", 500.1),
            ("P", -1.0),
            ("K", 501.0),
            ("temperature", -25.0),
            ("temperature", 65.0),
            ("humidity", -1.0),
            ("humidity", 101.0),
            ("ph", -0.1),
            ("ph", 14.1),
            ("rainfall", -1.0),
            ("rainfall", 5001.0),
        ],
    )
    def test_rejects_out_of_domain_bounds(
        self,
        sample_crop_input: CropInput,
        field_name: str,
        bad_value: float,
    ) -> None:
        """Verify values outside `DOMAIN_BOUNDS` raise ValidationError.

        Args:
            sample_crop_input: Valid baseline payload fixture.
            field_name: Name of the field under test.
            bad_value: Out-of-bounds value expected to fail validation.
        """
        data = sample_crop_input.model_dump()
        data[field_name] = bad_value
        with pytest.raises(ValidationError):
            CropInput(**data)

    @pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_values(
        self,
        sample_crop_input: CropInput,
        bad_value: float,
    ) -> None:
        """Verify NaN and infinite values raise ValidationError.

        Args:
            sample_crop_input: Valid baseline payload fixture.
            bad_value: Non-finite float value expected to fail validation.
        """
        data = sample_crop_input.model_dump()
        data["N"] = bad_value
        with pytest.raises(ValidationError):
            CropInput(**data)

    def test_forbids_extra_fields(self, sample_crop_input: CropInput) -> None:
        """Verify unexpected extra keys raise ValidationError.

        Args:
            sample_crop_input: Valid baseline payload fixture.
        """
        data = sample_crop_input.model_dump()
        data["unexpected"] = "bad"
        with pytest.raises(ValidationError):
            CropInput(**data)


class TestPredictionSchemas:
    """Test suite for metadata and prediction output schemas."""

    def test_model_metadata_and_prediction_result_round_trip(self) -> None:
        """Verify `ModelMetadata` and `PredictionResult` accept valid payloads."""
        meta = ModelMetadata(
            model_name="GaussianNB",
            artifact_version="1.0.0",
            trained_at="2026-11-13T18:06:18+00:00",
            features=list(FEATURE_ORDER),
            classes=["rice", "jute"],
            metrics=ModelMetrics(
                cv_mean=0.99,
                cv_std=0.01,
                test_accuracy=0.995,
                test_f1_macro=0.995,
            ),
            feature_ranges={"N": FeatureRange(min=0.0, max=140.0, mean=50.5)},
        )
        assert meta.model_name == "GaussianNB"
        assert meta.feature_ranges["N"].mean == 50.5

        result = PredictionResult(
            recommended_crop="rice",
            confidence=0.98,
            top_predictions=[PredictionItem(crop="rice", probability=0.98)],
            out_of_range_features=[],
            latency_ms=1.25,
        )
        assert result.recommended_crop == "rice"
        assert result.top_predictions[0].probability == 0.98

    def test_prediction_item_rejects_probability_outside_unit_interval(self) -> None:
        """Verify `PredictionItem` rejects probabilities above 1.0."""
        with pytest.raises(ValidationError):
            PredictionItem(crop="rice", probability=1.2)
