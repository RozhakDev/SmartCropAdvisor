"""Verify pure UI helper functions for CSV templates and widget defaults.

This module tests `_build_template_csv` and `_default_feature_value` outside
a live Streamlit runtime.
"""

import io

import pandas as pd

from src.components.batch_upload import _build_template_csv
from src.components.single_form import _default_feature_value
from src.schemas.prediction import DOMAIN_BOUNDS, FEATURE_ORDER, FeatureRange


class TestBatchHelpers:
    """Test suite for batch CSV template generation."""

    def test_build_template_csv_has_required_columns_and_bounds(self) -> None:
        """Verify template CSV contains `FEATURE_ORDER` columns within bounds."""
        ranges = {"N": FeatureRange(min=0.0, max=140.0, mean=50.5)}
        csv_bytes = _build_template_csv(ranges)
        df = pd.read_csv(io.BytesIO(csv_bytes))

        assert list(df.columns) == list(FEATURE_ORDER)
        assert len(df) == 1
        assert df.loc[0, "N"] == 50.5
        for col in FEATURE_ORDER:
            low, high = DOMAIN_BOUNDS[col]
            assert low <= float(df.loc[0, col]) <= high


class TestSingleFormHelpers:
    """Test suite for single-sample widget default calculations."""

    def test_default_feature_value_clamps_out_of_bound_mean(self) -> None:
        """Verify out-of-domain metadata means clamp to `DOMAIN_BOUNDS`."""
        ranges = {"N": FeatureRange(min=0.0, max=1000.0, mean=999.0)}
        assert _default_feature_value("N", ranges) == DOMAIN_BOUNDS["N"][1]

    def test_default_feature_value_falls_back_to_midpoint(self) -> None:
        """Verify missing feature statistics fall back to domain midpoints."""
        low, high = DOMAIN_BOUNDS["ph"]
        expected = round((low + high) / 2.0, 2)
        assert _default_feature_value("ph", {}) == expected
