"""Render the single-sample crop recommendation workbench view.

This module exposes `render_single_prediction` to collect the seven soil and
climate parameters side by side with ranked probability charts, latency metrics,
and training-distribution comparisons.
"""

import altair as alt
import pandas as pd
from pydantic import ValidationError
import streamlit as st

from src.config import AppConfig
from src.schemas.prediction import DOMAIN_BOUNDS, CropInput, FeatureRange, ModelMetadata
from src.services.predictor import CropPredictor


_SOIL_FIELDS: tuple[tuple[str, str, float], ...] = (
    ("N", "Nitrogen (N)", 1.0),
    ("P", "Phosphorus (P)", 1.0),
    ("K", "Potassium (K)", 1.0),
    ("ph", "Soil pH", 0.1),
)

_CLIMATE_FIELDS: tuple[tuple[str, str, float], ...] = (
    ("temperature", "Temperature (°C)", 0.5),
    ("humidity", "Humidity (%)", 0.5),
    ("rainfall", "Rainfall (mm)", 1.0),
)


def _default_feature_value(
    field_name: str,
    metadata_ranges: dict[str, FeatureRange],
) -> float:
    """Compute a domain-clamped default input value for a feature widget.

    Args:
        field_name: Canonical feature name from `FEATURE_ORDER`.
        metadata_ranges: Training feature statistics keyed by feature name.

    Returns:
        Default float rounded to 2 decimal places within `DOMAIN_BOUNDS`.
    """
    low, high = DOMAIN_BOUNDS[field_name]
    raw_default = (
        float(metadata_ranges[field_name].mean)
        if field_name in metadata_ranges
        else (low + high) / 2.0
    )
    return round(min(max(raw_default, float(low)), float(high)), 2)


def _render_field_pair(
    fields: tuple[tuple[str, str, float], ...],
    metadata_ranges: dict[str, FeatureRange],
    inputs: dict[str, float],
) -> None:
    """Render a group of numeric input widgets in a two-column grid.

    Args:
        fields: Sequence of `(field_name, label, step)` tuples to render.
        metadata_ranges: Training feature statistics for default values.
        inputs: Mutable mapping populated with widget values keyed by feature name.
    """
    cols = st.columns(2)
    for idx, (field_name, label, step) in enumerate(fields):
        low, high = DOMAIN_BOUNDS[field_name]
        inputs[field_name] = cols[idx % 2].number_input(
            label,
            min_value=float(low),
            max_value=float(high),
            value=_default_feature_value(field_name, metadata_ranges),
            step=step,
            key=f"single_input_{field_name}",
        )


def _build_probability_chart(top_df: pd.DataFrame) -> alt.Chart:
    """Build a horizontal Altair bar chart of candidate crop probabilities.

    Args:
        top_df: DataFrame containing `Crop` and `ProbabilityPct` columns.

    Returns:
        Configured Altair bar chart scaled from 0% to 100%.
    """
    return (
        alt.Chart(top_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "ProbabilityPct:Q",
                title="Predicted probability (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y("Crop:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("Crop:N", title="Crop"),
                alt.Tooltip("ProbabilityPct:Q", title="Probability (%)", format=".2f"),
            ],
        )
        .properties(height=max(140, len(top_df) * 38))
    )


def _render_empty_decision_state(metadata: ModelMetadata) -> None:
    """Render guidance and training parameter bounds before the first submission.

    Args:
        metadata: Active model metadata containing `feature_ranges`.
    """
    with st.container(border=True):
        st.subheader("Decision output", anchor=False)
        st.caption(
            "Adjust soil nutrients and climate parameters on the left, then select Predict crop "
            "to rank candidate crops by posterior probability."
        )
        if metadata.feature_ranges:
            bounds_rows = [
                {
                    "Parameter": label,
                    "Training min": round(metadata.feature_ranges[name].min, 2),
                    "Training mean": round(metadata.feature_ranges[name].mean, 2),
                    "Training max": round(metadata.feature_ranges[name].max, 2),
                }
                for name, label, _ in (*_SOIL_FIELDS, *_CLIMATE_FIELDS)
                if name in metadata.feature_ranges
            ]
            st.dataframe(
                pd.DataFrame(bounds_rows),
                width="stretch",
                hide_index=True,
            )


def _render_input_comparison(
    inputs: dict[str, float],
    metadata_ranges: dict[str, FeatureRange],
) -> None:
    """Render an expander comparing submitted values against training statistics.

    Args:
        inputs: Submitted feature values keyed by feature name.
        metadata_ranges: Training min, mean, and max values per feature.
    """
    if not metadata_ranges:
        return
    rows = [
        {
            "Parameter": label,
            "Your input": round(inputs[name], 2),
            "Training mean": round(metadata_ranges[name].mean, 2),
            "Training range": f"{metadata_ranges[name].min:.1f} - {metadata_ranges[name].max:.1f}",
        }
        for name, label, _ in (*_SOIL_FIELDS, *_CLIMATE_FIELDS)
        if name in metadata_ranges
    ]
    with st.expander("Input vs. training distribution"):
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
        )


def render_single_prediction(
    predictor: CropPredictor,
    metadata: ModelMetadata,
    config: AppConfig,
) -> None:
    """Render the side-by-side input workbench and ranked prediction output.

    Args:
        predictor: Stateless inference service for single-sample scoring.
        metadata: Active model metadata containing classes and feature ranges.
        config: Application configuration specifying `default_top_k` and `max_top_k`.
    """
    metadata_ranges = metadata.feature_ranges
    max_k = max(1, min(config.max_top_k, len(metadata.classes)))
    default_k = min(config.default_top_k, max_k)

    input_col, output_col = st.columns([1.0, 1.25], gap="large")

    with input_col:
        with st.form("single_crop_form"):
            st.markdown("**Soil nutrients & pH**")
            inputs: dict[str, float] = {}
            _render_field_pair(_SOIL_FIELDS, metadata_ranges, inputs)

            st.markdown("**Climate conditions**")
            _render_field_pair(_CLIMATE_FIELDS, metadata_ranges, inputs)

            top_k = st.slider(
                "Candidate crops to rank",
                min_value=1,
                max_value=max_k,
                value=default_k,
            )
            submitted = st.form_submit_button(
                "Predict crop",
                type="primary",
                icon=":material/eco:",
                width="stretch",
            )

    with output_col:
        if not submitted:
            _render_empty_decision_state(metadata)
            return

        try:
            payload = CropInput(**inputs)
        except ValidationError as exc:
            st.error(
                f"Input validation failed: {exc}",
                title="Invalid input",
                icon=":material/error:",
            )
            return

        with st.spinner("Running inference..."):
            result = predictor.predict_single(payload, top_k=top_k)

        if result.out_of_range_features:
            flagged = ", ".join(result.out_of_range_features)
            st.warning(
                f"Features outside training bounds ({flagged}) may lower prediction reliability.",
                title="Out-of-range input",
                icon=":material/warning:",
            )

        with st.container(border=True):
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Recommended crop", result.recommended_crop.capitalize())
            res_col2.metric("Confidence", f"{result.confidence:.2%}")
            res_col3.metric("Latency", f"{result.latency_ms:.2f} ms")

        top_df = pd.DataFrame(
            [
                {
                    "Crop": item.crop.capitalize(),
                    "Probability": round(item.probability, 4),
                    "ProbabilityPct": round(item.probability * 100.0, 2),
                }
                for item in result.top_predictions
            ]
        )

        with st.container(border=True):
            st.markdown("**Candidate probability ranking**")
            st.altair_chart(
                _build_probability_chart(top_df),
                width="stretch",
            )
            st.dataframe(
                top_df[["Crop", "Probability"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "Crop": st.column_config.TextColumn("Crop"),
                    "Probability": st.column_config.ProgressColumn(
                        "Probability",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.4f",
                    ),
                },
            )

        _render_input_comparison(inputs, metadata_ranges)
