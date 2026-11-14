"""Render the batch CSV upload, scoring, and export workbench view.

This module exposes `render_batch_upload` to provide a downloadable CSV schema
template, validate uploaded CSV files, score batches via `CropPredictor`, and
render distribution charts alongside sanitized CSV exports.
"""

import altair as alt
import pandas as pd
import streamlit as st

from src.config import AppConfig
from src.schemas.prediction import DOMAIN_BOUNDS, FEATURE_ORDER, FeatureRange
from src.services.predictor import BatchValidationError, CropPredictor


def _build_template_csv(feature_ranges: dict[str, FeatureRange]) -> bytes:
    """Build a single-row UTF-8 sample CSV template matching `FEATURE_ORDER`.

    Args:
        feature_ranges: Per-feature training statistics used to seed defaults.

    Returns:
        UTF-8 encoded CSV bytes with one representative sample row.
    """
    row: dict[str, float] = {}
    for col in FEATURE_ORDER:
        low, high = DOMAIN_BOUNDS[col]
        val = float(feature_ranges[col].mean) if col in feature_ranges else (low + high) / 2.0
        row[col] = round(min(max(val, float(low)), float(high)), 2)
    sample = pd.DataFrame([row], columns=list(FEATURE_ORDER))
    return sample.to_csv(index=False).encode("utf-8")


def _build_distribution_chart(summary_df: pd.DataFrame) -> alt.Chart:
    """Build a horizontal Altair bar chart of predicted crop counts.

    Args:
        summary_df: Aggregated DataFrame with `Crop` and `Count` columns.

    Returns:
        Configured Altair bar chart sorted by descending sample count.
    """
    return (
        alt.Chart(summary_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("Count:Q", title="Sample count"),
            y=alt.Y("Crop:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("Crop:N", title="Crop"),
                alt.Tooltip("Count:Q", title="Samples"),
            ],
        )
        .properties(height=max(160, len(summary_df) * 30))
    )


def render_batch_upload(
    predictor: CropPredictor,
    config: AppConfig,
    feature_ranges: dict[str, FeatureRange],
) -> None:
    """Render the CSV upload workflow with schema validation and sanitized export.

    Args:
        predictor: Stateless inference service for batch validation and scoring.
        config: Application configuration specifying `max_csv_bytes` limits.
        feature_ranges: Training feature statistics used to populate the CSV template.
    """
    with st.container(border=True):
        top_left, top_right = st.columns([1.8, 1.0], vertical_alignment="center")
        with top_left:
            st.markdown("**Batch CSV scoring**")
            st.caption("Required columns: " + ", ".join(f"`{col}`" for col in FEATURE_ORDER))
        with top_right:
            st.download_button(
                label="Download CSV template",
                data=_build_template_csv(feature_ranges),
                file_name="smartcrop_batch_template.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
            )

        uploaded = st.file_uploader(
            "Upload UTF-8 CSV file",
            type=["csv"],
            accept_multiple_files=False,
        )

    if uploaded is None:
        st.caption("Upload a CSV file or download the sample template above to score multiple samples.")
        return

    if uploaded.size == 0:
        st.error(
            "The uploaded CSV file is empty (0 bytes).",
            title="Empty file",
            icon=":material/error:",
        )
        return

    if uploaded.size > config.max_csv_bytes:
        max_mb = config.max_csv_bytes / (1024 * 1024)
        st.error(
            f"File size exceeds the {max_mb:.1f} MB limit.",
            title="File too large",
            icon=":material/error:",
        )
        return

    try:
        raw_df = pd.read_csv(uploaded)
    except pd.errors.EmptyDataError:
        st.error(
            "CSV file has no header or data rows.",
            title="Malformed CSV",
            icon=":material/error:",
        )
        return
    except UnicodeDecodeError:
        st.error(
            "Unsupported file encoding. Save the CSV as UTF-8 and retry.",
            title="Encoding error",
            icon=":material/error:",
        )
        return
    except Exception as exc:
        st.error(
            f"Could not parse CSV file: {exc}",
            title="Read error",
            icon=":material/error:",
        )
        return

    try:
        with st.spinner("Validating schema and scoring batch..."):
            scored_df = predictor.predict_batch(raw_df)
    except BatchValidationError as exc:
        st.error(
            str(exc),
            title="Batch validation failed",
            icon=":material/error:",
        )
        return

    summary_df = (
        scored_df["recommended_crop"]
        .str.capitalize()
        .value_counts()
        .rename_axis("Crop")
        .reset_index(name="Count")
    )
    top_crop = str(summary_df.iloc[0]["Crop"]) if not summary_df.empty else "-"
    mean_conf = float(scored_df["confidence"].mean()) if "confidence" in scored_df else 0.0

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Rows scored", f"{len(scored_df):,}", border=True)
    kpi_cols[1].metric("Unique crops", f"{len(summary_df)}", border=True)
    kpi_cols[2].metric("Most frequent", top_crop, border=True)
    kpi_cols[3].metric("Mean confidence", f"{mean_conf:.2%}", border=True)

    chart_col, table_col = st.columns([1.0, 1.5], gap="large")
    with chart_col:
        with st.container(border=True):
            st.markdown("**Predicted crop distribution**")
            st.altair_chart(
                _build_distribution_chart(summary_df),
                width="stretch",
            )

    with table_col:
        with st.container(border=True):
            header_col, dl_col = st.columns([1.2, 1.0], vertical_alignment="center")
            header_col.markdown("**Scored predictions**")
            with dl_col:
                st.download_button(
                    label="Export results CSV",
                    data=predictor.sanitize_csv_export(scored_df),
                    file_name="smartcrop_predictions.csv",
                    mime="text/csv",
                    type="primary",
                    icon=":material/download:",
                    width="stretch",
                )
            st.dataframe(
                scored_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.4f",
                    ),
                },
            )
