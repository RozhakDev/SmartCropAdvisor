"""Render the sidebar provenance panel for the active model artifact.

This module exposes `render_sidebar` to display the trained estimator name,
artifact version, training date, verified evaluation metrics, and supported
crop classes.
"""

import streamlit as st

from src.schemas.prediction import ModelMetadata


def render_sidebar(metadata: ModelMetadata) -> None:
    """Render model provenance and verified evaluation metrics in the sidebar.

    Args:
        metadata: Validated metadata contract for the active model artifact.
    """
    with st.sidebar:
        st.subheader("Model Provenance", anchor=False)
        st.badge(metadata.model_name, icon=":material/memory:", color="green")
        st.caption(f"Artifact v{metadata.artifact_version} · Trained {metadata.trained_at[:10]} UTC")

        col1, col2 = st.columns(2)
        col1.metric("CV F1 macro", f"{metadata.metrics.cv_mean:.4f}", border=True)
        col2.metric("Test accuracy", f"{metadata.metrics.test_accuracy:.4f}", border=True)

        with st.expander(f"Supported crops ({len(metadata.classes)})"):
            st.caption(", ".join(crop.capitalize() for crop in metadata.classes))
