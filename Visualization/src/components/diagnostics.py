"""Render model evaluation metrics and saved EDA/evaluation figure galleries.

This module exposes `render_diagnostics` to display cross-validation and test
metrics alongside a two-column gallery switched by plot category.
"""

from pathlib import Path

import streamlit as st

from src.services.artifact_loader import ArtifactBundle


_PLOT_GROUPS: tuple[str, ...] = (
    "Model evaluation",
    "Exploratory analysis",
)


def _render_plot_grid(group_name: str, plots: tuple[Path, ...]) -> None:
    """Render saved PNG figures in a two-column card grid.

    Args:
        group_name: Display name of the plot category for empty-state messaging.
        plots: Tuple of verified PNG file paths to render.
    """
    if not plots:
        st.info(
            f"No plots found for {group_name.lower()}. Run the training notebooks to generate figures.",
            title="No plots available",
            icon=":material/info:",
        )
        return

    for idx in range(0, len(plots), 2):
        cols = st.columns(2, gap="medium")
        for col, plot_path in zip(cols, plots[idx : idx + 2]):
            label = plot_path.stem.replace("_", " ").capitalize()
            with col:
                with st.container(border=True):
                    st.markdown(f"**{label}**")
                    st.image(
                        str(plot_path),
                        width="stretch",
                    )


def render_diagnostics(bundle: ArtifactBundle) -> None:
    """Render verified training metrics and saved EDA or evaluation plots.

    Args:
        bundle: Validated artifact bundle containing metrics and plot paths.
    """
    metrics = bundle.metadata.metrics

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CV F1 macro (mean)", f"{metrics.cv_mean:.4f}", border=True)
    col2.metric("CV F1 macro (std)", f"{metrics.cv_std:.4f}", border=True)
    col3.metric("Test accuracy", f"{metrics.test_accuracy:.4f}", border=True)
    col4.metric("Test F1 macro", f"{metrics.test_f1_macro:.4f}", border=True)

    selected_group = st.segmented_control(
        "Diagnostic figure group",
        options=list(_PLOT_GROUPS),
        default=_PLOT_GROUPS[0],
        label_visibility="collapsed",
        key="diagnostics_group",
    )

    active_group = selected_group or _PLOT_GROUPS[0]
    if active_group == "Model evaluation":
        _render_plot_grid(active_group, bundle.eval_plots)
    else:
        _render_plot_grid(active_group, bundle.eda_plots)
