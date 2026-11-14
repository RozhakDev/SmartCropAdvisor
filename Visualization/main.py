"""Provide the Streamlit entrypoint for the SmartCropAdvisor workbench.

This module configures page metadata, loads and caches trained model artifacts,
handles startup configuration errors, and routes navigation across the single
sample, batch CSV, and model diagnostics views.
"""

import streamlit as st

from src.components import (
    render_batch_upload,
    render_diagnostics,
    render_sidebar,
    render_single_prediction,
)
from src.config import AppConfig
from src.logger import get_logger
from src.services import (
    ArtifactBundle,
    ArtifactLoadError,
    ArtifactLoader,
    ArtifactNotFoundError,
    CropPredictor,
)


logger = get_logger(__name__)

_VIEW_OPTIONS: tuple[str, ...] = (
    "Single sample",
    "Batch CSV",
    "Model & EDA",
)

_VIEW_LABELS: dict[str, str] = {
    "Single sample": ":material/eco: Single sample",
    "Batch CSV": ":material/table_view: Batch CSV",
    "Model & EDA": ":material/monitoring: Model & EDA",
}

st.set_page_config(
    page_title="SmartCropAdvisor",
    page_icon=":material/eco:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_cached_bundle(artifacts_dir_key: str) -> ArtifactBundle:
    """Load and cache the validated model artifact bundle.

    Resolves application configuration from the environment and loads the
    serialized pipeline, metadata, and diagnostic plots once per cache key.

    Args:
        artifacts_dir_key: Resolved artifacts directory path used as the cache
            key to invalidate cached resources when the directory changes.

    Returns:
        Validated artifact bundle containing the pipeline, metadata, and plots.
    """
    _ = artifacts_dir_key
    config = AppConfig.from_env()
    return ArtifactLoader(config).load()


def main() -> None:
    """Render the SmartCropAdvisor workbench and handle startup errors.

    Displays the application header, loads cached model artifacts, renders the
    sidebar provenance panel, and dispatches the active workbench view.
    """
    header_col, nav_col = st.columns([1.3, 1.0], vertical_alignment="bottom")
    with header_col:
        st.title("SmartCropAdvisor", anchor=False)
        st.caption("Soil nutrient and climate decision workbench for crop recommendation.")

    try:
        config = AppConfig.from_env()
        bundle = load_cached_bundle(str(config.artifacts_dir))
    except ValueError as exc:
        logger.error("Invalid application configuration: %s", exc)
        st.error(
            f"Invalid environment configuration: {exc}",
            title="Configuration error",
            icon=":material/error:",
        )
        return
    except (ArtifactNotFoundError, ArtifactLoadError) as exc:
        logger.error("Artifact loading failed: %s", exc)
        st.error(
            f"Model artifacts are not ready: {exc}. Run the MachineLearning notebooks first.",
            title="Artifacts unavailable",
            icon=":material/error:",
        )
        return

    with nav_col:
        with st.container(horizontal_alignment="right"):
            active_view = st.segmented_control(
                "Workbench view",
                options=list(_VIEW_OPTIONS),
                default=_VIEW_OPTIONS[0],
                format_func=lambda opt: _VIEW_LABELS[opt],
                label_visibility="collapsed",
                key="active_view",
            )

    render_sidebar(bundle.metadata)
    predictor = CropPredictor(bundle=bundle, config=config)

    selected_view = active_view or _VIEW_OPTIONS[0]
    if selected_view == "Single sample":
        render_single_prediction(predictor, bundle.metadata, config)
    elif selected_view == "Batch CSV":
        render_batch_upload(predictor, config, bundle.metadata.feature_ranges)
    else:
        render_diagnostics(bundle)


if __name__ == "__main__":
    main()
