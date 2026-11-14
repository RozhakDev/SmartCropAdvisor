"""Expose Streamlit UI presentation components for SmartCropAdvisor.

Provides sidebar provenance, single-sample inference, batch CSV scoring,
and model diagnostic view renderers.
"""

from src.components.batch_upload import render_batch_upload
from src.components.diagnostics import render_diagnostics
from src.components.sidebar import render_sidebar
from src.components.single_form import render_single_prediction

__all__ = [
    "render_batch_upload",
    "render_diagnostics",
    "render_sidebar",
    "render_single_prediction",
]
