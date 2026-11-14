"""Verify end-to-end Streamlit workbench behavior via headless `AppTest`.

This module tests initial rendering, view switching, single-sample inference,
out-of-range warning banners, and graceful error states for missing artifacts.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


_MAIN_SCRIPT = str(Path(__file__).resolve().parents[2] / "main.py")


class TestStreamlitWorkbench:
    """Test suite for Streamlit `main.py` workbench interactions."""

    @staticmethod
    def _launch() -> AppTest:
        """Instantiate and run `main.py` in a headless `AppTest` session.

        Returns:
            Executed `AppTest` instance after initial render.
        """
        app = AppTest.from_file(_MAIN_SCRIPT)
        app.run(timeout=20)
        return app

    def test_initial_render_and_view_navigation(self) -> None:
        """Verify all primary views and diagnostic sub-views render cleanly."""
        app = self._launch()
        assert not app.exception
        assert any("SmartCropAdvisor" in t.value for t in app.title)
        assert any("Decision output" in s.value for s in app.subheader)

        app.segmented_control(key="active_view").set_value("Batch CSV").run(timeout=20)
        assert not app.exception

        app.segmented_control(key="active_view").set_value("Model & EDA").run(timeout=20)
        assert not app.exception
        assert len(app.metric) >= 4

        app.segmented_control(key="diagnostics_group").set_value("Exploratory analysis").run(timeout=20)
        assert not app.exception

    def test_single_prediction_submit_displays_decision_metrics(self) -> None:
        """Verify submitting default inputs renders crop, confidence, and latency."""
        app = self._launch()
        assert not app.exception

        app.button[0].click().run(timeout=20)
        assert not app.exception

        metric_labels = [m.label for m in app.metric]
        assert "Recommended crop" in metric_labels
        assert "Confidence" in metric_labels
        assert "Latency" in metric_labels

    def test_single_prediction_warns_on_out_of_training_range_input(self) -> None:
        """Verify out-of-training-range inputs display a Streamlit warning."""
        app = self._launch()
        assert not app.exception

        app.number_input(key="single_input_N").set_value(450.0)
        app.button[0].click().run(timeout=20)
        assert not app.exception
        assert len(app.warning) >= 1

    def test_missing_artifacts_dir_renders_error_without_crashing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Verify missing artifacts directory renders `st.error` without raising.

        Args:
            monkeypatch: Pytest environment patch fixture.
            tmp_path: Temporary directory containing no artifacts.
        """
        monkeypatch.setenv("SMARTCROP_ARTIFACTS_DIR", str(tmp_path / "missing"))
        app = self._launch()
        assert not app.exception
        assert len(app.error) >= 1
