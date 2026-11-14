"""Configure pytest import paths and shared fixtures for Visualization tests.

This module ensures `Visualization` is on `sys.path` and provides reusable
session-scoped bundles, predictors, sample payloads, and isolated temporary
artifact factories.
"""

from pathlib import Path
import shutil
import sys
from typing import Callable

import pandas as pd
import pytest


VISUALIZATION_ROOT = Path(__file__).resolve().parent.parent
if str(VISUALIZATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VISUALIZATION_ROOT))

from src.config import AppConfig  # noqa: E402
from src.schemas.prediction import FEATURE_ORDER, CropInput  # noqa: E402
from src.services.artifact_loader import ArtifactBundle, ArtifactLoader  # noqa: E402
from src.services.predictor import CropPredictor  # noqa: E402


@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    """Resolve application configuration using repository artifacts.

    Returns:
        Session-scoped `AppConfig` instance pointing to repository artifacts.
    """
    return AppConfig.from_env(visualization_root=VISUALIZATION_ROOT)


@pytest.fixture(scope="session")
def loaded_bundle(app_config: AppConfig) -> ArtifactBundle:
    """Load validated model pipeline, metadata, and diagnostic plots.

    Args:
        app_config: Session-scoped application configuration.

    Returns:
        Validated `ArtifactBundle` loaded once per test session.
    """
    return ArtifactLoader(app_config).load()


@pytest.fixture(scope="session")
def loaded_predictor(loaded_bundle: ArtifactBundle, app_config: AppConfig) -> CropPredictor:
    """Create a `CropPredictor` bound to the session artifact bundle.

    Args:
        loaded_bundle: Validated session artifact bundle.
        app_config: Session-scoped application configuration.

    Returns:
        Configured `CropPredictor` instance.
    """
    return CropPredictor(bundle=loaded_bundle, config=app_config)


@pytest.fixture
def sample_crop_input() -> CropInput:
    """Provide a valid single-sample `CropInput` for rice conditions.

    Returns:
        Validated `CropInput` payload within training feature ranges.
    """
    return CropInput(
        N=90.0,
        P=42.0,
        K=43.0,
        temperature=20.88,
        humidity=82.0,
        ph=6.5,
        rainfall=202.94,
    )


@pytest.fixture
def sample_batch_df() -> pd.DataFrame:
    """Provide a valid two-row DataFrame matching `FEATURE_ORDER`.

    Returns:
        Two-row pandas DataFrame for batch prediction tests.
    """
    return pd.DataFrame(
        [
            [90.0, 42.0, 43.0, 20.88, 82.0, 6.5, 202.94],
            [20.0, 130.0, 200.0, 22.0, 90.0, 6.0, 110.0],
        ],
        columns=list(FEATURE_ORDER),
    )


@pytest.fixture
def build_temp_config() -> Callable[..., AppConfig]:
    """Provide a factory for isolated `AppConfig` instances.

    Returns:
        Callable accepting `base_dir` and optional integer overrides.
    """

    def _factory(base_dir: Path, **overrides: int) -> AppConfig:
        plots_root = base_dir / "plots"
        return AppConfig(
            artifacts_dir=base_dir,
            model_path=base_dir / "model_pipeline.joblib",
            metadata_path=base_dir / "metadata.json",
            eda_plots_dir=plots_root / "eda",
            eval_plots_dir=plots_root / "eval",
            **overrides,
        )

    return _factory


@pytest.fixture
def build_valid_artifact_dir(
    app_config: AppConfig,
    tmp_path: Path,
) -> Callable[[], Path]:
    """Provide a factory copying real artifacts into an isolated directory.

    Args:
        app_config: Session configuration locating source artifacts.
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Zero-argument callable returning the populated temporary directory path.
    """

    def _factory() -> Path:
        target = tmp_path / "artifacts"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(app_config.model_path, target / "model_pipeline.joblib")
        shutil.copy2(app_config.metadata_path, target / "metadata.json")
        return target

    return _factory
