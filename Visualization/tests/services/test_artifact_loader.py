"""Verify `ArtifactLoader` contract checks, error handling, and PNG filtering.

This module tests successful repository bundle loading, missing artifact
exceptions, corrupted joblib/JSON detection, estimator capability checks, and
safe PNG plot discovery.
"""

import json
from pathlib import Path
from typing import Callable

import joblib
import pytest

from src.config import AppConfig
from src.services.artifact_loader import (
    ArtifactLoadError,
    ArtifactLoader,
    ArtifactNotFoundError,
)


class _DummyWithoutProba:
    """Minimal stub estimator omitting `predict_proba` for contract tests."""

    def predict(self, x: object) -> object:
        """Return the input unchanged.

        Args:
            x: Arbitrary input object.

        Returns:
            The unmodified input object.
        """
        return x


class TestArtifactLoader:
    """Test suite for `ArtifactLoader` validation and asset discovery."""

    def test_loads_valid_repository_bundle(self, app_config: AppConfig) -> None:
        """Verify repository artifacts and plot assets load without error.

        Args:
            app_config: Session-scoped application configuration fixture.
        """
        bundle = ArtifactLoader(app_config).load()
        assert bundle.metadata.model_name
        assert len(bundle.metadata.classes) >= 1
        assert len(bundle.eval_plots) >= 1
        assert len(bundle.eda_plots) >= 1

    def test_raises_when_model_missing(
        self,
        tmp_path: Path,
        build_temp_config: Callable[..., AppConfig],
    ) -> None:
        """Verify missing `model_pipeline.joblib` raises `ArtifactNotFoundError`.

        Args:
            tmp_path: Empty temporary directory without artifacts.
            build_temp_config: Factory creating `AppConfig` for `tmp_path`.
        """
        cfg = build_temp_config(tmp_path)
        with pytest.raises(ArtifactNotFoundError):
            ArtifactLoader(cfg).load()

    def test_raises_when_metadata_missing(
        self,
        build_valid_artifact_dir: Callable[[], Path],
        build_temp_config: Callable[..., AppConfig],
    ) -> None:
        """Verify missing `metadata.json` raises `ArtifactNotFoundError`.

        Args:
            build_valid_artifact_dir: Factory creating an isolated artifact copy.
            build_temp_config: Factory creating `AppConfig` for that directory.
        """
        artifact_dir = build_valid_artifact_dir()
        (artifact_dir / "metadata.json").unlink()
        cfg = build_temp_config(artifact_dir)
        with pytest.raises(ArtifactNotFoundError):
            ArtifactLoader(cfg).load()

    def test_raises_on_corrupted_joblib(
        self,
        build_valid_artifact_dir: Callable[[], Path],
        build_temp_config: Callable[..., AppConfig],
    ) -> None:
        """Verify unreadable pickle bytes raise `ArtifactLoadError`.

        Args:
            build_valid_artifact_dir: Factory creating an isolated artifact copy.
            build_temp_config: Factory creating `AppConfig` for that directory.
        """
        artifact_dir = build_valid_artifact_dir()
        (artifact_dir / "model_pipeline.joblib").write_bytes(b"not-a-valid-pickle")
        cfg = build_temp_config(artifact_dir)
        with pytest.raises(ArtifactLoadError):
            ArtifactLoader(cfg).load()

    def test_raises_when_model_lacks_predict_proba(
        self,
        build_valid_artifact_dir: Callable[[], Path],
        build_temp_config: Callable[..., AppConfig],
    ) -> None:
        """Verify estimators without `predict_proba` raise `ArtifactLoadError`.

        Args:
            build_valid_artifact_dir: Factory creating an isolated artifact copy.
            build_temp_config: Factory creating `AppConfig` for that directory.
        """
        artifact_dir = build_valid_artifact_dir()
        joblib.dump(_DummyWithoutProba(), artifact_dir / "model_pipeline.joblib")
        cfg = build_temp_config(artifact_dir)
        with pytest.raises(ArtifactLoadError):
            ArtifactLoader(cfg).load()

    def test_raises_on_corrupted_metadata_json(
        self,
        build_valid_artifact_dir: Callable[[], Path],
        build_temp_config: Callable[..., AppConfig],
    ) -> None:
        """Verify malformed `metadata.json` raises `ArtifactLoadError`.

        Args:
            build_valid_artifact_dir: Factory creating an isolated artifact copy.
            build_temp_config: Factory creating `AppConfig` for that directory.
        """
        artifact_dir = build_valid_artifact_dir()
        (artifact_dir / "metadata.json").write_text("{corrupt json", encoding="utf-8")
        cfg = build_temp_config(artifact_dir)
        with pytest.raises(ArtifactLoadError):
            ArtifactLoader(cfg).load()

    def test_raises_on_mismatched_features(
        self,
        build_valid_artifact_dir: Callable[[], Path],
        build_temp_config: Callable[..., AppConfig],
    ) -> None:
        """Verify metadata features differing from `FEATURE_ORDER` fail loading.

        Args:
            build_valid_artifact_dir: Factory creating an isolated artifact copy.
            build_temp_config: Factory creating `AppConfig` for that directory.
        """
        artifact_dir = build_valid_artifact_dir()
        meta_path = artifact_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["features"] = ["N", "P"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        cfg = build_temp_config(artifact_dir)
        with pytest.raises(ArtifactLoadError):
            ArtifactLoader(cfg).load()

    def test_collects_only_non_empty_pngs_and_ignores_missing_dirs(
        self,
        build_valid_artifact_dir: Callable[[], Path],
        build_temp_config: Callable[..., AppConfig],
    ) -> None:
        """Verify `_collect_pngs` skips empty files, non-PNGs, and missing dirs.

        Args:
            build_valid_artifact_dir: Factory creating an isolated artifact copy.
            build_temp_config: Factory creating `AppConfig` for that directory.
        """
        artifact_dir = build_valid_artifact_dir()
        eda_dir = artifact_dir / "plots" / "eda"
        eda_dir.mkdir(parents=True, exist_ok=True)
        (eda_dir / "valid.png").write_bytes(b"\x89PNG\r\n\x1a\ncontent")
        (eda_dir / "empty.png").write_bytes(b"")
        (eda_dir / "notes.txt").write_text("ignore", encoding="utf-8")

        cfg = build_temp_config(artifact_dir)
        bundle = ArtifactLoader(cfg).load()
        assert [p.name for p in bundle.eda_plots] == ["valid.png"]
        assert bundle.eval_plots == ()
