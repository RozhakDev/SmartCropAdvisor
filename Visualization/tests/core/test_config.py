"""Verify `AppConfig` environment variable resolution and validation rules.

This module tests environment overrides, blank-string fallbacks, and error
handling for invalid positive integer settings.
"""

from pathlib import Path

import pytest

from src.config import AppConfig


class TestAppConfig:
    """Test suite for `AppConfig` environment parsing and defaults."""

    def test_from_env_applies_overrides(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Verify environment variables override default paths and batch limits.

        Args:
            monkeypatch: Pytest environment patch fixture.
            tmp_path: Temporary directory path used as the artifacts root.
        """
        monkeypatch.setenv("SMARTCROP_ARTIFACTS_DIR", str(tmp_path))
        monkeypatch.setenv("SMARTCROP_MAX_CSV_BYTES", "2048")
        monkeypatch.setenv("SMARTCROP_MAX_BATCH_ROWS", "250")

        cfg = AppConfig.from_env()
        assert cfg.artifacts_dir == tmp_path.resolve()
        assert cfg.model_path == tmp_path.resolve() / "model_pipeline.joblib"
        assert cfg.metadata_path == tmp_path.resolve() / "metadata.json"
        assert cfg.max_csv_bytes == 2048
        assert cfg.max_batch_rows == 250

    def test_from_env_treats_blank_env_as_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Verify blank or whitespace-only overrides fall back to defaults.

        Args:
            monkeypatch: Pytest environment patch fixture.
            tmp_path: Temporary directory path passed as `visualization_root`.
        """
        monkeypatch.setenv("SMARTCROP_ARTIFACTS_DIR", "   ")
        monkeypatch.setenv("SMARTCROP_MAX_CSV_BYTES", "")
        monkeypatch.setenv("SMARTCROP_MAX_BATCH_ROWS", "   ")

        cfg = AppConfig.from_env(visualization_root=tmp_path)
        assert cfg.artifacts_dir == (tmp_path.parent / "MachineLearning" / "artifacts").resolve()
        assert cfg.max_csv_bytes == 5 * 1024 * 1024
        assert cfg.max_batch_rows == 10_000

    @pytest.mark.parametrize(
        ("env_key", "bad_value"),
        [
            ("SMARTCROP_MAX_CSV_BYTES", "not-an-int"),
            ("SMARTCROP_MAX_BATCH_ROWS", "abc"),
            ("SMARTCROP_MAX_CSV_BYTES", "0"),
            ("SMARTCROP_MAX_BATCH_ROWS", "-5"),
        ],
    )
    def test_from_env_rejects_invalid_positive_ints(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_key: str,
        bad_value: str,
    ) -> None:
        """Verify non-integer or non-positive integer overrides raise ValueError.

        Args:
            monkeypatch: Pytest environment patch fixture.
            env_key: Target environment variable name.
            bad_value: Invalid string value expected to fail validation.
        """
        monkeypatch.setenv(env_key, bad_value)
        with pytest.raises(ValueError):
            AppConfig.from_env()
