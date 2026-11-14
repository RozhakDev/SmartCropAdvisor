"""Resolve application configuration and operational limits.

This module provides `AppConfig` and environment parsing helpers to locate
trained model artifacts, diagnostic plot directories, and batch limits.
"""

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_positive_int(env_key: str, default: int) -> int:
    """Parse a positive integer environment variable with a default fallback.

    Args:
        env_key: Environment variable name to read.
        default: Fallback value when the variable is unset or blank.

    Returns:
        Parsed integer greater than or equal to 1.

    Raises:
        ValueError: If the value is non-numeric or less than 1.
    """
    raw = os.getenv(env_key)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{env_key} must be an integer, got: {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{env_key} must be >= 1, got: {value}")
    return value


@dataclass(frozen=True)
class AppConfig:
    """Immutable configuration for artifact paths and operational limits.

    Attributes:
        artifacts_dir: Root directory containing model and metadata artifacts.
        model_path: Path to the serialized scikit-learn pipeline file.
        metadata_path: Path to the JSON metadata contract file.
        eda_plots_dir: Directory containing exploratory data analysis PNGs.
        eval_plots_dir: Directory containing model evaluation PNGs.
        max_csv_bytes: Maximum allowed uploaded CSV file size in bytes.
        max_batch_rows: Maximum allowed row count per batch inference request.
        default_top_k: Default number of top candidate crops to display.
        max_top_k: Upper bound for user-selected top-K candidate crops.
    """

    artifacts_dir: Path
    model_path: Path
    metadata_path: Path
    eda_plots_dir: Path
    eval_plots_dir: Path
    max_csv_bytes: int = 5 * 1024 * 1024
    max_batch_rows: int = 10_000
    default_top_k: int = 3
    max_top_k: int = 5

    @classmethod
    def from_env(cls, visualization_root: Path | None = None) -> "AppConfig":
        """Build configuration from environment variables or repository defaults.

        Args:
            visualization_root: Optional override for the Visualization root
                directory used to resolve default repository artifact paths.

        Returns:
            Validated immutable `AppConfig` instance.

        Raises:
            ValueError: If numeric environment overrides are invalid or < 1.
        """
        root = visualization_root or Path(__file__).resolve().parent.parent
        env_artifacts = os.getenv("SMARTCROP_ARTIFACTS_DIR")
        artifacts_dir = (
            Path(env_artifacts).expanduser().resolve()
            if env_artifacts and env_artifacts.strip()
            else (root.parent / "MachineLearning" / "artifacts").resolve()
        )
        plots_root = artifacts_dir / "plots"
        return cls(
            artifacts_dir=artifacts_dir,
            model_path=artifacts_dir / "model_pipeline.joblib",
            metadata_path=artifacts_dir / "metadata.json",
            eda_plots_dir=plots_root / "eda",
            eval_plots_dir=plots_root / "eval",
            max_csv_bytes=_parse_positive_int("SMARTCROP_MAX_CSV_BYTES", 5 * 1024 * 1024),
            max_batch_rows=_parse_positive_int("SMARTCROP_MAX_BATCH_ROWS", 10_000),
        )
