"""Provide validated loading for trained model artifacts and plot assets.

This module deserializes the trained scikit-learn pipeline, validates the
accompanying `metadata.json` contract, and collects verified PNG figures for
exploratory data analysis and model evaluation.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
from pydantic import ValidationError
from sklearn.pipeline import Pipeline

from src.config import AppConfig
from src.logger import get_logger
from src.schemas.prediction import FEATURE_ORDER, ModelMetadata


logger = get_logger(__name__)


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when required model or metadata artifact files are missing."""


class ArtifactLoadError(RuntimeError):
    """Raised when an artifact fails deserialization or contract validation."""


@dataclass(frozen=True)
class ArtifactBundle:
    """Validated runtime bundle consumed by inference and diagnostics views.

    Attributes:
        pipeline: Trained scikit-learn classification pipeline.
        metadata: Validated model metadata and feature summary statistics.
        eda_plots: Tuple of verified exploratory data analysis PNG paths.
        eval_plots: Tuple of verified model evaluation PNG paths.
    """

    pipeline: Pipeline
    metadata: ModelMetadata
    eda_plots: tuple[Path, ...]
    eval_plots: tuple[Path, ...]


class ArtifactLoader:
    """Load and validate trained model artifacts from configured paths."""

    def __init__(self, config: AppConfig) -> None:
        """Initialize the loader with application configuration.

        Args:
            config: Resolved application configuration specifying artifact paths.
        """
        self._config = config

    def load(self) -> ArtifactBundle:
        """Load and validate the pipeline, metadata, and diagnostic PNG plots.

        Returns:
            Validated `ArtifactBundle` ready for inference and diagnostics.

        Raises:
            ArtifactNotFoundError: If the model pipeline or metadata file does
                not exist at the configured path.
            ArtifactLoadError: If deserialization fails, the estimator lacks
                `predict`/`predict_proba`, metadata validation fails, or the
                metadata feature list mismatches `FEATURE_ORDER`.
        """
        if not self._config.model_path.exists():
            logger.error("Model artifact missing at %s", self._config.model_path)
            raise ArtifactNotFoundError(
                f"Model artifact not found at {self._config.model_path}"
            )
        if not self._config.metadata_path.exists():
            logger.error("Metadata artifact missing at %s", self._config.metadata_path)
            raise ArtifactNotFoundError(
                f"Metadata artifact not found at {self._config.metadata_path}"
            )

        try:
            pipeline = joblib.load(self._config.model_path)
        except Exception as exc:
            logger.exception("Failed to deserialize model pipeline")
            raise ArtifactLoadError(f"Failed to load model pipeline: {exc}") from exc

        if not hasattr(pipeline, "predict") or not hasattr(pipeline, "predict_proba"):
            raise ArtifactLoadError("Loaded model does not expose predict and predict_proba.")

        try:
            raw_meta = json.loads(self._config.metadata_path.read_text(encoding="utf-8"))
            metadata = ModelMetadata.model_validate(raw_meta)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.exception("Invalid metadata.json contract")
            raise ArtifactLoadError(f"Invalid metadata.json contract: {exc}") from exc

        if set(metadata.features) != set(FEATURE_ORDER):
            raise ArtifactLoadError(
                f"Metadata features {metadata.features} do not match expected {list(FEATURE_ORDER)}."
            )

        eda_plots = self._collect_pngs(self._config.eda_plots_dir)
        eval_plots = self._collect_pngs(self._config.eval_plots_dir)
        logger.info(
            "Loaded model=%s version=%s classes=%d",
            metadata.model_name,
            metadata.artifact_version,
            len(metadata.classes),
        )
        return ArtifactBundle(
            pipeline=pipeline,
            metadata=metadata,
            eda_plots=eda_plots,
            eval_plots=eval_plots,
        )

    @staticmethod
    def _collect_pngs(directory: Path) -> tuple[Path, ...]:
        """Collect non-empty PNG files directly inside a plot directory.

        Resolves symlinks and ignores empty files or paths outside the target
        directory.

        Args:
            directory: Directory path to scan for `.png` files.

        Returns:
            Sorted tuple of resolved non-empty PNG paths, or an empty tuple if
            the directory does not exist.
        """
        if not directory.exists() or not directory.is_dir():
            return ()
        resolved_dir = directory.resolve()
        safe_files: list[Path] = []
        for path in sorted(resolved_dir.glob("*.png")):
            resolved = path.resolve()
            if (
                resolved.parent == resolved_dir
                and resolved.is_file()
                and resolved.stat().st_size > 0
            ):
                safe_files.append(resolved)
        return tuple(safe_files)
