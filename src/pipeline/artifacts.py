"""Serialisable bundle of everything needed to score live data.

A :class:`ModelArtifact` carries the fitted ensemble, calibrator, conformal
predictor, the selected feature list and the feature parameters, plus metadata
(training span, metrics, gate result). Persisting them together guarantees the
live scanner scores with exactly the pipeline that was validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from ..features.engineering import FeatureParams
from ..logging_utils import get_logger

logger = get_logger(__name__)

ARTIFACT_VERSION = 1


@dataclass
class ModelArtifact:
    ensemble: Any
    calibrator: Any
    conformal: Any
    feature_names: list[str]
    feature_params: FeatureParams
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = ARTIFACT_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def save_artifact(artifact: ModelArtifact, path: str | Path) -> Path:
    """Persist an artifact to ``path`` (creating parent directories)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    logger.info("Saved model artifact to %s", path)
    return path


def load_artifact(path: str | Path) -> ModelArtifact:
    """Load an artifact, checking its version for forward compatibility."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model artifact not found: {path}")
    artifact = joblib.load(path)
    if getattr(artifact, "version", None) != ARTIFACT_VERSION:
        logger.warning(
            "Artifact version %s != expected %s; proceed with caution",
            getattr(artifact, "version", None),
            ARTIFACT_VERSION,
        )
    return artifact
