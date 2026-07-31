"""Distribution drift detection (PSI / Jensen-Shannon)."""

from .detector import (
    DriftDetector,
    jensen_shannon_divergence,
    population_stability_index,
)

__all__ = [
    "DriftDetector",
    "jensen_shannon_divergence",
    "population_stability_index",
]
