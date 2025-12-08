"""Plateau detection strategies for adaptive precision cascading.

This module provides pluggable strategies for detecting convergence plateaus
in iterative eigenvalue algorithms. Different detection strategies can be
swapped via dependency injection.

Key Strategies:
- MultiCriteriaDetector: Combines relative improvement, variance, and acceleration
- RelativeImprovementDetector: Simple window-based progress monitoring
- ThresholdDetector: Basic error magnitude threshold
- VarianceDetector: Stability detection via variance analysis

References:
- Figueira et al., "Multiple Criteria Decision Analysis" (2005)
- Box & Jenkins, "Time Series Analysis" (2015)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from precision_lab.data.precision_types import get_tolerance


@dataclass(frozen=True, slots=True)
class PlateauResult:
    """Result of plateau detection."""

    detected: bool
    """Whether a plateau was detected."""

    score: float
    """Detection score (for debugging/logging)."""


class PlateauDetector(ABC):
    """Abstract base class for plateau detection strategies.

    All detector implementations must:
    1. Implement detect() method
    2. Implement get_config() method
    """

    @abstractmethod
    def detect(
        self,
        error_history: list[float],
        precision: str,
        iteration: int,
    ) -> PlateauResult:
        """Detect if convergence has plateaued.

        Args:
            error_history: List of residual norms (oldest to newest).
            precision: Current precision level ('fp8', 'fp16', 'fp32', 'fp64').
            iteration: Current iteration number within this precision level.

        Returns:
            PlateauResult with detection status and score.
        """

    @abstractmethod
    def get_config(self, precision: str) -> dict:
        """Get configuration parameters for given precision level."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@dataclass
class MultiCriteriaConfig:
    """Configuration for MultiCriteriaDetector."""

    relative_threshold: float
    """Minimum improvement over window to NOT trigger plateau."""

    variance_threshold: float
    """Log-variance threshold for stability detection."""

    acceleration_threshold: float
    """Second derivative threshold."""

    min_iterations: int
    """Minimum iterations before checking."""


@dataclass
class MultiCriteriaDetector(PlateauDetector):
    """Multi-criteria fusion plateau detector.

    Combines three independent statistical indicators:
    1. Relative improvement (window-based progress)
    2. Variance stability (error oscillation detection)
    3. Acceleration (second derivative of convergence)

    Requires weighted score ≥ threshold for plateau detection.

    Args:
        window_size: Sliding window for metrics (default 15).
        score_threshold: Required confidence for plateau (default 0.65).
        criteria_weights: Weights for each criterion (must sum to 1.0).
    """

    window_size: int = 15
    score_threshold: float = 0.65
    criteria_weights: dict[str, float] = field(
        default_factory=lambda: {
            "relative_improvement": 0.4,
            "variance": 0.3,
            "acceleration": 0.3,
        }
    )

    def __post_init__(self) -> None:
        total = sum(self.criteria_weights.values())
        if abs(total - 1.0) > 1e-6:
            msg = f"Weights must sum to 1.0, got {total}"
            raise ValueError(msg)

        # Precision-specific configurations
        self._configs: dict[str, MultiCriteriaConfig] = {
            "fp8": MultiCriteriaConfig(
                relative_threshold=0.10,
                variance_threshold=0.1,
                acceleration_threshold=0.01,
                min_iterations=30,
            ),
            "fp8_e4m3": MultiCriteriaConfig(
                relative_threshold=0.10,
                variance_threshold=0.1,
                acceleration_threshold=0.01,
                min_iterations=30,
            ),
            "fp8_e5m2": MultiCriteriaConfig(
                relative_threshold=0.10,
                variance_threshold=0.1,
                acceleration_threshold=0.01,
                min_iterations=30,
            ),
            "fp16": MultiCriteriaConfig(
                relative_threshold=0.06,
                variance_threshold=0.05,
                acceleration_threshold=0.005,
                min_iterations=50,
            ),
            "fp32": MultiCriteriaConfig(
                relative_threshold=0.025,
                variance_threshold=0.001,
                acceleration_threshold=0.0005,
                min_iterations=40,
            ),
        }

        # Precision floor thresholds
        self._precision_floors: dict[str, float] = {
            "fp8": 5e-1,
            "fp8_e4m3": 5e-1,
            "fp8_e5m2": 5e-1,
            "fp16": 1e-2,
            "fp32": 1e-5,
            "fp64": 1e-14,
        }

        self._min_iter_reduced: set[str] = set()

    def detect(
        self,
        error_history: list[float],
        precision: str,
        iteration: int,
    ) -> PlateauResult:
        """Multi-criteria fusion detection."""
        precision = precision.lower()
        config = self._configs.get(precision)

        if not config:
            # FP64 or unknown: no plateau detection
            return PlateauResult(detected=False, score=0.0)

        # Check precision floor
        current_residual = error_history[-1] if error_history else float("inf")
        effective_min = self._get_effective_min_iterations(
            precision, config.min_iterations, current_residual
        )

        if iteration < effective_min:
            return PlateauResult(detected=False, score=0.0)

        if len(error_history) < self.window_size:
            return PlateauResult(detected=False, score=0.0)

        score = self._compute_score(error_history, config)
        detected = score >= self.score_threshold

        return PlateauResult(detected=detected, score=score)

    def _get_effective_min_iterations(
        self, precision: str, base_min: int, current_residual: float
    ) -> int:
        """Reduce min_iterations when approaching precision floor."""
        floor_threshold = self._precision_floors.get(precision)
        if floor_threshold is None:
            return base_min

        if current_residual <= floor_threshold:
            if precision not in self._min_iter_reduced:
                self._min_iter_reduced.add(precision)
            return base_min // 2

        if precision in self._min_iter_reduced:
            return base_min // 2

        return base_min

    def _compute_score(
        self, error_history: list[float], config: MultiCriteriaConfig
    ) -> float:
        """Compute weighted multi-criteria score."""
        recent_errors = np.array(error_history[-self.window_size :])
        log_errors = np.log10(np.maximum(recent_errors, 1e-16))

        score = 0.0

        # Criterion 1: Relative Improvement
        error_start = recent_errors[0]
        error_end = recent_errors[-1]

        rel_improvement = 0.0
        if error_start > 0:
            rel_improvement = (error_start - error_end) / error_start
            if rel_improvement < config.relative_threshold:
                score += self.criteria_weights["relative_improvement"]

        # Criterion 2: Variance Analysis
        variance = np.var(log_errors)
        half = self.window_size // 2
        mean_first = np.mean(log_errors[:half])
        mean_second = np.mean(log_errors[half:])
        net_improvement = mean_first - mean_second

        if variance < config.variance_threshold and net_improvement < 0.1 or variance > config.variance_threshold * 5 and net_improvement < 0.05:
            score += self.criteria_weights["variance"]

        # Criterion 3: Acceleration
        slopes = np.diff(log_errors)
        if len(slopes) > 5:
            accelerations = np.diff(slopes)
            mean_accel = np.abs(np.mean(accelerations[-5:]))

            if mean_accel < config.acceleration_threshold or mean_accel > config.acceleration_threshold * 100 and rel_improvement < 0.01:
                score += self.criteria_weights["acceleration"]

        return score

    def get_config(self, precision: str) -> dict:
        """Get configuration for precision level."""
        config = self._configs.get(precision.lower())
        if config:
            return {
                "relative_threshold": config.relative_threshold,
                "variance_threshold": config.variance_threshold,
                "acceleration_threshold": config.acceleration_threshold,
                "min_iterations": config.min_iterations,
            }
        return {}

    def __repr__(self) -> str:
        return f"MultiCriteriaDetector(window={self.window_size}, threshold={self.score_threshold})"


@dataclass
class RelativeImprovementDetector(PlateauDetector):
    """Simple relative improvement plateau detector.

    Monitors relative change in error over a sliding window. Plateau detected
    when improvement rate falls below precision-specific threshold.

    Args:
        window_size: Sliding window for improvement calculation (default 15).
        min_iterations: Minimum iterations before checking (default 30).
    """

    window_size: int = 15
    min_iterations: int = 30

    def __post_init__(self) -> None:
        self._thresholds: dict[str, float] = {
            "fp8": get_tolerance("fp8_e4m3", "residual_tol"),
            "fp8_e4m3": get_tolerance("fp8_e4m3", "residual_tol"),
            "fp8_e5m2": get_tolerance("fp8_e5m2", "residual_tol"),
            "fp16": get_tolerance("fp16", "residual_tol"),
            "fp32": get_tolerance("fp32", "residual_tol"),
        }

    def detect(
        self,
        error_history: list[float],
        precision: str,
        iteration: int,
    ) -> PlateauResult:
        """Relative improvement detection."""
        precision = precision.lower()
        if precision not in self._thresholds:
            return PlateauResult(detected=False, score=0.0)

        if iteration < self.min_iterations:
            return PlateauResult(detected=False, score=0.0)

        if len(error_history) < self.window_size:
            return PlateauResult(detected=False, score=0.0)

        recent = error_history[-self.window_size :]
        error_start, error_end = recent[0], recent[-1]

        if error_start <= 0:
            return PlateauResult(detected=False, score=0.0)

        rel_improvement = (error_start - error_end) / error_start
        threshold = self._thresholds[precision]
        plateau = rel_improvement < threshold

        return PlateauResult(detected=plateau, score=rel_improvement)

    def get_config(self, precision: str) -> dict:
        """Get configuration for precision level."""
        return {
            "threshold": self._thresholds.get(precision.lower(), 0.0),
            "window_size": self.window_size,
            "min_iterations": self.min_iterations,
        }

    def __repr__(self) -> str:
        return f"RelativeImprovementDetector(window={self.window_size})"


@dataclass
class ThresholdDetector(PlateauDetector):
    """Original threshold-based plateau detector (baseline).

    Simple error magnitude threshold: plateau when error < threshold.

    Args:
        stagnation_threshold: Iterations without improvement before transition.
    """

    stagnation_threshold: int = 100

    def __post_init__(self) -> None:
        self._thresholds: dict[str, float] = {
            "fp8": get_tolerance("fp8_e4m3", "residual_tol"),
            "fp8_e4m3": get_tolerance("fp8_e4m3", "residual_tol"),
            "fp8_e5m2": get_tolerance("fp8_e5m2", "residual_tol"),
            "fp16": get_tolerance("fp16", "residual_tol"),
            "fp32": get_tolerance("fp32", "residual_tol"),
        }

    def detect(
        self,
        error_history: list[float],
        precision: str,
        _iteration: int,  # noqa: ARG002
    ) -> PlateauResult:
        """Threshold-based detection."""
        precision = precision.lower()
        if not error_history or precision not in self._thresholds:
            return PlateauResult(detected=False, score=0.0)

        current_error = error_history[-1]
        threshold = self._thresholds[precision]
        plateau = current_error < threshold

        return PlateauResult(detected=plateau, score=current_error)

    def get_config(self, precision: str) -> dict:
        """Get configuration for precision level."""
        return {
            "threshold": self._thresholds.get(precision.lower(), 0.0),
            "stagnation_threshold": self.stagnation_threshold,
        }

    def __repr__(self) -> str:
        return "ThresholdDetector()"


def create_detector(detector_type: str = "multi_criteria", **kwargs) -> PlateauDetector:
    """Factory function to create plateau detectors.

    Args:
        detector_type: Type of detector ('multi_criteria', 'relative', 'threshold').
        **kwargs: Detector-specific parameters.

    Returns:
        PlateauDetector instance.

    Example:
        >>> detector = create_detector('multi_criteria', score_threshold=0.7)
        >>> detector = create_detector('relative', window_size=20)
    """
    detectors: dict[str, type[PlateauDetector]] = {
        "multi_criteria": MultiCriteriaDetector,
        "relative": RelativeImprovementDetector,
        "threshold": ThresholdDetector,
    }

    if detector_type not in detectors:
        msg = f"Unknown detector: {detector_type}. Available: {list(detectors.keys())}"
        raise ValueError(msg)

    return detectors[detector_type](**kwargs)


__all__ = [
    "PlateauResult",
    "PlateauDetector",
    "MultiCriteriaDetector",
    "RelativeImprovementDetector",
    "ThresholdDetector",
    "create_detector",
]
