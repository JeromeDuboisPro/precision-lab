"""Cascading precision power method with adaptive precision escalation.

Implements the cascading precision strategy: FP8 → FP16 → FP32 → FP64
- Start fast with FP8 for rapid initial convergence
- Transition based on pluggable plateau detection strategies
- Escalate to FP32/FP64 only when higher accuracy needed
- Carry eigenvector state across transitions for efficiency

Performance Model (Simulated):
    Uses theoretical peak throughput ratios for demonstration:
    - FP8: 6× throughput vs FP64 (theoretical tensor core peak)
    - FP16: 4× throughput vs FP64 (theoretical half-precision units)
    - FP32: 2× throughput vs FP64
    - FP64: baseline reference

    Note: These are theoretical maximum speedups. Real-world performance
    varies with memory bandwidth (power method is memory-bound).

References:
- Higham, N.J.: "Accuracy and Stability of Numerical Algorithms" (2002)
- Modern GPU tensor core architecture whitepapers
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from precision_lab.algorithms.matrices import (
    ExperimentSetup,
    create_experiment,
)
from precision_lab.algorithms.plateau_detection import (
    MultiCriteriaDetector,
    PlateauDetector,
)
from precision_lab.algorithms.power_method import PowerIteration
from precision_lab.data.precision_types import PrecisionFormat

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    """Configuration for a single precision level."""

    name: str
    """Precision name (FP8, FP16, FP32, FP64)."""

    format: PrecisionFormat
    """Precision format enum."""

    bytes_per_element: int
    """Bytes per floating-point element."""

    time_speedup_h100: float
    """H100 time speedup factor (for performance display)."""

    iteration_speedup_h100: float
    """H100 iteration budget factor (for fair comparison)."""


# Default precision cascade configuration (H100 model)
DEFAULT_PRECISION_CASCADE: tuple[PrecisionConfig, ...] = (
    PrecisionConfig(
        name="FP8",
        format=PrecisionFormat.FP8_E4M3,
        bytes_per_element=1,
        time_speedup_h100=6.0,
        iteration_speedup_h100=6.0,
    ),
    PrecisionConfig(
        name="FP16",
        format=PrecisionFormat.FP16,
        bytes_per_element=2,
        time_speedup_h100=4.0,
        iteration_speedup_h100=4.0,
    ),
    PrecisionConfig(
        name="FP32",
        format=PrecisionFormat.FP32,
        bytes_per_element=4,
        time_speedup_h100=1.0,
        iteration_speedup_h100=2.0,
    ),
    PrecisionConfig(
        name="FP64",
        format=PrecisionFormat.FP64,
        bytes_per_element=8,
        time_speedup_h100=1.0,
        iteration_speedup_h100=1.0,
    ),
)


@dataclass(frozen=True, slots=True)
class SegmentResult:
    """Result of running power method at a single precision level."""

    precision: str
    """Precision name."""

    iterations: int
    """Actual iterations performed."""

    effective_iterations: float
    """FP64-equivalent iterations (weighted by speedup)."""

    start_iteration: int
    """Global iteration at segment start."""

    end_iteration: int
    """Global iteration at segment end."""

    start_residual: float
    """Residual norm at segment start."""

    end_residual: float
    """Residual norm at segment end."""

    start_error: float
    """Relative error at segment start."""

    end_error: float
    """Relative error at segment end."""

    segment_time: float
    """Wall clock time for segment (seconds)."""

    converged: bool
    """Whether target was reached in this segment."""

    plateau_score: float | None
    """Plateau detection score if plateau triggered."""


@dataclass(frozen=True, slots=True)
class CascadeTrace:
    """Complete trace of cascading precision execution."""

    iterations: int
    """Total actual iterations."""

    effective_iterations: float
    """Total FP64-equivalent iterations."""

    final_eigenvalue: float
    """Final eigenvalue estimate."""

    final_error: float
    """Final relative error."""

    final_residual: float
    """Final normalized residual."""

    converged: bool
    """Whether target residual was achieved."""

    total_time: float
    """Total execution time (seconds)."""

    segments: tuple[SegmentResult, ...]
    """Per-precision segment results."""

    history: tuple[dict[str, Any], ...]
    """Per-iteration metrics (tuple for immutability)."""


@dataclass
class CascadingPowerMethod:
    """Cascading precision power method with adaptive escalation.

    Orchestrates the precision cascade from FP8 to FP64, using plateau
    detection to determine when to escalate precision.

    Args:
        matrix_size: Matrix dimension (default 1024).
        condition_number: Desired condition number.
        plateau_detector: Plateau detection strategy (default: MultiCriteriaDetector).
        seed: Random seed for reproducibility.
        convergence_type: Matrix spectrum type ("slow", "linear", "geometric").

    Example:
        >>> cascading = CascadingPowerMethod(1024, 100)
        >>> trace = cascading.run(target_residual=1e-6)
        >>> print(f"Converged: {trace.converged}, iterations: {trace.iterations}")

        >>> # Custom plateau detector
        >>> from precision_lab.algorithms.plateau_detection import RelativeImprovementDetector
        >>> cascading = CascadingPowerMethod(
        ...     1024, 100,
        ...     plateau_detector=RelativeImprovementDetector(window_size=20)
        ... )
    """

    matrix_size: int = 1024
    condition_number: float = 100.0
    plateau_detector: PlateauDetector = field(default_factory=MultiCriteriaDetector)
    seed: int = 42
    convergence_type: str = "slow"

    # Internal state (set in __post_init__)
    _experiment: ExperimentSetup = field(init=False, repr=False)
    _precision_cascade: tuple[PrecisionConfig, ...] = field(
        init=False, repr=False, default=DEFAULT_PRECISION_CASCADE
    )

    def __post_init__(self) -> None:
        """Initialize experiment setup."""
        self._experiment = create_experiment(
            self.matrix_size,
            self.condition_number,
            seed=self.seed,
            convergence_type=self.convergence_type,
        )

    @property
    def true_eigenvalue(self) -> float:
        """True eigenvalue (ground truth)."""
        return self._experiment.true_eigenvalue

    @property
    def matrix_fingerprint(self) -> dict[str, Any]:
        """Matrix fingerprint for reproducibility verification."""
        return self._experiment.fingerprint.to_dict()

    def run(
        self,
        target_residual: float = 1e-6,
        max_effective_iterations: int | None = None,
    ) -> CascadeTrace:
        """Execute cascading precision power method.

        Runs until residual_norm < target_residual OR effective iteration
        budget exhausted.

        Args:
            target_residual: Target residual norm (default 1e-6).
            max_effective_iterations: Maximum FP64-equivalent iterations.
                Default: 5 * max(n, 1000). Lower precision can iterate MORE
                (weighted by speedup factor).

        Returns:
            CascadeTrace with complete execution history.
        """
        n = self.matrix_size

        if max_effective_iterations is None:
            max_effective_iterations = 5 * max(n, 1000)

        # Use reproducible initial vector
        x: NDArray[np.floating] = self._experiment.initial_vector.copy()

        # Tracking
        full_history: list[dict[str, Any]] = []
        segments: list[SegmentResult] = []

        total_iterations = 0
        effective_iterations = 0.0
        start_time = time.perf_counter()
        current_precision_idx = 0

        # Cumulative time tracking across segments
        cumulative_algorithm_time = 0.0
        cumulative_wall_time = 0.0

        # Iterate through precision cascade
        while (
            current_precision_idx < len(self._precision_cascade)
            and effective_iterations < max_effective_iterations
        ):
            precision_config = self._precision_cascade[current_precision_idx]
            iteration_speedup = precision_config.iteration_speedup_h100

            # Calculate remaining budget
            remaining_effective = max_effective_iterations - effective_iterations
            if remaining_effective <= 0:
                break

            # Convert to actual iterations allowed at this precision
            max_actual_iters = int(remaining_effective * iteration_speedup)

            # Run segment
            segment_history, x, converged, plateau_score = self._run_segment(
                precision_config,
                x,
                target_residual,
                max_actual_iters,
                total_iterations,
                cumulative_algorithm_time,
                cumulative_wall_time,
            )

            if not segment_history:
                break

            # Calculate effective iterations consumed
            actual_iters = len(segment_history)
            effective_iters = actual_iters / iteration_speedup

            # Create segment result
            segment = SegmentResult(
                precision=precision_config.name,
                iterations=actual_iters,
                effective_iterations=effective_iters,
                start_iteration=total_iterations,
                end_iteration=total_iterations + actual_iters,
                start_residual=segment_history[0]["residual_norm"],
                end_residual=segment_history[-1]["residual_norm"],
                start_error=segment_history[0]["relative_error"],
                end_error=segment_history[-1]["relative_error"],
                segment_time=sum(h["wall_time"] for h in segment_history),
                converged=converged,
                plateau_score=plateau_score,
            )
            segments.append(segment)

            # Update tracking
            full_history.extend(segment_history)
            total_iterations += actual_iters
            effective_iterations += effective_iters

            # Update cumulative offsets
            if segment_history:
                cumulative_algorithm_time = segment_history[-1][
                    "cumulative_algorithm_time"
                ]
                cumulative_wall_time = segment_history[-1]["cumulative_wall_time"]

            # Check if target reached
            if converged and segment_history[-1]["residual_norm"] <= target_residual:
                break

            current_precision_idx += 1

        total_time = time.perf_counter() - start_time

        # Extract final values
        if full_history:
            final = full_history[-1]
            final_eigenvalue = final["eigenvalue"]
            final_error = final["relative_error"]
            final_residual = final["residual_norm"]
            converged = final_residual <= target_residual
        else:
            final_eigenvalue = float("nan")
            final_error = float("nan")
            final_residual = float("nan")
            converged = False

        return CascadeTrace(
            iterations=total_iterations,
            effective_iterations=effective_iterations,
            final_eigenvalue=final_eigenvalue,
            final_error=final_error,
            final_residual=final_residual,
            converged=converged,
            total_time=total_time,
            segments=tuple(segments),
            history=tuple(full_history),
        )

    def _run_segment(
        self,
        precision_config: PrecisionConfig,
        x: NDArray[np.floating],
        target_residual: float,
        max_iter: int,
        iteration_offset: int,
        cumulative_algorithm_time: float,
        cumulative_wall_time: float,
    ) -> tuple[list[dict[str, Any]], NDArray[np.floating], bool, float | None]:
        """Run power method at a single precision level.

        Args:
            precision_config: Precision configuration.
            x: Initial eigenvector estimate.
            target_residual: Target residual norm.
            max_iter: Maximum iterations for this segment.
            iteration_offset: Global iteration counter offset.
            cumulative_algorithm_time: Time offset from previous segments.
            cumulative_wall_time: Wall time offset from previous segments.

        Returns:
            (history, final_vector, converged, plateau_score)
        """
        precision_name = precision_config.name
        precision_format = precision_config.format

        history: list[dict[str, Any]] = []
        converged = False
        residual_history: list[float] = []
        plateau_score: float | None = None

        # Create PowerIteration engine
        engine = PowerIteration(
            self._experiment.matrix,
            precision_format,
            A_fp64=self._experiment.matrix,
        )
        engine.set_initial_vector(x)

        for iteration in range(max_iter):
            # Execute iteration
            iter_result = engine.iterate()

            if np.isnan(iter_result.eigenvalue):
                break

            # Check convergence
            conv_result = engine.check_convergence(
                iter_result.eigenvalue,
                self._experiment.true_eigenvalue,
            )

            # Total wall time
            wall_time = iter_result.algorithm_time + conv_result.check_time
            cumulative_algorithm_time += iter_result.algorithm_time
            cumulative_wall_time += wall_time

            # Store iteration data
            history.append(
                {
                    "iteration": iteration_offset + iteration,
                    "precision": precision_name,
                    "wall_time": wall_time,
                    "algorithm_time": iter_result.algorithm_time,
                    "convergence_check_time": conv_result.check_time,
                    "cumulative_wall_time": cumulative_wall_time,
                    "cumulative_algorithm_time": cumulative_algorithm_time,
                    "eigenvalue": iter_result.eigenvalue,
                    "relative_error": conv_result.relative_error,
                    "residual_norm": conv_result.residual_norm,
                    "vector_norm": engine.vector_norm,
                }
            )

            residual_history.append(conv_result.residual_norm)

            # Check target reached
            if conv_result.residual_norm <= target_residual:
                converged = True
                break

            # Plateau detection (not for FP64)
            if precision_name != "FP64":
                result = self.plateau_detector.detect(
                    residual_history,
                    precision_name,
                    iteration,
                )
                plateau_score = result.score

                if result.detected:
                    converged = False
                    break

        final_x = engine.current_vector.copy()
        return history, final_x, converged, plateau_score

    def to_dict(self, trace: CascadeTrace) -> dict[str, Any]:
        """Convert trace to dictionary for JSON serialization.

        Args:
            trace: Cascade trace to convert.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        return {
            "metadata": {
                "algorithm": "cascading_precision",
                "matrix_size": self.matrix_size,
                "condition_number": self.condition_number,
                "true_eigenvalue": self.true_eigenvalue,
                "seed": self.seed,
                "convergence_type": self.convergence_type,
                "timestamp": datetime.now(UTC).isoformat(),
                "final_residual": trace.final_residual,
                "final_error": trace.final_error,
                "converged": trace.converged,
                "plateau_detector": {
                    "type": type(self.plateau_detector).__name__,
                    "config": self.plateau_detector.get_config("fp32"),
                },
                "matrix_fingerprint": self.matrix_fingerprint,
            },
            "summary": {
                "total_iterations": trace.iterations,
                "effective_iterations": trace.effective_iterations,
                "total_time_seconds": trace.total_time,
                "precision_levels_used": len(trace.segments),
            },
            "segments": [
                {
                    "precision": s.precision,
                    "iterations": s.iterations,
                    "effective_iterations": s.effective_iterations,
                    "start_iteration": s.start_iteration,
                    "end_iteration": s.end_iteration,
                    "start_residual": s.start_residual,
                    "end_residual": s.end_residual,
                    "start_error": s.start_error,
                    "end_error": s.end_error,
                    "segment_time": s.segment_time,
                    "converged": s.converged,
                    "plateau_score": s.plateau_score,
                }
                for s in trace.segments
            ],
            "trace": list(trace.history),
        }


__all__ = [
    "PrecisionConfig",
    "DEFAULT_PRECISION_CASCADE",
    "SegmentResult",
    "CascadeTrace",
    "CascadingPowerMethod",
]
