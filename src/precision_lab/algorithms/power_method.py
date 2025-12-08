"""Power iteration algorithm with ping-pong buffer optimization.

Implements the power method for computing the dominant eigenvalue and
eigenvector of a symmetric positive definite matrix.

Key Optimizations:
- Ping-pong buffer pattern eliminates allocations in hot loop
- Pre-computed FP64 reference matrix for accurate residual computation
- Self-timing for performance analysis

References:
- Golub & Van Loan: "Matrix Computations" (4th ed.), §7.3
- Cache-Oblivious Algorithms (Frigo et al., 1999)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from precision_lab.data.precision_types import (
    PrecisionFormat,
    get_dtype,
    get_tolerance,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray


@dataclass(frozen=True, slots=True)
class IterationResult:
    """Result of a single power iteration."""

    eigenvalue: float
    """Rayleigh quotient estimate."""

    algorithm_time: float
    """Time for iteration (seconds)."""


@dataclass(frozen=True, slots=True)
class ConvergenceResult:
    """Result of convergence check."""

    residual_norm: float
    """Normalized residual ||A*x - λ*x|| / (||A|| * ||x||)."""

    relative_error: float
    """Relative error |λ - λ_true| / |λ_true|."""

    eigenvalue_converged: bool
    """True if relative error < tolerance."""

    residual_converged: bool
    """True if normalized residual < tolerance."""

    check_time: float
    """Time for convergence computation (seconds)."""


class PowerIteration:
    """Power method engine with ping-pong buffer optimization.

    Eliminates array allocations in hot loop by alternating between two
    pre-allocated vectors in an Nx2 matrix (column-major for BLAS efficiency).

    Example:
        >>> from precision_lab.algorithms.matrices import create_experiment
        >>> experiment = create_experiment(100, condition_number=10)
        >>> engine = PowerIteration(experiment.matrix, "fp32")
        >>> for i in range(100):
        ...     result = engine.iterate()
        ...     conv = engine.check_convergence(result.eigenvalue, experiment.true_eigenvalue)
        ...     if conv.eigenvalue_converged and conv.residual_converged:
        ...         break
    """

    __slots__ = (
        "_A",
        "_A_fp64",
        "_dtype",
        "_precision_format",
        "_n",
        "_vectors",
        "_current_idx",
        "_eigenvalue_tol",
        "_residual_tol",
    )

    def __init__(
        self,
        A: NDArray[np.floating],
        precision: PrecisionFormat | str,
        *,
        A_fp64: NDArray[np.float64] | None = None,
        initial_vector: NDArray[np.floating] | None = None,
    ) -> None:
        """Initialize power iteration engine.

        Args:
            A: Input matrix (any precision).
            precision: Target precision format.
            A_fp64: Pre-computed FP64 matrix (optional, computed if None).
            initial_vector: Initial vector (optional, random if None).

        Note:
            A_fp64 is stored for residual computation (computed once).
            Working precision matrix is converted once during init.
        """
        # Normalize precision format
        if isinstance(precision, str):
            precision_str = precision.lower()
            self._precision_format = PrecisionFormat(precision_str)
        else:
            self._precision_format = precision
            precision_str = precision.value

        # Get dtype for working precision
        self._dtype: DTypeLike = get_dtype(self._precision_format)

        # Store FP64 reference for residual computation
        if A_fp64 is not None:
            self._A_fp64 = A_fp64
        else:
            self._A_fp64 = A.astype(np.float64)

        # Convert to working precision
        if A.dtype == self._dtype:
            self._A = A
        else:
            self._A = self._A_fp64.astype(self._dtype)

        # Initialize ping-pong buffer (Nx2, column-major for BLAS)
        self._n = A.shape[0]
        self._vectors = np.zeros((self._n, 2), dtype=self._dtype, order="F")
        self._current_idx = 0

        # Initialize starting vector
        if initial_vector is not None:
            self._vectors[:, 0] = initial_vector.astype(self._dtype)
            norm = np.linalg.norm(self._vectors[:, 0])
            if norm > 0:
                self._vectors[:, 0] /= norm
        else:
            rng = np.random.default_rng()
            vec = rng.standard_normal(self._n)
            self._vectors[:, 0] = vec.astype(self._dtype)
            norm = np.linalg.norm(self._vectors[:, 0])
            self._vectors[:, 0] /= norm

        # Precision-aware tolerances
        # Normalize FP8 variants for tolerance lookup
        tol_key = self._precision_format
        if precision_str.startswith("fp8"):
            tol_key = PrecisionFormat.FP8_E4M3  # Use E4M3 tolerances for all FP8

        self._eigenvalue_tol = get_tolerance(tol_key, "eigenvalue_tol")
        self._residual_tol = get_tolerance(tol_key, "residual_tol")

    def iterate(self) -> IterationResult:
        """Execute single power iteration with self-timing.

        Algorithm:
            1. Matrix-vector multiply: y = A @ x
            2. Normalize: x_new = y / ||y||
            3. Rayleigh quotient: λ = x_new^T @ A @ x_new

        Returns:
            IterationResult with eigenvalue and timing.
        """
        start = time.perf_counter()

        # Ping-pong: alternate between vectors
        next_idx = 1 - self._current_idx
        current_vec = self._vectors[:, self._current_idx]
        next_vec = self._vectors[:, next_idx]

        # Matrix-vector multiply (in-place write)
        next_vec[:] = self._A @ current_vec

        # Normalize (in-place)
        norm = np.linalg.norm(next_vec)

        # Check for numerical breakdown
        if norm < 1e-10:
            return IterationResult(
                eigenvalue=float("nan"),
                algorithm_time=time.perf_counter() - start,
            )

        next_vec[:] /= norm

        # Rayleigh quotient (reuse current_vec as temp buffer)
        current_vec[:] = self._A @ next_vec
        eigenvalue = float(next_vec @ current_vec)

        algorithm_time = time.perf_counter() - start

        # Update current index
        self._current_idx = next_idx

        return IterationResult(
            eigenvalue=eigenvalue,
            algorithm_time=algorithm_time,
        )

    def check_convergence(
        self,
        eigenvalue: float,
        true_eigenvalue: float,
        *,
        target_error: float | None = None,
    ) -> ConvergenceResult:
        """Check convergence using FP64 reference matrix.

        Implements precision-aware convergence criteria:
        - Normalized residual: ||A*x - λ*x|| / (||A|| * ||x||) < tol
        - Relative error: |λ - λ_true| / |λ_true| < tol

        Args:
            eigenvalue: Current eigenvalue estimate.
            true_eigenvalue: Reference eigenvalue from FP64 eigensolver.
            target_error: Optional user-specified threshold (overrides defaults).

        Returns:
            ConvergenceResult with metrics and convergence status.
        """
        start = time.perf_counter()

        current_vec = self._vectors[:, self._current_idx]

        # Compute residual in FP64 for accuracy
        x_fp64 = current_vec.astype(np.float64)
        eigenvalue_fp64 = float(eigenvalue)

        # Residual: r = A*x - λ*x
        residual = self._A_fp64 @ x_fp64 - eigenvalue_fp64 * x_fp64
        residual_norm = np.linalg.norm(residual)

        # Normalized residual: ||r|| / (||A|| * ||x||)
        A_norm = abs(eigenvalue_fp64)
        x_norm = np.linalg.norm(x_fp64)

        if A_norm < 1e-14 or x_norm < 1e-14:
            normalized_residual = float("inf")
        else:
            normalized_residual = residual_norm / (A_norm * x_norm)

        # Relative error
        relative_error = abs(eigenvalue - true_eigenvalue) / abs(true_eigenvalue)

        # Convergence checks
        if target_error is not None:
            if target_error == float("inf"):
                eigenvalue_converged = False
                residual_converged = False
            else:
                eigenvalue_converged = relative_error < target_error
                residual_converged = normalized_residual < target_error
        else:
            eigenvalue_converged = relative_error < self._eigenvalue_tol
            residual_converged = normalized_residual < self._residual_tol

        check_time = time.perf_counter() - start

        return ConvergenceResult(
            residual_norm=normalized_residual,
            relative_error=relative_error,
            eigenvalue_converged=eigenvalue_converged,
            residual_converged=residual_converged,
            check_time=check_time,
        )

    @property
    def current_vector(self) -> NDArray[np.floating]:
        """Return view of current eigenvector (no copy)."""
        return self._vectors[:, self._current_idx]

    @property
    def vector_norm(self) -> float:
        """Get norm of current eigenvector (should be ~1.0)."""
        return float(np.linalg.norm(self._vectors[:, self._current_idx]))

    def set_initial_vector(self, x: NDArray[np.floating]) -> None:
        """Set initial eigenvector (for cascading state transfer).

        Used when transitioning between precision levels.

        Args:
            x: Initial vector (any precision, will be converted and normalized).
        """
        self._current_idx = 0
        self._vectors[:, 0] = x.astype(self._dtype)
        norm = np.linalg.norm(self._vectors[:, 0])
        if norm > 1e-10:
            self._vectors[:, 0] /= norm


@dataclass(frozen=True, slots=True)
class PowerMethodTrace:
    """Complete trace of power method execution."""

    iterations: int
    """Number of iterations performed."""

    final_eigenvalue: float
    """Final eigenvalue estimate."""

    final_error: float
    """Final relative error."""

    final_residual: float
    """Final normalized residual."""

    converged: bool
    """Whether convergence criteria were met."""

    total_time: float
    """Total execution time (seconds)."""

    history: list[dict]
    """Per-iteration metrics."""


def run_power_method(
    A: NDArray[np.floating],
    precision: PrecisionFormat | str,
    true_eigenvalue: float,
    *,
    max_iterations: int = 1000,
    target_error: float | None = None,
    initial_vector: NDArray[np.floating] | None = None,
) -> PowerMethodTrace:
    """Run power method to convergence.

    Convenience function that handles iteration loop and tracking.

    Args:
        A: Input matrix.
        precision: Target precision format.
        true_eigenvalue: Reference eigenvalue for error computation.
        max_iterations: Maximum iterations.
        target_error: Optional convergence threshold.
        initial_vector: Optional starting vector.

    Returns:
        PowerMethodTrace with complete execution history.
    """
    engine = PowerIteration(A, precision, initial_vector=initial_vector)

    history: list[dict] = []
    start_time = time.perf_counter()
    cumulative_algo_time = 0.0

    for iteration in range(max_iterations):
        # Iterate
        iter_result = engine.iterate()

        if np.isnan(iter_result.eigenvalue):
            break

        cumulative_algo_time += iter_result.algorithm_time

        # Check convergence
        conv_result = engine.check_convergence(
            iter_result.eigenvalue,
            true_eigenvalue,
            target_error=target_error,
        )

        # Record history
        history.append(
            {
                "iteration": iteration,
                "eigenvalue": iter_result.eigenvalue,
                "relative_error": conv_result.relative_error,
                "residual_norm": conv_result.residual_norm,
                "algorithm_time": iter_result.algorithm_time,
                "cumulative_algorithm_time": cumulative_algo_time,
            }
        )

        # Check convergence
        if conv_result.eigenvalue_converged and conv_result.residual_converged:
            break

    total_time = time.perf_counter() - start_time
    final = history[-1] if history else {}

    return PowerMethodTrace(
        iterations=len(history),
        final_eigenvalue=final.get("eigenvalue", float("nan")),
        final_error=final.get("relative_error", float("nan")),
        final_residual=final.get("residual_norm", float("nan")),
        converged=conv_result.eigenvalue_converged and conv_result.residual_converged
        if history
        else False,
        total_time=total_time,
        history=history,
    )


__all__ = [
    "IterationResult",
    "ConvergenceResult",
    "PowerIteration",
    "PowerMethodTrace",
    "run_power_method",
]
