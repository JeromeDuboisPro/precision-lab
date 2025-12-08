"""Matrix generation utilities for precision experiments.

This module provides functions for creating symmetric positive definite (SPD)
matrices with controlled eigenvalue distributions for numerical experiments.

Key Features:
- Reproducible matrix generation with seed control
- Multiple eigenvalue spectrum types (linear, slow convergence, geometric)
- Matrix fingerprinting for experiment verification

References:
- Golub & Van Loan: "Matrix Computations" (4th ed.), Section 7.3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


DEFAULT_SEED: int = 42
"""Default random seed for reproducible experiments."""


@dataclass(frozen=True, slots=True)
class MatrixFingerprint:
    """Fingerprint for matrix identification and verification.

    Used to verify that different experiments use identical matrices.
    """

    eigenvalue_signature: tuple[float, ...]
    """Top eigenvalues (sorted descending)."""

    eigenvalue_ratio: float
    """λ₂/λ₁ ratio - key convergence rate indicator."""

    condition_number: float
    """Actual κ(A) = λ_max/λ_min."""

    matrix_size: int
    """Matrix dimension n."""

    frobenius_norm: float
    """||A||_F for additional verification."""

    seed: int
    """Random seed used for generation."""

    convergence_type: str
    """Matrix type: 'slow', 'linear', or 'geometric'."""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "eigenvalue_signature": list(self.eigenvalue_signature),
            "eigenvalue_ratio_lambda2_lambda1": self.eigenvalue_ratio,
            "condition_number_actual": self.condition_number,
            "matrix_size": self.matrix_size,
            "frobenius_norm": self.frobenius_norm,
            "random_seed": self.seed,
            "convergence_type": self.convergence_type,
        }


def create_linear_spectrum_matrix(
    n: int,
    condition_number: float,
    *,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Create SPD matrix with linearly spaced eigenvalues.

    Eigenvalue distribution: [1.0, ..., κ] (linearly spaced)

    This produces FAST convergence for the power method due to
    large gaps between consecutive eigenvalues.

    Mathematical Construction:
        λ_i = 1 + (κ-1) * (i-1)/(n-1)  for i = 1, ..., n
        A = Q @ diag(λ) @ Q^T  where Q is random orthogonal

    Args:
        n: Matrix dimension.
        condition_number: Desired condition number κ = λ_max / λ_min.
        seed: Random seed for reproducibility.

    Returns:
        n×n symmetric positive definite matrix.

    Example:
        >>> A = create_linear_spectrum_matrix(100, condition_number=100, seed=42)
        >>> eigenvalues = np.linalg.eigvalsh(A)
        >>> print(f"κ = {eigenvalues[-1]/eigenvalues[0]:.2f}")
        κ = 100.00
    """
    rng = np.random.default_rng(seed)

    # Eigenvalues linearly spaced from 1 to κ
    eigenvalues = np.linspace(1.0, condition_number, n)

    # Random orthogonal matrix via QR decomposition
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))

    # Construct A = Q @ diag(λ) @ Q^T
    return Q @ np.diag(eigenvalues) @ Q.T


def create_slow_convergence_matrix(
    n: int,
    condition_number: float,
    *,
    eigenvalue_gap: float = 1.1,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Create SPD matrix with small gap between dominant eigenvalues.

    Eigenvalue distribution:
        λ₁ = κ (largest)
        λ₂ = κ / eigenvalue_gap (only ~10% smaller by default)
        λ₃...λₙ = geometric decay from λ₂ to 1.0

    This produces SLOW convergence for the power method, making it
    ideal for visualizing convergence behavior over many iterations.

    Convergence rate: (λ₂/λ₁)^k = (1/1.1)^k ≈ (0.909)^k

    Args:
        n: Matrix dimension.
        condition_number: Desired condition number κ = λ_max / λ_min.
        eigenvalue_gap: Ratio λ₁/λ₂ (default 1.1 for 10% gap).
        seed: Random seed for reproducibility.

    Returns:
        n×n symmetric positive definite matrix with slow convergence.

    Example:
        >>> A = create_slow_convergence_matrix(100, condition_number=100, seed=42)
        >>> eigenvalues = np.linalg.eigvalsh(A)
        >>> gap = eigenvalues[-1] / eigenvalues[-2]
        >>> print(f"λ₁/λ₂ = {gap:.3f}")
        λ₁/λ₂ = 1.100
    """
    rng = np.random.default_rng(seed)

    eigenvalues = np.zeros(n)
    eigenvalues[0] = condition_number
    eigenvalues[1] = condition_number / eigenvalue_gap

    if n > 2:
        remaining = np.geomspace(eigenvalues[1], 1.0, n - 1)
        eigenvalues[1:] = remaining

    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))

    return Q @ np.diag(eigenvalues) @ Q.T


def create_geometric_spectrum_matrix(
    n: int,
    condition_number: float,
    *,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Create SPD matrix with geometrically spaced eigenvalues.

    Eigenvalue distribution: [1.0, ..., κ] (geometrically spaced)

    Produces intermediate convergence rate (between linear and slow).

    Mathematical Construction:
        λ_i = κ^((i-1)/(n-1))  for i = 1, ..., n
        A = Q @ diag(λ) @ Q^T  where Q is random orthogonal

    Args:
        n: Matrix dimension.
        condition_number: Desired condition number κ = λ_max / λ_min.
        seed: Random seed for reproducibility.

    Returns:
        n×n symmetric positive definite matrix.
    """
    rng = np.random.default_rng(seed)

    eigenvalues = np.geomspace(1.0, condition_number, n)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))

    return Q @ np.diag(eigenvalues) @ Q.T


def compute_fingerprint(
    matrix: NDArray[np.float64],
    *,
    num_eigenvalues: int = 5,
    seed: int = DEFAULT_SEED,
    convergence_type: str = "unknown",
) -> MatrixFingerprint:
    """Compute fingerprint for matrix identification.

    Args:
        matrix: Input matrix.
        num_eigenvalues: Number of top eigenvalues to include.
        seed: Random seed used for generation.
        convergence_type: Matrix type identifier.

    Returns:
        MatrixFingerprint for verification.
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]

    return MatrixFingerprint(
        eigenvalue_signature=tuple(eigenvalues_sorted[:num_eigenvalues].tolist()),
        eigenvalue_ratio=float(eigenvalues_sorted[1] / eigenvalues_sorted[0]),
        condition_number=float(eigenvalues_sorted[0] / eigenvalues_sorted[-1]),
        matrix_size=int(matrix.shape[0]),
        frobenius_norm=float(np.linalg.norm(matrix, "fro")),
        seed=seed,
        convergence_type=convergence_type,
    )


@dataclass(frozen=True, slots=True)
class ExperimentSetup:
    """Container for complete experiment setup with metadata."""

    matrix: NDArray[np.float64]
    """The n×n SPD matrix."""

    fingerprint: MatrixFingerprint
    """Matrix fingerprint for verification."""

    initial_vector: NDArray[np.float64]
    """Normalized random starting vector."""

    true_eigenvalue: float
    """Largest eigenvalue (ground truth)."""


# Backwards compatibility alias
ExperimentMatrix = ExperimentSetup


def create_experiment(
    n: int,
    condition_number: float,
    *,
    seed: int = DEFAULT_SEED,
    convergence_type: str = "slow",
    eigenvalue_gap: float = 1.1,
) -> ExperimentSetup:
    """Create complete experiment setup: matrix, initial vector, and metadata.

    Uses a single RNG instance (PCG64 via default_rng) to generate both
    the matrix and initial vector in natural sequence. This is consistent
    with standalone matrix functions like create_slow_convergence_matrix().

    Args:
        n: Matrix dimension.
        condition_number: Desired condition number κ = λ_max / λ_min.
        seed: Random seed (default: 42 for reproducibility).
        convergence_type: "slow" (10% gap), "linear", or "geometric".
        eigenvalue_gap: For "slow" type, ratio λ₁/λ₂ (default 1.1).

    Returns:
        ExperimentSetup with matrix, fingerprint, initial vector, and true eigenvalue.

    Example:
        >>> experiment = create_experiment(100, condition_number=100.0)
        >>> print(f"True λ = {experiment.true_eigenvalue:.6f}")

    Note:
        For exact reproduction of precision-lens post 3 traces, use
        create_legacy_experiment() which matches the original RNG behavior.
    """
    # Single RNG instance using PCG64 (NumPy default)
    # Consistent with standalone matrix functions
    rng = np.random.default_rng(seed)

    # Build eigenvalue spectrum based on convergence type
    if convergence_type == "slow":
        eigenvalues = np.zeros(n)
        eigenvalues[0] = condition_number
        eigenvalues[1] = condition_number / eigenvalue_gap
        if n > 2:
            remaining = np.geomspace(eigenvalues[1], 1.0, n - 1)
            eigenvalues[1:] = remaining
    elif convergence_type == "linear":
        eigenvalues = np.linspace(1.0, condition_number, n)
    elif convergence_type == "geometric":
        eigenvalues = np.geomspace(1.0, condition_number, n)
    else:
        msg = f"Unknown convergence_type: {convergence_type}"
        raise ValueError(msg)

    # Random orthogonal matrix via QR decomposition
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))

    # Construct A = Q @ diag(λ) @ Q^T
    matrix = Q @ np.diag(eigenvalues) @ Q.T

    # Initial vector follows naturally in the RNG sequence
    initial_vector = rng.standard_normal(n)
    initial_vector = initial_vector / np.linalg.norm(initial_vector)

    fingerprint = compute_fingerprint(
        matrix, seed=seed, convergence_type=convergence_type
    )

    true_eigenvalue = float(np.max(np.linalg.eigvalsh(matrix)))

    return ExperimentSetup(
        matrix=matrix,
        fingerprint=fingerprint,
        initial_vector=initial_vector,
        true_eigenvalue=true_eigenvalue,
    )


# Backwards compatibility alias
create_experiment_matrix = create_experiment


def create_legacy_experiment(
    n: int,
    condition_number: float,
    *,
    seed: int = DEFAULT_SEED,
    convergence_type: str = "slow",
    eigenvalue_gap: float = 1.1,
) -> ExperimentSetup:
    """Create experiment setup matching precision-lens post 3 behavior exactly.

    This function reproduces the exact RNG behavior from precision-lens v3,
    enabling comparison with historical results and validation.

    Historical Context:
        precision-lens used a mixed RNG approach for experiment setup:
        - Matrix generation: PCG64 (np.random.default_rng)
        - Initial vector: MT19937 (np.random.seed) with skip pattern

        The skip pattern consumed n×n random calls before generating the
        initial vector, matching the number of calls used for matrix generation.

    Args:
        n: Matrix dimension.
        condition_number: Desired condition number κ = λ_max / λ_min.
        seed: Random seed (default: 42 for reproducibility).
        convergence_type: "slow" (10% gap), "linear", or "geometric".
        eigenvalue_gap: For "slow" type, ratio λ₁/λ₂ (default 1.1).

    Returns:
        ExperimentSetup matching precision-lens post 3 output exactly.

    Example:
        >>> # Reproduce precision-lens post 3 traces
        >>> legacy = create_legacy_experiment(10, condition_number=100.0)
        >>> modern = create_experiment(10, condition_number=100.0)
        >>> # legacy.matrix == modern.matrix (same PCG64 seed)
        >>> # legacy.initial_vector != modern.initial_vector (different RNG path)
    """
    # Matrix: PCG64 (same as standalone functions and create_experiment)
    rng = np.random.default_rng(seed)

    # Build eigenvalue spectrum based on convergence type
    if convergence_type == "slow":
        eigenvalues = np.zeros(n)
        eigenvalues[0] = condition_number
        eigenvalues[1] = condition_number / eigenvalue_gap
        if n > 2:
            remaining = np.geomspace(eigenvalues[1], 1.0, n - 1)
            eigenvalues[1:] = remaining
    elif convergence_type == "linear":
        eigenvalues = np.linspace(1.0, condition_number, n)
    elif convergence_type == "geometric":
        eigenvalues = np.geomspace(1.0, condition_number, n)
    else:
        msg = f"Unknown convergence_type: {convergence_type}"
        raise ValueError(msg)

    # Random orthogonal matrix via QR decomposition (PCG64)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    matrix = Q @ np.diag(eigenvalues) @ Q.T

    # Initial vector: MT19937 with legacy skip pattern
    # This matches precision-lens post 3 behavior exactly
    np.random.seed(seed)  # noqa: NPY002 - intentional legacy RNG for compatibility
    _ = np.random.randn(n * n)  # noqa: NPY002 - skip matrix generation calls
    initial_vector = np.random.randn(n)  # noqa: NPY002 - legacy RNG
    initial_vector = initial_vector / np.linalg.norm(initial_vector)

    fingerprint = compute_fingerprint(
        matrix, seed=seed, convergence_type=convergence_type
    )

    true_eigenvalue = float(np.max(np.linalg.eigvalsh(matrix)))

    return ExperimentSetup(
        matrix=matrix,
        fingerprint=fingerprint,
        initial_vector=initial_vector,
        true_eigenvalue=true_eigenvalue,
    )


__all__ = [
    "DEFAULT_SEED",
    "MatrixFingerprint",
    "ExperimentSetup",
    "ExperimentMatrix",  # Backwards compatibility alias
    "create_linear_spectrum_matrix",
    "create_slow_convergence_matrix",
    "create_geometric_spectrum_matrix",
    "compute_fingerprint",
    "create_experiment",
    "create_experiment_matrix",  # Backwards compatibility alias
    "create_legacy_experiment",
]
