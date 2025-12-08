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
class ExperimentMatrix:
    """Container for experiment matrix with metadata."""

    matrix: NDArray[np.float64]
    """The n×n SPD matrix."""

    fingerprint: MatrixFingerprint
    """Matrix fingerprint for verification."""

    initial_vector: NDArray[np.float64]
    """Normalized random starting vector."""

    true_eigenvalue: float
    """Largest eigenvalue (ground truth)."""


def create_experiment_matrix(
    n: int,
    condition_number: float,
    *,
    seed: int = DEFAULT_SEED,
    convergence_type: str = "slow",
) -> ExperimentMatrix:
    """Create matrix for experiments with full metadata.

    This is the canonical function for creating matrices in experiments.
    It returns both the matrix and metadata for verification.

    Args:
        n: Matrix dimension.
        condition_number: Desired condition number.
        seed: Random seed (default: 42 for reproducibility).
        convergence_type: "slow" (10% gap), "linear", or "geometric".

    Returns:
        ExperimentMatrix with matrix, fingerprint, and initial vector.

    Example:
        >>> exp = create_experiment_matrix(1024, 100)
        >>> print(f"True λ = {exp.true_eigenvalue:.6f}")
    """
    if convergence_type == "slow":
        matrix = create_slow_convergence_matrix(n, condition_number, seed=seed)
    elif convergence_type == "linear":
        matrix = create_linear_spectrum_matrix(n, condition_number, seed=seed)
    elif convergence_type == "geometric":
        matrix = create_geometric_spectrum_matrix(n, condition_number, seed=seed)
    else:
        msg = f"Unknown convergence_type: {convergence_type}"
        raise ValueError(msg)

    fingerprint = compute_fingerprint(
        matrix, seed=seed, convergence_type=convergence_type
    )

    # Create initial vector using same RNG sequence as matrix creation
    # This matches precision-lens behavior: re-seed, skip matrix Q calls, then generate
    np.random.seed(seed)
    _ = np.random.randn(n, n)  # Skip the random calls used for matrix Q
    initial_vector = np.random.randn(n)
    initial_vector = initial_vector / np.linalg.norm(initial_vector)

    true_eigenvalue = float(np.max(np.linalg.eigvalsh(matrix)))

    return ExperimentMatrix(
        matrix=matrix,
        fingerprint=fingerprint,
        initial_vector=initial_vector,
        true_eigenvalue=true_eigenvalue,
    )


__all__ = [
    "DEFAULT_SEED",
    "MatrixFingerprint",
    "ExperimentMatrix",
    "create_linear_spectrum_matrix",
    "create_slow_convergence_matrix",
    "create_geometric_spectrum_matrix",
    "compute_fingerprint",
    "create_experiment_matrix",
]
