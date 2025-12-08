"""Tests for matrix generation utilities."""

import numpy as np
import pytest

from precision_lab.algorithms.matrices import (
    DEFAULT_SEED,
    ExperimentMatrix,
    MatrixFingerprint,
    compute_fingerprint,
    create_experiment_matrix,
    create_geometric_spectrum_matrix,
    create_linear_spectrum_matrix,
    create_slow_convergence_matrix,
)


class TestCreateLinearSpectrumMatrix:
    """Tests for create_linear_spectrum_matrix function."""

    def test_creates_correct_shape(self) -> None:
        """Matrix should be n×n."""
        n = 50
        A = create_linear_spectrum_matrix(n, condition_number=100, seed=42)
        assert A.shape == (n, n)

    def test_symmetric(self) -> None:
        """Matrix should be symmetric."""
        A = create_linear_spectrum_matrix(50, condition_number=100, seed=42)
        assert np.allclose(A, A.T)

    def test_positive_definite(self) -> None:
        """Matrix should be positive definite (all eigenvalues > 0)."""
        A = create_linear_spectrum_matrix(50, condition_number=100, seed=42)
        eigenvalues = np.linalg.eigvalsh(A)
        assert np.all(eigenvalues > 0)

    def test_condition_number(self) -> None:
        """Condition number should match specified value."""
        kappa = 100.0
        A = create_linear_spectrum_matrix(50, condition_number=kappa, seed=42)
        eigenvalues = np.linalg.eigvalsh(A)
        actual_kappa = eigenvalues.max() / eigenvalues.min()
        assert np.isclose(actual_kappa, kappa, rtol=1e-10)

    def test_reproducibility(self) -> None:
        """Same seed should produce identical matrix."""
        A1 = create_linear_spectrum_matrix(50, condition_number=100, seed=42)
        A2 = create_linear_spectrum_matrix(50, condition_number=100, seed=42)
        assert np.allclose(A1, A2)

    def test_different_seeds_produce_different_matrices(self) -> None:
        """Different seeds should produce different matrices."""
        A1 = create_linear_spectrum_matrix(50, condition_number=100, seed=42)
        A2 = create_linear_spectrum_matrix(50, condition_number=100, seed=43)
        assert not np.allclose(A1, A2)


class TestCreateSlowConvergenceMatrix:
    """Tests for create_slow_convergence_matrix function."""

    def test_creates_correct_shape(self) -> None:
        """Matrix should be n×n."""
        n = 50
        A = create_slow_convergence_matrix(n, condition_number=100, seed=42)
        assert A.shape == (n, n)

    def test_symmetric(self) -> None:
        """Matrix should be symmetric."""
        A = create_slow_convergence_matrix(50, condition_number=100, seed=42)
        assert np.allclose(A, A.T)

    def test_eigenvalue_gap(self) -> None:
        """Should have small gap between dominant eigenvalues."""
        A = create_slow_convergence_matrix(
            50, condition_number=100, eigenvalue_gap=1.1, seed=42
        )
        eigenvalues = np.sort(np.linalg.eigvalsh(A))[::-1]
        gap = eigenvalues[0] / eigenvalues[1]
        assert np.isclose(gap, 1.1, rtol=1e-10)

    def test_custom_eigenvalue_gap(self) -> None:
        """Custom eigenvalue gap should be respected."""
        gap = 1.2
        A = create_slow_convergence_matrix(
            50, condition_number=100, eigenvalue_gap=gap, seed=42
        )
        eigenvalues = np.sort(np.linalg.eigvalsh(A))[::-1]
        actual_gap = eigenvalues[0] / eigenvalues[1]
        assert np.isclose(actual_gap, gap, rtol=1e-10)


class TestCreateGeometricSpectrumMatrix:
    """Tests for create_geometric_spectrum_matrix function."""

    def test_creates_correct_shape(self) -> None:
        """Matrix should be n×n."""
        n = 50
        A = create_geometric_spectrum_matrix(n, condition_number=100, seed=42)
        assert A.shape == (n, n)

    def test_symmetric(self) -> None:
        """Matrix should be symmetric."""
        A = create_geometric_spectrum_matrix(50, condition_number=100, seed=42)
        assert np.allclose(A, A.T)

    def test_condition_number(self) -> None:
        """Condition number should match specified value."""
        kappa = 100.0
        A = create_geometric_spectrum_matrix(50, condition_number=kappa, seed=42)
        eigenvalues = np.linalg.eigvalsh(A)
        actual_kappa = eigenvalues.max() / eigenvalues.min()
        assert np.isclose(actual_kappa, kappa, rtol=1e-10)


class TestComputeFingerprint:
    """Tests for compute_fingerprint function."""

    def test_returns_fingerprint(self) -> None:
        """Should return MatrixFingerprint instance."""
        A = create_linear_spectrum_matrix(50, condition_number=100, seed=42)
        fp = compute_fingerprint(A, seed=42, convergence_type="linear")
        assert isinstance(fp, MatrixFingerprint)

    def test_fingerprint_fields(self) -> None:
        """Fingerprint should have all required fields."""
        A = create_linear_spectrum_matrix(50, condition_number=100, seed=42)
        fp = compute_fingerprint(A, seed=42, convergence_type="linear")

        assert len(fp.eigenvalue_signature) == 5
        assert fp.eigenvalue_ratio > 0
        assert fp.eigenvalue_ratio <= 1  # λ₂/λ₁ ≤ 1
        assert fp.condition_number > 0
        assert fp.matrix_size == 50
        assert fp.frobenius_norm > 0
        assert fp.seed == 42
        assert fp.convergence_type == "linear"

    def test_fingerprint_to_dict(self) -> None:
        """to_dict() should return serializable dictionary."""
        A = create_linear_spectrum_matrix(50, condition_number=100, seed=42)
        fp = compute_fingerprint(A, seed=42, convergence_type="linear")
        d = fp.to_dict()

        assert isinstance(d, dict)
        assert "eigenvalue_signature" in d
        assert "eigenvalue_ratio_lambda2_lambda1" in d
        assert "condition_number_actual" in d
        assert "matrix_size" in d

    def test_identical_matrices_same_fingerprint(self) -> None:
        """Identical matrices should have identical fingerprints."""
        A1 = create_linear_spectrum_matrix(50, condition_number=100, seed=42)
        A2 = create_linear_spectrum_matrix(50, condition_number=100, seed=42)
        fp1 = compute_fingerprint(A1, seed=42, convergence_type="linear")
        fp2 = compute_fingerprint(A2, seed=42, convergence_type="linear")

        assert fp1.eigenvalue_signature == fp2.eigenvalue_signature
        assert np.isclose(fp1.eigenvalue_ratio, fp2.eigenvalue_ratio)
        assert np.isclose(fp1.frobenius_norm, fp2.frobenius_norm)


class TestCreateExperimentMatrix:
    """Tests for create_experiment_matrix function."""

    def test_returns_experiment_matrix(self) -> None:
        """Should return ExperimentMatrix instance."""
        exp = create_experiment_matrix(50, 100)
        assert isinstance(exp, ExperimentMatrix)

    def test_experiment_matrix_fields(self) -> None:
        """ExperimentMatrix should have all required fields."""
        exp = create_experiment_matrix(50, 100)

        assert exp.matrix.shape == (50, 50)
        assert isinstance(exp.fingerprint, MatrixFingerprint)
        assert exp.initial_vector.shape == (50,)
        assert isinstance(exp.true_eigenvalue, float)

    def test_initial_vector_normalized(self) -> None:
        """Initial vector should be normalized."""
        exp = create_experiment_matrix(50, 100)
        norm = np.linalg.norm(exp.initial_vector)
        assert np.isclose(norm, 1.0)

    def test_true_eigenvalue_is_largest(self) -> None:
        """True eigenvalue should be the largest eigenvalue."""
        exp = create_experiment_matrix(50, 100)
        eigenvalues = np.linalg.eigvalsh(exp.matrix)
        assert np.isclose(exp.true_eigenvalue, eigenvalues.max())

    @pytest.mark.parametrize("conv_type", ["slow", "linear", "geometric"])
    def test_convergence_types(self, conv_type: str) -> None:
        """All convergence types should work."""
        exp = create_experiment_matrix(50, 100, convergence_type=conv_type)
        assert exp.fingerprint.convergence_type == conv_type

    def test_unknown_convergence_type_raises(self) -> None:
        """Unknown convergence type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown convergence_type"):
            create_experiment_matrix(50, 100, convergence_type="invalid")

    def test_default_seed(self) -> None:
        """Default seed should be DEFAULT_SEED."""
        exp = create_experiment_matrix(50, 100)
        assert exp.fingerprint.seed == DEFAULT_SEED

    def test_reproducibility(self) -> None:
        """Same parameters should produce identical results."""
        exp1 = create_experiment_matrix(50, 100, seed=42)
        exp2 = create_experiment_matrix(50, 100, seed=42)

        assert np.allclose(exp1.matrix, exp2.matrix)
        assert np.allclose(exp1.initial_vector, exp2.initial_vector)
        assert exp1.true_eigenvalue == exp2.true_eigenvalue
