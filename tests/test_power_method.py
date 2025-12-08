"""Tests for power method algorithm."""

import numpy as np
import pytest

from precision_lab.algorithms.matrices import create_experiment_matrix
from precision_lab.algorithms.power_method import (
    ConvergenceResult,
    IterationResult,
    PowerIteration,
    PowerMethodTrace,
    run_power_method,
)
from precision_lab.data.precision_types import PrecisionFormat


class TestIterationResult:
    """Tests for IterationResult dataclass."""

    def test_immutable(self) -> None:
        """IterationResult should be immutable."""
        result = IterationResult(eigenvalue=5.0, algorithm_time=0.001)
        with pytest.raises(AttributeError):
            result.eigenvalue = 6.0  # type: ignore[misc]

    def test_slots(self) -> None:
        """IterationResult should use slots (no __dict__)."""
        result = IterationResult(eigenvalue=5.0, algorithm_time=0.001)
        assert not hasattr(result, "__dict__")


class TestConvergenceResult:
    """Tests for ConvergenceResult dataclass."""

    def test_immutable(self) -> None:
        """ConvergenceResult should be immutable."""
        result = ConvergenceResult(
            residual_norm=1e-6,
            relative_error=1e-8,
            eigenvalue_converged=True,
            residual_converged=True,
            check_time=0.001,
        )
        with pytest.raises(AttributeError):
            result.residual_norm = 1e-7  # type: ignore[misc]


class TestPowerIteration:
    """Tests for PowerIteration class."""

    @pytest.fixture
    def experiment(self):
        """Create experiment matrix for testing."""
        return create_experiment_matrix(100, 100, seed=42)

    def test_initialization(self, experiment) -> None:
        """PowerIteration should initialize correctly."""
        engine = PowerIteration(experiment.matrix, "fp64")
        assert engine.vector_norm > 0

    def test_initialization_with_precision_format(self, experiment) -> None:
        """Should accept PrecisionFormat enum."""
        engine = PowerIteration(experiment.matrix, PrecisionFormat.FP64)
        assert engine.vector_norm > 0

    def test_iterate_returns_result(self, experiment) -> None:
        """iterate() should return IterationResult."""
        engine = PowerIteration(experiment.matrix, "fp64")
        result = engine.iterate()
        assert isinstance(result, IterationResult)

    def test_iterate_eigenvalue_positive(self, experiment) -> None:
        """Eigenvalue should be positive for SPD matrix."""
        engine = PowerIteration(experiment.matrix, "fp64")
        result = engine.iterate()
        assert result.eigenvalue > 0

    def test_iterate_has_timing(self, experiment) -> None:
        """iterate() should record algorithm time."""
        engine = PowerIteration(experiment.matrix, "fp64")
        result = engine.iterate()
        assert result.algorithm_time > 0

    def test_check_convergence_returns_result(self, experiment) -> None:
        """check_convergence() should return ConvergenceResult."""
        engine = PowerIteration(experiment.matrix, "fp64")
        iter_result = engine.iterate()
        conv_result = engine.check_convergence(
            iter_result.eigenvalue, experiment.true_eigenvalue
        )
        assert isinstance(conv_result, ConvergenceResult)

    def test_convergence_improves_over_iterations(self, experiment) -> None:
        """Error should generally decrease over iterations."""
        engine = PowerIteration(experiment.matrix, "fp64")

        errors = []
        for _ in range(50):
            iter_result = engine.iterate()
            conv_result = engine.check_convergence(
                iter_result.eigenvalue, experiment.true_eigenvalue
            )
            errors.append(conv_result.relative_error)

        # Final error should be much smaller than initial
        assert errors[-1] < errors[0] * 0.1

    def test_vector_norm_stays_normalized(self, experiment) -> None:
        """Eigenvector should stay approximately normalized."""
        engine = PowerIteration(experiment.matrix, "fp64")

        for _ in range(20):
            engine.iterate()

        assert np.isclose(engine.vector_norm, 1.0, rtol=1e-10)

    def test_current_vector_property(self, experiment) -> None:
        """current_vector should return correct shape."""
        engine = PowerIteration(experiment.matrix, "fp64")
        engine.iterate()
        assert engine.current_vector.shape == (100,)

    def test_set_initial_vector(self, experiment) -> None:
        """set_initial_vector should update state."""
        engine = PowerIteration(experiment.matrix, "fp64")

        # Set a specific initial vector
        new_vec = np.ones(100)
        engine.set_initial_vector(new_vec)

        # Should be normalized
        assert np.isclose(engine.vector_norm, 1.0, rtol=1e-10)

    @pytest.mark.parametrize("precision", ["fp64", "fp32", "fp16"])
    def test_different_precisions(self, experiment, precision: str) -> None:
        """Should work with different precisions."""
        engine = PowerIteration(experiment.matrix, precision)
        result = engine.iterate()
        assert not np.isnan(result.eigenvalue)


class TestRunPowerMethod:
    """Tests for run_power_method convenience function."""

    @pytest.fixture
    def experiment(self):
        """Create experiment matrix for testing."""
        return create_experiment_matrix(100, 100, seed=42)

    def test_returns_trace(self, experiment) -> None:
        """Should return PowerMethodTrace."""
        trace = run_power_method(
            experiment.matrix,
            "fp64",
            experiment.true_eigenvalue,
            max_iterations=50,
        )
        assert isinstance(trace, PowerMethodTrace)

    def test_trace_has_iterations(self, experiment) -> None:
        """Trace should record iterations."""
        trace = run_power_method(
            experiment.matrix,
            "fp64",
            experiment.true_eigenvalue,
            max_iterations=50,
        )
        assert trace.iterations > 0
        assert trace.iterations <= 50

    def test_trace_has_history(self, experiment) -> None:
        """Trace should have per-iteration history."""
        trace = run_power_method(
            experiment.matrix,
            "fp64",
            experiment.true_eigenvalue,
            max_iterations=50,
        )
        assert len(trace.history) == trace.iterations
        assert "eigenvalue" in trace.history[0]
        assert "relative_error" in trace.history[0]

    def test_converges_with_enough_iterations(self, experiment) -> None:
        """Should converge with sufficient iterations."""
        trace = run_power_method(
            experiment.matrix,
            "fp64",
            experiment.true_eigenvalue,
            max_iterations=500,
        )
        assert trace.converged

    def test_final_eigenvalue_close_to_true(self, experiment) -> None:
        """Final eigenvalue should be close to true value."""
        trace = run_power_method(
            experiment.matrix,
            "fp64",
            experiment.true_eigenvalue,
            max_iterations=500,
        )
        rel_error = abs(trace.final_eigenvalue - experiment.true_eigenvalue) / abs(
            experiment.true_eigenvalue
        )
        assert rel_error < 1e-10

    def test_respects_max_iterations(self, experiment) -> None:
        """Should stop at max_iterations if not converged."""
        trace = run_power_method(
            experiment.matrix,
            "fp64",
            experiment.true_eigenvalue,
            max_iterations=5,
        )
        assert trace.iterations == 5

    def test_total_time_positive(self, experiment) -> None:
        """Total time should be positive."""
        trace = run_power_method(
            experiment.matrix,
            "fp64",
            experiment.true_eigenvalue,
            max_iterations=50,
        )
        assert trace.total_time > 0

    def test_with_initial_vector(self, experiment) -> None:
        """Should accept custom initial vector."""
        initial = np.ones(100)
        trace = run_power_method(
            experiment.matrix,
            "fp64",
            experiment.true_eigenvalue,
            max_iterations=50,
            initial_vector=initial,
        )
        assert trace.iterations > 0

    def test_target_error_affects_convergence(self, experiment) -> None:
        """Target error should affect convergence criteria."""
        # With loose target, should converge faster
        trace_loose = run_power_method(
            experiment.matrix,
            "fp64",
            experiment.true_eigenvalue,
            max_iterations=500,
            target_error=1e-2,
        )

        # With tight target, needs more iterations
        trace_tight = run_power_method(
            experiment.matrix,
            "fp64",
            experiment.true_eigenvalue,
            max_iterations=500,
            target_error=1e-12,
        )

        # Loose target should converge in fewer iterations
        assert trace_loose.iterations <= trace_tight.iterations
