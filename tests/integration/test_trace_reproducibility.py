"""Integration tests for trace reproducibility.

These tests verify that the algorithms produce identical results across runs
by comparing against golden trace files. Any change to numerical behavior
will cause these tests to fail.

To regenerate golden files after intentional changes:
    python tests/integration/generate_golden.py
"""

import json
from pathlib import Path

import numpy as np
import pytest

from precision_lab.algorithms.cascading import CascadingPowerMethod
from precision_lab.algorithms.matrices import create_experiment_matrix
from precision_lab.algorithms.power_method import PowerIteration

# Test parameters - MUST match generate_golden.py
MATRIX_SIZE = 100
CONDITION_NUMBER = 100.0
SEED = 42
CONVERGENCE_TYPE = "linear"
MAX_ITERATIONS = 100

GOLDEN_DIR = Path(__file__).parent / "golden"

# Tolerance configuration per precision level
# Lower precision = looser tolerance due to quantization effects
TOLERANCES = {
    "fp64": {"eigenvalue": 1e-14, "residual": 1e-14, "error": 1e-14},
    "fp32": {"eigenvalue": 1e-6, "residual": 1e-6, "error": 1e-6},
    "fp16": {"eigenvalue": 1e-3, "residual": 1e-3, "error": 1e-3},
    "fp8_e4m3": {"eigenvalue": 1e-1, "residual": 1e-1, "error": 1e-1},
    "cascade": {"eigenvalue": 1e-10, "residual": 1e-10, "error": 1e-10},
}


def load_golden(filename: str) -> dict | list:
    """Load golden data from JSON file."""
    filepath = GOLDEN_DIR / filename
    if not filepath.exists():
        pytest.skip(f"Golden file not found: {filepath}. Run generate_golden.py first.")
    with open(filepath) as f:
        return json.load(f)


class TestSinglePrecisionReproducibility:
    """Test that single-precision traces are reproducible."""

    @pytest.fixture
    def experiment(self):
        """Create experiment matrix with fixed parameters."""
        return create_experiment_matrix(
            MATRIX_SIZE,
            CONDITION_NUMBER,
            seed=SEED,
            convergence_type=CONVERGENCE_TYPE,
        )

    def run_precision_trace(self, experiment, precision: str) -> list[dict]:
        """Run power method and collect trace."""
        engine = PowerIteration(
            experiment.matrix,
            precision,
            A_fp64=experiment.matrix,
        )
        engine.set_initial_vector(experiment.initial_vector.copy())

        trace = []
        for iteration in range(MAX_ITERATIONS):
            iter_result = engine.iterate()
            conv_result = engine.check_convergence(
                iter_result.eigenvalue,
                experiment.true_eigenvalue,
            )

            trace.append({
                "iteration": iteration,
                "eigenvalue": float(iter_result.eigenvalue),
                "relative_error": float(conv_result.relative_error),
                "residual_norm": float(conv_result.residual_norm),
                "vector_norm": float(engine.vector_norm),
            })

        return trace

    def compare_traces(
        self, actual: list[dict], expected: list[dict], precision: str
    ) -> None:
        """Compare actual trace against expected with appropriate tolerances."""
        tol = TOLERANCES[precision]

        assert len(actual) == len(expected), (
            f"Trace length mismatch: {len(actual)} vs {len(expected)}"
        )

        for i, (act, exp) in enumerate(zip(actual, expected)):
            assert act["iteration"] == exp["iteration"], f"Iteration mismatch at {i}"

            # Compare eigenvalue
            assert np.isclose(act["eigenvalue"], exp["eigenvalue"], rtol=tol["eigenvalue"]), (
                f"Eigenvalue mismatch at iteration {i}: "
                f"{act['eigenvalue']} vs {exp['eigenvalue']}"
            )

            # Compare residual norm
            assert np.isclose(act["residual_norm"], exp["residual_norm"], rtol=tol["residual"]), (
                f"Residual mismatch at iteration {i}: "
                f"{act['residual_norm']} vs {exp['residual_norm']}"
            )

            # Compare relative error
            assert np.isclose(act["relative_error"], exp["relative_error"], rtol=tol["error"]), (
                f"Error mismatch at iteration {i}: "
                f"{act['relative_error']} vs {exp['relative_error']}"
            )

    def test_fp64_reproducibility(self, experiment) -> None:
        """FP64 trace should exactly match golden file."""
        expected = load_golden("fp64_trace.json")
        actual = self.run_precision_trace(experiment, "fp64")
        self.compare_traces(actual, expected, "fp64")

    def test_fp32_reproducibility(self, experiment) -> None:
        """FP32 trace should match golden file within tolerance."""
        expected = load_golden("fp32_trace.json")
        actual = self.run_precision_trace(experiment, "fp32")
        self.compare_traces(actual, expected, "fp32")

    def test_fp16_reproducibility(self, experiment) -> None:
        """FP16 trace should match golden file within tolerance."""
        expected = load_golden("fp16_trace.json")
        actual = self.run_precision_trace(experiment, "fp16")
        self.compare_traces(actual, expected, "fp16")

    def test_fp8_reproducibility(self, experiment) -> None:
        """FP8 trace should match golden file within tolerance."""
        expected = load_golden("fp8_e4m3_trace.json")
        actual = self.run_precision_trace(experiment, "fp8_e4m3")
        self.compare_traces(actual, expected, "fp8_e4m3")

    def test_convergence_monotonic_fp64(self, experiment) -> None:
        """FP64 residual should generally decrease (with some noise)."""
        trace = self.run_precision_trace(experiment, "fp64")
        residuals = [t["residual_norm"] for t in trace]

        # Check that final residual is smaller than initial
        # With condition_number=100, convergence rate is ~0.99 per iteration
        # After 100 iterations: 0.99^100 ≈ 0.37, so expect ~60% reduction minimum
        assert residuals[-1] < residuals[0] * 0.5, (
            "FP64 should show convergence progress"
        )

    def test_eigenvalue_converges_to_true(self, experiment) -> None:
        """Eigenvalue should converge toward true value."""
        trace = self.run_precision_trace(experiment, "fp64")
        final_eigenvalue = trace[-1]["eigenvalue"]
        true_eigenvalue = experiment.true_eigenvalue

        rel_error = abs(final_eigenvalue - true_eigenvalue) / true_eigenvalue
        # With condition_number=100 and 100 iterations, expect ~1% relative error
        assert rel_error < 0.02, (
            f"Eigenvalue should converge: {final_eigenvalue} vs {true_eigenvalue}"
        )


class TestCascadeReproducibility:
    """Test that cascading precision traces are reproducible."""

    def test_cascade_metadata_matches(self) -> None:
        """Cascade metadata should match golden file."""
        expected = load_golden("cascade_trace.json")

        cascading = CascadingPowerMethod(
            matrix_size=MATRIX_SIZE,
            condition_number=CONDITION_NUMBER,
            seed=SEED,
            convergence_type=CONVERGENCE_TYPE,
        )

        trace = cascading.run(
            target_residual=1e-12,
            max_effective_iterations=500,
        )

        # Check metadata
        assert trace.iterations == expected["metadata"]["total_iterations"]
        assert np.isclose(
            cascading.true_eigenvalue,
            expected["metadata"]["true_eigenvalue"],
            rtol=1e-10,
        )

    def test_cascade_segments_match(self) -> None:
        """Cascade segment boundaries should match golden file."""
        expected = load_golden("cascade_trace.json")

        cascading = CascadingPowerMethod(
            matrix_size=MATRIX_SIZE,
            condition_number=CONDITION_NUMBER,
            seed=SEED,
            convergence_type=CONVERGENCE_TYPE,
        )

        trace = cascading.run(
            target_residual=1e-12,
            max_effective_iterations=500,
        )

        # Check segment count
        assert len(trace.segments) == len(expected["segments"]), (
            f"Segment count mismatch: {len(trace.segments)} vs {len(expected['segments'])}"
        )

        # Check each segment
        for i, (actual, exp) in enumerate(zip(trace.segments, expected["segments"])):
            assert actual.precision == exp["precision"], (
                f"Segment {i} precision mismatch: {actual.precision} vs {exp['precision']}"
            )
            assert actual.iterations == exp["iterations"], (
                f"Segment {i} iteration count mismatch: "
                f"{actual.iterations} vs {exp['iterations']}"
            )
            assert actual.start_iteration == exp["start_iteration"], (
                f"Segment {i} start mismatch"
            )
            assert actual.end_iteration == exp["end_iteration"], (
                f"Segment {i} end mismatch"
            )

    def test_cascade_trace_reproducibility(self) -> None:
        """Cascade trace iterations should match golden file."""
        expected = load_golden("cascade_trace.json")
        tol = TOLERANCES["cascade"]

        cascading = CascadingPowerMethod(
            matrix_size=MATRIX_SIZE,
            condition_number=CONDITION_NUMBER,
            seed=SEED,
            convergence_type=CONVERGENCE_TYPE,
        )

        trace = cascading.run(
            target_residual=1e-12,
            max_effective_iterations=500,
        )

        # Compare first 100 iterations
        actual_history = list(trace.history)[:MAX_ITERATIONS]
        expected_history = expected["trace"]

        assert len(actual_history) == len(expected_history), (
            f"History length mismatch: {len(actual_history)} vs {len(expected_history)}"
        )

        for i, (act, exp) in enumerate(zip(actual_history, expected_history)):
            assert act["iteration"] == exp["iteration"], f"Iteration mismatch at {i}"
            assert act["precision"] == exp["precision"], (
                f"Precision mismatch at iteration {i}: "
                f"{act['precision']} vs {exp['precision']}"
            )

            assert np.isclose(act["eigenvalue"], exp["eigenvalue"], rtol=tol["eigenvalue"]), (
                f"Eigenvalue mismatch at iteration {i}: "
                f"{act['eigenvalue']} vs {exp['eigenvalue']}"
            )

            assert np.isclose(act["residual_norm"], exp["residual_norm"], rtol=tol["residual"]), (
                f"Residual mismatch at iteration {i}: "
                f"{act['residual_norm']} vs {exp['residual_norm']}"
            )

    def test_cascade_precision_progression(self) -> None:
        """Cascade should progress through precisions in order."""
        expected = load_golden("cascade_trace.json")

        cascading = CascadingPowerMethod(
            matrix_size=MATRIX_SIZE,
            condition_number=CONDITION_NUMBER,
            seed=SEED,
            convergence_type=CONVERGENCE_TYPE,
        )

        trace = cascading.run(
            target_residual=1e-12,
            max_effective_iterations=500,
        )

        # Extract precision sequence
        precisions = [s.precision for s in trace.segments]
        expected_precisions = [s["precision"] for s in expected["segments"]]

        assert precisions == expected_precisions, (
            f"Precision sequence mismatch: {precisions} vs {expected_precisions}"
        )

        # Verify order is correct (no going back)
        valid_order = ["FP8", "FP16", "FP32", "FP64"]
        indices = [valid_order.index(p) for p in precisions]
        assert indices == sorted(indices), "Precisions should progress in order"

    def test_cascade_residual_improvement_per_segment(self) -> None:
        """Each segment should show residual improvement."""
        cascading = CascadingPowerMethod(
            matrix_size=MATRIX_SIZE,
            condition_number=CONDITION_NUMBER,
            seed=SEED,
            convergence_type=CONVERGENCE_TYPE,
        )

        trace = cascading.run(
            target_residual=1e-12,
            max_effective_iterations=500,
        )

        for segment in trace.segments:
            # End residual should be less than or equal to start
            # (allowing for some noise in low-precision segments)
            assert segment.end_residual <= segment.start_residual * 1.1, (
                f"{segment.precision} segment should not significantly increase residual: "
                f"{segment.start_residual} -> {segment.end_residual}"
            )


class TestTraceStatistics:
    """Statistical tests on trace behavior."""

    @pytest.fixture
    def experiment(self):
        """Create experiment matrix with fixed parameters."""
        return create_experiment_matrix(
            MATRIX_SIZE,
            CONDITION_NUMBER,
            seed=SEED,
            convergence_type=CONVERGENCE_TYPE,
        )

    def test_fp64_convergence_rate(self, experiment) -> None:
        """FP64 should show linear convergence in log scale."""
        engine = PowerIteration(experiment.matrix, "fp64", A_fp64=experiment.matrix)
        engine.set_initial_vector(experiment.initial_vector.copy())

        residuals = []
        for _ in range(MAX_ITERATIONS):
            iter_result = engine.iterate()
            conv_result = engine.check_convergence(
                iter_result.eigenvalue,
                experiment.true_eigenvalue,
            )
            residuals.append(conv_result.residual_norm)

        # Log residuals should decrease roughly linearly
        log_residuals = np.log10(residuals)
        # Check that later residuals are lower (at least 1 order of magnitude)
        # With condition_number=100, convergence is slow (~0.99 per iteration)
        assert log_residuals[-1] < log_residuals[0] - 1, (
            "Should achieve at least 1 order of magnitude improvement"
        )

    def test_precision_comparison(self, experiment) -> None:
        """Higher precision should achieve lower final residual."""
        final_residuals = {}

        for precision in ["fp64", "fp32", "fp16"]:
            engine = PowerIteration(
                experiment.matrix,
                precision,
                A_fp64=experiment.matrix,
            )
            engine.set_initial_vector(experiment.initial_vector.copy())

            for _ in range(MAX_ITERATIONS):
                iter_result = engine.iterate()
                conv_result = engine.check_convergence(
                    iter_result.eigenvalue,
                    experiment.true_eigenvalue,
                )

            final_residuals[precision] = conv_result.residual_norm

        # FP64 should be best or equal, FP32 second or equal, FP16 third
        # At convergence, higher precision may achieve same residual (limited by
        # convergence rate, not precision), so use <= rather than strict <
        assert final_residuals["fp64"] <= final_residuals["fp32"] * 1.01, (
            "FP64 should be at least as good as FP32"
        )
        assert final_residuals["fp32"] <= final_residuals["fp16"] * 1.01, (
            "FP32 should be at least as good as FP16"
        )
