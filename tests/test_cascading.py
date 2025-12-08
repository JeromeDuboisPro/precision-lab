"""Tests for cascading precision algorithm."""

import numpy as np
import pytest

from precision_lab.algorithms.cascading import (
    DEFAULT_PRECISION_CASCADE,
    CascadeTrace,
    CascadingPowerMethod,
    PrecisionConfig,
    SegmentResult,
)
from precision_lab.algorithms.plateau_detection import (
    RelativeImprovementDetector,
    ThresholdDetector,
)
from precision_lab.data.precision_types import PrecisionFormat


class TestPrecisionConfig:
    """Tests for PrecisionConfig dataclass."""

    def test_immutable(self) -> None:
        """PrecisionConfig should be immutable."""
        config = PrecisionConfig(
            name="FP16",
            format=PrecisionFormat.FP16,
            bytes_per_element=2,
            time_speedup_h100=4.0,
            iteration_speedup_h100=4.0,
        )
        with pytest.raises(AttributeError):
            config.name = "FP32"  # type: ignore[misc]

    def test_slots(self) -> None:
        """PrecisionConfig should use slots (no __dict__)."""
        config = PrecisionConfig(
            name="FP16",
            format=PrecisionFormat.FP16,
            bytes_per_element=2,
            time_speedup_h100=4.0,
            iteration_speedup_h100=4.0,
        )
        assert not hasattr(config, "__dict__")


class TestDefaultPrecisionCascade:
    """Tests for DEFAULT_PRECISION_CASCADE."""

    def test_has_four_precisions(self) -> None:
        """Should have FP8, FP16, FP32, FP64."""
        assert len(DEFAULT_PRECISION_CASCADE) == 4

    def test_ordered_by_precision(self) -> None:
        """Should be ordered from lowest to highest precision."""
        names = [c.name for c in DEFAULT_PRECISION_CASCADE]
        assert names == ["FP8", "FP16", "FP32", "FP64"]

    def test_speedup_factors_decrease(self) -> None:
        """Speedup factors should generally decrease with precision."""
        speedups = [c.iteration_speedup_h100 for c in DEFAULT_PRECISION_CASCADE]
        # FP8 should have highest speedup
        assert speedups[0] >= speedups[-1]

    def test_bytes_increase(self) -> None:
        """Bytes per element should increase with precision."""
        bytes_list = [c.bytes_per_element for c in DEFAULT_PRECISION_CASCADE]
        assert bytes_list == [1, 2, 4, 8]


class TestSegmentResult:
    """Tests for SegmentResult dataclass."""

    def test_immutable(self) -> None:
        """SegmentResult should be immutable."""
        result = SegmentResult(
            precision="FP16",
            iterations=100,
            effective_iterations=25.0,
            start_iteration=0,
            end_iteration=100,
            start_residual=1.0,
            end_residual=0.01,
            start_error=0.5,
            end_error=0.001,
            segment_time=1.0,
            converged=True,
            plateau_score=None,
        )
        with pytest.raises(AttributeError):
            result.iterations = 50  # type: ignore[misc]


class TestCascadeTrace:
    """Tests for CascadeTrace dataclass."""

    def test_immutable(self) -> None:
        """CascadeTrace should be immutable."""
        trace = CascadeTrace(
            iterations=100,
            effective_iterations=50.0,
            final_eigenvalue=5.0,
            final_error=1e-6,
            final_residual=1e-8,
            converged=True,
            total_time=1.0,
            segments=(),
            history=(),
        )
        with pytest.raises(AttributeError):
            trace.iterations = 50  # type: ignore[misc]


class TestCascadingPowerMethod:
    """Tests for CascadingPowerMethod class."""

    def test_initialization(self) -> None:
        """Should initialize with default parameters."""
        cascading = CascadingPowerMethod(matrix_size=100, condition_number=100)
        assert cascading.matrix_size == 100
        assert cascading.condition_number == 100.0
        assert cascading.seed == 42

    def test_true_eigenvalue_property(self) -> None:
        """true_eigenvalue should return ground truth."""
        cascading = CascadingPowerMethod(matrix_size=100, condition_number=100)
        assert cascading.true_eigenvalue > 0

    def test_matrix_fingerprint_property(self) -> None:
        """matrix_fingerprint should return dictionary."""
        cascading = CascadingPowerMethod(matrix_size=100, condition_number=100)
        fp = cascading.matrix_fingerprint
        assert isinstance(fp, dict)
        assert "eigenvalue_signature" in fp

    def test_run_returns_trace(self) -> None:
        """run() should return CascadeTrace."""
        cascading = CascadingPowerMethod(matrix_size=100, condition_number=100)
        trace = cascading.run(target_residual=1e-2, max_effective_iterations=100)
        assert isinstance(trace, CascadeTrace)

    def test_run_has_iterations(self) -> None:
        """Trace should have iteration count."""
        cascading = CascadingPowerMethod(matrix_size=100, condition_number=100)
        trace = cascading.run(target_residual=1e-2, max_effective_iterations=100)
        assert trace.iterations > 0

    def test_run_has_segments(self) -> None:
        """Trace should have segment results."""
        cascading = CascadingPowerMethod(matrix_size=100, condition_number=100)
        trace = cascading.run(target_residual=1e-2, max_effective_iterations=100)
        assert len(trace.segments) > 0

    def test_run_has_history(self) -> None:
        """Trace should have per-iteration history."""
        cascading = CascadingPowerMethod(matrix_size=100, condition_number=100)
        trace = cascading.run(target_residual=1e-2, max_effective_iterations=100)
        assert len(trace.history) == trace.iterations
        if trace.history:
            assert "eigenvalue" in trace.history[0]
            assert "residual_norm" in trace.history[0]

    def test_effective_iterations_weighted(self) -> None:
        """Effective iterations should account for speedup factors."""
        cascading = CascadingPowerMethod(matrix_size=100, condition_number=100)
        trace = cascading.run(target_residual=1e-2, max_effective_iterations=100)
        # Effective iterations should be less than or equal to actual iterations
        # (because lower precision iterations count for less)
        assert trace.effective_iterations <= trace.iterations

    def test_segments_add_up(self) -> None:
        """Segment iterations should sum to total iterations."""
        cascading = CascadingPowerMethod(matrix_size=100, condition_number=100)
        trace = cascading.run(target_residual=1e-2, max_effective_iterations=100)
        segment_sum = sum(s.iterations for s in trace.segments)
        assert segment_sum == trace.iterations

    def test_convergence_with_sufficient_iterations(self) -> None:
        """Should converge with sufficient iterations."""
        cascading = CascadingPowerMethod(
            matrix_size=100,
            condition_number=100,
            convergence_type="linear",  # Faster convergence
        )
        trace = cascading.run(target_residual=1e-4, max_effective_iterations=5000)
        assert trace.converged

    def test_respects_max_effective_iterations(self) -> None:
        """Should stop at max_effective_iterations."""
        cascading = CascadingPowerMethod(matrix_size=100, condition_number=100)
        trace = cascading.run(target_residual=1e-15, max_effective_iterations=10)
        assert trace.effective_iterations <= 10

    def test_final_eigenvalue_reasonable(self) -> None:
        """Final eigenvalue should be close to true value."""
        cascading = CascadingPowerMethod(
            matrix_size=100,
            condition_number=100,
            convergence_type="linear",
        )
        trace = cascading.run(target_residual=1e-4, max_effective_iterations=5000)
        if trace.converged:
            rel_error = abs(trace.final_eigenvalue - cascading.true_eigenvalue) / abs(
                cascading.true_eigenvalue
            )
            assert rel_error < 1e-3

    def test_total_time_positive(self) -> None:
        """Total time should be positive."""
        cascading = CascadingPowerMethod(matrix_size=100, condition_number=100)
        trace = cascading.run(target_residual=1e-2, max_effective_iterations=100)
        assert trace.total_time > 0

    def test_custom_plateau_detector(self) -> None:
        """Should accept custom plateau detector."""
        detector = RelativeImprovementDetector(window_size=10)
        cascading = CascadingPowerMethod(
            matrix_size=100, condition_number=100, plateau_detector=detector
        )
        trace = cascading.run(target_residual=1e-2, max_effective_iterations=100)
        assert trace.iterations > 0

    def test_threshold_detector(self) -> None:
        """Should work with ThresholdDetector."""
        detector = ThresholdDetector()
        cascading = CascadingPowerMethod(
            matrix_size=100, condition_number=100, plateau_detector=detector
        )
        trace = cascading.run(target_residual=1e-2, max_effective_iterations=100)
        assert trace.iterations > 0

    @pytest.mark.parametrize("conv_type", ["slow", "linear", "geometric"])
    def test_convergence_types(self, conv_type: str) -> None:
        """Should work with different convergence types."""
        cascading = CascadingPowerMethod(
            matrix_size=100, condition_number=100, convergence_type=conv_type
        )
        trace = cascading.run(target_residual=1e-2, max_effective_iterations=100)
        assert trace.iterations > 0

    def test_reproducibility(self) -> None:
        """Same parameters should produce identical results."""
        cascading1 = CascadingPowerMethod(
            matrix_size=100, condition_number=100, seed=42
        )
        cascading2 = CascadingPowerMethod(
            matrix_size=100, condition_number=100, seed=42
        )

        trace1 = cascading1.run(target_residual=1e-3, max_effective_iterations=100)
        trace2 = cascading2.run(target_residual=1e-3, max_effective_iterations=100)

        assert trace1.iterations == trace2.iterations
        assert np.isclose(trace1.final_eigenvalue, trace2.final_eigenvalue)

    def test_to_dict_returns_dict(self) -> None:
        """to_dict should return serializable dictionary."""
        cascading = CascadingPowerMethod(matrix_size=100, condition_number=100)
        trace = cascading.run(target_residual=1e-2, max_effective_iterations=100)
        d = cascading.to_dict(trace)

        assert isinstance(d, dict)
        assert "metadata" in d
        assert "summary" in d
        assert "segments" in d
        assert "trace" in d

    def test_to_dict_metadata(self) -> None:
        """to_dict metadata should have key information."""
        cascading = CascadingPowerMethod(matrix_size=100, condition_number=100)
        trace = cascading.run(target_residual=1e-2, max_effective_iterations=100)
        d = cascading.to_dict(trace)

        metadata = d["metadata"]
        assert metadata["algorithm"] == "cascading_precision"
        assert metadata["matrix_size"] == 100
        assert metadata["condition_number"] == 100.0
        assert "plateau_detector" in metadata
        assert "matrix_fingerprint" in metadata

    def test_to_dict_summary(self) -> None:
        """to_dict summary should have execution statistics."""
        cascading = CascadingPowerMethod(matrix_size=100, condition_number=100)
        trace = cascading.run(target_residual=1e-2, max_effective_iterations=100)
        d = cascading.to_dict(trace)

        summary = d["summary"]
        assert "total_iterations" in summary
        assert "effective_iterations" in summary
        assert "total_time_seconds" in summary
        assert "precision_levels_used" in summary


class TestCascadingPrecisionProgression:
    """Tests for precision cascade progression."""

    def test_starts_with_fp8(self) -> None:
        """First segment should be FP8."""
        cascading = CascadingPowerMethod(matrix_size=100, condition_number=100)
        trace = cascading.run(target_residual=1e-2, max_effective_iterations=100)
        if trace.segments:
            assert trace.segments[0].precision == "FP8"

    def test_history_has_precision_field(self) -> None:
        """History entries should include precision."""
        cascading = CascadingPowerMethod(matrix_size=100, condition_number=100)
        trace = cascading.run(target_residual=1e-2, max_effective_iterations=100)
        if trace.history:
            assert "precision" in trace.history[0]

    def test_precision_transitions(self) -> None:
        """Should transition through precisions."""
        cascading = CascadingPowerMethod(
            matrix_size=100,
            condition_number=100,
            convergence_type="slow",  # More likely to use multiple precisions
        )
        trace = cascading.run(target_residual=1e-6, max_effective_iterations=1000)

        # With enough iterations and tight target, should use multiple precisions
        if len(trace.segments) > 1:
            precisions = [s.precision for s in trace.segments]
            # Each precision should appear at most once (no going back)
            assert len(precisions) == len(set(precisions))

    def test_segments_have_correct_iteration_ranges(self) -> None:
        """Segment start/end iterations should be consecutive."""
        cascading = CascadingPowerMethod(matrix_size=100, condition_number=100)
        trace = cascading.run(target_residual=1e-2, max_effective_iterations=100)

        for i, segment in enumerate(trace.segments):
            if i == 0:
                assert segment.start_iteration == 0
            else:
                assert segment.start_iteration == trace.segments[i - 1].end_iteration

            assert segment.end_iteration == segment.start_iteration + segment.iterations
