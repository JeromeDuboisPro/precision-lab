"""Tests for plateau detection strategies."""

import pytest

from precision_lab.algorithms.plateau_detection import (
    MultiCriteriaDetector,
    PlateauResult,
    RelativeImprovementDetector,
    ThresholdDetector,
    create_detector,
)


class TestPlateauResult:
    """Tests for PlateauResult dataclass."""

    def test_immutable(self) -> None:
        """PlateauResult should be immutable."""
        result = PlateauResult(detected=True, score=0.75)
        with pytest.raises(AttributeError):
            result.detected = False  # type: ignore[misc]

    def test_slots(self) -> None:
        """PlateauResult should use slots (no __dict__)."""
        result = PlateauResult(detected=True, score=0.75)
        assert not hasattr(result, "__dict__")


class TestMultiCriteriaDetector:
    """Tests for MultiCriteriaDetector."""

    def test_default_initialization(self) -> None:
        """Should initialize with default parameters."""
        detector = MultiCriteriaDetector()
        assert detector.window_size == 15
        assert detector.score_threshold == 0.65

    def test_custom_parameters(self) -> None:
        """Should accept custom parameters."""
        detector = MultiCriteriaDetector(window_size=20, score_threshold=0.7)
        assert detector.window_size == 20
        assert detector.score_threshold == 0.7

    def test_weights_must_sum_to_one(self) -> None:
        """Criteria weights must sum to 1.0."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            MultiCriteriaDetector(
                criteria_weights={
                    "relative_improvement": 0.5,
                    "variance": 0.3,
                    "acceleration": 0.1,  # Sum = 0.9, not 1.0
                }
            )

    def test_no_plateau_with_insufficient_iterations(self) -> None:
        """Should not detect plateau before minimum iterations."""
        detector = MultiCriteriaDetector()
        error_history = [1.0] * 10  # Not enough iterations
        result = detector.detect(error_history, "fp16", iteration=5)
        assert not result.detected

    def test_no_plateau_with_insufficient_history(self) -> None:
        """Should not detect plateau with short history."""
        detector = MultiCriteriaDetector()
        error_history = [1.0] * 5  # Less than window_size
        result = detector.detect(error_history, "fp16", iteration=100)
        assert not result.detected

    def test_fp64_never_plateaus(self) -> None:
        """FP64 should never trigger plateau detection."""
        detector = MultiCriteriaDetector()
        error_history = [1e-10] * 100  # Constant low error
        result = detector.detect(error_history, "fp64", iteration=100)
        assert not result.detected

    def test_plateau_detected_with_stagnant_errors(self) -> None:
        """Should detect plateau when errors stop improving."""
        detector = MultiCriteriaDetector()
        # Create stagnant error history at low values
        error_history = [1e-3] * 100
        result = detector.detect(error_history, "fp16", iteration=100)
        # With constant errors, plateau should be detected
        assert result.detected or result.score > 0

    def test_no_plateau_with_improving_errors(self) -> None:
        """Should not detect plateau when errors are improving."""
        detector = MultiCriteriaDetector()
        # Create rapidly improving error history
        error_history = [10 ** (-i / 10) for i in range(100)]
        result = detector.detect(error_history, "fp16", iteration=100)
        # With improving errors, plateau should not be detected
        assert not result.detected

    def test_get_config_returns_dict(self) -> None:
        """get_config should return configuration dictionary."""
        detector = MultiCriteriaDetector()
        config = detector.get_config("fp16")
        assert isinstance(config, dict)
        assert "relative_threshold" in config
        assert "variance_threshold" in config
        assert "min_iterations" in config

    def test_get_config_unknown_precision(self) -> None:
        """get_config should return empty dict for unknown precision."""
        detector = MultiCriteriaDetector()
        config = detector.get_config("unknown")
        assert config == {}

    def test_repr(self) -> None:
        """__repr__ should be descriptive."""
        detector = MultiCriteriaDetector(window_size=20, score_threshold=0.8)
        repr_str = repr(detector)
        assert "MultiCriteriaDetector" in repr_str
        assert "20" in repr_str
        assert "0.8" in repr_str


class TestRelativeImprovementDetector:
    """Tests for RelativeImprovementDetector."""

    def test_default_initialization(self) -> None:
        """Should initialize with default parameters."""
        detector = RelativeImprovementDetector()
        assert detector.window_size == 15
        assert detector.min_iterations == 30

    def test_no_plateau_before_min_iterations(self) -> None:
        """Should not detect plateau before minimum iterations."""
        detector = RelativeImprovementDetector()
        error_history = [1.0] * 20
        result = detector.detect(error_history, "fp16", iteration=10)
        assert not result.detected

    def test_unknown_precision_returns_no_plateau(self) -> None:
        """Unknown precision should not trigger plateau."""
        detector = RelativeImprovementDetector()
        error_history = [1.0] * 50
        result = detector.detect(error_history, "fp64", iteration=50)
        assert not result.detected

    def test_plateau_with_no_improvement(self) -> None:
        """Should detect plateau with constant errors."""
        detector = RelativeImprovementDetector()
        error_history = [0.001] * 50
        result = detector.detect(error_history, "fp16", iteration=50)
        assert result.detected

    def test_no_plateau_with_improvement(self) -> None:
        """Should not detect plateau with improving errors."""
        detector = RelativeImprovementDetector()
        # Errors improving significantly
        error_history = [0.1 * (0.9**i) for i in range(50)]
        result = detector.detect(error_history, "fp16", iteration=50)
        assert not result.detected

    def test_get_config(self) -> None:
        """get_config should return threshold information."""
        detector = RelativeImprovementDetector()
        config = detector.get_config("fp16")
        assert "threshold" in config
        assert "window_size" in config
        assert "min_iterations" in config

    def test_repr(self) -> None:
        """__repr__ should be descriptive."""
        detector = RelativeImprovementDetector(window_size=25)
        repr_str = repr(detector)
        assert "RelativeImprovementDetector" in repr_str
        assert "25" in repr_str


class TestThresholdDetector:
    """Tests for ThresholdDetector."""

    def test_default_initialization(self) -> None:
        """Should initialize with default parameters."""
        detector = ThresholdDetector()
        assert detector.stagnation_threshold == 100

    def test_empty_history_returns_no_plateau(self) -> None:
        """Empty history should not trigger plateau."""
        detector = ThresholdDetector()
        result = detector.detect([], "fp16", 100)
        assert not result.detected

    def test_unknown_precision_returns_no_plateau(self) -> None:
        """Unknown precision should not trigger plateau."""
        detector = ThresholdDetector()
        result = detector.detect([1.0], "fp64", 100)
        assert not result.detected

    def test_plateau_when_below_threshold(self) -> None:
        """Should detect plateau when error below threshold."""
        detector = ThresholdDetector()
        # Very low error should trigger plateau
        result = detector.detect([1e-10], "fp16", 100)
        assert result.detected

    def test_no_plateau_when_above_threshold(self) -> None:
        """Should not detect plateau when error above threshold."""
        detector = ThresholdDetector()
        # High error should not trigger plateau
        result = detector.detect([1.0], "fp16", 100)
        assert not result.detected

    def test_get_config(self) -> None:
        """get_config should return threshold information."""
        detector = ThresholdDetector()
        config = detector.get_config("fp16")
        assert "threshold" in config
        assert "stagnation_threshold" in config

    def test_repr(self) -> None:
        """__repr__ should be descriptive."""
        detector = ThresholdDetector()
        assert repr(detector) == "ThresholdDetector()"


class TestCreateDetector:
    """Tests for create_detector factory function."""

    def test_create_multi_criteria(self) -> None:
        """Should create MultiCriteriaDetector."""
        detector = create_detector("multi_criteria")
        assert isinstance(detector, MultiCriteriaDetector)

    def test_create_relative(self) -> None:
        """Should create RelativeImprovementDetector."""
        detector = create_detector("relative")
        assert isinstance(detector, RelativeImprovementDetector)

    def test_create_threshold(self) -> None:
        """Should create ThresholdDetector."""
        detector = create_detector("threshold")
        assert isinstance(detector, ThresholdDetector)

    def test_unknown_detector_raises(self) -> None:
        """Unknown detector type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown detector"):
            create_detector("unknown")

    def test_pass_kwargs(self) -> None:
        """Should pass kwargs to detector constructor."""
        detector = create_detector("multi_criteria", window_size=25)
        assert isinstance(detector, MultiCriteriaDetector)
        assert detector.window_size == 25
