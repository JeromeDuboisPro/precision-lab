"""Numerical algorithms module.

This module contains implementations of:
- Power method for eigenvalue computation
- Cascading precision algorithm (FP8 → FP16 → FP32 → FP64)
- Matrix generation utilities with controlled spectra
- Plateau detection strategies for adaptive precision transitions
"""

from precision_lab.algorithms.cascading import (
    DEFAULT_PRECISION_CASCADE,
    CascadeTrace,
    CascadingPowerMethod,
    PrecisionConfig,
    SegmentResult,
)
from precision_lab.algorithms.matrices import (
    DEFAULT_SEED,
    ExperimentMatrix,
    ExperimentSetup,
    MatrixFingerprint,
    compute_fingerprint,
    create_experiment,
    create_experiment_matrix,
    create_geometric_spectrum_matrix,
    create_legacy_experiment,
    create_linear_spectrum_matrix,
    create_slow_convergence_matrix,
)
from precision_lab.algorithms.plateau_detection import (
    MultiCriteriaDetector,
    PlateauDetector,
    PlateauResult,
    RelativeImprovementDetector,
    ThresholdDetector,
    create_detector,
)
from precision_lab.algorithms.power_method import (
    ConvergenceResult,
    IterationResult,
    PowerIteration,
    PowerMethodTrace,
    run_power_method,
)

__all__ = [
    # Cascading precision
    "CascadeTrace",
    "CascadingPowerMethod",
    "DEFAULT_PRECISION_CASCADE",
    "PrecisionConfig",
    "SegmentResult",
    # Matrix generation
    "DEFAULT_SEED",
    "ExperimentSetup",
    "ExperimentMatrix",  # Backwards compatibility alias
    "MatrixFingerprint",
    "compute_fingerprint",
    "create_experiment",
    "create_experiment_matrix",  # Backwards compatibility alias
    "create_geometric_spectrum_matrix",
    "create_legacy_experiment",
    "create_linear_spectrum_matrix",
    "create_slow_convergence_matrix",
    # Plateau detection
    "MultiCriteriaDetector",
    "PlateauDetector",
    "PlateauResult",
    "RelativeImprovementDetector",
    "ThresholdDetector",
    "create_detector",
    # Power method
    "ConvergenceResult",
    "IterationResult",
    "PowerIteration",
    "PowerMethodTrace",
    "run_power_method",
]
