"""
Precision Format Definitions - Single Source of Truth

This module defines all supported floating-point precision formats with their
mathematical properties, machine epsilon values, and H100 performance characteristics.

References:
    - IEEE 754-2019 Standard for Floating-Point Arithmetic
    - Golub & Van Loan: "Matrix Computations" (4th ed.), Section 7.3
    - Higham: "Accuracy and Stability of Numerical Algorithms" (2nd ed.)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import numpy as np
from numpy.typing import DTypeLike

# Try to import ml_dtypes for FP8 support
try:
    import ml_dtypes

    HAS_FP8 = True
except ImportError:
    ml_dtypes = None  # type: ignore[assignment,unused-ignore]
    HAS_FP8 = False


class PrecisionFormat(Enum):
    """Supported floating-point precision formats."""

    FP64 = "fp64"
    FP32 = "fp32"
    FP16 = "fp16"
    FP8_E4M3 = "fp8_e4m3"  # 4 exponent, 3 mantissa bits (H100 tensor cores)
    FP8_E5M2 = "fp8_e5m2"  # 5 exponent, 2 mantissa bits (wider range)


@dataclass(frozen=True, slots=True)
class PrecisionSpec:
    """Specification for a floating-point precision format."""

    format: PrecisionFormat
    bits: int
    mantissa_bits: int
    exponent_bits: int
    machine_epsilon: float
    h100_time_speedup: float  # Relative to FP64 baseline
    h100_iteration_budget: float  # Iterations per FP64 iteration

    @property
    def bytes(self) -> int:
        """Number of bytes for this format."""
        return self.bits // 8


# =============================================================================
# PRECISION SPECIFICATIONS
# =============================================================================
# Machine epsilon: 2^(-mantissa_bits)
# H100 speedup factors: Theoretical peak throughput ratios for demonstration.
# Note: Real-world performance depends on memory bandwidth utilization.
# Power method is memory-bound, so actual speedups may differ.

_PRECISION_SPECS: dict[PrecisionFormat, PrecisionSpec] = {
    PrecisionFormat.FP64: PrecisionSpec(
        format=PrecisionFormat.FP64,
        bits=64,
        mantissa_bits=52,
        exponent_bits=11,
        machine_epsilon=2.22e-16,  # 2^(-52)
        h100_time_speedup=1.0,
        h100_iteration_budget=1.0,
    ),
    PrecisionFormat.FP32: PrecisionSpec(
        format=PrecisionFormat.FP32,
        bits=32,
        mantissa_bits=23,
        exponent_bits=8,
        machine_epsilon=1.19e-7,  # 2^(-23)
        h100_time_speedup=2.0,  # 2x faster than FP64
        h100_iteration_budget=2.0,
    ),
    PrecisionFormat.FP16: PrecisionSpec(
        format=PrecisionFormat.FP16,
        bits=16,
        mantissa_bits=10,
        exponent_bits=5,
        machine_epsilon=9.77e-4,  # 2^(-10)
        h100_time_speedup=4.0,  # 4x faster than FP64
        h100_iteration_budget=4.0,
    ),
    PrecisionFormat.FP8_E4M3: PrecisionSpec(
        format=PrecisionFormat.FP8_E4M3,
        bits=8,
        mantissa_bits=3,
        exponent_bits=4,
        machine_epsilon=0.125,  # 2^(-3)
        h100_time_speedup=6.0,  # Tensor core acceleration
        h100_iteration_budget=6.0,
    ),
    PrecisionFormat.FP8_E5M2: PrecisionSpec(
        format=PrecisionFormat.FP8_E5M2,
        bits=8,
        mantissa_bits=2,
        exponent_bits=5,
        machine_epsilon=0.25,  # 2^(-2)
        h100_time_speedup=6.0,
        h100_iteration_budget=6.0,
    ),
}


# =============================================================================
# CONVERGENCE TOLERANCES
# =============================================================================
# Based on research: residual_tol ≈ 10 * sqrt(ε_machine) * sqrt(κ)
# Using κ = 100 as baseline for well-conditioned matrices

_CONVERGENCE_TOLERANCES: dict[PrecisionFormat, dict[str, float | int]] = {
    PrecisionFormat.FP64: {
        "eigenvalue_tol": 1e-12,
        "residual_tol": 1e-11,
        "stagnation_window": 50,
    },
    PrecisionFormat.FP32: {
        "eigenvalue_tol": 1e-6,
        "residual_tol": 1e-5,
        "stagnation_window": 20,
    },
    PrecisionFormat.FP16: {
        "eigenvalue_tol": 1e-3,
        "residual_tol": 1e-2,
        "stagnation_window": 10,
    },
    PrecisionFormat.FP8_E4M3: {
        "eigenvalue_tol": 5e-2,
        "residual_tol": 5e-1,
        "stagnation_window": 5,
    },
    PrecisionFormat.FP8_E5M2: {
        "eigenvalue_tol": 5e-2,
        "residual_tol": 5e-1,
        "stagnation_window": 5,
    },
}


# =============================================================================
# PUBLIC API
# =============================================================================


def get_spec(fmt: PrecisionFormat | str) -> PrecisionSpec:
    """
    Get the full specification for a precision format.

    Args:
        fmt: Precision format (enum or string like 'fp32', 'FP16', 'fp8-e4m3')

    Returns:
        PrecisionSpec with all format properties

    Raises:
        ValueError: If format is unknown

    Example:
        >>> spec = get_spec("fp32")
        >>> spec.machine_epsilon
        1.19e-07
    """
    if isinstance(fmt, str):
        fmt = _parse_format(fmt)
    return _PRECISION_SPECS[fmt]


def get_dtype(fmt: PrecisionFormat | str) -> DTypeLike:
    """
    Get the numpy dtype for a precision format.

    Args:
        fmt: Precision format

    Returns:
        Numpy dtype object

    Raises:
        ValueError: If format is unknown
        ImportError: If FP8 format requested but ml_dtypes not installed

    Example:
        >>> get_dtype("fp32")
        dtype('float32')
    """
    if isinstance(fmt, str):
        fmt = _parse_format(fmt)

    dtype_map: dict[PrecisionFormat, Any] = {
        PrecisionFormat.FP64: np.float64,
        PrecisionFormat.FP32: np.float32,
        PrecisionFormat.FP16: np.float16,
    }

    if fmt in dtype_map:
        return cast("DTypeLike", dtype_map[fmt])

    # FP8 formats require ml_dtypes
    if not HAS_FP8:
        raise ImportError(
            f"FP8 format '{fmt.value}' requires ml_dtypes package. "
            "Install with: pip install ml-dtypes"
        )

    fp8_map: dict[PrecisionFormat, Any] = {
        PrecisionFormat.FP8_E4M3: ml_dtypes.float8_e4m3fn,
        PrecisionFormat.FP8_E5M2: ml_dtypes.float8_e5m2,
    }
    return cast("DTypeLike", fp8_map[fmt])


def get_eps(fmt: PrecisionFormat | str) -> float:
    """
    Get machine epsilon for a precision format.

    Machine epsilon is the smallest positive number ε such that 1.0 + ε ≠ 1.0
    in the given floating-point representation.

    Args:
        fmt: Precision format

    Returns:
        Machine epsilon value

    Example:
        >>> get_eps("fp64")
        2.22e-16
    """
    return get_spec(fmt).machine_epsilon


def get_tolerance(
    fmt: PrecisionFormat | str,
    tolerance_type: str = "residual_tol",
) -> float | int:
    """
    Get convergence tolerance for a precision format.

    Args:
        fmt: Precision format
        tolerance_type: One of 'eigenvalue_tol', 'residual_tol', 'stagnation_window'

    Returns:
        Tolerance value

    Example:
        >>> get_tolerance("fp32", "residual_tol")
        1e-05
    """
    if isinstance(fmt, str):
        fmt = _parse_format(fmt)

    tols = _CONVERGENCE_TOLERANCES[fmt]
    if tolerance_type not in tols:
        valid = list(tols.keys())
        raise ValueError(f"Unknown tolerance type: {tolerance_type}. Valid: {valid}")

    return tols[tolerance_type]


def get_precision_hierarchy() -> list[PrecisionFormat]:
    """
    Get precision formats in order from lowest to highest precision.

    This order is used for cascading precision algorithms.

    Returns:
        List of PrecisionFormat from FP8 to FP64

    Example:
        >>> get_precision_hierarchy()
        [<PrecisionFormat.FP8_E4M3: 'fp8_e4m3'>, ..., <PrecisionFormat.FP64: 'fp64'>]
    """
    return [
        PrecisionFormat.FP8_E4M3,
        PrecisionFormat.FP8_E5M2,
        PrecisionFormat.FP16,
        PrecisionFormat.FP32,
        PrecisionFormat.FP64,
    ]


def list_available_formats() -> list[PrecisionFormat]:
    """
    List all precision formats available in current environment.

    FP8 formats are only available if ml_dtypes is installed.

    Returns:
        List of available PrecisionFormat values
    """
    available = [PrecisionFormat.FP64, PrecisionFormat.FP32, PrecisionFormat.FP16]

    if HAS_FP8:
        available.extend([PrecisionFormat.FP8_E4M3, PrecisionFormat.FP8_E5M2])

    return available


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _parse_format(name: str) -> PrecisionFormat:
    """Parse a string into a PrecisionFormat enum."""
    normalized = name.lower().replace("-", "_").replace(" ", "_")

    for fmt in PrecisionFormat:
        if fmt.value == normalized:
            return fmt

    valid = [f.value for f in PrecisionFormat]
    raise ValueError(f"Unknown precision format: '{name}'. Valid: {valid}")


# =============================================================================
# FP8 QUANTIZATION INFRASTRUCTURE
# =============================================================================
# GPU-ready quantization utilities for FP8 formats.
# Designed for future integration with CUDA/cuDNN tensor core operations.


@dataclass
class QuantizationResult:
    """Result of FP8 quantization operation.

    Attributes:
        quantized: Quantized tensor in target FP8 format.
        scale: Scale factor used for quantization (tensor = quantized * scale).
        format: FP8 format used (e4m3 or e5m2).
        clipped_count: Number of values clipped to representable range.
    """

    quantized: "np.ndarray[Any, np.dtype[Any]]"
    scale: float
    format: str
    clipped_count: int


def compute_fp8_scale(
    tensor: "np.ndarray[Any, np.dtype[np.floating[Any]]]",
    format: str = "e4m3",
) -> float:
    """Compute optimal scale factor for FP8 quantization.

    Uses the maximum absolute value to determine scale, ensuring
    the tensor fits within the FP8 representable range.

    Args:
        tensor: Input tensor to quantize.
        format: FP8 format ('e4m3' or 'e5m2').

    Returns:
        Scale factor such that tensor/scale fits in FP8 range.

    Note:
        FP8 E4M3 max value: 448 (no inf/nan representation)
        FP8 E5M2 max value: 57344 (with inf/nan)
    """
    # FP8 format maximum representable values
    max_values = {
        "e4m3": 448.0,  # E4M3: no inf/nan, max exponent used for values
        "e5m2": 57344.0,  # E5M2: includes inf/nan representation
    }

    if format not in max_values:
        raise ValueError(
            f"Unknown FP8 format: {format}. Valid: {list(max_values.keys())}"
        )

    abs_max = float(np.max(np.abs(tensor)))
    if abs_max == 0:
        return 1.0  # Avoid division by zero for zero tensors

    # Scale to use ~90% of dynamic range (leave headroom for accumulation)
    target_max = max_values[format] * 0.9
    return abs_max / target_max


def quantize_to_fp8(
    tensor: "np.ndarray[Any, np.dtype[np.floating[Any]]]",
    format: str = "e4m3",
    scale: float | None = None,
) -> QuantizationResult:
    """Quantize tensor to FP8 format with proper scaling.

    Implements production-ready FP8 quantization with:
    - Automatic scale factor computation
    - Clipping to representable range
    - Support for both E4M3 and E5M2 formats

    Args:
        tensor: Input tensor (any floating-point type).
        format: FP8 format ('e4m3' or 'e5m2').
        scale: Optional pre-computed scale factor.

    Returns:
        QuantizationResult with quantized tensor, scale, and metadata.

    Raises:
        ImportError: If ml_dtypes is not installed.

    Example:
        >>> matrix = np.random.randn(256, 256).astype(np.float32)
        >>> result = quantize_to_fp8(matrix, format='e4m3')
        >>> print(f"Scale: {result.scale:.4f}, Clipped: {result.clipped_count}")
    """
    if not HAS_FP8:
        raise ImportError(
            "FP8 quantization requires ml_dtypes package. "
            "Install with: pip install ml-dtypes"
        )

    # Compute scale if not provided
    if scale is None:
        scale = compute_fp8_scale(tensor, format)

    # Scale down and quantize
    scaled = tensor / scale

    # Get FP8 dtype
    if format == "e4m3":
        fp8_dtype = ml_dtypes.float8_e4m3fn
        max_val = 448.0
    elif format == "e5m2":
        fp8_dtype = ml_dtypes.float8_e5m2
        max_val = 57344.0
    else:
        raise ValueError(f"Unknown FP8 format: {format}")

    # Count values that will be clipped
    clipped_count = int(np.sum(np.abs(scaled) > max_val))

    # Clip and convert to FP8
    clipped = np.clip(scaled, -max_val, max_val)
    quantized = clipped.astype(fp8_dtype)

    return QuantizationResult(
        quantized=quantized,
        scale=scale,
        format=format,
        clipped_count=clipped_count,
    )


def dequantize_from_fp8(
    quantized: "np.ndarray[Any, np.dtype[Any]]",
    scale: float,
    target_dtype: DTypeLike = np.float32,
) -> "np.ndarray[Any, np.dtype[np.floating[Any]]]":
    """Dequantize FP8 tensor back to higher precision.

    Args:
        quantized: FP8 tensor to dequantize.
        scale: Scale factor from quantization.
        target_dtype: Target precision (default: float32).

    Returns:
        Dequantized tensor in target precision.

    Example:
        >>> result = quantize_to_fp8(matrix, format='e4m3')
        >>> recovered = dequantize_from_fp8(result.quantized, result.scale)
    """
    return (quantized.astype(target_dtype) * scale).astype(target_dtype)
