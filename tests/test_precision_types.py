"""Tests for precision_types module."""

import numpy as np
import pytest

from precision_lab.data.precision_types import (
    HAS_FP8,
    PrecisionFormat,
    get_dtype,
    get_eps,
    get_precision_hierarchy,
    get_spec,
    get_tolerance,
    list_available_formats,
)


class TestPrecisionFormat:
    """Tests for PrecisionFormat enum."""

    def test_all_formats_defined(self) -> None:
        """Verify all expected formats exist."""
        expected = {"fp64", "fp32", "fp16", "fp8_e4m3", "fp8_e5m2"}
        actual = {f.value for f in PrecisionFormat}
        assert actual == expected

    def test_format_values_lowercase(self) -> None:
        """Format values should be lowercase."""
        for fmt in PrecisionFormat:
            assert fmt.value == fmt.value.lower()


class TestGetSpec:
    """Tests for get_spec function."""

    @pytest.mark.parametrize(
        "fmt,expected_bits",
        [
            (PrecisionFormat.FP64, 64),
            (PrecisionFormat.FP32, 32),
            (PrecisionFormat.FP16, 16),
            ("fp64", 64),
            ("FP32", 32),
            ("fp16", 16),
        ],
    )
    def test_get_spec_bits(self, fmt: PrecisionFormat | str, expected_bits: int) -> None:
        """Verify bit counts for each format."""
        spec = get_spec(fmt)
        assert spec.bits == expected_bits

    def test_spec_mantissa_plus_exponent_plus_sign(self) -> None:
        """Mantissa + exponent + sign bit should equal total bits."""
        for fmt in [PrecisionFormat.FP64, PrecisionFormat.FP32, PrecisionFormat.FP16]:
            spec = get_spec(fmt)
            # IEEE 754: 1 sign bit + exponent + mantissa = total
            assert 1 + spec.exponent_bits + spec.mantissa_bits == spec.bits

    def test_bytes_property(self) -> None:
        """Bytes should be bits / 8."""
        for fmt in PrecisionFormat:
            spec = get_spec(fmt)
            assert spec.bytes == spec.bits // 8

    def test_unknown_format_raises(self) -> None:
        """Unknown format should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown precision format"):
            get_spec("fp128")


class TestGetDtype:
    """Tests for get_dtype function."""

    def test_fp64_dtype(self) -> None:
        """FP64 should return float64."""
        assert get_dtype("fp64") == np.float64

    def test_fp32_dtype(self) -> None:
        """FP32 should return float32."""
        assert get_dtype("fp32") == np.float32

    def test_fp16_dtype(self) -> None:
        """FP16 should return float16."""
        assert get_dtype("fp16") == np.float16

    @pytest.mark.skipif(not HAS_FP8, reason="ml_dtypes not installed")
    def test_fp8_e4m3_dtype(self) -> None:
        """FP8_E4M3 should return float8_e4m3fn when available."""
        import ml_dtypes

        assert get_dtype("fp8_e4m3") == ml_dtypes.float8_e4m3fn

    @pytest.mark.skipif(HAS_FP8, reason="ml_dtypes is installed")
    def test_fp8_without_mldtypes_raises(self) -> None:
        """FP8 should raise ImportError when ml_dtypes not available."""
        with pytest.raises(ImportError, match="ml_dtypes"):
            get_dtype("fp8_e4m3")


class TestGetEps:
    """Tests for get_eps function."""

    def test_epsilon_decreases_with_precision(self) -> None:
        """Higher precision should have smaller epsilon."""
        eps_fp16 = get_eps("fp16")
        eps_fp32 = get_eps("fp32")
        eps_fp64 = get_eps("fp64")

        assert eps_fp16 > eps_fp32 > eps_fp64

    def test_fp64_epsilon_matches_numpy(self) -> None:
        """FP64 epsilon should match numpy's finfo."""
        # Allow some tolerance as our value is theoretical
        assert abs(get_eps("fp64") - np.finfo(np.float64).eps) < 1e-17


class TestGetTolerance:
    """Tests for get_tolerance function."""

    def test_residual_tolerance_exists(self) -> None:
        """All formats should have residual_tol defined."""
        for fmt in PrecisionFormat:
            tol = get_tolerance(fmt, "residual_tol")
            assert isinstance(tol, float)
            assert tol > 0

    def test_tolerance_increases_with_lower_precision(self) -> None:
        """Lower precision should have higher tolerance."""
        tol_fp64 = get_tolerance("fp64", "residual_tol")
        tol_fp32 = get_tolerance("fp32", "residual_tol")
        tol_fp16 = get_tolerance("fp16", "residual_tol")

        assert tol_fp16 > tol_fp32 > tol_fp64

    def test_unknown_tolerance_type_raises(self) -> None:
        """Unknown tolerance type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown tolerance type"):
            get_tolerance("fp64", "invalid_tol")


class TestPrecisionHierarchy:
    """Tests for get_precision_hierarchy function."""

    def test_hierarchy_order(self) -> None:
        """Hierarchy should go from lowest to highest precision."""
        hierarchy = get_precision_hierarchy()

        # FP8 variants should come first
        assert hierarchy[0] in [PrecisionFormat.FP8_E4M3, PrecisionFormat.FP8_E5M2]
        assert hierarchy[1] in [PrecisionFormat.FP8_E4M3, PrecisionFormat.FP8_E5M2]

        # FP64 should be last (highest precision)
        assert hierarchy[-1] == PrecisionFormat.FP64

    def test_hierarchy_contains_all_formats(self) -> None:
        """Hierarchy should contain all precision formats."""
        hierarchy = get_precision_hierarchy()
        assert set(hierarchy) == set(PrecisionFormat)


class TestListAvailableFormats:
    """Tests for list_available_formats function."""

    def test_standard_formats_always_available(self) -> None:
        """FP16, FP32, FP64 should always be available."""
        available = list_available_formats()

        assert PrecisionFormat.FP64 in available
        assert PrecisionFormat.FP32 in available
        assert PrecisionFormat.FP16 in available

    @pytest.mark.skipif(not HAS_FP8, reason="ml_dtypes not installed")
    def test_fp8_available_with_mldtypes(self) -> None:
        """FP8 formats should be available when ml_dtypes installed."""
        available = list_available_formats()

        assert PrecisionFormat.FP8_E4M3 in available
        assert PrecisionFormat.FP8_E5M2 in available

    @pytest.mark.skipif(HAS_FP8, reason="ml_dtypes is installed")
    def test_fp8_unavailable_without_mldtypes(self) -> None:
        """FP8 formats should not be available without ml_dtypes."""
        available = list_available_formats()

        assert PrecisionFormat.FP8_E4M3 not in available
        assert PrecisionFormat.FP8_E5M2 not in available
