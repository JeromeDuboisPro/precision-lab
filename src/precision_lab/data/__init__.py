"""Data module for precision types and matrix generation."""

from precision_lab.data.precision_types import (
    HAS_FP8,
    PrecisionFormat,
    PrecisionSpec,
    get_dtype,
    get_eps,
    get_precision_hierarchy,
    get_spec,
    get_tolerance,
    list_available_formats,
)

__all__ = [
    "HAS_FP8",
    "PrecisionFormat",
    "PrecisionSpec",
    "get_dtype",
    "get_eps",
    "get_precision_hierarchy",
    "get_spec",
    "get_tolerance",
    "list_available_formats",
]
