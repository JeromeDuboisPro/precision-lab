"""Precision Lab: Exploring precision-performance tradeoffs in numerical computing."""

__version__ = "0.1.0"

from precision_lab.data.precision_types import (
    PrecisionFormat,
    get_dtype,
    get_eps,
    get_precision_hierarchy,
)

__all__ = [
    "__version__",
    "PrecisionFormat",
    "get_dtype",
    "get_eps",
    "get_precision_hierarchy",
]
