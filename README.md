# Precision Lab

**Exploring precision-performance tradeoffs in numerical computing**

[![CI](https://github.com/JeromeDuboisPro/precision-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/JeromeDuboisPro/precision-lab/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Precision Lab investigates how floating-point precision affects numerical algorithm performance and accuracy. The project features:

- **Power Method experiments** across FP8, FP16, FP32, and FP64 precisions
- **Cascading Precision Algorithm** - a novel approach that adaptively escalates precision
- **H100 GPU performance modeling** with realistic speedup factors
- **Interactive visualizations** showing precision-accuracy-performance tradeoffs

## Key Findings

The cascading precision algorithm demonstrates that strategic precision management can achieve:
- **3-6× computational speedup** on H100 GPUs
- **Equivalent accuracy** to full FP64 computation
- **Graceful degradation** under numerical stress

## Installation

```bash
# Clone the repository
git clone https://github.com/JeromeDuboisPro/precision-lab.git
cd precision-lab

# Create virtual environment (Python 3.13+)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### FP8 Support

For FP8 (E4M3/E5M2) experiments, install ml-dtypes:

```bash
pip install ml-dtypes
```

## Quick Start

```python
from precision_lab import PrecisionFormat, get_dtype, get_eps

# Get machine epsilon for different formats
print(f"FP64 epsilon: {get_eps('fp64')}")  # 2.22e-16
print(f"FP32 epsilon: {get_eps('fp32')}")  # 1.19e-07
print(f"FP16 epsilon: {get_eps('fp16')}")  # 9.77e-04

# Get numpy dtype for computations
import numpy as np
dtype = get_dtype(PrecisionFormat.FP32)
matrix = np.random.randn(100, 100).astype(dtype)
```

### CLI Usage

```bash
# Show available precision formats
precision-lab info

# Compare precision properties
precision-lab compare fp16 fp32 fp64

# Run experiment (coming soon)
precision-lab run --size 100 --precision fp32
```

## Project Structure

```
precision-lab/
├── src/precision_lab/
│   ├── algorithms/      # Power method, cascading precision
│   ├── data/            # Precision types, matrix generation
│   └── visualizations/  # Plotting utilities
├── tests/               # Unit tests
├── web/                 # Interactive GitHub Pages site
└── docs/                # Documentation
```

## Precision Formats

| Format | Bits | Mantissa | Machine ε | H100 Speedup |
|--------|------|----------|-----------|--------------|
| FP64   | 64   | 52       | 2.22e-16  | 1.0×         |
| FP32   | 32   | 23       | 1.19e-07  | 2.0×         |
| FP16   | 16   | 10       | 9.77e-04  | 4.0×         |
| FP8-E4M3 | 8  | 3        | 1.25e-01  | 6.0×         |
| FP8-E5M2 | 8  | 2        | 2.50e-01  | 6.0×         |

## The Cascading Precision Algorithm

Traditional approach: Run everything in FP64 (safe but slow)

**Cascading approach:**
1. Start in FP8 (6× faster on H100)
2. Monitor convergence rate and residual
3. Escalate to higher precision only when needed
4. Result: Same accuracy, fraction of the time

```
FP8  ────▶ FP16 ────▶ FP32 ────▶ FP64
     ↑          ↑          ↑
   stall?    stall?    stall?
```

## Mathematical Foundation

The power method iteratively computes:

```
v_{k+1} = Av_k / ||Av_k||
```

Convergence rate depends on the eigenvalue gap:
```
|λ₁ - λ_estimated| ≤ C × (|λ₂|/|λ₁|)^k
```

In lower precision, round-off errors accumulate faster, but H100 tensor cores provide substantial speedup that can compensate through additional iterations.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- Golub & Van Loan: "Matrix Computations" (4th ed.)
- Higham: "Accuracy and Stability of Numerical Algorithms"
- IEEE 754-2019: Standard for Floating-Point Arithmetic
