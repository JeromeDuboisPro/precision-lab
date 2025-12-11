# Precision Lab

[![CI](https://github.com/JeromeDuboisPro/precision-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/JeromeDuboisPro/precision-lab/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Type Checked: mypy](https://img.shields.io/badge/type--checked-mypy-informational)](http://mypy-lang.org/)

**30-Second Pitch:** Modern GPU accelerators offer significant throughput gains at reduced precisions (theoretical peak: up to 6× at FP8 vs FP64). But can you still solve numerical problems accurately? This project demonstrates how reduced floating-point precision (FP8/FP16/FP32/FP64) affects convergence of iterative algorithms, and introduces a novel "cascading precision" strategy that achieves FP64-quality results while spending most iterations in faster, lower-precision formats.

**[Live Demo →](https://jeromeduboispro.github.io/precision-lab/)** *(Interactive visualizations showing precision race and cascading precision algorithm)*

---

## Key Findings

Precision-performance tradeoffs for power method eigenvalue computation on a 1024×1024 matrix (condition number κ=100):

| Precision | Machine ε | Residual Floor | Iterations to 1e-6 | Simulated Speedup* | Time Advantage |
|-----------|-----------|----------------|-------------------|--------------|----------------|
| **FP8** (E4M3) | ~0.125 | ~1e-3 | Plateaus early | 6× | Fast initial progress |
| **FP16** | 9.77e-4 | ~1e-4 | Plateaus early | 4× | Good for moderate targets |
| **FP32** | 1.19e-7 | ~1e-7 | ~450 | 2× | Standard engineering precision |
| **FP64** | 2.22e-16 | ~1e-15 | ~450 | 1× (baseline) | Scientific reference |
| **Cascading** | Adaptive | ~1e-15 | ~250 (effective) | 2.5-3× | **Best of both worlds** |

*\*Speedup factors are theoretical peak throughput ratios based on tensor core specifications. Actual performance depends on memory bandwidth (power method is memory-bound).*

**Cascading Strategy:** Start at FP8 (fastest throughput) → escalate to FP16 → FP32 → FP64 as needed. Achieves FP64 accuracy with 40-60% fewer effective iterations by spending early iterations in faster precisions.

---

## Installation

**Requirements:** Python 3.13+

```bash
# Clone repository
git clone https://github.com/JeromeDuboisPro/precision-lab.git
cd precision-lab

# Install package
pip install -e .

# Or with development tools
pip install -e ".[dev]"
```

---

## Quick Start

```python
from precision_lab.algorithms.cascading import CascadingPowerMethod

# Run cascading precision algorithm
cascading = CascadingPowerMethod(
    matrix_size=1024,
    condition_number=100,
    seed=42
)

trace = cascading.run(target_residual=1e-6)

print(f"Converged: {trace.converged}")
print(f"Total iterations: {trace.iterations}")
print(f"Effective iterations: {trace.effective_iterations:.1f}")
print(f"Final residual: {trace.final_residual:.2e}")

# View precision transitions
for segment in trace.segments:
    print(f"{segment.precision}: {segment.iterations} iters, "
          f"residual: {segment.end_residual:.2e}")
```

**Example output:**
```
Converged: True
Total iterations: 876
Effective iterations: 257.3
Final residual: 9.84e-07

FP8: 360 iters, residual: 1.23e-03
FP16: 288 iters, residual: 8.45e-05
FP32: 180 iters, residual: 3.21e-07
FP64: 48 iters, residual: 9.84e-07
```

---

## What is Cascading Precision?

Traditional approach: Run entire computation in single precision (either too slow in FP64 or insufficient accuracy in FP8/FP16).

**Cascading precision approach:**

1. **Start Fast (FP8):** Leverage 6× throughput for rapid initial convergence
2. **Detect Plateau:** Monitor residual improvement to identify precision floor
3. **Escalate Strategically:** Transition to FP16 → FP32 → FP64 only when needed
4. **Preserve State:** Carry eigenvector estimate across transitions (no wasted work)

**Why it works:** Iterative algorithms make rapid progress early (insensitive to precision), then slow down as they approach the solution (require higher precision). Cascading precision exploits this by matching precision to convergence phase.

**Performance model (simulated):** Based on theoretical peak throughput ratios for tensor cores:
- **FP8:** 6× iteration budget (theoretical tensor core peak)
- **FP16:** 4× iteration budget (theoretical half-precision units)
- **FP32:** 2× iteration budget (standard single precision)
- **FP64:** 1× baseline (reference precision)

*Note: These are theoretical maximum speedups. Real-world performance varies with memory bandwidth utilization, matrix size, and implementation efficiency.*

---

## Algorithm: Power Method

Iterative algorithm for computing dominant eigenvalue λ₁ of matrix **A**:

```
1. Start with random vector x₀
2. Iterate: x_{k+1} = A·x_k / ||A·x_k||
3. Eigenvalue estimate: λ = x_k^T·A·x_k
4. Converge when residual ||A·x - λ·x|| < tolerance
```

**Convergence characteristics:**
- **Rate:** Depends on eigenvalue gap (λ₁/λ₂) and condition number κ(**A**)
- **Precision sensitivity:** Ill-conditioned matrices (κ > 1000) require higher precision
- **Residual floor:** Each precision format has characteristic accuracy limit

**Why this algorithm?** Power method is representative of many iterative numerical algorithms (Krylov methods, gradient descent, fixed-point iteration) that exhibit similar precision-performance tradeoffs.

---

## Project Structure

```
precision-lab/
├── src/precision_lab/          # Python package
│   ├── algorithms/             # Power method & cascading implementations
│   ├── data/                   # Precision format definitions
│   └── visualization/          # Trace generation utilities
├── docs/                       # GitHub Pages (live demos)
├── experiments/                # Reproducible experiment scripts
├── tests/                      # pytest test suite
└── web/                        # Web visualization assets
```

---

## Use Cases

**For ML Engineers:**
- Understand precision limits when training with FP8/FP16 on modern GPUs
- Design mixed-precision training strategies for optimal throughput/accuracy

**For HPC Developers:**
- Explore reduced-precision iterative solvers for large-scale linear algebra
- Evaluate precision requirements for specific condition numbers

**For Numerical Computing Researchers:**
- Study stability of algorithms under precision reduction
- Benchmark precision-performance tradeoffs for eigensolvers

**For GPU Programming:**
- Quantify actual performance gains from tensor core utilization
- Design adaptive precision algorithms for real applications

---

## References

**Numerical Methods:**
- Golub & Van Loan: "Matrix Computations" (4th ed.) - Power method theory
- Higham: "Accuracy and Stability of Numerical Algorithms" - Precision analysis

**Mixed Precision Computing:**
- Micikevicius et al.: "Mixed Precision Training" (ICLR 2018)
- IEEE 754 Standard: Floating-Point Arithmetic

**GPU Architecture:**
- Modern GPU tensor core architecture whitepapers
- ML-accelerator precision formats (FP8 E4M3/E5M2)

---

## Contributing

This is an educational research project exploring precision-performance frontiers in numerical computing. Issues and pull requests welcome for:

- Additional numerical algorithms (conjugate gradient, GMRES, etc.)
- Alternative plateau detection strategies
- GPU implementation benchmarks (CUDA/HIP)
- Extended precision formats (bfloat16, TensorFloat-32)

**Development setup:**
```bash
pip install -e ".[dev]"
pre-commit install
pytest tests/
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
