# Precision Lab - Project Context for Claude Code

## 🎯 Project Mission

**Explore precision-performance tradeoffs in numerical computing through interactive visualization**

This project demonstrates how reduced floating-point precision (FP8/FP16/FP32/FP64) affects convergence of numerical algorithms - a critical question for modern GPU math libraries and AI accelerators.

---

## 🔧 Project Management with Beads

This project uses **beads** (`bd` CLI) for lightweight issue tracking with dependency support.

### Quick Reference
```bash
bd list          # List all issues
bd show <id>     # Show issue details
bd ready         # Show unblocked work
bd blocked       # Show blocked issues
bd close <id>    # Close completed issue
```

### ⚠️ CONFIDENTIAL PROTOCOL
**CRITICAL**: A git pre-commit hook automatically guards against accidental exposure.

The hook blocks commits containing sensitive terms. If triggered:
1. Review flagged files
2. Remove or rephrase sensitive content
3. Re-stage and commit

**Public framing**: Educational research project exploring precision-performance frontiers.

---

## 📐 Mathematical Foundation

### Power Method Algorithm
Iterative algorithm for computing dominant eigenvalue λ₁ of matrix A:

1. Start with random vector x₀
2. Iterate: x_{k+1} = A·x_k / ||A·x_k||
3. Eigenvalue estimate: λ = x_k^T·A·x_k
4. Converge when residual norm < tolerance

**Convergence Rate**: The power method converges linearly with rate ρ = |λ₂/λ₁|

- **Convergence ratio**: ρ = |λ₂/λ₁| (second to first eigenvalue)
- **Error reduction per iteration**: error_k ≈ ρᵏ · error_0
- **Iterations to precision ε**: k ≈ log(ε) / log(ρ)
- **Example**: ρ = 0.9 → ~44 iterations per decade of accuracy

The `convergence_type` parameter controls ρ:
- `"fast"` → ρ ≈ 0.5 (λ₂/λ₁ gap = 50%)
- `"slow"` → ρ ≈ 0.909 (λ₂/λ₁ gap = 10%)

### ⚠️ IMPORTANT: Use Residual Norm, NOT Relative Error

**Always use normalized residual ||Av - λv|| / (|λ| · ||v||) as the convergence metric.**

- **Normalized Residual**: ||Av - λv|| / (|λ| · ||v||) measures convergence independent of scale
- Uses |λ| as approximation for ||A||₂ (valid for SPD matrices where ||A||₂ = λ_max)
- It's the mathematically proper convergence criterion for iterative eigensolvers
- Shows correct precision floor behavior for each floating-point format

### Condition Number κ
Ratio of largest to smallest eigenvalue: κ = λ_max / λ_min

- **Well-conditioned** (κ < 100): Fast, stable convergence
- **Moderately conditioned** (100 ≤ κ ≤ 1000): Slower convergence
- **Ill-conditioned** (κ > 1000): Very slow, sensitive to precision

---

## 🔬 Precision Formats

### FP64 (Double Precision)
- **Format**: 1 sign + 11 exponent + 52 mantissa bits
- **Machine Epsilon**: 2.22e-16
- **Use**: Scientific computing, reference baseline

### FP32 (Single Precision)
- **Format**: 1 sign + 8 exponent + 23 mantissa bits
- **Machine Epsilon**: 1.19e-7
- **Use**: Most engineering/ML training

### FP16 (Half Precision)
- **Format**: 1 sign + 5 exponent + 10 mantissa bits
- **Machine Epsilon**: 9.77e-4
- **Use**: ML training, well-conditioned problems

### FP8 (via ml_dtypes)
- **E4M3 Format**: 1 sign + 4 exponent + 3 mantissa bits
- **E5M2 Format**: 1 sign + 5 exponent + 2 mantissa bits
- **Machine Epsilon**: E4M3: ~0.125, E5M2: ~0.25
- **Use**: ML training/inference on modern GPU tensor cores

---

## 🏗️ Project Structure

```
precision-lab/
├── src/precision_lab/          # Python package
│   ├── algorithms/             # Numerical algorithms
│   │   └── power_method/       # Power method implementations
│   ├── precision/              # FP8/16/32/64 handling
│   └── visualization/          # Trace generation
├── docs/                       # GitHub Pages
│   ├── index.html             # Landing page
│   ├── race.html              # Precision race visualization
│   └── cascading.html         # Cascading precision demo
├── experiments/                # Reproducible scripts
├── tests/                      # pytest suite
└── .github/workflows/          # CI + Pages deployment
```

---

## ⚡ Key Algorithms

### Standard Power Method
Compare convergence across FP8/FP16/FP32/FP64 for same matrix.

### Cascading Precision (Novel Contribution)
Dynamic precision escalation: **FP8 → FP16 → FP32 → FP64**

**Strategy**:
1. Start at FP8 (fastest throughput)
2. Detect stagnation or precision threshold
3. Escalate to next precision level
4. Carry eigenvector state across transitions

**Results**: 2-3× speedup vs FP64-only for same accuracy.

---

## 🎯 H100 Performance Modeling

### Time Speedup (Simulated)
**Note**: These are *theoretical maximum* speedup factors for demonstration purposes.
Actual performance varies based on memory bandwidth, matrix size, and implementation.

Scale CPU time to simulate GPU performance:
- **FP8**: 6× speedup (theoretical tensor core peak)
- **FP16**: 4× speedup (theoretical half-precision units)
- **FP32**: 1× (baseline)
- **FP64**: 1× (reference)

*Real-world power method is memory-bound, not compute-bound. Actual speedups
may be lower depending on memory bandwidth utilization.*

### Iteration Budget (Fair Comparison)
Allocate more iterations to faster precisions:
- **FP8**: 6× iterations
- **FP16**: 4× iterations
- **FP32**: 2× iterations
- **FP64**: 1× (baseline)

---

## ✅ Quality Standards

### Numerical Correctness
- Convergence must satisfy precision-appropriate tolerance
- Residual norm computed correctly
- State preserved across precision transitions

### Performance Claims
- Fair comparison (same matrix, same seed)
- Correct FLOPS count (2n² + n per iteration)
- **Never run benchmarks in parallel** (corrupts timing)

### Code Quality
- Type hints throughout
- Google-style docstrings
- pytest test coverage
- ruff + mypy clean

---

## 📚 Key References

### Numerical Methods
- Golub & Van Loan: "Matrix Computations" (power method theory)
- Higham: "Accuracy and Stability of Numerical Algorithms"

### Mixed Precision
- Micikevicius et al.: "Mixed Precision Training" (ICLR 2018)
- IEEE 754 Standard: Floating-Point Arithmetic

---

## 🚦 Development Workflow

### Session Start
```bash
bd list               # ALWAYS check open beads first!
bd ready              # See actionable tasks
bd show <bead-id>     # Check details
```

> **⚠️ IMPORTANT**: Always run `bd list` at session start to check open beads before starting work.

### During Work
- Validate mathematical correctness
- Run tests frequently
- Document uncertainties

### Before Commit
```bash
# Pre-commit hook runs automatically - no manual check needed
pytest tests/
ruff check .
mypy src/
```

### Session End
```bash
bd close <completed-beads>
git add -A && git commit
```

---

## 📊 Visualization Data Generation

### Trace Generation Parameters
When regenerating traces for the interactive visualizations:

```python
# Configuration for fair comparison
matrix_size = 1024           # 1024×1024 matrix
condition_number = 100.0     # κ=100 (moderately conditioned)
seed = 42                    # Reproducibility
convergence_type = "slow"    # 10% eigenvalue gap (λ₂/λ₁ = 0.909)
```

### Convergence Targets
- **Cascading**: `target_residual=1e-12` → Forces use of all 4 precision levels (FP8→FP16→FP32→FP64)
- **FP64 reference**: `target_error=1e-12` → Must match cascading's residual target for fair comparison

### Expected Results (1024×1024, κ=100)
| Method | Raw Iterations | Effective Iterations | Final Residual |
|--------|---------------|---------------------|----------------|
| Cascading | ~283 | ~165 | ~9.45e-13 |
| FP64-only | ~259 | 259 | ~9.81e-13 |

**Speedup**: Cascading achieves same accuracy in ~165 effective iterations vs FP64's 259 = **1.57× faster**

### Effective Iteration Calculation
X-axis shows "Effective FP64 Iterations" (normalized by speedup):
- FP8 iterations ÷ 6
- FP16 iterations ÷ 4
- FP32 iterations ÷ 2
- FP64 iterations ÷ 1 (baseline)

---

**This is an educational research project demonstrating precision-performance tradeoffs in numerical computing.**
