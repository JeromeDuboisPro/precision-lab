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

**Convergence Rate**: Depends on eigenvalue gap (λ₁/λ₂) and condition number κ(A)

### ⚠️ IMPORTANT: Use Residual Norm, NOT Relative Error

**Always use residual norm ||Av - λv|| as the convergence metric.**

- **Residual Norm**: ||Av - λv|| measures how well the eigenvector equation is satisfied
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

### Time Speedup (Simulation)
Scale CPU time to simulate GPU performance:
- **FP8**: 6× speedup (tensor cores + memory bandwidth)
- **FP16**: 4× speedup (half-precision units)
- **FP32**: 1× (baseline)
- **FP64**: 1× (reference)

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
bd ready              # See actionable tasks
bd show <bead-id>     # Check details
```

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

**This is an educational research project demonstrating precision-performance tradeoffs in numerical computing.**
