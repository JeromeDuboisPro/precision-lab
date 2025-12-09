# Precision Lab Experiments

This directory contains experiment scripts and trace data for precision-lab visualizations.

## Directory Structure

```
experiments/
├── generate_traces.py    # Script to generate convergence traces
├── traces/               # Generated JSON trace files
│   ├── trace_fp8_e4m3.json
│   ├── trace_fp16.json
│   ├── trace_fp32.json
│   ├── trace_fp64.json
│   └── trace_cascading.json
└── README.md
```

## Generated Trace Files

### Single Precision Traces

Individual precision format traces show convergence behavior for fixed-precision algorithms:

- **`trace_fp8_e4m3.json`** - FP8 (E4M3 format) power method trace
- **`trace_fp16.json`** - FP16 power method trace
- **`trace_fp32.json`** - FP32 power method trace
- **`trace_fp64.json`** - FP64 power method trace

Each file contains:
- Metadata: algorithm parameters, matrix properties, convergence status
- Summary: total iterations, execution time, final eigenvalue
- Trace: per-iteration metrics (eigenvalue, error, residual, timing)

### Cascading Precision Trace

**`trace_cascading.json`** - Adaptive precision escalation (FP8→FP16→FP32→FP64)

Shows how the cascading algorithm automatically transitions between precision levels based on plateau detection, demonstrating the efficiency gains from mixed-precision computation.

Additional data:
- Segments: breakdown by precision level (iterations, residuals, plateau scores)
- Trace includes `precision` field showing which format was used per iteration

## Regenerating Traces

To regenerate the trace data:

```bash
python experiments/generate_traces.py
```

Configuration parameters (editable in script):
- `matrix_size`: Matrix dimension (default: 256)
- `condition_number`: Matrix condition number (default: 100.0)
- `seed`: Random seed for reproducibility (default: 42)

## JSON Schema

### Single Precision Trace

```json
{
  "metadata": {
    "algorithm": "power_method",
    "precision": "fp32",
    "matrix_size": 256,
    "condition_number": 100.0,
    "true_eigenvalue": 100.0,
    "converged": true,
    "matrix_fingerprint": { ... }
  },
  "summary": {
    "iterations": 90,
    "total_time_seconds": 0.123
  },
  "trace": [
    {
      "iteration": 0,
      "eigenvalue": 65.71,
      "relative_error": 0.342,
      "residual_norm": 0.321,
      "algorithm_time": 0.001,
      "cumulative_algorithm_time": 0.001
    }
  ]
}
```

### Cascading Trace

```json
{
  "metadata": { ... },
  "summary": {
    "total_iterations": 120,
    "effective_iterations": 37.6,
    "precision_levels_used": 3
  },
  "segments": [
    {
      "precision": "FP8",
      "iterations": 35,
      "effective_iterations": 5.8,
      "start_residual": 0.321,
      "end_residual": 0.045,
      "plateau_score": 0.89
    }
  ],
  "trace": [
    {
      "iteration": 0,
      "precision": "FP8",
      "eigenvalue": 65.71,
      "relative_error": 0.342,
      "residual_norm": 0.321,
      "wall_time": 0.002
    }
  ]
}
```

## Experiment Parameters

Current traces use:
- **Matrix**: 256×256 symmetric positive definite
- **Condition number**: κ = 100
- **Convergence type**: "slow" (small eigenvalue gap for visualization)
- **Seed**: 42 (reproducible)
- **Target residual**: 1e-6 (cascading only)

These parameters provide good visualization data while keeping computation time reasonable for demos.

## Usage in Visualizations

These JSON files are designed for direct consumption by web-based visualization tools:

1. **Convergence plots**: Plot `residual_norm` vs `iteration` for each precision
2. **Error tracking**: Plot `relative_error` vs `iteration`
3. **Cascading visualization**: Show precision transitions using `segments` data
4. **Performance analysis**: Compare `cumulative_algorithm_time` across precisions
5. **Eigenvalue convergence**: Plot `eigenvalue` trajectory toward `true_eigenvalue`

## Notes

- All traces use the same matrix (verified via `matrix_fingerprint`)
- FP8 traces may plateau earlier due to limited precision
- Cascading trace demonstrates automatic precision escalation
- Timing data is wall-clock time on the generation machine (informational only)
