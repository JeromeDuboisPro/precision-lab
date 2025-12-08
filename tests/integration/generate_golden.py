#!/usr/bin/env python
"""Generate golden trace files for integration testing.

Run this script to regenerate golden files after intentional algorithm changes.
Usage: python tests/integration/generate_golden.py
"""

import json
from pathlib import Path

from precision_lab.algorithms.cascading import CascadingPowerMethod
from precision_lab.algorithms.matrices import create_experiment_matrix
from precision_lab.algorithms.power_method import PowerIteration

# Test parameters - MUST match test_trace_reproducibility.py
MATRIX_SIZE = 100
CONDITION_NUMBER = 100.0
SEED = 42
CONVERGENCE_TYPE = "linear"
MAX_ITERATIONS = 100

GOLDEN_DIR = Path(__file__).parent / "golden"


def generate_single_precision_trace(precision: str) -> list[dict]:
    """Generate trace for a single precision level."""
    experiment = create_experiment_matrix(
        MATRIX_SIZE,
        CONDITION_NUMBER,
        seed=SEED,
        convergence_type=CONVERGENCE_TYPE,
    )

    engine = PowerIteration(
        experiment.matrix,
        precision,
        A_fp64=experiment.matrix,
    )
    engine.set_initial_vector(experiment.initial_vector.copy())

    trace = []
    for iteration in range(MAX_ITERATIONS):
        iter_result = engine.iterate()
        conv_result = engine.check_convergence(
            iter_result.eigenvalue,
            experiment.true_eigenvalue,
        )

        trace.append({
            "iteration": iteration,
            "eigenvalue": float(iter_result.eigenvalue),
            "relative_error": float(conv_result.relative_error),
            "residual_norm": float(conv_result.residual_norm),
            "vector_norm": float(engine.vector_norm),
        })

    return trace


def generate_cascade_trace() -> dict:
    """Generate trace for cascading precision algorithm."""
    cascading = CascadingPowerMethod(
        matrix_size=MATRIX_SIZE,
        condition_number=CONDITION_NUMBER,
        seed=SEED,
        convergence_type=CONVERGENCE_TYPE,
    )

    # Run with enough iterations to potentially use all precisions
    trace = cascading.run(
        target_residual=1e-12,  # Tight target to force precision escalation
        max_effective_iterations=500,
    )

    # Extract first 100 iterations (or all if less)
    history = list(trace.history)[:MAX_ITERATIONS]

    return {
        "metadata": {
            "matrix_size": MATRIX_SIZE,
            "condition_number": CONDITION_NUMBER,
            "seed": SEED,
            "convergence_type": CONVERGENCE_TYPE,
            "true_eigenvalue": float(cascading.true_eigenvalue),
            "total_iterations": int(trace.iterations),
            "converged": bool(trace.converged),
            "final_residual": float(trace.final_residual),
        },
        "segments": [
            {
                "precision": s.precision,
                "iterations": s.iterations,
                "start_iteration": s.start_iteration,
                "end_iteration": s.end_iteration,
                "start_residual": float(s.start_residual),
                "end_residual": float(s.end_residual),
            }
            for s in trace.segments
        ],
        "trace": [
            {
                "iteration": h["iteration"],
                "precision": h["precision"],
                "eigenvalue": float(h["eigenvalue"]),
                "relative_error": float(h["relative_error"]),
                "residual_norm": float(h["residual_norm"]),
            }
            for h in history
        ],
    }


def save_golden(data: dict | list, filename: str) -> None:
    """Save golden data to JSON file."""
    filepath = GOLDEN_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✓ Generated {filepath}")


def main() -> None:
    """Generate all golden files."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating golden trace files...")
    print(f"  Matrix: {MATRIX_SIZE}×{MATRIX_SIZE}, κ={CONDITION_NUMBER}")
    print(f"  Seed: {SEED}, Convergence: {CONVERGENCE_TYPE}")
    print(f"  Iterations: {MAX_ITERATIONS}")
    print()

    # Generate single precision traces
    for precision in ["fp64", "fp32", "fp16", "fp8_e4m3"]:
        print(f"Generating {precision.upper()} trace...")
        trace = generate_single_precision_trace(precision)
        save_golden(trace, f"{precision}_trace.json")

    # Generate cascade trace
    print("Generating CASCADE trace...")
    cascade_data = generate_cascade_trace()
    save_golden(cascade_data, "cascade_trace.json")

    print()
    print("Done! Golden files generated in:", GOLDEN_DIR)


if __name__ == "__main__":
    main()
