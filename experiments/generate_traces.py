"""Generate convergence traces for precision-lab visualizations.

This script generates JSON trace files for:
- Individual precision formats (FP8, FP16, FP32, FP64)
- Cascading precision execution

Output JSON files are suitable for web-based visualization.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from precision_lab.algorithms.cascading import CascadingPowerMethod
from precision_lab.algorithms.matrices import create_experiment
from precision_lab.algorithms.power_method import run_power_method
from precision_lab.data.precision_types import PrecisionFormat


def generate_single_precision_traces(
    matrix_size: int = 256,
    condition_number: float = 100.0,
    seed: int = 42,
    output_dir: Path | None = None,
) -> None:
    """Generate convergence traces for individual precision formats.

    Args:
        matrix_size: Matrix dimension (default 256 for demo).
        condition_number: Matrix condition number.
        seed: Random seed for reproducibility.
        output_dir: Output directory (defaults to experiments/traces/).
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "traces"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating single precision traces (n={matrix_size}, κ={condition_number})..."
    )

    # Create experiment setup once
    experiment = create_experiment(
        matrix_size, condition_number, seed=seed, convergence_type="slow"
    )

    # Iteration budgets per precision format
    # Lower precision formats get more iterations to show their behavior at the precision floor
    # FP64 is the reference (500), others are scaled based on throughput advantage
    iteration_budgets = {
        PrecisionFormat.FP8_E4M3: 3000,  # 6x throughput → 6x iterations
        PrecisionFormat.FP16: 2000,  # 4x throughput → 4x iterations
        PrecisionFormat.FP32: 1000,  # 2x throughput → 2x iterations
        PrecisionFormat.FP64: 500,  # Reference budget
    }

    for precision, max_iters in iteration_budgets.items():
        print(
            f"  Running {precision.value} ({max_iters} iterations)...",
            end=" ",
            flush=True,
        )

        # Use float('inf') as target to disable early stopping
        # This ensures all precisions run for max_iters so we see real data
        # (the fluctuations around the precision floor, not repeated final values)
        trace = run_power_method(
            experiment.matrix,
            precision,
            experiment.true_eigenvalue,
            max_iterations=max_iters,
            target_error=float("inf"),  # Disable convergence-based stopping
            initial_vector=experiment.initial_vector,
        )

        # Prepare JSON-serializable output
        output = {
            "metadata": {
                "algorithm": "power_method",
                "precision": precision.value,
                "matrix_size": matrix_size,
                "condition_number": condition_number,
                "true_eigenvalue": experiment.true_eigenvalue,
                "seed": seed,
                "convergence_type": "slow",
                "timestamp": datetime.now(UTC).isoformat(),
                "final_residual": trace.final_residual,
                "final_error": trace.final_error,
                "converged": trace.converged,
                "matrix_fingerprint": experiment.fingerprint.to_dict(),
            },
            "summary": {
                "iterations": trace.iterations,
                "total_time_seconds": trace.total_time,
                "final_eigenvalue": trace.final_eigenvalue,
            },
            "trace": [
                {
                    "iteration": h["iteration"],
                    "eigenvalue": h["eigenvalue"],
                    "relative_error": h["relative_error"],
                    "residual_norm": h["residual_norm"],
                    "algorithm_time": h["algorithm_time"],
                    "cumulative_algorithm_time": h["cumulative_algorithm_time"],
                }
                for h in trace.history
            ],
        }

        # Write to JSON file
        output_file = output_dir / f"trace_{precision.value}.json"
        with output_file.open("w") as f:
            json.dump(output, f, indent=2)

        print(f"✓ {trace.iterations} iterations, converged={trace.converged}")

    print(f"\nSingle precision traces saved to: {output_dir}")


def generate_cascading_trace(
    matrix_size: int = 256,
    condition_number: float = 100.0,
    seed: int = 42,
    output_dir: Path | None = None,
) -> None:
    """Generate cascading precision convergence trace.

    Args:
        matrix_size: Matrix dimension (default 256 for demo).
        condition_number: Matrix condition number.
        seed: Random seed for reproducibility.
        output_dir: Output directory (defaults to experiments/traces/).
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "traces"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\nGenerating cascading precision trace (n={matrix_size}, κ={condition_number})..."
    )

    # Create cascading power method instance
    cascading = CascadingPowerMethod(
        matrix_size=matrix_size,
        condition_number=condition_number,
        seed=seed,
        convergence_type="slow",
    )

    # Run cascading execution with tight tolerance to require FP64
    # Using 1e-12 ensures we need FP64 (FP32 floor is ~1e-7)
    print("  Running cascading precision (FP8→FP16→FP32→FP64)...", end=" ", flush=True)
    trace = cascading.run(target_residual=1e-12, max_effective_iterations=5000)

    # Convert to JSON-serializable format using built-in method
    output = cascading.to_dict(trace)

    # Write to JSON file
    output_file = output_dir / "trace_cascading.json"
    with output_file.open("w") as f:
        json.dump(output, f, indent=2)

    print(
        f"✓ {trace.iterations} iterations ({trace.effective_iterations:.1f} effective), "
        f"converged={trace.converged}"
    )

    # Print segment summary
    print("\n  Precision segments:")
    for segment in trace.segments:
        print(
            f"    {segment.precision}: {segment.iterations} iterations "
            f"({segment.effective_iterations:.1f} effective)"
        )

    print(f"\nCascading trace saved to: {output_dir}")


def main() -> None:
    """Generate all traces for precision-lab visualizations."""
    print("=" * 70)
    print("Precision Lab - Trace Data Generation")
    print("=" * 70)

    # Configuration - matching README specs
    matrix_size = 1024  # 1024x1024 matrix as documented
    condition_number = 100.0  # κ=100
    seed = 42
    # Note: eigenvalue_gap=1.1 (default) gives 10% difference between λ₁ and λ₂

    output_dir = Path(__file__).parent / "traces"

    # Generate all traces
    generate_single_precision_traces(
        matrix_size=matrix_size,
        condition_number=condition_number,
        seed=seed,
        output_dir=output_dir,
    )

    generate_cascading_trace(
        matrix_size=matrix_size,
        condition_number=condition_number,
        seed=seed,
        output_dir=output_dir,
    )

    print("\n" + "=" * 70)
    print("✓ All traces generated successfully!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    for trace_file in sorted(output_dir.glob("trace_*.json")):
        size_kb = trace_file.stat().st_size / 1024
        print(f"  - {trace_file.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
