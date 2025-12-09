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

    print(f"Generating single precision traces (n={matrix_size}, κ={condition_number})...")

    # Create experiment setup once
    experiment = create_experiment(
        matrix_size, condition_number, seed=seed, convergence_type="slow"
    )

    # Precision formats to test
    precisions = [
        PrecisionFormat.FP8_E4M3,
        PrecisionFormat.FP16,
        PrecisionFormat.FP32,
        PrecisionFormat.FP64,
    ]

    for precision in precisions:
        print(f"  Running {precision.value}...", end=" ", flush=True)

        # Run power method
        trace = run_power_method(
            experiment.matrix,
            precision,
            experiment.true_eigenvalue,
            max_iterations=2000,
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
        with open(output_file, "w") as f:
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

    # Run cascading execution
    print("  Running cascading precision (FP8→FP16→FP32→FP64)...", end=" ", flush=True)
    trace = cascading.run(target_residual=1e-6, max_effective_iterations=5000)

    # Convert to JSON-serializable format using built-in method
    output = cascading.to_dict(trace)

    # Write to JSON file
    output_file = output_dir / "trace_cascading.json"
    with open(output_file, "w") as f:
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

    # Configuration
    matrix_size = 256
    condition_number = 100.0
    seed = 42

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
