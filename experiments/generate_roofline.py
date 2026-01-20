#!/usr/bin/env python3
"""Generate roofline model for memory-bound iterative solvers.

Demonstrates why power method speedup is limited by memory bandwidth,
not tensor core peak FLOPS. Shows theoretical vs achievable performance
for FP8/FP16/FP32/FP64 precision formats.

References:
- Williams et al.: "Roofline: An Insightful Visual Performance Model" (2009)
- H100 GPU specifications: Memory bandwidth and tensor core peak FLOPS
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# H100 specifications (80GB HBM3)
H100_MEMORY_BANDWIDTH_GB_S = 3350  # GB/s (theoretical peak)
H100_FP64_TFLOPS = 60  # Tensor core peak (FP64)
H100_FP32_TFLOPS = 120  # 2× FP64
H100_FP16_TFLOPS = 240  # 4× FP64
H100_FP8_TFLOPS = 480  # 8× FP64

# Bytes per element
BYTES_PER_ELEMENT = {"FP64": 8, "FP32": 4, "FP16": 2, "FP8": 1}

# Theoretical tensor core speedups (compute-bound assumption)
THEORETICAL_SPEEDUP = {"FP64": 1.0, "FP32": 2.0, "FP16": 4.0, "FP8": 8.0}


def power_method_operational_intensity(precision: str) -> float:
    """Calculate operational intensity (FLOPS/byte) for power method.

    Power method per iteration:
    - Matrix-vector multiply: 2n² FLOPS (n² multiply-adds)
    - Vector normalize: n FLOPS
    - Total: ~2n² FLOPS

    Memory traffic:
    - Read matrix A: n² * bytes_per_element
    - Read vector x: n * bytes_per_element
    - Write vector y: n * bytes_per_element
    - Total: (n² + 2n) * bytes_per_element ≈ n² * bytes_per_element

    Operational intensity = FLOPS / bytes = 2n² / (n² * bytes) = 2 / bytes

    Args:
        precision: Precision format (FP8, FP16, FP32, FP64).

    Returns:
        Operational intensity in FLOPS/byte.
    """
    bytes_per_elem = BYTES_PER_ELEMENT[precision]
    return 2.0 / bytes_per_elem  # FLOPS per byte transferred


def memory_bandwidth_ceiling(
    bandwidth_gb_s: float, operational_intensity: float
) -> float:
    """Calculate memory bandwidth ceiling (GFLOPS).

    For memory-bound kernels: Achievable GFLOPS = bandwidth * OI

    Args:
        bandwidth_gb_s: Memory bandwidth in GB/s.
        operational_intensity: Operational intensity in FLOPS/byte.

    Returns:
        Performance ceiling in GFLOPS.
    """
    return bandwidth_gb_s * operational_intensity


def generate_roofline_plot(output_path: Path) -> None:
    """Generate roofline model plot for power method.

    Shows:
    - Horizontal rooflines for tensor core peak FLOPS (compute ceiling)
    - Diagonal rooflines for memory bandwidth ceiling
    - Power method operating point (low OI = memory-bound)

    Args:
        output_path: Path to save plot image.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Operational intensity range (FLOPS/byte)
    oi_range = np.logspace(-2, 2, 100)  # 0.01 to 100 FLOPS/byte

    # Color scheme
    colors = {"FP64": "#1f77b4", "FP32": "#ff7f0e", "FP16": "#2ca02c", "FP8": "#d62728"}

    # Plot rooflines for each precision
    for precision in ["FP64", "FP32", "FP16", "FP8"]:
        # Compute ceiling (horizontal line)
        if precision == "FP64":
            compute_peak_gflops = H100_FP64_TFLOPS * 1000
        elif precision == "FP32":
            compute_peak_gflops = H100_FP32_TFLOPS * 1000
        elif precision == "FP16":
            compute_peak_gflops = H100_FP16_TFLOPS * 1000
        else:  # FP8
            compute_peak_gflops = H100_FP8_TFLOPS * 1000

        # Memory bandwidth ceiling (diagonal line)
        mem_ceiling_gflops = H100_MEMORY_BANDWIDTH_GB_S * oi_range

        # Combined roofline (min of compute and memory)
        roofline = np.minimum(compute_peak_gflops, mem_ceiling_gflops)

        ax.loglog(
            oi_range,
            roofline,
            label=f"{precision} roofline",
            color=colors[precision],
            linewidth=2,
        )

        # Mark power method operating point
        pm_oi = power_method_operational_intensity(precision)
        pm_perf = memory_bandwidth_ceiling(H100_MEMORY_BANDWIDTH_GB_S, pm_oi)

        ax.scatter(
            pm_oi,
            pm_perf,
            s=150,
            marker="o",
            color=colors[precision],
            edgecolor="black",
            linewidth=2,
            zorder=10,
        )

        # Annotate with actual speedup (position based on precision to avoid overlaps)
        bytes_per_elem = BYTES_PER_ELEMENT[precision]
        actual_speedup = BYTES_PER_ELEMENT["FP64"] / bytes_per_elem
        theoretical_speedup = THEORETICAL_SPEEDUP[precision]

        # Strategic annotation positions to avoid overlap
        annotation_positions = {
            "FP64": (15, -40),  # Below
            "FP32": (25, 25),  # Upper right
            "FP16": (30, -30),  # Lower right
            "FP8": (35, 35),  # Far upper right
        }

        ax.annotate(
            f"{precision}\nActual: {actual_speedup:.1f}×\nTheory: {theoretical_speedup:.1f}×",
            xy=(pm_oi, pm_perf),
            xytext=annotation_positions[precision],
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.6",
                "facecolor": colors[precision],
                "alpha": 0.85,
                "edgecolor": "black",
                "linewidth": 1.5,
            },
            arrowprops={
                "arrowstyle": "->",
                "connectionstyle": "arc3,rad=0.2",
                "linewidth": 2,
                "color": "black",
            },
        )

    # Styling
    ax.set_xlabel("Operational Intensity (FLOPS/byte)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Performance (GFLOPS)", fontsize=16, fontweight="bold")
    ax.set_title(
        "Roofline Model: Power Method on H100 GPU\n"
        "Memory Bandwidth Limits Mixed-Precision Speedup",
        fontsize=18,
        fontweight="bold",
        pad=25,
    )
    ax.grid(True, which="both", alpha=0.3, linestyle="--", linewidth=0.8)
    ax.legend(
        loc="lower right",
        fontsize=12,
        framealpha=0.95,
        edgecolor="black",
        fancybox=True,
    )

    # Add text box with key insight (repositioned to top-right for better visibility)
    textstr = (
        "Key Insight:\n\n"
        "Power method: Low operational intensity\n"
        "(0.25-2 FLOPS/byte)\n\n"
        "→ Memory-bound, NOT compute-bound\n"
        "→ Speedup from bytes_ratio, not FLOPS\n"
        "→ FP8: 8× bandwidth advantage"
    )
    props = {
        "boxstyle": "round,pad=0.8",
        "facecolor": "lightyellow",
        "alpha": 0.95,
        "edgecolor": "black",
        "linewidth": 2,
    }
    ax.text(
        0.97,
        0.97,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Roofline plot saved to {output_path}")


def print_analysis() -> None:
    """Print detailed analysis of memory bandwidth impact."""
    print("\n" + "=" * 70)
    print("MEMORY BANDWIDTH ROOFLINE ANALYSIS")
    print("=" * 70)

    print("\n📊 H100 Specifications:")
    print(f"  Memory Bandwidth: {H100_MEMORY_BANDWIDTH_GB_S} GB/s")
    print(f"  FP64 Tensor Core: {H100_FP64_TFLOPS} TFLOPS")
    print(f"  FP32 Tensor Core: {H100_FP32_TFLOPS} TFLOPS")
    print(f"  FP16 Tensor Core: {H100_FP16_TFLOPS} TFLOPS")
    print(f"  FP8 Tensor Core:  {H100_FP8_TFLOPS} TFLOPS")

    print("\n🔢 Power Method Operational Intensity:")
    for precision in ["FP64", "FP32", "FP16", "FP8"]:
        oi = power_method_operational_intensity(precision)
        print(f"  {precision}: {oi:.2f} FLOPS/byte")

    print("\n⚡ Achievable Performance (Memory-Bound):")
    for precision in ["FP64", "FP32", "FP16", "FP8"]:
        oi = power_method_operational_intensity(precision)
        perf_gflops = memory_bandwidth_ceiling(H100_MEMORY_BANDWIDTH_GB_S, oi)
        print(f"  {precision}: {perf_gflops:.1f} GFLOPS")

    print("\n📈 Speedup Analysis:")
    fp64_oi = power_method_operational_intensity("FP64")
    fp64_perf = memory_bandwidth_ceiling(H100_MEMORY_BANDWIDTH_GB_S, fp64_oi)

    print(f"  Baseline (FP64): {fp64_perf:.1f} GFLOPS")
    for precision in ["FP32", "FP16", "FP8"]:
        oi = power_method_operational_intensity(precision)
        perf_gflops = memory_bandwidth_ceiling(H100_MEMORY_BANDWIDTH_GB_S, oi)
        actual_speedup = perf_gflops / fp64_perf
        theoretical_speedup = THEORETICAL_SPEEDUP[precision]
        efficiency = (actual_speedup / theoretical_speedup) * 100

        print(
            f"  {precision}: {perf_gflops:.1f} GFLOPS "
            f"→ {actual_speedup:.1f}× actual speedup "
            f"({efficiency:.0f}% of {theoretical_speedup:.1f}× theory)"
        )

    print("\n💡 Key Takeaway:")
    print("  Power method is MEMORY-BOUND, not compute-bound.")
    print("  Reduced precision helps by moving more data per bandwidth unit,")
    print("  NOT by utilizing higher tensor core FLOPS.")
    print("  Speedup ≈ bytes_ratio (8× for FP8 vs FP64), not FLOPS_ratio.")
    print("=" * 70 + "\n")


def main() -> None:
    """Generate roofline plot and analysis."""
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "roofline_power_method.png"
    generate_roofline_plot(output_path)
    print_analysis()


if __name__ == "__main__":
    main()
