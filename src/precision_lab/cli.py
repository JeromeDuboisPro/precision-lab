"""
Command-line interface for Precision Lab.

Usage:
    precision-lab info           Show available precision formats
    precision-lab compare        Compare precision format properties
    precision-lab run            Run power method experiments
"""

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from precision_lab import __version__
from precision_lab.data import (
    PrecisionFormat,
    get_spec,
    list_available_formats,
)

app = typer.Typer(
    name="precision-lab",
    help="Exploring precision-performance tradeoffs in numerical computing",
    add_completion=False,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"precision-lab version {__version__}")
        raise typer.Exit()


@app.callback()  # type: ignore[misc]
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Precision Lab - Numerical precision experiments."""
    pass


@app.command()  # type: ignore[misc]
def info() -> None:
    """Display information about available precision formats."""
    table = Table(title="Available Precision Formats")

    table.add_column("Format", style="cyan", no_wrap=True)
    table.add_column("Bits", justify="right")
    table.add_column("Mantissa", justify="right")
    table.add_column("Machine ε", justify="right")
    table.add_column("H100 Speedup", justify="right")
    table.add_column("Available", justify="center")

    available = set(list_available_formats())

    for fmt in PrecisionFormat:
        spec = get_spec(fmt)
        is_available = "✓" if fmt in available else "✗"
        style = "" if fmt in available else "dim"

        table.add_row(
            fmt.value.upper(),
            str(spec.bits),
            str(spec.mantissa_bits),
            f"{spec.machine_epsilon:.2e}",
            f"{spec.h100_time_speedup:.1f}×",
            is_available,
            style=style,
        )

    console.print(table)

    if PrecisionFormat.FP8_E4M3 not in available:
        console.print(
            "\n[yellow]Note:[/] FP8 formats require ml-dtypes package. "
            "Install with: [bold]pip install ml-dtypes[/]"
        )


@app.command()  # type: ignore[misc]
def compare(
    formats: Annotated[
        list[str] | None,
        typer.Argument(help="Formats to compare (e.g., fp32 fp64)"),
    ] = None,
) -> None:
    """Compare properties of precision formats."""
    if formats is None:
        formats = ["fp16", "fp32", "fp64"]

    table = Table(title="Precision Format Comparison")

    table.add_column("Property", style="bold")
    for fmt in formats:
        table.add_column(fmt.upper(), justify="right")

    specs = [get_spec(f) for f in formats]

    # Add rows for each property
    table.add_row("Bits", *[str(s.bits) for s in specs])
    table.add_row("Bytes", *[str(s.bytes) for s in specs])
    table.add_row("Mantissa bits", *[str(s.mantissa_bits) for s in specs])
    table.add_row("Exponent bits", *[str(s.exponent_bits) for s in specs])
    table.add_row("Machine ε", *[f"{s.machine_epsilon:.2e}" for s in specs])
    table.add_row("H100 speedup", *[f"{s.h100_time_speedup:.1f}×" for s in specs])
    table.add_row(
        "Iteration budget",
        *[f"{s.h100_iteration_budget:.1f}×" for s in specs],
    )

    console.print(table)


@app.command()  # type: ignore[misc]
def run(
    matrix_size: Annotated[
        int,
        typer.Option("--size", "-n", help="Matrix dimension"),
    ] = 100,
    precision: Annotated[
        str,
        typer.Option("--precision", "-p", help="Precision format to use"),
    ] = "fp64",
    max_iterations: Annotated[
        int,
        typer.Option("--max-iter", "-i", help="Maximum iterations"),
    ] = 1000,
) -> None:
    """Run power method experiment (placeholder)."""
    console.print("[bold]Power Method Experiment[/]")
    console.print(f"  Matrix size: {matrix_size}×{matrix_size}")
    console.print(f"  Precision: {precision.upper()}")
    console.print(f"  Max iterations: {max_iterations}")
    console.print("\n[yellow]Note:[/] Full implementation coming soon!")


if __name__ == "__main__":
    app()
