"""Command-line interface for the data-quality-agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .agent import DQAgent
from .profiler import DataProfiler

console = Console()


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="data-quality-agent")
def cli() -> None:
    """Data Quality Agent - automated profiling, anomaly detection, and lineage tracking."""


# ---------------------------------------------------------------------------
# profile command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    "-o",
    default="report.md",
    show_default=True,
    help="Output path for the Markdown report.",
)
@click.option(
    "--baseline",
    "-b",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a baseline JSON profile for schema-drift detection.",
)
@click.option(
    "--save-profile",
    "save_profile",
    default=None,
    type=click.Path(dir_okay=False),
    help="Save the computed profile as a JSON file (useful as a future baseline).",
)
@click.option(
    "--model",
    default="claude-sonnet-4-6",
    show_default=True,
    help="Claude model to use.",
)
@click.option(
    "--api-key",
    envvar="ANTHROPIC_API_KEY",
    default=None,
    help="Anthropic API key (defaults to ANTHROPIC_API_KEY env var).",
)
def profile(
    path: str,
    output: str,
    baseline: Optional[str],
    save_profile: Optional[str],
    model: str,
    api_key: Optional[str],
) -> None:
    """Generate a full data quality report for a CSV or Parquet file.

    PATH is the file to analyse (.csv or .parquet).

    Example:

        dq-agent profile data.csv --output report.md
    """
    baseline_profile: dict | None = None
    if baseline:
        try:
            baseline_profile = json.loads(Path(baseline).read_text())
            console.print(f"[dim]Loaded baseline profile from {baseline}[/dim]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Failed to load baseline: {exc}[/red]")
            sys.exit(1)

    # Optionally save a fresh profile before the AI analysis
    if save_profile:
        profiler = DataProfiler()
        try:
            prof = profiler.profile(path)
            sp = Path(save_profile)
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_text(json.dumps(prof, indent=2, default=str))
            console.print(f"[green]Profile saved to {save_profile}[/green]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Warning: Could not save profile: {exc}[/yellow]")

    agent = DQAgent(api_key=api_key, model=model)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Running data quality analysis...", total=None)
        try:
            report = agent.analyse(path, baseline_profile=baseline_profile)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Analysis failed: {exc}[/red]")
            sys.exit(1)

    # Write report to file
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    console.print(f"\n[bold green]Report saved to {output}[/bold green]")

    # Pretty-print the report
    console.print()
    console.print(Markdown(report))


# ---------------------------------------------------------------------------
# compare command
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--baseline",
    "-b",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the baseline JSON profile.",
)
@click.option(
    "--current",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the current CSV/Parquet file (or a saved profile JSON).",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Output drift results as JSON instead of a table.",
)
def compare(
    baseline: str,
    current: str,
    as_json: bool,
) -> None:
    """Compare a baseline profile with the current dataset for schema drift.

    BASELINE is a JSON file produced by a previous --save-profile run.
    CURRENT is either a CSV/Parquet data file or a saved profile JSON.

    Example:

        dq-agent compare --baseline baseline.json --current current.csv
    """
    # Load baseline
    try:
        baseline_profile: dict = json.loads(Path(baseline).read_text())
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to load baseline: {exc}[/red]")
        sys.exit(1)

    # Load or compute current profile
    current_path = Path(current)
    if current_path.suffix.lower() == ".json":
        try:
            current_profile: dict = json.loads(current_path.read_text())
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Failed to load current profile JSON: {exc}[/red]")
            sys.exit(1)
    else:
        profiler = DataProfiler()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Profiling current dataset...", total=None)
            try:
                current_profile = profiler.profile(current)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Failed to profile current file: {exc}[/red]")
                sys.exit(1)

    profiler = DataProfiler()
    drift = profiler.compare_schemas(baseline_profile, current_profile)

    if as_json:
        click.echo(json.dumps(drift, indent=2))
        return

    # Rich table output
    status_color = "red" if drift["drift_detected"] else "green"
    status_text = "DRIFT DETECTED" if drift["drift_detected"] else "NO DRIFT"
    console.print(
        Panel(
            f"[bold {status_color}]{status_text}[/bold {status_color}]",
            title="Schema Comparison",
            expand=False,
        )
    )

    # Shape comparison
    shape_table = Table(title="Shape Comparison", show_header=True)
    shape_table.add_column("Metric", style="bold")
    shape_table.add_column("Baseline")
    shape_table.add_column("Current")
    before = drift.get("shape_before") or ["-", "-"]
    after = drift.get("shape_after") or ["-", "-"]
    shape_table.add_row("Rows", str(before[0]), str(after[0]))
    shape_table.add_row("Columns", str(before[1]), str(after[1]))
    console.print(shape_table)

    # Added columns
    if drift["added_columns"]:
        tbl = Table(title="Added Columns", show_header=True)
        tbl.add_column("Column", style="green")
        for col in drift["added_columns"]:
            tbl.add_row(col)
        console.print(tbl)

    # Removed columns
    if drift["removed_columns"]:
        tbl = Table(title="Removed Columns", show_header=True)
        tbl.add_column("Column", style="red")
        for col in drift["removed_columns"]:
            tbl.add_row(col)
        console.print(tbl)

    # Type changes
    if drift["type_changes"]:
        tbl = Table(title="Type Changes", show_header=True)
        tbl.add_column("Column", style="bold")
        tbl.add_column("Before", style="yellow")
        tbl.add_column("After", style="cyan")
        for change in drift["type_changes"]:
            tbl.add_row(change["column"], change["before"], change["after"])
        console.print(tbl)

    if not drift["drift_detected"]:
        console.print("[green]All columns match baseline schema.[/green]")


# ---------------------------------------------------------------------------
# quick-profile command (no LLM, just stats)
# ---------------------------------------------------------------------------


@cli.command(name="quick-profile")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(dir_okay=False),
    help="Save profile JSON to this path.",
)
def quick_profile(path: str, output: Optional[str]) -> None:
    """Compute and display statistics without invoking Claude.

    Useful for quickly inspecting a dataset or generating a baseline JSON.

    Example:

        dq-agent quick-profile data.csv --output baseline.json
    """
    profiler = DataProfiler()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Profiling...", total=None)
        try:
            prof = profiler.profile(path)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Profiling failed: {exc}[/red]")
            sys.exit(1)

    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(prof, indent=2, default=str))
        console.print(f"[green]Profile saved to {output}[/green]")

    # Summary table
    console.print()
    console.print(
        Panel(
            f"File: [bold]{prof['file']}[/bold]\n"
            f"Shape: {prof['shape'][0]} rows x {prof['shape'][1]} columns",
            title="Dataset Overview",
        )
    )

    tbl = Table(title="Column Statistics", show_header=True, show_lines=True)
    tbl.add_column("Column", style="bold")
    tbl.add_column("Dtype")
    tbl.add_column("Non-null", justify="right")
    tbl.add_column("Null %", justify="right")
    tbl.add_column("Unique", justify="right")
    tbl.add_column("Mean / Top Value")

    for col, stats in prof["stats"].items():
        null_pct = f"{stats['null_rate'] * 100:.1f}%"
        null_color = "red" if stats["null_rate"] > 0.2 else "white"
        if "mean" in stats and stats["mean"] is not None:
            extra = f"{stats['mean']:.4g}"
        else:
            top = next(iter(stats.get("top_values", {})), "-")
            extra = str(top)
        tbl.add_row(
            col,
            stats["dtype"],
            str(stats["count"]),
            f"[{null_color}]{null_pct}[/{null_color}]",
            str(stats["unique_count"]),
            extra,
        )

    console.print(tbl)


def main() -> None:
    """Entry point for the ``dq-agent`` command."""
    cli()


if __name__ == "__main__":
    main()
