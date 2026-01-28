"""Generate hit-picking lists from demultiplexing results."""

from typing import Optional
from pathlib import Path
import csv
import json

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

PROJECT_STATE_FILE = "usortm_project.json"


def pick(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to uSort-M project directory (with demux results).",
        exists=True,
    ),
    targets: Optional[Path] = typer.Option(
        None,
        "--targets", "-t",
        help="CSV of specific variants to pick (columns: variant, count)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file path for hit-picking list",
    ),
    volume: float = typer.Option(
        5.0,
        "--volume", "-v",
        help="Transfer volume in µL",
    ),
    target_format: int = typer.Option(
        384,
        "--target-format",
        help="Target plate format (96 or 384)",
    ),
    fill_order: str = typer.Option(
        "column",
        "--fill-order",
        help="Fill order for target plate (column or row)",
    ),
    unique_only: bool = typer.Option(
        True,
        "--unique-only/--all-hits",
        help="Pick only one well per unique variant",
    ),
):
    """
    Generate hit-picking list from demultiplexing results.

    Output is formatted for [bold]Integra ASSIST PLUS[/bold] liquid handling robots
    as semicolon-delimited CSV.

    [bold]Example:[/bold]

        usortm pick my_project/ --unique-only --volume 5.0
    """
    # Load project state
    state_file = project_dir / PROJECT_STATE_FILE
    if not state_file.exists():
        console.print(f"[red]Error:[/red] Not a valid uSort-M project (missing {PROJECT_STATE_FILE})")
        console.print(f"Run 'usortm plan' first to create a project.")
        raise typer.Exit(1)

    with open(state_file) as f:
        project = json.load(f)

    # Check if demux has been run
    if "workflow_steps" not in project or not project["workflow_steps"].get("demux", {}).get("completed"):
        console.print("[red]Error:[/red] No demultiplexing results found.")
        console.print("Run 'usortm demux' first to process sequencing data.")
        raise typer.Exit(1)

    console.print()
    console.print(Panel.fit(
        "[bold blue]uSort-M[/bold blue] Hit Picking",
        border_style="blue",
    ))
    console.print()

    # Load demux results
    demux_output = project_dir / "demux_output"
    well_assignments_file = demux_output / "well_assignments.csv"

    if not well_assignments_file.exists():
        console.print(f"[red]Error:[/red] Well assignments not found: {well_assignments_file}")
        raise typer.Exit(1)

    well_data = _load_well_assignments(well_assignments_file)
    console.print(f"[green]✓[/green] Loaded {len(well_data)} wells with data")

    # Load target variants if specified
    target_variants = None
    if targets:
        target_variants = _load_targets(targets)
        console.print(f"[green]✓[/green] Loaded {len(target_variants)} target variants")

    # Generate pick list
    pick_list = _generate_pick_list(
        well_data=well_data,
        target_variants=target_variants,
        unique_only=unique_only,
        target_format=target_format,
        fill_order=fill_order,
    )

    if len(pick_list) == 0:
        console.print("[yellow]Warning:[/yellow] No hits to pick!")
        console.print("Check your demux results and target criteria.")
        raise typer.Exit(1)

    # Determine output file path
    output_file = output
    if output_file is None:
        output_file = project_dir / "hitlist.csv"

    # Save pick list in Integra ASSIST PLUS format
    _save_pick_list(pick_list, output_file, volume)

    # Display summary
    console.print()
    summary_table = Table(
        title="Hit Picking Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", justify="right")

    unique_variants = len(set(hit["variant"] for hit in pick_list))
    summary_table.add_row("Total hits", f"{len(pick_list)}")
    summary_table.add_row("Unique variants", f"{unique_variants}")
    summary_table.add_row("Transfer volume", f"{volume} µL")
    summary_table.add_row("Target format", f"{target_format}-well")
    summary_table.add_row("Fill order", fill_order)

    console.print(summary_table)
    console.print()

    console.print("[green]✓[/green] Pick list generated!")
    console.print(f"  Output: {output_file}")
    console.print()
    console.print("[bold]Next step:[/bold]")
    console.print(f"  [cyan]usortm report {project_dir}/[/cyan]  → Generate final report")
    console.print()


def _load_well_assignments(assignments_file: Path) -> list:
    """Load well assignments from demux output."""
    well_data = []

    with open(assignments_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            well_data.append({
                "plate": row["plate"],
                "well": row["well"],
                "variant": row["variant"],
                "reads": int(row["reads"]),
                "consensus_fraction": float(row["consensus_fraction"]),
            })

    return well_data


def _load_targets(targets_file: Path) -> set:
    """Load target variants from CSV."""
    targets = set()

    with open(targets_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "variant" in row:
                targets.add(row["variant"])

    return targets


def _generate_pick_list(
    well_data: list,
    target_variants: Optional[set],
    unique_only: bool,
    target_format: int,
    fill_order: str,
) -> list:
    """Generate pick list from well data."""
    pick_list = []
    seen_variants = set()

    # Sort by reads (descending) to pick highest quality wells first
    sorted_wells = sorted(well_data, key=lambda x: x["reads"], reverse=True)

    for well in sorted_wells:
        variant = well["variant"]

        # Filter by target variants if specified
        if target_variants and variant not in target_variants:
            continue

        # Skip if we've already picked this variant and unique_only is True
        if unique_only and variant in seen_variants:
            continue

        pick_list.append({
            "variant": variant,
            "source_plate": well["plate"],
            "source_well": well["well"],
            "reads": well["reads"],
            "consensus_fraction": well["consensus_fraction"],
        })

        seen_variants.add(variant)

    # Assign target wells based on fill order
    _assign_target_wells(pick_list, target_format, fill_order)

    return pick_list


def _assign_target_wells(pick_list: list, target_format: int, fill_order: str):
    """Assign target plate and well positions."""
    if target_format == 96:
        rows, cols = 8, 12
    elif target_format == 384:
        rows, cols = 16, 24
    else:
        rows, cols = 16, 24  # Default to 384

    target_plate = 0
    well_index = 0

    for hit in pick_list:
        # Calculate target well position
        if fill_order == "column":
            # Fill column-wise (A1, B1, C1... then A2, B2, C2...)
            col = well_index // rows
            row = well_index % rows
        else:  # row
            # Fill row-wise (A1, A2, A3... then B1, B2, B3...)
            row = well_index // cols
            col = well_index % cols

        # Convert to well name (e.g., A1, B2, etc.)
        row_letter = chr(ord('A') + row)
        col_number = col + 1
        target_well = f"{row_letter}{col_number}"

        hit["target_plate"] = str(target_plate)
        hit["target_well"] = target_well

        well_index += 1

        # Move to next plate if current is full
        if well_index >= rows * cols:
            target_plate += 1
            well_index = 0


def _save_pick_list(pick_list: list, output_file: Path, volume: float):
    """Save pick list in Integra ASSIST PLUS format."""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")

        # Header for Integra ASSIST PLUS
        writer.writerow([
            "SampleID",
            "SourcePlateID",
            "SourceWell",
            "TargetPlateID",
            "TargetWell",
            "TransferVolume",
        ])

        for hit in pick_list:
            writer.writerow([
                hit["variant"],
                hit["source_plate"],
                hit["source_well"],
                hit["target_plate"],
                hit["target_well"],
                f"{volume:.1f}",
            ])
