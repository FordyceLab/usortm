"""Generate hit-picking lists from demultiplexed uSort-M project."""

from typing import Optional
from pathlib import Path
import csv
import json
from datetime import datetime

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
        help="Path to uSort-M project directory.",
        exists=True,
    ),
    target_variants: Optional[Path] = typer.Option(
        None,
        "--targets", "-t",
        help="CSV of specific variants to pick (default: all recovered variants).",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output path for hit list (default: project_dir/hitlist.csv).",
    ),
    transfer_volume: float = typer.Option(
        5.0,
        "--volume", "-v",
        help="Transfer volume in µL.",
    ),
    target_format: str = typer.Option(
        "384",
        "--target-format",
        help="Target plate format: '96' or '384'.",
    ),
    fill_order: str = typer.Option(
        "column",
        "--fill-order",
        help="Fill order: 'column' (A1,B1,...) or 'row' (A1,A2,...).",
    ),
    unique_only: bool = typer.Option(
        True,
        "--unique-only/--all-hits",
        help="Pick only one well per unique variant.",
    ),
):
    """
    Generate hit-picking list from [blue]uSort-M[/blue] demultiplexing results.
    
    Creates an Integra ASSIST PLUS compatible CSV for automated cherry-picking
    of sequence-verified clones.
    
    [bold]Example:[/bold]
    
        usortm pick my_project/
        usortm pick my_project/ --targets priority_variants.csv
    """
    # Load project state
    state_file = project_dir / PROJECT_STATE_FILE
    if not state_file.exists():
        console.print(f"[red]Error:[/red] Not a valid uSort-M project.")
        raise typer.Exit(1)
    
    with open(state_file) as f:
        project = json.load(f)
    
    # Check demux completed
    if not project["workflow_steps"]["demux"].get("completed"):
        console.print("[red]Error:[/red] Demultiplexing not completed.")
        console.print(f"Run 'usortm demux {project_dir}/' first.")
        raise typer.Exit(1)
    
    console.print()
    console.print(Panel.fit(
        "[bold blue]uSort-M[/bold blue] Hit-Picking",
        border_style="blue",
    ))
    console.print()
    
    # Load demux results
    demux_output = project_dir / "demux_output"
    assignments_file = demux_output / "well_assignments.csv"
    
    if not assignments_file.exists():
        console.print("[red]Error:[/red] Well assignments not found.")
        console.print(f"Expected: {assignments_file}")
        raise typer.Exit(1)
    
    # Read well assignments
    well_data = []
    with open(assignments_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            well_data.append({
                "plate": int(row["plate"]),
                "well": row["well"],
                "variant": row["variant"],
                "reads": int(row["reads"]),
                "consensus_fraction": float(row["consensus_fraction"]),
            })
    
    console.print(f"[green]✓[/green] Loaded {len(well_data)} well assignments")
    
    # Filter by target variants if specified
    if target_variants:
        targets = set()
        with open(target_variants, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                targets.add(row.get("variant") or row.get("name") or list(row.values())[0])
        
        well_data = [w for w in well_data if w["variant"] in targets]
        console.print(f"[green]✓[/green] Filtered to {len(well_data)} target variants")
    
    # Pick unique variants (best well for each)
    if unique_only:
        variant_wells = {}
        for w in well_data:
            variant = w["variant"]
            if variant not in variant_wells or w["reads"] > variant_wells[variant]["reads"]:
                variant_wells[variant] = w
        well_data = list(variant_wells.values())
        console.print(f"[green]✓[/green] Selected {len(well_data)} unique variants")
    
    # Generate target well assignments
    max_wells = 384 if target_format == "384" else 96
    rows = "ABCDEFGHIJKLMNOP" if target_format == "384" else "ABCDEFGH"
    cols = 24 if target_format == "384" else 12
    
    target_wells = []
    if fill_order == "column":
        for c in range(1, cols + 1):
            for r in rows:
                target_wells.append(f"{r}{c}")
    else:
        for r in rows:
            for c in range(1, cols + 1):
                target_wells.append(f"{r}{c}")
    
    # Build hit list
    hit_list = []
    current_target_plate = 0
    current_well_idx = 0
    
    for w in well_data:
        if current_well_idx >= max_wells:
            current_target_plate += 1
            current_well_idx = 0
        
        hit_list.append({
            "SampleID": w["variant"],
            "SourcePlateID": w["plate"],
            "SourceWell": w["well"],
            "TargetPlateID": current_target_plate,
            "TargetWell": target_wells[current_well_idx],
            "TransferVolume": transfer_volume,
        })
        current_well_idx += 1
    
    # Write output
    output_path = output or (project_dir / "hitlist.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["SampleID", "SourcePlateID", "SourceWell",
                       "TargetPlateID", "TargetWell", "TransferVolume"],
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(hit_list)
    
    # Update project state
    project["workflow_steps"]["pick"] = {
        "completed": True,
        "timestamp": datetime.now().isoformat(),
        "variants_picked": len(hit_list),
        "target_plates": current_target_plate + 1,
    }
    
    with open(state_file, "w") as f:
        json.dump(project, f, indent=2)
    
    # Display summary
    console.print()
    n_target_plates = current_target_plate + 1
    source_plates = sorted(set(h["SourcePlateID"] for h in hit_list))
    
    summary_table = Table(
        title="Hit List Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", justify="right")
    
    summary_table.add_row("Variants to pick", f"{len(hit_list)}")
    summary_table.add_row("Source plates", f"{len(source_plates)}")
    summary_table.add_row("Target plates", f"{n_target_plates}")
    summary_table.add_row("Target format", f"{target_format}-well")
    summary_table.add_row("Transfer volume", f"{transfer_volume} µL")
    
    console.print(summary_table)
    console.print()
    console.print(f"[green]✓[/green] Hit list saved to: [cyan]{output_path}[/cyan]")
    console.print()
    console.print("[bold]Next step:[/bold]")
    console.print(f"  [cyan]usortm report {project_dir}/[/cyan]  → Generate final plate maps")
    console.print()
