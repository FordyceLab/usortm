"""Generate Integra ASSIST PLUS hit-picking input files."""

from typing import Optional
from pathlib import Path
import csv

import typer
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def integra(
    input_file: Path = typer.Argument(
        ...,
        help="CSV file with variant assignments (columns: variant, plate, well).",
        exists=True,
    ),
    output: Path = typer.Option(
        Path("hitlist.csv"),
        "--output", "-o",
        help="Output file path for the ASSIST PLUS hit list.",
    ),
    transfer_volume: float = typer.Option(
        5.0,
        "--volume", "-v",
        help="Transfer volume in µL.",
    ),
    variant_col: str = typer.Option(
        "variant",
        "--variant-col",
        help="Column name for variant identifiers.",
    ),
    plate_col: str = typer.Option(
        "plate",
        "--plate-col",
        help="Column name for source plate numbers.",
    ),
    well_col: str = typer.Option(
        "well",
        "--well-col",
        help="Column name for source well positions.",
    ),
    target_format: str = typer.Option(
        "384",
        "--target-format",
        help="Target plate format: '96' or '384'.",
    ),
    fill_order: str = typer.Option(
        "column",
        "--fill-order",
        help="Fill order for target plate: 'column' (A1,B1,...) or 'row' (A1,A2,...).",
    ),
):
    """
    Generate Integra ASSIST PLUS hit-picking input files.
    
    Converts sequencing analysis output to the semicolon-delimited CSV format
    required by the Integra ASSIST PLUS liquid handling robot for automated
    cherry-picking of verified clones.
    
    [bold]Input CSV format:[/bold]
    
        variant,plate,well
        K44A,1,K23
        G45A,1,A11
        T46G,1,G7
    
    [bold]Output format (semicolon-delimited):[/bold]
    
        SampleID;SourcePlateID;SourceWell;TargetPlateID;TargetWell;TransferVolume
    
    [bold]Example:[/bold]
    
        usortm integra demux_results.csv --output hitlist.csv --volume 5
    """
    # Read input file
    variants = []
    with open(input_file, newline="") as f:
        reader = csv.DictReader(f)
        
        # Validate columns exist
        if reader.fieldnames is None:
            console.print("[red]Error:[/red] Could not read CSV headers.")
            raise typer.Exit(1)
        
        missing_cols = []
        for col in [variant_col, plate_col, well_col]:
            if col not in reader.fieldnames:
                missing_cols.append(col)
        
        if missing_cols:
            console.print(f"[red]Error:[/red] Missing columns: {', '.join(missing_cols)}")
            console.print(f"Available columns: {', '.join(reader.fieldnames)}")
            raise typer.Exit(1)
        
        for row in reader:
            variants.append({
                "variant": row[variant_col],
                "plate": int(row[plate_col]),
                "well": row[well_col],
            })
    
    if not variants:
        console.print("[red]Error:[/red] No variants found in input file.")
        raise typer.Exit(1)
    
    console.print()
    console.print(f"[bold blue]Generating Integra ASSIST Hit List[/bold blue]")
    console.print(f"  Input: {input_file}")
    console.print(f"  Variants: {len(variants)}")
    console.print()
    
    # Generate target well assignments
    max_wells = 384 if target_format == "384" else 96
    rows = "ABCDEFGHIJKLMNOP" if target_format == "384" else "ABCDEFGH"
    cols = 24 if target_format == "384" else 12
    
    target_wells = []
    if fill_order == "column":
        # A1, B1, C1, ... A2, B2, C2, ...
        for c in range(1, cols + 1):
            for r in rows:
                target_wells.append(f"{r}{c}")
    else:
        # A1, A2, A3, ... B1, B2, B3, ...
        for r in rows:
            for c in range(1, cols + 1):
                target_wells.append(f"{r}{c}")
    
    # Assign target plates and wells
    output_rows = []
    current_target_plate = 0
    current_well_idx = 0
    
    for var in variants:
        if current_well_idx >= max_wells:
            current_target_plate += 1
            current_well_idx = 0
        
        output_rows.append({
            "SampleID": var["variant"],
            "SourcePlateID": var["plate"],  # 1-indexed
            "SourceWell": var["well"],
            "TargetPlateID": current_target_plate,  # 0-indexed
            "TargetWell": target_wells[current_well_idx],
            "TransferVolume": transfer_volume,
        })
        current_well_idx += 1
    
    # Write output file (semicolon-delimited)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["SampleID", "SourcePlateID", "SourceWell", 
                       "TargetPlateID", "TargetWell", "TransferVolume"],
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(output_rows)
    
    # Summary
    n_target_plates = current_target_plate + 1
    source_plates = sorted(set(v["plate"] for v in variants))
    
    summary_table = Table(
        title="Hit List Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", justify="right")
    
    summary_table.add_row("Total variants", f"{len(variants)}")
    summary_table.add_row("Source plates", f"{len(source_plates)} ({min(source_plates)}-{max(source_plates)})")
    summary_table.add_row("Target plates", f"{n_target_plates}")
    summary_table.add_row("Target format", f"{target_format}-well")
    summary_table.add_row("Transfer volume", f"{transfer_volume} µL")
    
    console.print(summary_table)
    console.print()
    console.print(f"[green]✓[/green] Hit list saved to: [cyan]{output}[/cyan]")
    console.print()
    
    # Show preview
    console.print("[bold]Preview (first 5 rows):[/bold]")
    preview_table = Table(box=box.SIMPLE)
    for col in ["SampleID", "SourcePlateID", "SourceWell", "TargetPlateID", "TargetWell", "TransferVolume"]:
        preview_table.add_column(col)
    
    for row in output_rows[:5]:
        preview_table.add_row(
            row["SampleID"],
            str(row["SourcePlateID"]),
            row["SourceWell"],
            str(row["TargetPlateID"]),
            row["TargetWell"],
            str(row["TransferVolume"]),
        )
    
    console.print(preview_table)
    console.print()
