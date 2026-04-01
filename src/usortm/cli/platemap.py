"""Generate demux plate map HTML from existing demux results."""
from __future__ import annotations

from typing import Optional
from pathlib import Path

import typer

from usortm.cli.theme import get_console

console = get_console()


def platemap(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to uSort-M project directory (with completed demux results).",
        exists=True,
    ),
    min_reads: int = typer.Option(
        100,
        "--min-reads", "-m",
        help="Minimum reads per well for full color on plate map.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output path for the plate map HTML. Defaults to <project_dir>/demux_output/plate_map.html.",
    ),
):
    """
    Generate demux plate map HTML from existing demux results.

    Reads ``demux_output/read_df.csv`` from a completed demux run and
    produces an interactive Bokeh plate map without re-running demux.

    [bold]Example:[/bold]

        usortm platemap ./my_project

        usortm platemap ./my_project --min-reads 50 --output my_map.html
    """
    import csv
    import json
    import pandas as pd
    from usortm.demux.viz import save_plate_map_html

    demux_output = project_dir / "demux_output"
    read_df_path = demux_output / "read_df.csv"

    if not read_df_path.exists():
        console.print(
            f"[red]Error:[/red] Could not find {read_df_path}\n"
            "Run [cyan]usortm demux[/cyan] first to generate demux results."
        )
        raise typer.Exit(1)

    console.print(f"Loading reads from [cyan]{read_df_path}[/cyan] ...")
    read_df = pd.read_csv(read_df_path)

    if read_df.empty:
        console.print(
            "[yellow]⚠[/yellow] Plate map skipped: read_df.csv is empty. "
            "No reads were assigned to wells during demux."
        )
        raise typer.Exit(1)

    # Load mutation/streakout labels from existing demux results
    streakout_wells: set = set()
    mutation_wells: set = set()
    silent_mutation_wells: set = set()

    so_csv = demux_output / "streakout" / "streakout_candidates.csv"
    if so_csv.exists():
        with open(so_csv) as f:
            for row in csv.DictReader(f):
                streakout_wells.add(f"{row['plate']}_{row['well']}")

    wa_csv = demux_output / "well_assignments.csv"
    if wa_csv.exists():
        with open(wa_csv) as f:
            for row in csv.DictReader(f):
                key = f"{row['plate']}_{row['well']}"
                cc = row.get("cons_check", "")
                reads = int(row.get("reads", 0))
                if cc in ("Error", "Other Error") and reads >= 20 and key not in streakout_wells:
                    mutation_wells.add(key)
                elif cc == "Silent Mutation":
                    silent_mutation_wells.add(key)

    plate_map_path = output if output is not None else demux_output / "plate_map.html"

    save_plate_map_html(
        read_df,
        str(plate_map_path),
        title="Demux Plate Map",
        min_reads=min_reads,
        streakout_wells=streakout_wells,
        mutation_wells=mutation_wells,
        silent_mutation_wells=silent_mutation_wells,
    )
    console.print(f"[green]✓[/green] Plate map saved to {plate_map_path}")
