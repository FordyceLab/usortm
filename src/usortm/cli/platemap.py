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

    plate_map_path = output if output is not None else demux_output / "plate_map.html"

    save_plate_map_html(
        read_df,
        str(plate_map_path),
        title="Demux Plate Map",
        min_reads=min_reads,
    )
    console.print(f"[green]✓[/green] Plate map saved to {plate_map_path}")
