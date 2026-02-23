"""CLI subcommands for managing uSort-M configuration and presets."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table
from rich import box

from usortm.cli.theme import get_console, BORDER_STYLE
from usortm.demux.presets import list_presets, add_preset

console = get_console()

config_app = typer.Typer(
    help="Manage barcode mask presets and configuration.",
    add_completion=False,
)


@config_app.command(name="list")
def config_list():
    """List available barcode mask presets."""
    presets = list_presets()

    if not presets:
        console.print("[yellow]No presets found.[/yellow]")
        console.print("Add one with: [cyan]usortm config add <file.toml>[/cyan]")
        return

    table = Table(
        title="Available Mask Presets",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Source", style="muted")

    for p in presets:
        table.add_row(p["name"], p["description"], p["source"])

    console.print()
    console.print(table)
    console.print()
    console.print(
        "Use a preset with: [cyan]usortm demux project/ --fastq data.fq "
        "--mask-config <preset-name>[/cyan]"
    )
    console.print()


@config_app.command(name="add")
def config_add(
    toml_file: Path = typer.Argument(
        ...,
        help="Path to a TOML mask config file to install as a preset.",
        exists=True,
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name", "-n",
        help="Preset name (defaults to filename without extension).",
    ),
):
    """Install a user preset from a TOML mask config file."""
    dest = add_preset(toml_file, name)
    console.print(f"[green]\u2713[/green] Preset installed: {dest}")
    console.print(
        f"  Use it with: [cyan]usortm demux project/ --fastq data.fq "
        f"--mask-config {dest.stem}[/cyan]"
    )
