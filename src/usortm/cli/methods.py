"""Collect the commands a project was built by, for a methods section."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import box
from rich.table import Table

from usortm import provenance
from usortm.cli.theme import get_console

console = get_console()

PROJECT_STATE_FILE = "usortm_project.json"


def methods(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to uSort-M project directory.",
        exists=True,
    ),
    output: Path = typer.Option(
        None,
        "--output", "-o",
        help=(
            "Where to write the commands. Defaults to commands.txt at the top "
            "of the project."
        ),
    ),
) -> None:
    """Write the commands this project was built by, in the order they ran.

    Each step records its own invocation as it completes, so this collects
    them rather than reconstructing them.  A step that ran before that was
    recorded is listed by its parameters and said to be missing a command,
    which is the honest form: a command line inferred from a parameter table
    is one that was never typed.

    Example:

        usortm methods my_project/
    """
    state_file = project_dir / PROJECT_STATE_FILE
    if not state_file.exists():
        console.print(f"[red]Error:[/red] No uSort-M project at {project_dir}")
        console.print("Expected " + PROJECT_STATE_FILE)
        raise typer.Exit(1)

    with open(state_file) as fh:
        project = json.load(fh)

    text = provenance.render_commands(project, project_dir)
    path = Path(output) if output else project_dir / provenance.COMMANDS_FILE
    path.write_text(text)

    steps = provenance._steps(project)
    recorded = sum(1 for *_, s in steps if s.get("command"))

    table = Table(title="Methods", box=box.ROUNDED, show_header=True,
                  header_style="bold cyan")
    table.add_column("Step", style="bold")
    table.add_column("When", style="muted")
    table.add_column("Command", justify="right")
    for when, rnd, step, s in steps:
        label = step if rnd == 1 else f"{step} (round {rnd})"
        table.add_row(label, (when or "")[:10] or "—",
                      "recorded" if s.get("command") else "not recorded")
    console.print()
    console.print(table)
    console.print()
    console.print(f"[green]✓[/green] {path}")
    if recorded < len(steps):
        console.print(
            f"[yellow]![/yellow] {len(steps) - recorded} step(s) ran before "
            f"their command was recorded and are shown by their parameters. "
            f"Re-running one records the command it was given."
        )
