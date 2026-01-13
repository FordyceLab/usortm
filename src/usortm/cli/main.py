"""Main CLI entry point for uSort-M."""

import typer
from rich.console import Console

from usortm import __version__

# Create the main app
app = typer.Typer(
    name="usortm",
    help="[blue]uSort-M[/blue]: Rapid and low-cost parsed variant library generation.",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"[bold blue]uSort-M[/bold blue] version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """
    [bold blue]uSort-M[/bold blue]: Rapid and low-cost parsed variant library generation.
    
    [bold]Workflow commands:[/bold]
    
        [cyan]plan[/cyan]      Initialize project from variant list
        [cyan]demux[/cyan]     Demultiplex sequencing data  
        [cyan]pick[/cyan]      Generate hit-picking list
        [cyan]report[/cyan]    Generate final plate maps
    
    [bold]Utility commands:[/bold]
    
        [cyan]estimate[/cyan]  Quick cost/effort estimation
        [cyan]integra[/cyan]   Generate Integra ASSIST file (standalone)
    """
    pass


# Import and register subcommands
from .estimate import estimate
from .plan import plan
from .demux_cmd import demux
from .pick import pick
from .report import report
from .integra import integra

app.command(name="estimate")(estimate)
app.command(name="plan")(plan)
app.command(name="demux")(demux)
app.command(name="pick")(pick)
app.command(name="report")(report)
app.command(name="integra")(integra)


if __name__ == "__main__":
    app()
