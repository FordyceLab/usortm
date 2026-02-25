"""Main CLI entry point for uSort-M."""

import typer

from usortm import __version__
from usortm.cli.theme import get_console, BORDER_STYLE

# Create the main app
app = typer.Typer(
    name="usortm",
    help="[#4096E3]uSort-M[/#4096E3]: Rapid and low-cost parsed variant library generation.",
    add_completion=False,
    rich_markup_mode="rich",
)

console = get_console()


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"[brand]uSort-M[/brand] version {__version__}")
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
    [bold #4096E3]uSort-M[/bold #4096E3]: Rapid and low-cost parsed variant library generation.

    [bold]Workflow commands:[/bold]

        [cyan]plan[/cyan]      Initialize project from variant list
        [cyan]demux[/cyan]     Demultiplex sequencing data
        [cyan]pick[/cyan]      Generate hit-picking list
        [cyan]reorder[/cyan]   Export synthesis order for dropout variants
        [cyan]report[/cyan]    Generate final plate maps

    [bold]Utility commands:[/bold]

        [cyan]estimate[/cyan]  Quick cost/effort estimation
        [cyan]platemap[/cyan]  Regenerate demux plate map HTML from existing results
    """
    pass


# Import and register subcommands
from .estimate import estimate
from .plan import plan
from .demux_cmd import demux
from .pick import pick
from .reorder import reorder
from .report import report
from .platemap import platemap

app.command(name="estimate")(estimate)
app.command(name="plan")(plan)
app.command(name="demux")(demux)
app.command(name="pick")(pick)
app.command(name="reorder")(reorder)
app.command(name="report")(report)
app.command(name="platemap")(platemap)

try:
    from .integra import integra
    app.command(name="integra")(integra)
except ImportError:
    pass


if __name__ == "__main__":
    app()
