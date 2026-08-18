"""Command-line interface for uSort-M."""

import typer

from usortm import __version__
from usortm.cli.theme import get_console

from .estimate import estimate
from .plan import plan
from .skew_cmd import skew
from .demux_cmd import demux
from .pick import pick
from .reorder import reorder
from .report import report
from .merge import merge
from .config_cmd import config_app
from .remote_cmd import remote_app
from .platemap import platemap
from .pileups import pileups

console = get_console()


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"[brand]uSort-M[/brand] version {__version__}")
        raise typer.Exit()


# Create the main Typer app
app = typer.Typer(
    name="usortm",
    help="[#4096E3]uSort-M[/#4096E3]: Rapid and low-cost parsed variant library generation.",
    add_completion=False,
    rich_markup_mode="rich",
)


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
        [cyan]skew[/cyan]      Measure library skew and recommend sorting depth
        [cyan]demux[/cyan]     Demultiplex sequencing data
        [cyan]pick[/cyan]      Generate hit-picking list
        [cyan]reorder[/cyan]   Export synthesis order for dropout variants
        [cyan]merge[/cyan]     Merge picks from multiple rounds
        [cyan]report[/cyan]    Generate final plate maps

    [bold]Utility commands:[/bold]

        [cyan]estimate[/cyan]  Quick cost/effort estimation
        [cyan]platemap[/cyan]  Regenerate demux plate map HTML from existing results
        [cyan]remote[/cyan]    Run demux on a remote server via SSH
    """
    pass


# Register commands
app.command(name="estimate")(estimate)
app.command(name="plan")(plan)
app.command(name="skew")(skew)
app.command(name="demux")(demux)
app.command(name="pick")(pick)
app.command(name="reorder")(reorder)
app.command(name="merge")(merge)
app.command(name="report")(report)
app.command(name="platemap")(platemap)
app.command(name="pileups")(pileups)
app.add_typer(config_app, name="config")
app.add_typer(remote_app, name="remote")

__all__ = ["app", "estimate", "plan", "skew", "demux", "pick", "reorder", "merge", "report", "platemap", "remote_app"]
