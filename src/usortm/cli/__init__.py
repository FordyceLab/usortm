"""Command-line interface for uSort-M."""

import typer

from .estimate import estimate
from .plan import plan
from .demux_cmd import demux
from .pick import pick
from .reorder import reorder
from .report import report
from .config_cmd import config_app
from .platemap import platemap

# Create the main Typer app
app = typer.Typer(
    name="usortm",
    help="uSort-M: Scalable and cost-effective arrayed gene library generation",
    add_completion=False,
)

# Register commands
app.command(name="estimate")(estimate)
app.command(name="plan")(plan)
app.command(name="demux")(demux)
app.command(name="pick")(pick)
app.command(name="reorder")(reorder)
app.command(name="report")(report)
app.command(name="platemap")(platemap)
app.add_typer(config_app, name="config")

__all__ = ["app", "estimate", "plan", "demux", "pick", "reorder", "report", "platemap"]
