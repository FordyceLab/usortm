"""Centralized Rich theme for the uSort-M CLI."""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

# Brand blue: visible on both dark and light terminal backgrounds
_BRAND = "#4096E3"
# Muted text: replaces "dim" which is illegible on light backgrounds
_MUTED = "#888888"

USORTM_THEME = Theme({
    "brand": f"bold {_BRAND}",
    "brand.plain": _BRAND,
    "muted": _MUTED,
})

# Border style for panels (can't reference theme styles in border_style param)
BORDER_STYLE = _BRAND


def get_console() -> Console:
    """Create a Console pre-configured with the uSort-M theme."""
    return Console(theme=USORTM_THEME)


def section(console: Console, title: str) -> None:
    """Start a new section of console output.

    Always preceded by a blank line.  Left to each call site the spacing came
    out uneven -- a heading sat flush against the last line of the section
    before it wherever whoever wrote that block had not thought to print one,
    which reads as though the heading belongs to the lines above rather than
    the ones below.

    Args:
        console: The console to write to.
        title: Heading text, without markup; it is emboldened here.
    """
    console.print()
    console.rule(f"[bold]{title}[/bold]", style=BORDER_STYLE, align="left")
