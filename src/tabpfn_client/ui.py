"""Utilities for consistent terminal output across the TabPFN client."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


def _should_use_color() -> bool:
    """Determine whether color output should be used."""

    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    return True


console = Console(soft_wrap=False, highlight=True, force_terminal=_should_use_color())


def setup_logging(verbosity: int = 0) -> None:
    """Configure logging to emit through Rich with a consistent style."""

    level = logging.WARNING - min(verbosity, 2) * 10
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_time=False,
                show_path=False,
            )
        ],
    )


def header(title: str, subtitle: str | None = None) -> None:
    """Render a section header."""

    console.print(
        Panel.fit(
            title if not subtitle else f"[bold]{title}[/bold]\n[dim]{subtitle}[/dim]"
        )
    )


logger = logging.getLogger(__name__)


def notify(message: str) -> None:
    """Status text for a human at a terminal, a log record anywhere else.

    Output that is not part of an interactive prompt goes through here, so a
    script or batch job gets a quiet stdout it can redirect and parse.
    """
    if sys.stdout.isatty():
        console.print(message)
    else:
        logger.info(message)


def success(message: str) -> None:
    console.print(f"[bold green]{message}[/bold green]")


def warn(message: str) -> None:
    console.print(f"[yellow]{message}[/yellow]")


def fail(message: str) -> None:
    console.print(f"[bold red]{message}[/bold red]")


def info(message: str) -> None:
    console.print(f"[blue]{message}[/blue]")


@contextmanager
def status(message: str) -> Generator[None]:
    with console.status(f"[bold]{message}[/bold]"):
        yield


def progress_bar(description: str = "Working...") -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


# =============================
# Branding: Prior Labs ASCII
# =============================

_PRIOR_LABS_ASCII = r"""
########  ########   ###  #########  #########       ###         #####     ########  ########
     ###        ##   ###  ###   ###        ###       ###        ###  ###   ##   ###  ###     
########  #######    ###  ###   ###  #######         ###        ########   ######    ########
###       ###   ##   ###  ###   ###  ###   ###       ###        ###  ###   ##   ###       ###
###       ###   ##   ###  #########  ###   ###       ########   ###  ###   ########  ########                                                     
"""

_PRIOR_LABS_ASCII_SMALL = r"""
[ PRIOR LABS ]
"""


def print_logo(subtitle=None) -> None:
    """Print the large Prior Labs ASCII logo with optional subtitle."""
    console.print(_PRIOR_LABS_ASCII, style="bold blue")
    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]", end="\n\n")


def print_logo_small(subtitle=None) -> None:
    """Print a small Prior Labs ASCII banner with optional subtitle."""
    console.print(_PRIOR_LABS_ASCII_SMALL, style="bold blue")
    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]")


__all__ = [
    "console",
    "fail",
    "header",
    "print_logo",
    "print_logo_small",
    "progress_bar",
    "setup_logging",
    "status",
    "success",
    "info",
    "warn",
]
