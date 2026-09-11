"""Utilities for consistent terminal output across the TabPFN client."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager

from rich.console import Console


def _should_use_color() -> bool:
    """Determine whether color output should be used."""

    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    return True


console = Console(soft_wrap=False, highlight=True, force_terminal=_should_use_color())

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


@contextmanager
def status(message: str) -> Generator[None]:
    with console.status(f"[bold]{message}[/bold]"):
        yield


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


def print_logo(subtitle=None) -> None:
    """Print the large Prior Labs ASCII logo with optional subtitle."""
    console.print(_PRIOR_LABS_ASCII, style="bold blue")
    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]", end="\n\n")


__all__ = [
    "console",
    "fail",
    "notify",
    "print_logo",
    "status",
    "success",
    "warn",
]
