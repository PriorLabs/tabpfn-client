"""Utilities for consistent terminal output across the TabPFN client."""
from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from typing import Iterator, List

from rich import traceback
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


def _collect_suppressed_modules() -> List[object]:
    suppressed: List[object] = []
    for name in ("typer", "click", "rich"):
        try:
            module = __import__(name)
        except ModuleNotFoundError:
            continue
        else:
            suppressed.append(module)
    return suppressed


traceback.install(show_locals=False, suppress=_collect_suppressed_modules())


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


def success(message: str) -> None:
    console.print(f":white_check_mark: [bold green]{message}[/bold green]")


def warn(message: str) -> None:
    console.print(f":warning: [yellow]{message}[/yellow]")


def fail(message: str) -> None:
    console.print(f":x: [bold red]{message}[/bold red]")


@contextmanager
def status(message: str) -> Iterator[None]:
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


__all__ = [
    "console",
    "fail",
    "header",
    "progress_bar",
    "setup_logging",
    "status",
    "success",
    "warn",
]
