#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
from __future__ import annotations

import sys
import textwrap

from tabpfn_client.constants import (
    URL_PRIOR_LABS_API_KEYS,
    URL_TABPFN_CLIENT_GITHUB_ISSUES,
)
from tabpfn_client.ui import (
    console,
    success,
    warn,
    print_logo,
)


def maybe_graceful_exit() -> None:
    try:
        from IPython import get_ipython  # type: ignore

        if get_ipython() is not None:
            return
    except ImportError:
        # We're in a script, just exit
        sys.exit(1)


class PromptAgent:
    def __new__(cls):
        raise RuntimeError(
            "This class should not be instantiated. Use classmethods instead."
        )

    @staticmethod
    def indent(text: str):
        indent_factor = 2
        indent_str = " " * indent_factor
        return textwrap.indent(text, indent_str)

    @staticmethod
    def _print(text: str) -> None:
        console.print(PromptAgent.indent(text))

    @classmethod
    def prompt_welcome(cls):
        # Large Prior Labs ASCII logo with a short tagline
        print_logo("Thanks for being part of the journey")
        console.print(
            cls.indent(
                "TabPFN is under active development, please help us improve and report any bugs/ideas you find."
            )
        )
        console.print(
            cls.indent(f"[cyan]Report issues: {URL_TABPFN_CLIENT_GITHUB_ISSUES}[/cyan]")
        )
        console.print(cls.indent("[cyan]Press Ctrl+C anytime to exit[/cyan]"))

    @classmethod
    def token_instructions(cls, rejected: bool = False) -> str:
        """The message shown when no usable access token is available.

        `rejected` distinguishes a token that the server turned down from no
        token having been supplied at all.
        """
        headline = (
            "Your TabPFN access token was rejected by the server."
            if rejected
            else "No TabPFN access token found."
        )
        return (
            f"{headline}\n"
            f"Please generate a token at {URL_PRIOR_LABS_API_KEYS} and either\n"
            "  - set it as the TABPFN_TOKEN environment variable, or\n"
            "  - pass it to tabpfn_client.set_access_token('<your-token>').\n"
            "\n"
            "In an interactive terminal you can instead run\n"
            "  from tabpfn_client import interactive_login; interactive_login()\n"
            "to log in or register via the browser."
        )

    @classmethod
    def prompt_for_token(cls) -> str | None:
        """Ask the user to paste an access token.

        Returns the token, or None if the user aborted. Only called when stdin
        is interactive; non-interactive callers get `token_instructions()` as an
        error instead.
        """
        console.print()
        warn("No TabPFN access token found.")
        console.print(
            f"  Please generate a token at [link={URL_PRIOR_LABS_API_KEYS}]"
            f"{URL_PRIOR_LABS_API_KEYS}[/link] and paste it here."
        )
        console.print(
            "  [cyan]Set TABPFN_TOKEN in your environment to skip this prompt.[/cyan]"
        )
        console.print("  [cyan]Press Ctrl+C to abort.[/cyan]")

        while True:
            token = console.input("\n[bold cyan]→[/bold cyan] Access token: ").strip()
            if token:
                return token
            warn("An access token is required.")

    @classmethod
    def prompt_token_accepted(cls):
        success("Access token accepted.")

    @classmethod
    def prompt_reusing_existing_token(cls):
        success("Found existing access token, reusing it for authentication.")

    @classmethod
    def prompt_retrieved_greeting_messages(cls, greeting_messages: list[str]):
        for message in greeting_messages:
            cls._print(message)

    CONFIRM_DELETION_PHRASE = "confirm deletion"

    @classmethod
    def confirm_user_account_deletion(cls) -> bool:
        warn(
            "You are about to delete your account. This is permanent and "
            "cannot be undone."
        )
        typed = console.input(
            f"Type '{cls.CONFIRM_DELETION_PHRASE}' to proceed (anything else cancels): "
        )
        return typed.strip().lower() == cls.CONFIRM_DELETION_PHRASE.lower()

    @classmethod
    def prompt_account_deleted(cls):
        success("Your account has been deleted.")
