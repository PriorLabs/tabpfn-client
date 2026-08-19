#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
from __future__ import annotations

import getpass
import sys
import textwrap

from password_strength import PasswordPolicy

from tabpfn_client.constants import (
    URL_PRIOR_LABS_API_KEYS,
    URL_PRIOR_LABS_PRIVACY_POLICY,
    URL_PRIOR_LABS_TERMS_AND_CONDITIONS,
    URL_TABPFN_CLIENT_GITHUB_ISSUES,
)
from tabpfn_client.service_wrapper import UserAuthenticationClient
from tabpfn_client.ui import (
    console,
    fail,
    status,
    success,
    warn,
    print_logo,
)

# The registration link the server expects; kept from the original flow.
VALIDATION_LINK = "tabpfn-2023"

ROLES = [
    "Data Scientist",
    "ML Engineer",
    "AI Engineer",
    "Software Engineer",
    "Product Manager",
    "Researcher",
    "Student",
    "Executive",
    "Other",
]


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

    # ------------------------------------------------------------------
    # Signup (reached only through interactive_login)
    # ------------------------------------------------------------------

    @staticmethod
    def password_req_to_policy(password_req: list[str]):
        """
        Convert password requirement strings like "Length(8)" into a PasswordPolicy.
        """
        requirements = {}
        for req in password_req:
            word_part, number_part = req.split("(")
            number = int(number_part[:-1])
            requirements[word_part.lower()] = number
        return PasswordPolicy.from_names(**requirements)

    @classmethod
    def display_requirement_status(
        cls, password: str, password_req: list[str], password_policy: PasswordPolicy
    ) -> None:
        """Display check marks for met/unmet requirements."""
        if not password:
            return

        failed_names = {test.name() for test in password_policy.test(password)}

        console.print("  Requirements:")
        for req in password_req:
            req_key = req.split("(")[0].lower()
            if req_key not in failed_names:
                console.print(f"    [green]✓[/green] {req}")
            else:
                console.print(
                    f"    [bright_black]•[/bright_black] [bright_black]{req}[/bright_black]"
                )

    @classmethod
    def prompt_multi_select(cls, options: list[str], prompt: str) -> str:
        """Creates an interactive single-choice menu over `options`."""
        console.print(f"\n[bold]{prompt}[/bold]")
        for i, option in enumerate(options):
            console.print(f"[bold cyan]\\[{chr(ord('a') + i)}][/bold cyan] {option}")

        valid_choices = [chr(ord("a") + i) for i in range(len(options))]
        while True:
            choice_letter = (
                console.input(
                    f"\n[bold cyan]→[/bold cyan] Choose ({'/'.join(valid_choices)}): "
                )
                .strip()
                .lower()
            )
            if choice_letter in valid_choices:
                return options[ord(choice_letter) - ord("a")]
            console.print(
                f"  [cyan]Hmm, that's not one of the options. Try {', '.join(valid_choices)}[/cyan]"
            )

    @classmethod
    def prompt_and_retry(cls, prompt: str, min_length: int = 2) -> str:
        """Prompt with a minimum-length check."""
        console.print(f"\n{prompt}:")
        while True:
            value = console.input("→ ").strip()
            if len(value) >= min_length:
                return value
            console.print(
                f"  [cyan]Could you add a bit more? We need at least {min_length} characters.[/cyan]"
            )

    @classmethod
    def _prompt_account_details(cls) -> tuple[str, str]:
        """Step 1: email and password. Returns (email, password)."""
        console.print("\n[bold cyan]Step 1/3[/bold cyan] - Account details")

        # Replaces the separate Terms and Data Privacy steps. Printed before the
        # prompt rather than after it, so it is on screen before the user
        # commits anything -- the CLI equivalent of sitting under the field.
        console.print(
            "\n[bright_black]By creating an account you agree to our "
            f"[link={URL_PRIOR_LABS_TERMS_AND_CONDITIONS}]Terms of Service[/link] "
            "and acknowledge our "
            f"[link={URL_PRIOR_LABS_PRIVACY_POLICY}]Privacy Policy[/link]. "
            "Do not upload personal, confidential or sensitive data.[/bright_black]"
        )

        while True:
            email = console.input("\nEmail: ").strip()
            if not email:
                warn("Email is required.")
                continue

            with status("Validating email"):
                is_valid, message = UserAuthenticationClient.validate_email(email)
            if is_valid:
                break
            warn(f"  {message}")
            console.print(
                "  [cyan]Please try a different email, or contact support if this seems incorrect.[/cyan]"
            )

        with status("Retrieving password policy"):
            password_req = UserAuthenticationClient.get_password_policy()
        password_policy = cls.password_req_to_policy(password_req)

        console.print("\n  Requirements:")
        for req in password_req:
            console.print(f"    [bright_black]•[/bright_black] {req}")

        while True:
            password = getpass.getpass("\nPassword: ")
            if password_policy.test(password):
                console.print()
                cls.display_requirement_status(password, password_req, password_policy)
                console.print(
                    "  [cyan]Enter a password that meets all requirements.[/cyan]"
                )
                continue

            if password == getpass.getpass("Confirm password: "):
                return email, password
            warn("Passwords do not match.")
            console.print("[cyan]Please re-enter your password.[/cyan]")

    @classmethod
    def _prompt_profile(cls) -> dict:
        """Step 2: profile and the marketing opt-in."""
        console.print("\n[bold cyan]Step 2/3[/bold cyan] - Complete your profile")

        while True:
            first_name = console.input("\nFirst name: ").strip()
            if first_name:
                break
            console.print("[cyan]We'd love to know what to call you![/cyan]")

        while True:
            last_name = console.input("Last name: ").strip()
            if last_name:
                break
            console.print("[cyan]And your last name too![/cyan]")

        company = cls.prompt_and_retry("Where do you work?")

        role = cls.prompt_multi_select(ROLES, "What is your current role?")
        if role == "Other":
            role = cls.prompt_and_retry("Please specify your role")

        console.print(
            "\nYes to emails from Prior Labs with product news, offers and resources. "
            "Unsubscribe anytime."
        )
        # Defaults to no: consent to marketing has to be an affirmative act.
        choice = (
            console.input("[bold cyan]→[/bold cyan] Subscribe? (y/N): ").strip().lower()
        )
        contact_via_email = choice in ("y", "yes")

        return {
            "first_name": first_name,
            "last_name": last_name,
            "company": company,
            "role": role,
            "contact_via_email": contact_via_email,
            # Accepted via the notice under the email field in step 1.
            "agreed_terms_and_cond": True,
            "agreed_personally_identifiable_information": True,
        }

    @classmethod
    def prompt_signup(cls) -> str | None:
        """Run the three-step signup. Returns an access token, or None if aborted.

        The token is only returned once the email is verified: an unverified
        account cannot authenticate, so handing the token back earlier would
        just fail downstream.
        """
        email, password = cls._prompt_account_details()
        additional_info = cls._prompt_profile()

        with status("Creating account"):
            is_created, message, access_token = (
                UserAuthenticationClient.set_token_by_registration(
                    email, password, password, VALIDATION_LINK, additional_info
                )
            )
        if not is_created:
            raise RuntimeError(f"User registration failed: {message}")

        console.print()
        success("Account created successfully!")
        console.print(
            "  [cyan]Almost done! Check your email for a verification code.[/cyan]"
        )

        console.print("\n[bold cyan]Step 3/3[/bold cyan] - Verify your email")
        if not cls._verify_user_email(access_token=access_token):
            return None
        return access_token

    @classmethod
    def _verify_user_email(cls, access_token: str | None) -> bool:
        if access_token is None:
            fail("No access token available for email verification.")
            return False
        console.print("Enter the verification code sent to your email.")
        console.print(
            "[cyan]Type 'resend' to get a new code, or 'quit' to exit.[/cyan]"
        )

        while True:
            token = console.input("\nVerification code: ").strip()

            if not token:
                warn("Please enter a verification code.")
                continue

            if token.lower() == "resend":
                with status("Sending new verification code"):
                    sent, resend_msg = UserAuthenticationClient.send_verification_email(
                        access_token
                    )
                if sent:
                    success("New verification code sent!")
                    console.print("[cyan]Check your email for the new code.[/cyan]")
                else:
                    fail(f"Failed to resend: {resend_msg}")
                continue

            if token.lower() == "quit":
                console.print("\n[yellow]Verification cancelled.[/yellow]")
                console.print(
                    "  [cyan]You can verify your email later by logging in again.[/cyan]"
                )
                return False

            with status("Verifying"):
                verified, message = UserAuthenticationClient.verify_email(
                    token, access_token
                )

            if verified:
                success("Email verified successfully!")
                return True

            warn(f"{message}")
            console.print(
                "  [cyan]Try again, type 'resend' for a new code, or 'quit' to exit.[/cyan]"
            )

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
