#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
import getpass
import sys
import textwrap
from rich.table import Table

from password_strength import PasswordPolicy

from tabpfn_client.service_wrapper import UserAuthenticationClient
from tabpfn_client.ui import (
    console,
    success,
    warn,
    fail,
    status,
    print_logo,
)


def maybe_graceful_exit() -> None:
    try:
        from IPython import get_ipython

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

    @staticmethod
    def show_password_requirements(
        password: str, password_policy: PasswordPolicy
    ) -> list[str]:
        """Show which password requirements are met/unmet. Returns list of failed tests."""
        if not password:
            return []

        failed_tests = password_policy.test(password)
        return failed_tests

    @staticmethod
    def display_requirement_status(
        password: str, password_req: list[str], password_policy: PasswordPolicy
    ) -> None:
        """Display check marks for met/unmet requirements."""
        if not password:
            return

        failed_tests = password_policy.test(password)
        failed_names = {test.name() for test in failed_tests}

        console.print("  Requirements:")
        for req in password_req:
            # Parse requirement like "Length(8)" -> ("length", "8")
            word_part, number_part = req.split("(")
            req_key = word_part.lower()

            # Check if this requirement is in failed tests
            is_met = req_key not in failed_names
            if is_met:
                icon = "[green]✓[/green]"
                text = req
            else:
                icon = "[bright_black]•[/bright_black]"
                text = f"[bright_black]{req}[/bright_black]"

            console.print(f"    {icon} {text}")

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
            cls.indent(
                "[cyan]Report issues: https://github.com/priorlabs/tabpfn-client/issues[/cyan]"
            )
        )
        console.print(cls.indent("[cyan]Press Ctrl+C anytime to exit[/cyan]"))

    @classmethod
    def prompt_and_set_token(cls) -> bool:
        """Prompt for login/registration. Returns True if successful, False if interrupted."""
        try:
            success, message = UserAuthenticationClient.try_browser_login()
            if success:
                console.print("[green]Login via browser successful![/green]")
                return True

            result = cls._prompt_and_set_token_impl()
            # If _prompt_and_set_token_impl returns False (user quit), propagate it
            # If it returns True or None (success), return True
            if result is False:
                return False
            return True
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Goodbye![/yellow]")
            maybe_graceful_exit()
            return False

    @classmethod
    def _prompt_and_set_token_impl(cls) -> bool:
        # Account access section — compact UI
        console.print(cls.indent("\n"))
        table = Table(box=None, show_header=False, pad_edge=False, show_edge=False)
        table.add_column("#", style="bold cyan", width=5)
        table.add_column("Action")
        table.add_row("\\[1]", "Create a TabPFN account")
        table.add_row("\\[2]", "Login to your TabPFN account")
        table.add_row("\\[q]", "Quit")
        console.print(table)

        # Prompt for a valid choice using Rich input
        valid_choices = {"1", "2", "q"}
        while True:
            choice = (
                console.input("\n[bold cyan]→[/bold cyan] Choose (1/2/q): ")
                .strip()
                .lower()
            )
            if choice in valid_choices:
                break
            warn("Invalid choice. Please enter 1, 2, or q.")

        if choice == "q":
            console.print("Goodbye!")
            maybe_graceful_exit()
            return False

        # Registration
        if choice == "1":
            validation_link = "tabpfn-2023"

            # Show time estimate
            console.print("\n[cyan]Registration: 6 steps (about 2 minutes)[/cyan]")

            # Step 1: Terms
            console.print("\n[bold cyan]Step 1/6[/bold cyan] - Terms & Conditions")
            agreed_terms_and_cond = cls.prompt_terms_and_cond()
            if not agreed_terms_and_cond:
                raise RuntimeError(
                    "You must agree to the terms and conditions to use TabPFN"
                )

            # Step 2: Email
            console.print("\n[bold cyan]Step 2/6[/bold cyan] - Account Details")
            while True:
                email = console.input("Email: ").strip()
                if not email:
                    warn("Email is required.")
                    continue

                with status("Validating email"):
                    is_valid, message = UserAuthenticationClient.validate_email(
                        str(email)
                    )
                if is_valid:
                    break
                warn(f"  {message}")
                console.print(
                    "  [cyan]Please try a different email or contact support if this seems incorrect.[/cyan]"
                )

            # Step 3: Password
            console.print("\n[bold cyan]Step 3/6[/bold cyan] - Create Password")

            with status("Retrieving password policy"):
                password_req = UserAuthenticationClient.get_password_policy()
            password_policy = cls.password_req_to_policy(password_req)

            # Show requirements upfront
            console.print("\n  Requirements:")
            for req in password_req:
                console.print(f"    [bright_black]•[/bright_black] {req}")

            password = None
            while True:
                password = getpass.getpass("\nPassword: ")

                # Validate password requirements
                failed_tests = password_policy.test(password)
                if len(failed_tests) != 0:
                    console.print()
                    cls.display_requirement_status(
                        password, password_req, password_policy
                    )
                    console.print(
                        "  [cyan]Enter a password that meets all requirements.[/cyan]"
                    )
                    continue

                # Confirm password
                password_confirm = getpass.getpass("Confirm password: ")
                if password == password_confirm:
                    break
                else:
                    warn("Passwords do not match.")
                    console.print("[cyan]Please re-enter your password.[/cyan]")
            # Step 4: Data Privacy
            console.print("\n[bold cyan]Step 4/6[/bold cyan] - Data Privacy")
            agreed_personally_identifiable_information = (
                cls.prompt_personally_identifiable_information()
            )
            if not agreed_personally_identifiable_information:
                raise RuntimeError("You must agree to not upload personal data.")

            # Step 5 & 6: User info
            additional_info = cls.prompt_add_user_information()
            additional_info["agreed_terms_and_cond"] = agreed_terms_and_cond
            additional_info["agreed_personally_identifiable_information"] = (
                agreed_personally_identifiable_information
            )
            with status("Creating account"):
                (
                    is_created,
                    message,
                    access_token,
                ) = UserAuthenticationClient.set_token_by_registration(
                    str(email),
                    str(password),
                    str(password_confirm),
                    validation_link,
                    additional_info,
                )
            if not is_created:
                raise RuntimeError("User registration failed: " + str(message) + "\n")

            console.print()
            success("Account created successfully!")
            console.print(
                "  [cyan]Almost done! Check your email for a verification code.[/cyan]\n"
            )
            # verify token from email
            verified = cls._verify_user_email(access_token=access_token)
            if not verified:
                # User quit verification
                return False
            return True

        # Login
        elif choice == "2":
            console.print("\n[bold]Login[/bold]")
            email = console.input("Email: ").strip()

            while True:
                password = getpass.getpass("Password: ")

                # Ensure both are strings for URL encoding
                if not password:
                    warn("Password is required.")
                    continue

                with status("Authenticating"):
                    (
                        access_token,
                        message,
                        status_code,
                    ) = UserAuthenticationClient.set_token_by_login(
                        str(email), str(password)
                    )

                if status_code == 200 and access_token is not None:
                    success("Login successful!")
                    return True

                if status_code == 403:
                    # 403 implies that the email is not verified
                    warn("Email not verified.")
                    verified = cls._verify_user_email(access_token=access_token)
                    if not verified:
                        # User quit verification
                        return False
                    # After verification, try login again
                    with status("Authenticating"):
                        (
                            access_token,
                            message,
                            status_code,
                        ) = UserAuthenticationClient.set_token_by_login(email, password)
                    if status_code == 200 and access_token is not None:
                        success("Login successful!")
                        return True
                    # If still failing, show error and continue loop
                    continue

                # Login failed - show options
                fail(f"Login failed: {message}")
                console.print("\n[bold]What would you like to do?[/bold]")
                console.print("[bold cyan]\\[1][/bold cyan] Try again with same email")
                console.print("[bold cyan]\\[2][/bold cyan] Login with different email")
                console.print("[bold cyan]\\[3][/bold cyan] Reset password via email")
                console.print("[bold cyan]\\[q][/bold cyan] Quit")

                retry_choice = (
                    console.input(
                        "\n[bold cyan]→[/bold cyan] Choose (1/2/3/q) [default: 1]: "
                    )
                    .strip()
                    .lower()
                    or "1"
                )

                if retry_choice == "1":
                    console.print(f"[cyan]Logging in as: {email}[/cyan]")
                    continue
                elif retry_choice == "3":
                    console.print("\n[bold]Password Reset[/bold]")
                    console.print("We'll send a reset link to your email.")
                    with status("Sending password reset email"):
                        sent, reset_msg = (
                            UserAuthenticationClient.send_reset_password_email(
                                str(email)
                            )
                        )

                    if sent:
                        success(f"Password reset email sent to {email}")
                        console.print(
                            "  [cyan]Please check your email and return here after resetting.[/cyan]"
                        )
                    else:
                        fail(f"Failed to send reset password: {reset_msg}")
                    return False
                elif retry_choice == "2":
                    email = console.input("\nEmail: ").strip()
                    console.print(f"[cyan]Switched to: {email}[/cyan]")
                    continue
                elif retry_choice == "q":
                    console.print("Goodbye!")
                    maybe_graceful_exit()
                    return False
                else:
                    # Invalid choice, use default (retry)
                    console.print(f"[cyan]Logging in as: {email}[/cyan]")
                    continue

    @classmethod
    def prompt_terms_and_cond(cls) -> bool:
        """Simplified terms prompt for registration flow."""
        console.print(
            "By using TabPFN, you agree to the terms and conditions at [link=https://www.priorlabs.ai/terms]https://www.priorlabs.ai/terms[/link]"
        )

        while True:
            choice = (
                console.input("[bold cyan]→[/bold cyan] I agree? (y/n): ")
                .strip()
                .lower()
            )
            if choice in ["y", "yes"]:
                return True
            elif choice in ["n", "no"]:
                return False
            else:
                warn("Please enter 'y' or 'n'.")

    @classmethod
    def prompt_personally_identifiable_information(cls) -> bool:
        """Simplified data privacy prompt for registration flow."""
        console.print("I agree not to upload personal, confidential or sensitive data.")

        while True:
            choice = (
                console.input("[bold cyan]→[/bold cyan] I agree (y/n): ")
                .strip()
                .lower()
            )
            if choice in ["y", "yes"]:
                return True
            elif choice in ["n", "no"]:
                return False
            else:
                warn("Please enter 'y' or 'n'.")

    @classmethod
    def clear_console(cls) -> None:
        console.clear()

    @classmethod
    def prompt_multi_select(
        cls, options: list[str], prompt: str, allow_back: bool = False
    ) -> str:
        """Creates an interactive multi select"""
        num_options = len(options)

        console.print(f"\n[bold]{prompt}[/bold]")

        # Print the lettered menu options
        for i, option in enumerate(options):
            letter = chr(ord("a") + i)
            console.print(f"[bold cyan]\\[{letter}][/bold cyan] {option}")

        if allow_back:
            console.print("[bold cyan]\\[b][/bold cyan] Back to previous menu")

        # Generate valid letter choices
        valid_choices = [chr(ord("a") + i) for i in range(num_options)]
        if allow_back:
            valid_choices.append("b")

        while True:
            choice_letter = (
                console.input(
                    f"\n[bold cyan]→[/bold cyan] Choose ({'/'.join(valid_choices)}): "
                )
                .strip()
                .lower()
            )

            if not choice_letter:
                console.print("[cyan]Please choose one of the options above[/cyan]")
                continue

            if choice_letter == "b" and allow_back:
                return "__BACK__"

            if choice_letter in valid_choices:
                selected_index = ord(choice_letter) - ord("a")
                return options[selected_index]
            else:
                console.print(
                    f"  [cyan]Hmm, that's not one of the options. Try {', '.join(valid_choices)}[/cyan]"
                )

    @classmethod
    def prompt_and_retry(
        cls, prompt: str, min_length: int = 2, example: str = None
    ) -> str:
        """Prompt with validation and optional example."""
        console.print(f"\n{prompt}:")
        if example:
            console.print(f"[cyan]Example: {example}[/cyan]")

        while True:
            value = console.input("→ ").strip()
            if len(value) >= min_length:
                return value
            console.print(
                f"  [cyan]Could you add a bit more? We need at least {min_length} characters.[/cyan]"
            )

    @classmethod
    def prompt_add_user_information(cls) -> dict:
        console.print("\n[bold cyan]Step 5/6[/bold cyan] - Your Information")
        console.print("[cyan]This helps us personalize your experience[/cyan]")

        # Name fields - ask separately to ensure both are provided
        first_name = ""
        while True:
            first_name = console.input("\nFirst name: ").strip()
            if first_name:
                break
            console.print("[cyan]We'd love to know what to call you![/cyan]")

        last_name = ""
        while True:
            last_name = console.input("Last name: ").strip()
            if last_name:
                break
            console.print("[cyan]And your last name too![/cyan]")

        console.print("\n[bold cyan]Step 6/6[/bold cyan] - Help Us Serve You Better")
        console.print("[cyan]Just a few quick questions to get you started[/cyan]")

        company = cls.prompt_and_retry("Where do you work?")

        role = cls.prompt_multi_select(
            ["Field practitioner", "Researcher", "Student", "Other"],
            "What is your current role?",
        )
        if role == "Other":
            role = cls.prompt_and_retry("Please specify your role")

        use_case = cls.prompt_and_retry(
            "What do you want to use TabPFN for?",
            min_length=10,
            example="Predicting customer churn in a SaaS application",
        )

        console.print()
        while True:
            choice = (
                console.input(
                    "[bold cyan]→[/bold cyan] Can we contact you via email for support? (y/n) [y]: "
                )
                .strip()
                .lower()
                or "y"
            )
            if choice in ["y", "yes"]:
                contact_via_email = True
                break
            elif choice in ["n", "no"]:
                contact_via_email = False
                break
            else:
                warn("Please enter 'y' or 'n'.")

        return {
            "first_name": first_name,
            "last_name": last_name,
            "company": company,
            "role": role,
            "use_case": use_case,
            "contact_via_email": contact_via_email,
        }

    @classmethod
    def prompt_reusing_existing_token(cls):
        success("Found existing access token, reusing it for authentication.")

    @classmethod
    def reverify_email(cls, access_token):
        """Prompt for email verification. Returns True if successful, 'restart' to show main menu, False to quit."""
        console.print("\n[bold]Email Verification Required[/bold]")
        console.print("Your account exists but email is not verified.")
        console.print()
        console.print("[bold cyan]\\[1][/bold cyan] Verify email now")
        console.print("[bold cyan]\\[2][/bold cyan] Start over (login/register)")
        console.print("[bold cyan]\\[q][/bold cyan] Quit")

        while True:
            choice = (
                console.input("\n[bold cyan]→[/bold cyan] Choose (1/2/q): ")
                .strip()
                .lower()
            )
            if choice in ["1"]:
                break
            elif choice in ["2"]:
                console.print("[cyan]Returning to main menu...[/cyan]")
                return "restart"  # Signal to show main menu
            elif choice in ["q", "quit"]:
                console.print("Goodbye!")
                maybe_graceful_exit()
                return False
            else:
                warn("Please enter 1, 2, or q.")

        # Go directly to verification - the prompt already has resend option
        # verify token from email
        verified = cls._verify_user_email(access_token=access_token)
        if verified:
            UserAuthenticationClient.set_token(access_token)
            return True
        return False  # User quit during verification

    @classmethod
    def prompt_retrieved_greeting_messages(cls, greeting_messages: list[str]):
        for message in greeting_messages:
            cls._print(message)

    @classmethod
    def prompt_confirm_password_for_user_account_deletion(cls) -> str:
        warn("You are about to delete your account.")
        confirm_pass = getpass.getpass("Please confirm by entering your password: ")

        return confirm_pass

    @classmethod
    def prompt_account_deleted(cls):
        success("Your account has been deleted.")

    @classmethod
    def _choice_with_retries(cls, prompt: str, choices: list) -> str:
        """
        Prompt text and give user infinitely many attempts to select one of the possible choices. If valid choice
        is selected, return choice in lowercase.
        """
        assert all(c.lower() == c for c in choices), "Choices need to be lower case."
        choice = console.input(prompt)

        # retry until valid choice is made
        while True:
            if choice.lower() not in choices:
                choices_str = (
                    "', '".join([f"'{choice}'" for choice in choices[:-1]])
                    + f" or '{choices[-1]}'"
                )
                choice = console.input(f"Invalid choice, please enter {choices_str}: ")
            else:
                break

        return choice.lower()

    @classmethod
    def _verify_user_email(cls, access_token: str) -> bool:
        console.print("\n[bold]Email Verification[/bold]")
        console.print("Enter the verification code sent to your email.")
        console.print(
            "[cyan]Type 'resend' to get a new code, or 'quit' to exit.[/cyan]"
        )

        while True:
            token = console.input("\nVerification code: ").strip()

            if not token:
                warn("Please enter a verification code.")
                continue

            # Handle special commands
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

            # Verify the code
            with status("Verifying"):
                verified, message = UserAuthenticationClient.verify_email(
                    token, access_token
                )

            if verified:
                success("Email verified successfully!")
                return True
            else:
                warn(f"{message}")
                console.print(
                    "  [cyan]Try again, type 'resend' for a new code, or 'quit' to exit.[/cyan]"
                )
