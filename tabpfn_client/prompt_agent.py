#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
import getpass
import textwrap
from rich.table import Table

from password_strength import PasswordPolicy

from tabpfn_client.service_wrapper import UserAuthenticationClient
from tabpfn_client.state_manager import RegistrationState, check_internet_connection
from tabpfn_client.ui import (
    console,
    success,
    warn,
    fail,
    status,
    print_logo,
)


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
    def show_password_requirements(password: str, password_policy) -> list[str]:
        """Show which password requirements are met/unmet. Returns list of failed tests."""
        if not password:
            return []

        failed_tests = password_policy.test(password)
        return failed_tests

    @staticmethod
    def display_requirement_status(
        password: str, password_req: list[str], password_policy
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
                "TabPFN is still under active development, please help us improve and report any bugs/ideas you find."
            )
        )
        console.print(
            cls.indent(
                "[blue]Report issues: https://github.com/priorlabs/tabpfn-client/issues[/blue]"
            )
        )

    @classmethod
    def prompt_and_set_token(cls) -> bool:
        """Prompt for login/registration. Returns True if successful, False if interrupted."""
        try:
            result = cls._prompt_and_set_token_impl()
            # If _prompt_and_set_token_impl returns False (user quit), propagate it
            # If it returns True or None (success), return True
            if result is False:
                return False
            return True
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Goodbye![/yellow]")
            return False

    @classmethod
    def _prompt_and_set_token_impl(cls):
        # Check internet connection
        if not check_internet_connection():
            warn(
                "No internet connection detected. TabPFN client requires Internet access."
            )
            console.print("[blue]Please check your connection and try again.[/blue]")
            return False

        # Check for interrupted registration
        state_mgr = RegistrationState()
        saved_state = state_mgr.load()

        if saved_state and saved_state.get("email"):
            console.print(
                f"\n[yellow]Found interrupted registration for: {saved_state['email']}[/yellow]"
            )
            resume = (
                console.input("[bold blue]→[/bold blue] Resume? (y/n) [y]: ")
                .strip()
                .lower()
                or "y"
            )

            if resume in ["y", "yes"]:
                console.print("[blue]Resuming registration...[/blue]")
                # Continue with saved email
                resume_result = cls._resume_registration(saved_state)
                return resume_result
            else:
                state_mgr.clear()

        # Account access section — compact UI
        console.print()
        table = Table(box=None, show_header=False, pad_edge=False, show_edge=False)
        table.add_column("#", style="bold blue", width=5)
        table.add_column("Action")
        table.add_row("\\[1]", "Create a TabPFN account")
        table.add_row("\\[2]", "Login to your TabPFN account")
        table.add_row("\\[q]", "Quit")
        console.print(table)

        # Prompt for a valid choice using Rich input
        console.print("\n  [blue]Press Ctrl+C anytime to exit[/blue]")
        valid_choices = {"1", "2", "q", "b"}
        while True:
            choice = (
                console.input("\n[bold blue]→[/bold blue] Choose (1/2/q): ")
                .strip()
                .lower()
            )
            if choice in valid_choices:
                break
            if choice == "b":
                # Back navigation (currently at top level, so same as quit)
                console.print("Goodbye!")
                return False
            warn("Invalid choice. Please enter 1, 2, or q.")

        if choice == "q":
            console.print("Goodbye!")
            return False
        email = ""

        # Registration
        if choice == "1":
            validation_link = "tabpfn-2023"
            state_mgr = RegistrationState()

            # Show time estimate
            console.print("\n[blue]Registration: 6 steps (about 2 minutes)[/blue]")

            # Step 1: Terms
            console.print("\n[bold blue]Step 1/6[/bold blue] - Terms & Conditions")
            agreed_terms_and_cond = cls.prompt_terms_and_cond_simple()
            if not agreed_terms_and_cond:
                state_mgr.clear()
                raise RuntimeError(
                    "You must agree to the terms and conditions to use TabPFN"
                )

            # Step 2: Email
            console.print("\n[bold blue]Step 2/6[/bold blue] - Account Details")
            email = ""
            while True:
                email = input("Email: ").strip()
                if not email:
                    warn("Email is required.")
                    continue

                # Save state in case of interruption
                state_mgr.save({"email": email, "step": "email_validation"})

                with status("Validating email"):
                    is_valid, message = UserAuthenticationClient.validate_email(email)
                if is_valid:
                    break
                warn(f"  {message}")
                console.print(
                    "  [blue]Please try a different email or contact support if this seems incorrect.[/blue]"
                )

            # Step 3: Password
            console.print("\n[bold blue]Step 3/6[/bold blue] - Create Password")

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
                        "  [blue]Enter a password that meets all requirements.[/blue]"
                    )
                    continue

                # Confirm password
                password_confirm = getpass.getpass("Confirm password: ")
                if password == password_confirm:
                    break
                else:
                    warn("Passwords do not match.")
                    console.print("[blue]Please re-enter your password.[/blue]")
            # Step 4: Data Privacy
            console.print("\n[bold blue]Step 4/6[/bold blue] - Data Privacy")
            agreed_personally_identifiable_information = (
                cls.prompt_personally_identifiable_information_simple()
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
                    email, password, password_confirm, validation_link, additional_info
                )
            if not is_created:
                raise RuntimeError("User registration failed: " + str(message) + "\n")

            console.print()
            success("Account created successfully!")
            console.print(
                "  [blue]Almost done! Check your email for a verification code.[/blue]\n"
            )
            # Clear saved state on success
            state_mgr.clear()
            # verify token from email
            verified = cls._verify_user_email(access_token=access_token)
            if not verified:
                # User quit verification
                return False
            return True

        # Login
        elif choice == "2":
            console.print("\n[bold]Login[/bold]")
            email = input("Email: ")

            while True:
                password = getpass.getpass("Password: ")

                with status("Authenticating"):
                    (
                        access_token,
                        message,
                        status_code,
                    ) = UserAuthenticationClient.set_token_by_login(email, password)

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
                console.print()
                console.print(
                    "[bold blue]\\[1][/bold blue] Try password again [blue](default)[/blue]"
                )
                console.print("[bold blue]\\[2][/bold blue] Reset password")
                console.print("[bold blue]\\[3][/bold blue] Change email")
                console.print("[bold blue]\\[q][/bold blue] Quit")

                retry_choice = (
                    console.input("\n[bold blue]→[/bold blue] Choose (1/2/3/q) [1]: ")
                    .strip()
                    .lower()
                    or "1"
                )

                if retry_choice == "1":
                    console.print(f"[blue]Logging in as: {email}[/blue]")
                    continue
                elif retry_choice == "2":
                    console.print("\n[bold]Password Reset[/bold]")
                    console.print("We'll send a reset link to your email.")
                    with status("Sending password reset email"):
                        sent, reset_msg = (
                            UserAuthenticationClient.send_reset_password_email(email)
                        )

                    if sent:
                        success(f"Password reset email sent to {email}")
                        console.print(
                            "  [blue]Please check your email and return here after resetting.[/blue]"
                        )
                    else:
                        fail(f"Failed to send reset email: {reset_msg}")
                    return False
                elif retry_choice == "3":
                    email = input("\nNew email: ")
                    console.print(f"[blue]Switched to: {email}[/blue]")
                    continue
                elif retry_choice == "q":
                    console.print("Goodbye!")
                    return False
                else:
                    # Invalid choice, use default (retry)
                    console.print(f"[blue]Logging in as: {email}[/blue]")
                    continue

    @classmethod
    def prompt_terms_and_cond(cls) -> bool:
        console.print("\n[bold]Terms & Conditions[/bold]")
        console.print(
            "Please review: [link=https://www.priorlabs.ai/terms]https://www.priorlabs.ai/terms[/link]"
        )
        console.print("By using TabPFN, you agree to our terms and conditions.")

        while True:
            choice = (
                console.input("\n[bold blue]→[/bold blue] Do you agree? (y/n): ")
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
    def prompt_terms_and_cond_simple(cls) -> bool:
        """Simplified terms prompt for registration flow."""
        console.print(
            "By using TabPFN, you agree to the terms and conditions at [link=https://www.priorlabs.ai/terms]https://www.priorlabs.ai/terms[/link]"
        )

        while True:
            choice = (
                console.input("[bold blue]→[/bold blue] Agree to terms? (y/n): ")
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
        console.print("\n[bold]Data Privacy[/bold]")
        console.print("Please do not upload personal, sensitive, or confidential data.")

        while True:
            choice = (
                console.input("[bold blue]→[/bold blue] Do you agree? (y/n): ")
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
    def prompt_personally_identifiable_information_simple(cls) -> bool:
        """Simplified data privacy prompt for registration flow."""
        console.print("Do not upload personal/sensitive data.")

        while True:
            choice = (
                console.input("[bold blue]→[/bold blue] I understand (y/n): ")
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
            console.print(f"[bold blue]\\[{letter}][/bold blue] {option}")

        if allow_back:
            console.print("[bold blue]\\[b][/bold blue] Back to previous menu")

        # Generate valid letter choices
        valid_choices = [chr(ord("a") + i) for i in range(num_options)]
        if allow_back:
            valid_choices.append("b")

        while True:
            choice_letter = (
                console.input(
                    f"\n[bold blue]→[/bold blue] Choose ({'/'.join(valid_choices)}): "
                )
                .strip()
                .lower()
            )

            if not choice_letter:
                console.print("[blue]Please choose one of the options above[/blue]")
                continue

            if choice_letter == "b" and allow_back:
                return "__BACK__"

            if choice_letter in valid_choices:
                selected_index = ord(choice_letter) - ord("a")
                return options[selected_index]
            else:
                console.print(
                    f"  [blue]Hmm, that's not one of the options. Try {', '.join(valid_choices)}[/blue]"
                )

    @classmethod
    def prompt_and_retry(
        cls, prompt: str, min_length: int = 2, example: str = None
    ) -> str:
        """Prompt with validation and optional example."""
        console.print(f"\n{prompt}:")
        if example:
            console.print(f"[blue]Example: {example}[/blue]")

        while True:
            value = input("→ ").strip()
            if len(value) >= min_length:
                return value
            console.print(
                f"  [blue]Could you add a bit more? We need at least {min_length} characters.[/blue]"
            )

    @classmethod
    def prompt_add_user_information(cls) -> dict:
        console.print("\n[bold blue]Step 5/6[/bold blue] - Your Information")
        console.print("[blue]This helps us personalize your experience[/blue]")

        # Name field - required but combined for better UX
        name = ""
        while True:
            name = input("\nYour name: ").strip()
            if name:
                break
            console.print("[blue]We'd love to know what to call you![/blue]")

        # Split name for backward compatibility
        name_parts = name.split(None, 1)  # Split on first space
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ""

        console.print("\n[bold blue]Step 6/6[/bold blue] - Help Us Serve You Better")
        console.print("[blue]Just a few quick questions to get you started[/blue]")

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
                    "[bold blue]→[/bold blue] Can we contact you via email for support? (y/n) [y]: "
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
        console.print("[bold blue]\\[1][/bold blue] Verify email now")
        console.print("[bold blue]\\[2][/bold blue] Start over (login/register)")
        console.print("[bold blue]\\[q][/bold blue] Quit")

        while True:
            choice = (
                console.input("\n[bold blue]→[/bold blue] Choose (1/2/q): ")
                .strip()
                .lower()
            )
            if choice in ["1"]:
                break
            elif choice in ["2"]:
                console.print("[blue]Returning to main menu...[/blue]")
                return "restart"  # Signal to show main menu
            elif choice in ["q", "quit"]:
                console.print("Goodbye!")
                return False  # Signal to quit without showing menu
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
        choice = input(prompt)

        # retry until valid choice is made
        while True:
            if choice.lower() not in choices:
                choices_str = (
                    "', '".join([f"'{choice}'" for choice in choices[:-1]])
                    + f" or '{choices[-1]}'"
                )
                choice = input(f"Invalid choice, please enter {choices_str}: ")
            else:
                break

        return choice.lower()

    @classmethod
    def _verify_user_email(cls, access_token: str):
        console.print("\n[bold]Email Verification[/bold]")
        console.print("Enter the verification code sent to your email.")
        console.print(
            "[blue]Type 'resend' to get a new code, or 'quit' to exit.[/blue]"
        )

        while True:
            token = input("\nVerification code: ").strip()

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
                    console.print("[blue]Check your email for the new code.[/blue]")
                else:
                    fail(f"Failed to resend: {resend_msg}")
                continue

            if token.lower() == "quit":
                console.print("\n[yellow]Verification cancelled.[/yellow]")
                console.print(
                    "  [blue]You can verify your email later by logging in again.[/blue]"
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
                    "  [blue]Try again, type 'resend' for a new code, or 'quit' to exit.[/blue]"
                )

    @classmethod
    def _resume_registration(cls, saved_state: dict) -> bool:
        """Resume an interrupted registration. Returns True if successful."""
        email = saved_state.get("email")
        state_mgr = RegistrationState()

        console.print("\n[bold]Resuming Registration[/bold]")
        console.print(f"Email: {email}")

        validation_link = "tabpfn-2023"

        # Skip terms if already agreed (we're resuming)
        agreed_terms_and_cond = True

        # Continue with password setup
        with status("Retrieving password policy"):
            password_req = UserAuthenticationClient.get_password_policy()
        password_policy = cls.password_req_to_policy(password_req)

        # Show requirements upfront
        console.print("\n[bold]Password Requirements[/bold]")
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
                cls.display_requirement_status(password, password_req, password_policy)
                console.print(
                    "  [blue]Enter a password that meets all requirements.[/blue]"
                )
                continue

            password_confirm = getpass.getpass("Confirm password: ")
            if password == password_confirm:
                break
            else:
                warn("Passwords do not match.")
                console.print("[blue]Please re-enter your password.[/blue]")

        agreed_personally_identifiable_information = (
            cls.prompt_personally_identifiable_information()
        )
        if not agreed_personally_identifiable_information:
            state_mgr.clear()
            raise RuntimeError("You must agree to not upload personal data.")

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
                email, password, password_confirm, validation_link, additional_info
            )

        if not is_created:
            raise RuntimeError("User registration failed: " + str(message) + "\n")

        console.print()
        success("Account created successfully!")
        console.print("Check your email for a verification code.\n")
        state_mgr.clear()
        verified = cls._verify_user_email(access_token=access_token)
        if not verified:
            # User quit verification
            return False
        return True
