#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import textwrap
import getpass
from password_strength import PasswordPolicy

from tabpfn_client.service_wrapper import UserAuthenticationClient


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
    def password_req_to_policy(password_req: list[str]):
        """
        Small function that receives password requirements as a list of
        strings like "Length(8)" and returns a corresponding
        PasswordPolicy object.
        """
        requirements = {}
        for req in password_req:
            word_part, number_part = req.split("(")
            number = int(number_part[:-1])
            requirements[word_part.lower()] = number
        return PasswordPolicy.from_names(**requirements)

    @classmethod
    def prompt_welcome(cls):
        prompt = "\n".join(
            [
                "Welcome to TabPFN!",
                "",
                "TabPFN is still under active development, and we are working hard to make it better.",
                "Please bear with us if you encounter any issues.",
                "",
            ]
        )

        print(cls.indent(prompt))

    @classmethod
    def prompt_and_set_token(cls):
        # Try browser login first
        success, message = UserAuthenticationClient.try_browser_login()
        if success:
            print(cls.indent("Login via browser successful!"))
            return

        # Rest of the existing CLI login code
        prompt = "\n".join(
            [
                "Please choose one of the following options:",
                "(1) Create a TabPFN account",
                "(2) Login to your TabPFN account",
                "",
                "Please enter your choice: ",
            ]
        )
        choice = cls._choice_with_retries(prompt, ["1", "2"])
        email = ""

        # Registration
        if choice == "1":
            validation_link = "tabpfn-2023"

            agreed_terms_and_cond = cls.prompt_terms_and_cond()
            if not agreed_terms_and_cond:
                raise RuntimeError(
                    "You must agree to the terms and conditions to use TabPFN"
                )

            while True:
                email = input(cls.indent("Please enter your email: "))
                # Send request to server to check if email is valid and not already taken.
                is_valid, message = UserAuthenticationClient.validate_email(email)
                if is_valid:
                    break
                else:
                    print(cls.indent(message + "\n"))

            password_req = UserAuthenticationClient.get_password_policy()
            password_policy = cls.password_req_to_policy(password_req)
            password_req_prompt = "\n".join(
                [
                    "",
                    "Password requirements (minimum):",
                    "\n".join([f". {req}" for req in password_req]),
                    "",
                    "Please enter your password: ",
                ]
            )
            while True:
                password = getpass.getpass(cls.indent(password_req_prompt))
                password_req_prompt = "Please enter your password: "
                if len(password_policy.test(password)) != 0:
                    print(cls.indent("Password requirements not satisfied.\n"))
                    continue

                password_confirm = getpass.getpass(
                    cls.indent("Please confirm your password: ")
                )
                if password == password_confirm:
                    break
                else:
                    print(
                        cls.indent(
                            "Entered password and confirmation password do not match, please try again.\n"
                        )
                    )
            agreed_personally_identifiable_information = (
                cls.prompt_personally_identifiable_information()
            )
            if not agreed_personally_identifiable_information:
                raise RuntimeError("You must agree to not upload personal data.")

            additional_info = cls.prompt_add_user_information()
            additional_info["agreed_terms_and_cond"] = agreed_terms_and_cond
            additional_info["agreed_personally_identifiable_information"] = (
                agreed_personally_identifiable_information
            )
            (
                is_created,
                message,
                access_token,
            ) = UserAuthenticationClient.set_token_by_registration(
                email, password, password_confirm, validation_link, additional_info
            )
            if not is_created:
                raise RuntimeError("User registration failed: " + str(message) + "\n")

            print(
                cls.indent(
                    "Account created successfully! To start using TabPFN please enter the verification code we sent you by mail."
                )
                + "\n"
            )
            # verify token from email
            cls._verify_user_email(access_token=access_token)

        # Login
        elif choice == "2":
            # login to account
            while True:
                email = input(cls.indent("Please enter your email: "))
                password = getpass.getpass(cls.indent("Please enter your password: "))

                (
                    access_token,
                    message,
                    status_code,
                ) = UserAuthenticationClient.set_token_by_login(email, password)
                if status_code == 200 and access_token is not None:
                    break
                print(cls.indent("Login failed: " + str(message)) + "\n")
                if status_code == 403:
                    # 403 implies that the email is not verified
                    cls._verify_user_email(access_token=access_token)
                    UserAuthenticationClient.set_token_by_login(email, password)
                    break

                prompt = "\n".join(
                    [
                        "Please choose one of the following options:",
                        "(1) Retry login",
                        "(2) Reset your password",
                        "",
                        "Please enter your choice: ",
                    ]
                )
                choice = cls._choice_with_retries(prompt, ["1", "2"])

                if choice == "1":
                    continue
                elif choice == "2":
                    sent = False
                    print(
                        cls.indent(
                            "We will send you an email with a link "
                            "that allows you to reset your password. \n"
                        )
                    )
                    while True:
                        (
                            sent,
                            message,
                        ) = UserAuthenticationClient.send_reset_password_email(email)
                        print("\n" + cls.indent(message))
                        if sent:
                            break
                        email = input(cls.indent("Please enter your email address: "))
                    print(
                        cls.indent(
                            "Once you have reset your password, you will be able to login here: "
                        )
                    )

            print(cls.indent("Login successful!") + "\n")

    @classmethod
    def prompt_terms_and_cond(cls) -> bool:
        t_and_c = "\n".join(
            [
                "\nPlease refer to our terms and conditions at: https://www.priorlabs.ai/terms "
                "By using TabPFN, you agree to the following terms and conditions:",
                "Do you agree to the above terms and conditions? (y/n): ",
            ]
        )
        choice = cls._choice_with_retries(t_and_c, ["y", "n"])
        return choice == "y"

    @classmethod
    def prompt_personally_identifiable_information(cls) -> bool:
        pii = "\n".join(
            [
                "Do you agree to not upload personal data? (y/n): ",
            ]
        )
        choice = cls._choice_with_retries(pii, ["y", "n"])
        return choice == "y"

    @classmethod
    def prompt_add_user_information(cls) -> dict:
        print(cls.indent("\nPlease provide your name:"))

        # Required fields
        while True:
            first_name = input(cls.indent("First Name: ")).strip()
            if not first_name:
                print(
                    cls.indent("First name is required. Please enter your first name.")
                )
                continue
            break

        while True:
            last_name = input(cls.indent("Last Name: ")).strip()
            if not last_name:
                print(cls.indent("Last name is required. Please enter your last name."))
                continue
            break

        print(
            cls.indent(
                "\nTo help us tailor our support and services to your needs, we have a few optional questions. "
                "Feel free to skip any question by leaving it blank."
            )
            + "\n"
        )

        company = input(cls.indent("Where do you work? "))
        role = input(cls.indent("What is your role? "))
        use_case = input(cls.indent("What do you want to use TabPFN for? "))

        choice_contact = cls._choice_with_retries(
            "Can we reach out to you via email to support you? (y/n):", ["y", "n"]
        )
        contact_via_email = True if choice_contact == "y" else False

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
        prompt = "\n".join(
            [
                "Welcome Back! Found existing access token, reusing it for authentication."
            ]
        )

        print(cls.indent(prompt))

    @classmethod
    def reverify_email(cls, access_token):
        prompt = "\n".join(
            [
                "Please check your inbox for the verification email.",
                "Note: The email might be in your spam folder or could have expired.",
            ]
        )
        print(cls.indent(prompt))
        retry_verification = "\n".join(
            [
                "Do you want to resend email verification link? (y/n): ",
            ]
        )
        choice = cls._choice_with_retries(retry_verification, ["y", "n"])
        if choice == "y":
            # get user email from UserAuthenticationClient and resend verification email
            sent, message = UserAuthenticationClient.send_verification_email(
                access_token
            )
            if not sent:
                print(cls.indent("Failed to send verification email: " + message))
            else:
                print(
                    cls.indent(
                        "A verification email has been sent, provided the details are correct!"
                    )
                    + "\n"
                )
        # verify token from email
        cls._verify_user_email(access_token=access_token)
        UserAuthenticationClient.set_token(access_token)
        return

    @classmethod
    def prompt_retrieved_greeting_messages(cls, greeting_messages: list[str]):
        for message in greeting_messages:
            print(cls.indent(message))

    @classmethod
    def prompt_confirm_password_for_user_account_deletion(cls) -> str:
        print(cls.indent("You are about to delete your account."))
        confirm_pass = getpass.getpass(
            cls.indent("Please confirm by entering your password: ")
        )

        return confirm_pass

    @classmethod
    def prompt_account_deleted(cls):
        print(cls.indent("Your account has been deleted."))

    @classmethod
    def _choice_with_retries(cls, prompt: str, choices: list) -> str:
        """
        Prompt text and give user infinitely many attempts to select one of the possible choices. If valid choice
        is selected, return choice in lowercase.
        """
        assert all(c.lower() == c for c in choices), "Choices need to be lower case."
        choice = input(cls.indent(prompt))

        # retry until valid choice is made
        while True:
            if choice.lower() not in choices:
                choices_str = (
                    ", ".join(f"'{item}'" for item in choices[:-1])
                    + f" or '{choices[-1]}'"
                )
                choice = input(
                    cls.indent(f"Invalid choice, please enter {choices_str}: ")
                )
            else:
                break

        return choice.lower()

    @classmethod
    def _verify_user_email(cls, access_token: str):
        verified = False
        while not verified:
            token = input(
                cls.indent(
                    "Please enter the correct verification code sent to your email: "
                )
            )
            verified, message = UserAuthenticationClient.verify_email(
                token, access_token
            )
            if not verified:
                print("\n" + cls.indent(str(message) + "Please try again!") + "\n")
            else:
                print(cls.indent("Email verified successfully!") + "\n")
                break
        return
