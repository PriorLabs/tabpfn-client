import unittest
from unittest.mock import patch
from tabpfn_client.prompt_agent import PromptAgent


class TestPromptAgent(unittest.TestCase):
    def test_password_req_to_policy(self):
        password_req = ["Length(8)", "Uppercase(1)", "Numbers(1)", "Special(1)"]
        password_policy = PromptAgent.password_req_to_policy(password_req)
        requirements = [repr(req) for req in password_policy.test("")]
        self.assertEqual(password_req, requirements)

    @patch("getpass.getpass", side_effect=["Password123!", "Password123!"])
    @patch(
        "rich.console.Console.input",
        side_effect=[
            "1",  # menu choice: registration
            "y",  # agree to terms (Step 1)
            "user@example.com",  # email (Step 2)
            "y",  # agree to data privacy (Step 4)
            "first",  # first name (Step 5)
            "last",  # last name (Step 5)
            "cool company",  # company (Step 6)
            "a",  # role choice (Step 6)
            "my awesome usecase! bla",  # use case (Step 6)
            "y",  # contact via email (Step 6)
            "test",  # verification code
        ],
    )
    def test_prompt_and_set_token_registration(self, mock_console_input, mock_getpass):
        # for some reason, it needs to be patched with a with-statement instead of a decorator
        with patch(
            "tabpfn_client.prompt_agent.UserAuthenticationClient"
        ) as mock_auth_client:
            mock_auth_client.try_browser_login.return_value = (False, None)
            mock_auth_client.get_password_policy.return_value = [
                "Length(8)",
                "Uppercase(1)",
                "Numbers(1)",
                "Special(1)",
            ]
            mock_auth_client.set_token_by_registration.return_value = (
                True,
                "Registration successful",
                "dummy_token",
            )
            mock_auth_client.validate_email.return_value = (True, "")
            mock_auth_client.verify_email.return_value = (
                True,
                "Verification successful",
            )

            PromptAgent.prompt_and_set_token()

            mock_auth_client.validate_email.assert_called_once_with("user@example.com")
            mock_auth_client.set_token_by_registration.assert_called_once_with(
                "user@example.com",
                "Password123!",
                "Password123!",
                "tabpfn-2023",
                {
                    "first_name": "first",
                    "last_name": "last",
                    "company": "cool company",
                    "role": "Field practitioner",
                    "use_case": "my awesome usecase! bla",
                    "contact_via_email": True,
                    "agreed_terms_and_cond": True,
                    "agreed_personally_identifiable_information": True,
                },
            )

    @patch("getpass.getpass", side_effect=["password123"])
    @patch(
        "rich.console.Console.input",
        side_effect=[
            "2",  # menu choice: login
            "user@gmail.com",  # email
        ],
    )
    def test_prompt_and_set_token_login(self, mock_console_input, mock_getpass):
        with patch(
            "tabpfn_client.prompt_agent.UserAuthenticationClient"
        ) as mock_auth_client:
            mock_auth_client.try_browser_login.return_value = (False, None)
            mock_auth_client.set_token_by_login.return_value = (
                True,
                "Login successful",
                200,
            )
            PromptAgent.prompt_and_set_token()
            mock_auth_client.set_token_by_login.assert_called_once()

    @patch("rich.console.Console.input", return_value="y")
    def test_prompt_terms_and_cond_returns_true(self, mock_input):
        result = PromptAgent.prompt_terms_and_cond()
        self.assertTrue(result)

    @patch("rich.console.Console.input", return_value="n")
    def test_prompt_terms_and_cond_returns_false(self, mock_input):
        result = PromptAgent.prompt_terms_and_cond()
        self.assertFalse(result)

    @patch("rich.console.Console.input", return_value="1")
    def test_choice_with_retries_valid_first_try(self, mock_console_input):
        result = PromptAgent._choice_with_retries(
            "Please enter your choice: ", ["1", "2"]
        )
        self.assertEqual(result, "1")

    @patch("rich.console.Console.input", side_effect=["3", "3", "1"])
    def test_choice_with_retries_valid_third_try(self, mock_console_input):
        result = PromptAgent._choice_with_retries(
            "Please enter your choice: ", ["1", "2"]
        )
        self.assertEqual(result, "1")
