import unittest
from unittest.mock import patch

from tabpfn_client.constants import URL_PRIOR_LABS_API_KEYS
from tabpfn_client.prompt_agent import ROLES, PromptAgent


class TestPromptAgent(unittest.TestCase):
    def test_token_instructions_point_at_the_api_keys_page(self):
        instructions = PromptAgent.token_instructions()
        self.assertIn(URL_PRIOR_LABS_API_KEYS, instructions)
        self.assertIn("TABPFN_TOKEN", instructions)
        self.assertIn("set_access_token", instructions)

    def test_password_req_to_policy(self):
        password_req = ["Length(8)", "Uppercase(1)", "Numbers(1)", "Special(1)"]
        password_policy = PromptAgent.password_req_to_policy(password_req)
        self.assertEqual(password_req, [repr(r) for r in password_policy.test("")])


# Menu letters are positional, so pin the ones the tests type.
ROLE_LETTER_ML_ENGINEER = "b"
ROLE_LETTER_OTHER = chr(ord("a") + len(ROLES) - 1)


class TestSignupFlow(unittest.TestCase):
    """Signup runs in three steps: credentials, profile, verification."""

    PASSWORD_REQ = ["Length(8)", "Uppercase(1)", "Numbers(1)", "Special(1)"]

    def _run_signup(self, inputs, passwords=("Password123!", "Password123!")):
        with patch("tabpfn_client.prompt_agent.UserAuthenticationClient") as mock_auth:
            mock_auth.validate_email.return_value = (True, "")
            mock_auth.get_password_policy.return_value = self.PASSWORD_REQ
            mock_auth.set_token_by_registration.return_value = (
                True,
                "Registration successful",
                "new_token",
            )
            mock_auth.verify_email.return_value = (True, "Verified")

            with patch("getpass.getpass", side_effect=list(passwords)):
                with patch("rich.console.Console.input", side_effect=inputs):
                    token = PromptAgent.prompt_signup()
        return token, mock_auth

    def _inputs(self, subscribe):
        return [
            "user@example.com",  # step 1: email
            "First",  # step 2: first name
            "Last",  # step 2: last name
            "Prior Labs",  # step 2: company
            ROLE_LETTER_ML_ENGINEER,  # step 2: role
            subscribe,  # step 2: marketing opt-in
            "123456",  # step 3: verification code
        ]

    def test_signup_sends_the_expected_payload(self):
        token, mock_auth = self._run_signup(self._inputs(subscribe=""))

        self.assertEqual("new_token", token)
        mock_auth.validate_email.assert_called_once_with("user@example.com")

        args = mock_auth.set_token_by_registration.call_args.args
        self.assertEqual("user@example.com", args[0])
        self.assertEqual(
            {
                "first_name": "First",
                "last_name": "Last",
                "company": "Prior Labs",
                "role": "ML Engineer",
                "contact_via_email": False,
                "agreed_terms_and_cond": True,
                "agreed_personally_identifiable_information": True,
            },
            args[4],
        )
        # The "What do you want to use TabPFN for?" input is gone.
        self.assertNotIn("use_case", args[4])

    def test_marketing_opt_in_defaults_to_no(self):
        """A bare Enter must not subscribe: consent needs an affirmative act."""
        _, mock_auth = self._run_signup(self._inputs(subscribe=""))
        self.assertFalse(
            mock_auth.set_token_by_registration.call_args.args[4]["contact_via_email"]
        )

    def test_marketing_opt_in_honours_an_explicit_yes(self):
        _, mock_auth = self._run_signup(self._inputs(subscribe="y"))
        self.assertTrue(
            mock_auth.set_token_by_registration.call_args.args[4]["contact_via_email"]
        )

    def test_other_role_falls_through_to_free_text(self):
        inputs = [
            "user@example.com",
            "First",
            "Last",
            "Prior Labs",
            ROLE_LETTER_OTHER,
            "Chief Vibes Officer",  # free-text role
            "",
            "123456",
        ]
        _, mock_auth = self._run_signup(inputs)
        self.assertEqual(
            "Chief Vibes Officer",
            mock_auth.set_token_by_registration.call_args.args[4]["role"],
        )

    def test_role_list_matches_the_agreed_set(self):
        self.assertEqual(
            [
                "Data Scientist",
                "ML Engineer",
                "AI Engineer",
                "Software Engineer",
                "Product Manager",
                "Researcher",
                "Student",
                "Executive",
                "Other",
            ],
            ROLES,
        )

    def test_quitting_verification_returns_no_token(self):
        inputs = self._inputs(subscribe="")
        inputs[-1] = "quit"
        token, _ = self._run_signup(inputs)
        self.assertIsNone(token)
