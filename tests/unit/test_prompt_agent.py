import unittest
from unittest.mock import patch

from tabpfn_client.constants import URL_PRIOR_LABS_API_KEYS
from tabpfn_client.prompt_agent import PromptAgent


class TestPromptAgent(unittest.TestCase):
    @patch("rich.console.Console.input", return_value="  dummy_token  ")
    def test_prompt_for_token_strips_whitespace(self, mock_input):
        self.assertEqual(PromptAgent.prompt_for_token(), "dummy_token")

    @patch("rich.console.Console.input", side_effect=["", "   ", "dummy_token"])
    def test_prompt_for_token_retries_on_empty_input(self, mock_input):
        self.assertEqual(PromptAgent.prompt_for_token(), "dummy_token")
        self.assertEqual(mock_input.call_count, 3)

    def test_token_instructions_point_at_the_api_keys_page(self):
        instructions = PromptAgent.token_instructions()
        self.assertIn(URL_PRIOR_LABS_API_KEYS, instructions)
        self.assertIn("TABPFN_TOKEN", instructions)
        self.assertIn("set_access_token", instructions)
