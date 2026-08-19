import unittest
from tabpfn_client.constants import URL_PRIOR_LABS_API_KEYS
from tabpfn_client.prompt_agent import PromptAgent


class TestPromptAgent(unittest.TestCase):
    def test_token_instructions_point_at_the_api_keys_page(self):
        instructions = PromptAgent.token_instructions()
        self.assertIn(URL_PRIOR_LABS_API_KEYS, instructions)
        self.assertIn("TABPFN_TOKEN", instructions)
        self.assertIn("set_access_token", instructions)
