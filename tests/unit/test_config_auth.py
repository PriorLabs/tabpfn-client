#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""Authentication is token-only: TABPFN_TOKEN, set_access_token, or a paste prompt."""

import os
import shutil
import unittest
from unittest.mock import patch

from tabpfn_client import config
from tabpfn_client.client import ServiceClient
from tabpfn_client.config import init, set_access_token
from tabpfn_client.constants import CACHE_DIR, URL_PRIOR_LABS_API_KEYS
from tabpfn_client.service_wrapper import UserAuthenticationClient
from tests.mock_tabpfn_server import with_mock_server


class TestTokenOnlyAuth(unittest.TestCase):
    def setUp(self):
        config.reset()
        ServiceClient.reset_authorization()
        os.environ.pop("TABPFN_TOKEN", None)

    def tearDown(self):
        config.reset()
        ServiceClient.reset_authorization()
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    @with_mock_server()
    def test_env_token_is_used_without_prompting(self, mock_server):
        # A valid token short-circuits the connection check, so `/` is never hit.
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        mock_server.router.get(
            mock_server.endpoints.retrieve_greeting_messages.path
        ).respond(200, json={"messages": []})

        with patch.dict(os.environ, {"TABPFN_TOKEN": "env_token"}):
            with patch.object(config, "_prompt_and_set_token") as mock_prompt:
                init(use_server=True)

        mock_prompt.assert_not_called()
        self.assertEqual("env_token", ServiceClient.get_access_token())

    @with_mock_server()
    def test_non_interactive_without_token_explains_how_to_get_one(self, mock_server):
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)

        with patch.object(config, "_stdin_is_interactive", return_value=False):
            with self.assertRaises(RuntimeError) as cm:
                init(use_server=True)

        message = str(cm.exception)
        self.assertIn(URL_PRIOR_LABS_API_KEYS, message)
        self.assertIn("TABPFN_TOKEN", message)
        self.assertIn("set_access_token", message)

    @with_mock_server()
    def test_non_interactive_rejected_token_says_so(self, mock_server):
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(401)

        with patch.dict(os.environ, {"TABPFN_TOKEN": "stale_token"}):
            with patch.object(config, "_stdin_is_interactive", return_value=False):
                with self.assertRaises(RuntimeError) as cm:
                    init(use_server=True)

        message = str(cm.exception)
        self.assertIn("rejected", message)
        self.assertIn(URL_PRIOR_LABS_API_KEYS, message)

    @with_mock_server()
    def test_interactive_prompt_accepts_pasted_token(self, mock_server):
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        # No token resolves, so the only protected-root call is the validation
        # of the pasted token.
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        mock_server.router.get(
            mock_server.endpoints.retrieve_greeting_messages.path
        ).respond(200, json={"messages": []})

        with patch.object(config, "_stdin_is_interactive", return_value=True):
            with patch(
                "tabpfn_client.prompt_agent.PromptAgent.prompt_for_token",
                return_value="pasted_token",
            ):
                init(use_server=True)

        self.assertEqual("pasted_token", ServiceClient.get_access_token())
        # The pasted token is cached so the next run does not re-prompt.
        self.assertEqual(
            "pasted_token", UserAuthenticationClient.CACHED_TOKEN_FILE.read_text()
        )

    @with_mock_server()
    def test_set_access_token_skips_init_entirely(self, mock_server):
        set_access_token("explicit_token")
        self.assertEqual("explicit_token", ServiceClient.get_access_token())

        with patch.object(config, "_prompt_and_set_token") as mock_prompt:
            init(use_server=True)
        mock_prompt.assert_not_called()
