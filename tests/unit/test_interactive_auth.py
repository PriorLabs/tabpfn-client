#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""interactive_login() is opt-in and never runs on the default auth path."""

import os
import shutil
import unittest
from unittest.mock import patch

from tabpfn_client import config, interactive_auth
from tabpfn_client.client import ServiceClient
from tabpfn_client.constants import CACHE_DIR, URL_PRIOR_LABS_API_KEYS
from tabpfn_client.interactive_auth import InteractiveLoginError, interactive_login
from tabpfn_client.service_wrapper import UserAuthenticationClient
from tests.mock_tabpfn_server import with_mock_server


class TestInteractiveLogin(unittest.TestCase):
    def setUp(self):
        config.reset()
        ServiceClient.reset_authorization()
        os.environ.pop("TABPFN_TOKEN", None)

    def tearDown(self):
        config.reset()
        ServiceClient.reset_authorization()
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    def test_requires_a_terminal(self):
        with patch.object(interactive_auth, "_stdin_is_interactive", return_value=False):
            with self.assertRaises(InteractiveLoginError) as cm:
                interactive_login()

        message = str(cm.exception)
        self.assertIn("interactive terminal", message)
        self.assertIn(URL_PRIOR_LABS_API_KEYS, message)

    @with_mock_server()
    def test_pasted_token_is_verified_and_installed(self, mock_server):
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)

        with patch.object(interactive_auth, "_stdin_is_interactive", return_value=True):
            with patch.object(interactive_auth, "_has_display", return_value=False):
                with patch.object(
                    interactive_auth, "_paste_only_login", return_value="pasted_token"
                ):
                    token = interactive_login()

        self.assertEqual("pasted_token", token)
        self.assertEqual("pasted_token", ServiceClient.get_access_token())
        # A later init() finds the token without prompting.
        self.assertTrue(config.Config.is_initialized)
        # interactive_login is the sole writer of the token cache, so a later
        # process picks the token up with no prompt.
        self.assertEqual(
            "pasted_token", UserAuthenticationClient.CACHED_TOKEN_FILE.read_text()
        )

    @with_mock_server()
    def test_rejected_token_raises(self, mock_server):
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(401)

        with patch.object(interactive_auth, "_stdin_is_interactive", return_value=True):
            with patch.object(interactive_auth, "_has_display", return_value=False):
                with patch.object(
                    interactive_auth, "_paste_only_login", return_value="bad_token"
                ):
                    with self.assertRaises(InteractiveLoginError) as cm:
                        interactive_login()

        self.assertIn("rejected", str(cm.exception))
        self.assertIsNone(ServiceClient.get_access_token())

    def test_aborted_login_raises(self):
        with patch.object(interactive_auth, "_stdin_is_interactive", return_value=True):
            with patch.object(interactive_auth, "_has_display", return_value=False):
                with patch.object(
                    interactive_auth, "_paste_only_login", return_value=None
                ):
                    with self.assertRaises(InteractiveLoginError) as cm:
                        interactive_login()

        self.assertIn("not completed", str(cm.exception))

    def test_open_browser_false_skips_the_browser(self):
        with patch.object(interactive_auth, "_stdin_is_interactive", return_value=True):
            with patch.object(interactive_auth, "_has_display", return_value=True):
                with patch.object(interactive_auth, "_browser_login") as mock_browser:
                    with patch.object(
                        interactive_auth, "_paste_only_login", return_value=None
                    ) as mock_paste:
                        with self.assertRaises(InteractiveLoginError):
                            interactive_login(open_browser=False)

        mock_browser.assert_not_called()
        mock_paste.assert_called_once()

    @with_mock_server()
    def test_init_never_triggers_interactive_login(self, mock_server):
        """The default path stays token-only."""
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)

        with patch.object(interactive_auth, "_browser_login") as mock_browser:
            with patch.object(interactive_auth, "_paste_only_login") as mock_paste:
                with self.assertRaises(RuntimeError):
                    config.init(use_server=True)

        mock_browser.assert_not_called()
        mock_paste.assert_not_called()


class TestBrowserCallbackRace(unittest.TestCase):
    """The paste prompt and the callback server race; either can win."""

    def test_stdin_paste_wins_over_a_silent_callback(self):
        import threading

        auth_event = threading.Event()
        received: list[str | None] = [None]

        with patch.object(interactive_auth.select, "select", return_value=([1], [], [])):
            with patch.object(
                interactive_auth.sys.stdin, "readline", return_value="typed_token\n"
            ):
                token = interactive_auth._poll_for_token(auth_event, received, 5.0)

        self.assertEqual("typed_token", token)

    def test_callback_wins_when_stdin_is_silent(self):
        import threading

        auth_event = threading.Event()
        auth_event.set()  # callback already delivered
        received: list[str | None] = ["callback_token"]

        token = interactive_auth._poll_for_token(auth_event, received, 5.0)
        self.assertEqual("callback_token", token)

    def test_poll_times_out_rather_than_hanging(self):
        import threading

        auth_event = threading.Event()
        received: list[str | None] = [None]

        # select never reports stdin ready and the callback never fires.
        with patch.object(interactive_auth.select, "select", return_value=([], [], [])):
            token = interactive_auth._poll_for_token(auth_event, received, 1.0)

        self.assertIsNone(token)
