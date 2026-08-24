#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""interactive_login() is opt-in and never runs on the default auth path."""

import io
import os
import shutil
import threading
import unittest
from unittest.mock import patch

import httpx

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
        with patch.object(
            interactive_auth, "_stdin_is_interactive", return_value=False
        ):
            with self.assertRaises(InteractiveLoginError) as cm:
                interactive_login()

        message = str(cm.exception)
        self.assertIn("interactive terminal", message)
        self.assertIn(URL_PRIOR_LABS_API_KEYS, message)

    @with_mock_server()
    def test_cached_token_short_circuits_the_prompt(self, mock_server):
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        UserAuthenticationClient.persist_token("cached_token")

        with patch.object(interactive_auth, "_prompt_menu") as mock_menu:
            token = interactive_login()

        self.assertEqual(token, "cached_token")
        mock_menu.assert_not_called()

    @with_mock_server()
    def test_already_logged_in_stays_off_stdout_when_not_a_tty(self, mock_server):
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        UserAuthenticationClient.persist_token("cached_token")

        buffer = io.StringIO()  # not a tty
        with patch("sys.stdout", buffer):
            interactive_login()

        self.assertNotIn("Already logged in", buffer.getvalue())

    @with_mock_server()
    def test_cached_token_does_not_short_circuit_a_non_tty(self, mock_server):
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        UserAuthenticationClient.persist_token("cached_token")

        with patch.object(
            interactive_auth, "_stdin_is_interactive", return_value=False
        ):
            self.assertEqual(interactive_login(), "cached_token")

    @with_mock_server()
    def test_force_relogin_ignores_the_cached_token(self, mock_server):
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        UserAuthenticationClient.persist_token("cached_token")

        with patch.object(interactive_auth, "_stdin_is_interactive", return_value=True):
            with patch.object(interactive_auth, "_prompt_menu", return_value="login"):
                with patch.object(interactive_auth, "_has_display", return_value=False):
                    with patch.object(
                        interactive_auth,
                        "_paste_only_login",
                        return_value="fresh_token",
                    ):
                        token = interactive_login(force_relogin=True)

        self.assertEqual(token, "fresh_token")

    @with_mock_server()
    def test_rejected_cached_token_falls_through_to_the_prompt(self, mock_server):
        mock_server.router.get(mock_server.endpoints.protected_root.path).mock(
            side_effect=[httpx.Response(401), httpx.Response(200)]
        )
        UserAuthenticationClient.persist_token("stale_token")

        with patch.object(interactive_auth, "_stdin_is_interactive", return_value=True):
            with patch.object(interactive_auth, "_prompt_menu", return_value="login"):
                with patch.object(interactive_auth, "_has_display", return_value=False):
                    with patch.object(
                        interactive_auth,
                        "_paste_only_login",
                        return_value="fresh_token",
                    ):
                        token = interactive_login()

        self.assertEqual(token, "fresh_token")

    @with_mock_server()
    def test_pasted_token_is_verified_and_installed(self, mock_server):
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)

        with patch.object(interactive_auth, "_stdin_is_interactive", return_value=True):
            with patch.object(interactive_auth, "_prompt_menu", return_value="login"):
                with patch.object(interactive_auth, "_has_display", return_value=False):
                    with patch.object(
                        interactive_auth,
                        "_paste_only_login",
                        return_value="pasted_token",
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
            with patch.object(interactive_auth, "_prompt_menu", return_value="login"):
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
            with patch.object(interactive_auth, "_prompt_menu", return_value="login"):
                with patch.object(interactive_auth, "_has_display", return_value=False):
                    with patch.object(
                        interactive_auth, "_paste_only_login", return_value=None
                    ):
                        with self.assertRaises(InteractiveLoginError) as cm:
                            interactive_login()

        self.assertIn("not completed", str(cm.exception))

    def test_open_browser_false_skips_the_browser(self):
        with patch.object(interactive_auth, "_stdin_is_interactive", return_value=True):
            with patch.object(interactive_auth, "_prompt_menu", return_value="login"):
                with patch.object(interactive_auth, "_has_display", return_value=True):
                    with patch.object(
                        interactive_auth, "_browser_login"
                    ) as mock_browser:
                        with patch.object(
                            interactive_auth, "_paste_only_login", return_value=None
                        ) as mock_paste:
                            with self.assertRaises(InteractiveLoginError):
                                interactive_login(open_browser=False)

        mock_browser.assert_not_called()
        mock_paste.assert_called_once()

    @with_mock_server()
    def test_menu_routes_to_signup(self, mock_server):
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)

        with patch.object(interactive_auth, "_stdin_is_interactive", return_value=True):
            with patch.object(interactive_auth, "_prompt_menu", return_value="signup"):
                with patch(
                    "tabpfn_client.prompt_agent.PromptAgent.prompt_signup",
                    return_value="signup_token",
                ) as mock_signup:
                    with patch.object(
                        interactive_auth, "_browser_login"
                    ) as mock_browser:
                        token = interactive_login()

        mock_signup.assert_called_once()
        # Signup never needs a browser -- that is the point of it.
        mock_browser.assert_not_called()
        self.assertEqual("signup_token", token)
        self.assertEqual(
            "signup_token", UserAuthenticationClient.CACHED_TOKEN_FILE.read_text()
        )

    def test_menu_quit_raises(self):
        with patch.object(interactive_auth, "_stdin_is_interactive", return_value=True):
            with patch.object(interactive_auth, "_prompt_menu", return_value="q"):
                with self.assertRaises(InteractiveLoginError) as cm:
                    interactive_login()

        self.assertIn("cancelled", str(cm.exception))

    def test_abandoned_signup_raises(self):
        """Quitting at email verification leaves no usable token."""
        with patch.object(interactive_auth, "_stdin_is_interactive", return_value=True):
            with patch.object(interactive_auth, "_prompt_menu", return_value="signup"):
                with patch(
                    "tabpfn_client.prompt_agent.PromptAgent.prompt_signup",
                    return_value=None,
                ):
                    with self.assertRaises(InteractiveLoginError) as cm:
                        interactive_login()

        self.assertIn("not completed", str(cm.exception))

    @with_mock_server()
    def test_init_never_triggers_interactive_login(self, mock_server):
        """The default path stays token-only."""
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)

        with patch.object(interactive_auth, "_browser_login") as mock_browser:
            with patch.object(interactive_auth, "_paste_only_login") as mock_paste:
                with patch(
                    "tabpfn_client.prompt_agent.PromptAgent.prompt_signup"
                ) as mock_signup:
                    with self.assertRaises(RuntimeError):
                        config.init(use_server=True)

        mock_browser.assert_not_called()
        mock_paste.assert_not_called()
        mock_signup.assert_not_called()


class TestBrowserCallbackRace(unittest.TestCase):
    """The paste prompt and the callback server race; either can win."""

    def test_stdin_paste_wins_over_a_silent_callback(self):
        import threading

        auth_event = threading.Event()
        received: list[str | None] = [None]

        with patch.object(
            interactive_auth.select, "select", return_value=([1], [], [])
        ):
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


class TestReviewFindings(unittest.TestCase):
    """Regression guards for the issues raised in review of #352."""

    def test_callback_server_binds_loopback_only(self):
        """An all-interfaces bind lets anyone on the network inject a token."""
        auth_event = threading.Event()
        received: list = [None]
        httpd, _ = interactive_auth._create_callback_server(
            "https://ux.priorlabs.ai", auth_event, received
        )
        try:
            self.assertEqual("127.0.0.1", httpd.socket.getsockname()[0])
        finally:
            httpd.server_close()

    def test_windows_does_not_select_on_stdin(self):
        """select() accepts only sockets on Windows; stdin needs its own thread."""
        with patch.object(interactive_auth.sys, "platform", "win32"):
            with patch.object(
                interactive_auth.select, "select", side_effect=OSError(10038, "nope")
            ):
                with patch.object(
                    interactive_auth, "_read_line", return_value="typed_token"
                ):
                    token = interactive_auth._poll_for_token(
                        threading.Event(), [None], 5.0
                    )
        self.assertEqual("typed_token", token)

    def test_notebook_counts_as_interactive(self):
        """A kernel has no TTY but can still ask the user a question."""
        with patch.object(interactive_auth, "_in_notebook", return_value=True):
            with patch.object(interactive_auth.sys.stdin, "isatty", return_value=False):
                self.assertTrue(interactive_auth._stdin_is_interactive())

    def test_notebook_skips_the_localhost_callback(self):
        """A kernel's localhost is not necessarily the reader's browser."""
        with patch.object(interactive_auth, "_stdin_is_interactive", return_value=True):
            with patch.object(interactive_auth, "_in_notebook", return_value=True):
                with patch.object(interactive_auth, "_has_display", return_value=True):
                    with patch.object(
                        interactive_auth, "_prompt_menu", return_value="login"
                    ):
                        with patch.object(
                            interactive_auth, "_browser_login"
                        ) as mock_browser:
                            with patch.object(
                                interactive_auth,
                                "_paste_only_login",
                                return_value=None,
                            ) as mock_paste:
                                with self.assertRaises(InteractiveLoginError):
                                    interactive_login(force_relogin=True)

        mock_browser.assert_not_called()
        mock_paste.assert_called_once()

    def test_unreachable_server_falls_through_to_login(self):
        """The already-logged-in shortcut must not surface a transport error."""
        with patch.object(
            UserAuthenticationClient, "resolve_token", return_value="some_token"
        ):
            with patch.object(
                interactive_auth.ServiceClient,
                "is_auth_token_outdated",
                side_effect=httpx.ConnectError("unreachable"),
            ):
                with patch.object(
                    interactive_auth, "_stdin_is_interactive", return_value=False
                ):
                    # Falls through to the TTY guard rather than raising ConnectError.
                    with self.assertRaises(InteractiveLoginError) as cm:
                        interactive_login()

        self.assertIn("interactive terminal", str(cm.exception))

    def test_failed_browser_open_is_reported(self):
        """Say so when no browser opened, instead of silently waiting."""
        auth_event = threading.Event()
        auth_event.set()  # resolve immediately; we only care about the message

        with patch.object(interactive_auth.webbrowser, "open", return_value=False):
            with patch.object(interactive_auth, "_poll_for_token", return_value="tok"):
                with patch("sys.stdout", new=io.StringIO()) as fake_out:
                    interactive_auth._browser_login("https://ux.priorlabs.ai", 1.0)

        self.assertIn("Could not open a browser", fake_out.getvalue())
