#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Opt-in interactive login and registration.

**This is not part of the default authentication path.** `init()` and the
estimators authenticate with a token only: TABPFN_TOKEN, `set_access_token()`,
or a token cached by an earlier run. Nothing in this module runs unless the
caller explicitly asks for it:

    from tabpfn_client import interactive_login
    interactive_login()

The flow opens the Prior Labs login page in a browser -- where the user can log
in *or* register, including via SSO -- and waits for the resulting API key.
Because an identity provider can drop the localhost callback on the way back,
the callback server and a manual paste prompt run at the same time: whichever
produces a key first wins. The key is verified against the server and cached, so
later runs authenticate without any prompt.

Adapted from the browser-auth flow in the TabPFN package.
"""

from __future__ import annotations

import base64
import http.server
import logging
import os
import select
import socketserver
import sys
import threading
import urllib.parse
import webbrowser

from tabpfn_client.client import ServiceClient
from tabpfn_client.constants import URL_PRIOR_LABS_API_KEYS

logger = logging.getLogger(__name__)

# How long to wait for the browser callback before giving up on it. The paste
# prompt stays available for the whole time, so this only bounds the case where
# the user closes the browser and walks away.
_CALLBACK_TIMEOUT_SECS = 600.0


class InteractiveLoginError(RuntimeError):
    """Raised when an interactive login cannot be completed."""


def _has_display() -> bool:
    """Heuristic: is a graphical display likely available for opening a browser?"""
    if sys.platform == "win32":
        return True
    if sys.platform == "darwin":
        # macOS has a display unless we are in a pure SSH session without X
        # forwarding.
        return not (os.environ.get("SSH_CONNECTION") and not os.environ.get("DISPLAY"))
    # Linux / other Unix: require X11 or Wayland.
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _stdin_is_interactive() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _copy_osc52(text: str) -> None:
    """Copy *text* to the system clipboard via the OSC 52 terminal escape.

    Works over SSH when the terminal emulator supports it (iTerm2, kitty,
    Windows Terminal, most modern terminals).
    """
    encoded = base64.b64encode(text.encode()).decode()
    sys.stdout.write(f"\033]52;c;{encoded}\a")
    sys.stdout.flush()


def _create_callback_server(
    gui_url: str,
    auth_event: threading.Event,
    received_token: list[str | None],
) -> tuple[socketserver.TCPServer, int]:
    """Serve the login callback on an ephemeral port. Returns ``(httpd, port)``."""

    class _CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            if "token" in query:
                received_token[0] = query["token"][0]

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()

            if received_token[0] is not None:
                body = (
                    "<h2>Login successful</h2>"
                    "<p>You can close this tab and return to your terminal.</p>"
                    f'<script>window.location.href = "{gui_url}/redirect-success";</script>'
                )
            else:
                body = (
                    "<h2>No API key received</h2>"
                    "<p>Please paste your API key in the terminal, or copy it from "
                    f'<a href="{URL_PRIOR_LABS_API_KEYS}">{URL_PRIOR_LABS_API_KEYS}</a>.</p>'
                )

            html = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<title>Prior Labs</title></head>"
                '<body style="font-family: -apple-system, Segoe UI, Roboto, sans-serif;'
                ' text-align: center; padding: 50px;">'
                f"{body}</body></html>"
            )
            self.wfile.write(html.encode())
            if received_token[0] is not None:
                auth_event.set()

        def log_message(self, format: str, *args: object) -> None:
            pass  # silence request logs

    httpd = socketserver.TCPServer(("", 0), _CallbackHandler)
    port = httpd.server_address[1]
    return httpd, port


def _serve_until_event(
    httpd: socketserver.TCPServer, auth_event: threading.Event
) -> None:
    """Handle requests until *auth_event* is set. Runs in a daemon thread."""
    httpd.timeout = 0.5
    while not auth_event.is_set():
        try:
            httpd.handle_request()
        except Exception:
            break


def _poll_for_token(
    auth_event: threading.Event,
    received_token: list[str | None],
    timeout: float,
) -> str | None:
    """Read a token from stdin or the browser callback, whichever arrives first."""
    prompt = "API key (or press Enter to keep waiting): "
    sys.stdout.write(prompt)
    sys.stdout.flush()

    waited = 0.0
    while not auth_event.is_set():
        if waited >= timeout:
            sys.stdout.write("\nTimed out waiting for login.\n")
            sys.stdout.flush()
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.5)
        if not ready:
            waited += 0.5
            continue
        line = sys.stdin.readline()
        if not line:  # EOF
            return None
        token = line.strip()
        if token:
            return token
        sys.stdout.write(prompt)
        sys.stdout.flush()
    return received_token[0]


def _paste_only_login(login_url: str, timeout: float) -> str | None:
    """Token acquisition without a browser, e.g. over SSH.

    Shows the login URL, offers clipboard copy via OSC 52, and waits for a
    pasted key.
    """
    print(
        "\nNo display detected. Open this URL in a browser on another device:\n"
        f"\n  {login_url}\n"
        f"\nAfter logging in, copy your API key from\n  {URL_PRIOR_LABS_API_KEYS}\n"
    )

    deadline_note = (
        "  Type [c] then Enter to copy the URL, or paste your API key:\n\n> "
    )
    try:
        while True:
            sys.stdout.write(deadline_note)
            sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:  # EOF
                return None
            text = line.strip()
            if text.lower() == "c":
                _copy_osc52(login_url)
                sys.stdout.write("✓ Copied to clipboard\n\n")
                sys.stdout.flush()
                continue
            if text:
                return text
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        return None


def _browser_login(gui_url: str, timeout: float) -> str | None:
    """Open the browser and race the localhost callback against a paste prompt."""
    auth_event = threading.Event()
    received_token: list[str | None] = [None]

    try:
        httpd, port = _create_callback_server(gui_url, auth_event, received_token)
    except Exception:
        logger.debug("Could not create callback server", exc_info=True)
        # Without a callback server the paste prompt is still perfectly usable.
        return _paste_only_login(f"{gui_url}/login", timeout)

    callback_url = f"http://localhost:{port}"
    login_url = f"{gui_url}/login?callback={callback_url}"

    server_thread = threading.Thread(
        target=_serve_until_event, args=(httpd, auth_event), daemon=True
    )
    server_thread.start()

    webbrowser.open(login_url)

    print(
        "\nOpening your browser to log in or register.\n"
        f"\n  {login_url}\n"
        "\nWaiting for login to complete...\n"
        "\nHaving trouble? You can also authenticate manually:\n"
        f"  1. Open {URL_PRIOR_LABS_API_KEYS} in a browser (log in or register)\n"
        "  2. Generate an API key and copy it\n"
        "  3. Paste it below\n"
    )

    try:
        token = _poll_for_token(auth_event, received_token, timeout)
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        token = None
    finally:
        auth_event.set()  # stop the server thread
        httpd.server_close()

    return token


def _prompt_menu() -> str:
    """Ask whether to log in or create an account. Returns 'login'/'signup'/'q'."""
    sys.stdout.write(
        "\n  [1] Log in to your TabPFN account"
        "\n  [2] Create a TabPFN account"
        "\n  [q] Quit\n"
    )
    sys.stdout.flush()
    while True:
        choice = input("\n> Choose (1/2/q): ").strip().lower()
        if choice in ("1", "login"):
            return "login"
        if choice in ("2", "signup"):
            return "signup"
        if choice in ("q", "quit"):
            return "q"
        sys.stdout.write("  Please enter 1, 2, or q.\n")
        sys.stdout.flush()


def interactive_login(
    *,
    force_relogin: bool = False,
    open_browser: bool = True,
    timeout: float = _CALLBACK_TIMEOUT_SECS,
) -> str:
    """Log in or create an account interactively, returning the access token.

    This is opt-in. The default authentication path never calls it: `init()`
    uses TABPFN_TOKEN, `set_access_token()`, or a cached token, and raises with
    instructions when none is available.

    Offers two routes. Logging in opens the Prior Labs login page in a browser
    (or prints the URL when there is no display) and waits for the resulting API
    key. Creating an account runs entirely in the terminal, which is what makes
    this usable from a hosted notebook where a browser tab is not an option.

    On success the token is verified, cached for future runs, and installed as
    the active token, so a subsequent `init()` needs no further input.

    :param force_relogin: Run the login flow even when a working token is
                          already available, for example to switch accounts.
                          By default an existing token is returned as-is.
    :param open_browser: Open the login page in a browser. When False (or when
                         no display is detected) the URL is printed and you
                         paste the API key instead. Does not affect signup,
                         which never needs a browser.
    :param timeout: Seconds to wait for the browser callback. The paste prompt
                    stays available throughout.
    :returns: The access token.
    :raises InteractiveLoginError: If no terminal is available, the flow was
                                   aborted, or the resulting token was rejected.
    """
    if not force_relogin:
        from tabpfn_client.service_wrapper import UserAuthenticationClient

        existing = UserAuthenticationClient.resolve_token()
        if existing is not None and ServiceClient.is_auth_token_outdated(existing):
            from tabpfn_client.config import set_access_token

            set_access_token(existing)
            print(
                "\nAlready logged in. "
                "Pass force_relogin=True to log in as a different user.\n"
            )
            return existing

    if not _stdin_is_interactive():
        raise InteractiveLoginError(
            "interactive_login() needs an interactive terminal.\n"
            f"Generate a token at {URL_PRIOR_LABS_API_KEYS} and either\n"
            "  - set it as the TABPFN_TOKEN environment variable, or\n"
            "  - pass it to tabpfn_client.set_access_token('<your-token>')."
        )

    from tabpfn_client.prompt_agent import PromptAgent

    PromptAgent.prompt_welcome()

    try:
        choice = _prompt_menu()
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        raise InteractiveLoginError("Login was cancelled.") from None

    if choice == "q":
        raise InteractiveLoginError("Login was cancelled.")

    gui_url = str(ServiceClient.server_config.gui_url)

    if choice == "signup":
        try:
            token = PromptAgent.prompt_signup()
        except KeyboardInterrupt:
            sys.stdout.write("\n")
            raise InteractiveLoginError("Signup was cancelled.") from None
    elif open_browser and _has_display():
        token = _browser_login(gui_url, timeout)
    else:
        token = _paste_only_login(f"{gui_url}/login", timeout)

    if not token:
        raise InteractiveLoginError(
            "Login was not completed.\n"
            f"Generate a token at {URL_PRIOR_LABS_API_KEYS} and either\n"
            "  - set it as the TABPFN_TOKEN environment variable, or\n"
            "  - pass it to tabpfn_client.set_access_token('<your-token>')."
        )

    is_valid = ServiceClient.is_auth_token_outdated(token)
    if is_valid is None:
        raise InteractiveLoginError(
            "Your TabPFN account's email address is not verified.\n"
            "Check your inbox for the verification email, or sign in at\n"
            f"  {gui_url}\n"
            "to request a new one, then run interactive_login() again."
        )
    if not is_valid:
        raise InteractiveLoginError(
            "The access token was rejected by the server. Please try again, or "
            f"generate a new token at {URL_PRIOR_LABS_API_KEYS}."
        )

    # Imported here to keep the module importable without pulling in config.
    from tabpfn_client.config import set_access_token
    from tabpfn_client.service_wrapper import UserAuthenticationClient

    set_access_token(token)
    # The only place the token cache is written. A token supplied via
    # TABPFN_TOKEN or set_access_token() stays in-process; an explicit login is
    # the one case where persisting is what the user asked for.
    UserAuthenticationClient.persist_token(token)
    print("\nLogin successful. The API key is cached for future runs.\n")
    return token
