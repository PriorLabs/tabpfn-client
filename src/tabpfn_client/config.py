#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import shutil
import sys

from httpx import ConnectError

from tabpfn_client.client import ServiceClient
from tabpfn_client.service_wrapper import UserAuthenticationClient
from tabpfn_client.constants import CACHE_DIR, URL_PRIOR_LABS_API_KEYS
from tabpfn_client.prompt_agent import PromptAgent, maybe_graceful_exit
from tabpfn_client.ui import console, warn
from tabpfn_client.options import reload_opts


CONNECTION_ERROR = RuntimeError(
    "TabPFN is inaccessible at the moment, please try again later."
)


def _stdin_is_interactive() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


class Config:
    def __new__(cls, *args, **kwargs):
        """
        This class is a singleton and should not be instantiated directly.
        Only use class methods.
        """
        raise TypeError("Cannot instantiate this class")

    is_initialized = False
    use_server = False
    token: str | None = None


def init(use_server=True):
    """
    Initializes the TabPFN client and authenticates with the TabPFN cloud service.

    Authentication is token-based. The token is taken from the TABPFN_TOKEN
    environment variable, or from a token previously passed to
    `set_access_token()`. If neither is available and the session is
    interactive, you are asked to paste a token; otherwise a RuntimeError
    explains how to obtain one.

    Generate a token at https://ux.priorlabs.ai/account/api-keys

    :param use_server: Whether to use the TabPFN cloud service. Currently, only
                       True is supported.
    :raises RuntimeError: If local inference is requested, if the server is
                          unreachable, or if no valid access token is available.
    """
    # initialize config
    Config.use_server = use_server

    if Config.is_initialized:
        # Only do the following if the initialization has not been done yet
        return

    reload_opts()

    if use_server:
        # Remember whether a token was supplied at all: a rejected token needs a
        # different message than a missing one, and the check below discards it.
        had_token = UserAuthenticationClient.resolve_token() is not None
        try:
            is_valid_token, unverified_token = (
                UserAuthenticationClient.try_reuse_existing_token()
            )
        except ConnectError:
            raise CONNECTION_ERROR

        if is_valid_token:
            PromptAgent.prompt_reusing_existing_token()
        elif unverified_token is not None:
            # The token is well-formed but the account's email is unverified,
            # which no token can work around.
            raise RuntimeError(
                "Your TabPFN account's email address is not verified. Please "
                "verify it before using the client."
            )
        else:
            if not UserAuthenticationClient.is_accessible_connection():
                raise CONNECTION_ERROR
            if not _prompt_and_set_token(token_was_rejected=had_token):
                # User interrupted - don't mark as initialized
                return

        # Print new greeting messages. If there are no new messages, nothing will be printed.
        PromptAgent.prompt_retrieved_greeting_messages(
            UserAuthenticationClient.retrieve_greeting_messages()
        )

        _ = ServiceClient.get_settings()

        Config.use_server = True
        Config.is_initialized = True
    else:
        raise RuntimeError("Local inference is not supported yet.")


def _prompt_and_set_token(token_was_rejected: bool = False) -> bool:
    """Ask for an access token interactively and validate it against the server.

    Returns True once a valid token is set, False if the user interrupted.
    Raises RuntimeError when the session is non-interactive, since there is no
    way to obtain a token without input.
    """
    if not _stdin_is_interactive():
        raise RuntimeError(PromptAgent.token_instructions(rejected=token_was_rejected))

    if token_was_rejected:
        warn("Your TabPFN access token was rejected by the server.")
        console.print(
            f"  [cyan]Generate a new one at {URL_PRIOR_LABS_API_KEYS}.[/cyan]"
        )
    else:
        PromptAgent.prompt_welcome()

    try:
        while True:
            token = PromptAgent.prompt_for_token()
            if token is None:
                return False

            try:
                is_valid = ServiceClient.is_auth_token_outdated(token)
            except ConnectError:
                raise CONNECTION_ERROR

            if is_valid is None:
                raise RuntimeError(
                    "Your TabPFN account's email address is not verified. Please "
                    "verify it before using the client."
                )
            if is_valid:
                UserAuthenticationClient.set_token(token)
                PromptAgent.prompt_token_accepted()
                return True

            warn("That access token was not accepted by the server.")
            console.print(
                f"  [cyan]Check that you copied it in full from {URL_PRIOR_LABS_API_KEYS}.[/cyan]"
            )
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted. Goodbye![/yellow]")
        maybe_graceful_exit()
        return False


def reset():
    """
    Resets the client state and clears local authentication caches.

    Use this function if you need to log out or clear stored session data
    from the local machine.
    """
    Config.is_initialized = False
    # reset user auth handler
    if Config.use_server:
        UserAuthenticationClient.reset_cache()

    # remove cache dir
    shutil.rmtree(CACHE_DIR, ignore_errors=True)


def get_access_token() -> str:
    """
    Retrieves the current active access token.

    If the client is not yet initialized, this will trigger the `init()` login flow.

    :return: The access token string used for API requests.
    """
    init()
    access_token = ServiceClient.get_access_token()
    if access_token is None:
        raise CONNECTION_ERROR
    return access_token


def set_access_token(access_token: str):
    """
    Manually sets the access token for the session.

    Use this in non-interactive environments (e.g. CI/CD, notebooks) as an
    alternative to the TABPFN_TOKEN environment variable.

    Generate a token at https://ux.priorlabs.ai/account/api-keys

    :param access_token: A valid TabPFN access token string.
    """
    UserAuthenticationClient.set_token(access_token)
    Config.is_initialized = True


def get_api_usage() -> str:
    """
    Fetches and formats the current API usage statistics for the user.

    :return: A human-readable string detailing current credit usage,
             the total limit, and when the limit resets.
    """
    access_token = get_access_token()
    response = ServiceClient.get_api_usage(access_token)
    return f"Currently, you have used {response['current_usage']} of the allowed limit of {'Unlimited' if int(response['usage_limit']) == -1 else response['usage_limit']} credits. The limit will reset at {response['reset_time']}."
