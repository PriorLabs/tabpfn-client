#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import shutil

from httpx import ConnectError

from tabpfn_client.client import ServiceClient
from tabpfn_client.service_wrapper import UserAuthenticationClient
from tabpfn_client.constants import CACHE_DIR
from tabpfn_client.prompt_agent import PromptAgent
from tabpfn_client.options import reload_opts


CONNECTION_ERROR = RuntimeError(
    "TabPFN is inaccessible at the moment, please try again later."
)


class Config:
    def __new__(cls, *args, **kwargs):
        """
        This class is a singleton and should not be instantiated directly.
        Only use class methods.
        """
        raise TypeError("Cannot instantiate this class")

    is_initialized = False
    use_server = False


def init(use_server=True):
    """
    Initializes the TabPFN client and authenticates with the TabPFN cloud service.

    Authentication is token-based and never interactive. The token comes from
    `set_access_token()` or the TABPFN_TOKEN environment variable, or from a
    token cached by an earlier `interactive_login()`. If none is available this
    raises, explaining how to obtain one -- it will not prompt.

    Generate a token at https://ux.priorlabs.ai/account/api-keys, or call
    `tabpfn_client.interactive_login()` to log in through the browser.

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
            # Never prompt from the default path: this is a library, so a
            # missing token is an error the caller resolves, either by
            # supplying one or by calling interactive_login() explicitly.
            raise RuntimeError(PromptAgent.token_instructions(rejected=had_token))

        # Print new greeting messages. If there are no new messages, nothing will be printed.
        PromptAgent.prompt_retrieved_greeting_messages(
            UserAuthenticationClient.retrieve_greeting_messages()
        )

        _ = ServiceClient.get_settings()

        Config.use_server = True
        Config.is_initialized = True
    else:
        raise RuntimeError("Local inference is not supported yet.")


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
