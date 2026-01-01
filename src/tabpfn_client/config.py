#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import shutil

from httpx import ConnectError

from tabpfn_client.client import ServiceClient
from tabpfn_client.service_wrapper import UserAuthenticationClient
from tabpfn_client.constants import CACHE_DIR
from tabpfn_client.prompt_agent import PromptAgent
from tabpfn_client.ui import console, warn


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
    Initializes the TabPFN client and handles user authentication.

    This function checks for existing credentials, prompts for login/registration 
    if necessary, and verifies email status. It must be called before performing 
    any inference tasks.

    :param use_server: Whether to use the TabPFN cloud service. Currently, only 
                       True is supported.
    :raises RuntimeError: If local inference is requested or if the server is unreachable.
    """
    # initialize config
    Config.use_server = use_server

    if Config.is_initialized:
        # Only do the following if the initialization has not been done yet
        return

    if use_server:
        try:
            is_valid_token, access_token = (
                UserAuthenticationClient.try_reuse_existing_token()
            )
        except ConnectError:
            raise CONNECTION_ERROR

        # TODO: no need to check connection again if token is valid, need to
        # adjust tests accordingly.
        if not UserAuthenticationClient.is_accessible_connection():
            raise CONNECTION_ERROR

        if is_valid_token:
            PromptAgent.prompt_reusing_existing_token()
        elif access_token is not None:
            # token holds invalid due to user email verification
            console.print()
            warn("Email not verified")
            console.print(
                "  [blue]You need to verify your email before continuing.[/blue]"
            )
            result = PromptAgent.reverify_email(access_token)

            if result == "restart":
                # User chose to start over - show main menu
                PromptAgent.prompt_welcome()
                success = PromptAgent.prompt_and_set_token()
                if not success:
                    return
            elif result is False:
                # User chose to quit - exit without showing menu
                return
            # else: result is True, verification successful, continue to greeting messages
        else:
            PromptAgent.prompt_welcome()
            # prompt for login / register
            success = PromptAgent.prompt_and_set_token()
            if not success:
                # User interrupted or quit - don't mark as initialized
                return

        # Print new greeting messages. If there are no new messages, nothing will be printed.
        PromptAgent.prompt_retrieved_greeting_messages(
            UserAuthenticationClient.retrieve_greeting_messages()
        )

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
    return ServiceClient.get_access_token()


def set_access_token(access_token: str):
    """
    Manually sets the access token for the session.

    This is useful for non-interactive environments
    (e.g., CI/CD, Notebooks) where you want to skip 
    the interactive login prompt.
    
    You can obtain your access token via the TabPFN 
    UX as explained at:
    https://docs.priorlabs.ai/api-reference/getting-started#1-get-your-access-token

    :param access_token: A valid TabPFN access token string.
    """
    UserAuthenticationClient.set_token(access_token)
    Config.is_initialized = True


def get_api_usage() -> dict:
    """
    Fetches and formats the current API usage statistics for the user.

    :return: A human-readable string detailing current credit usage, 
             the total limit, and when the limit resets.
    """
    access_token = get_access_token()
    response = ServiceClient.get_api_usage(access_token)
    return f"Currently, you have used {response['current_usage']} of the allowed limit of {'Unlimited' if int(response['usage_limit']) == -1 else response['usage_limit']} credits. The limit will reset at {response['reset_time']}."
