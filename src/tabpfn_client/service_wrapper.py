#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import logging
import os
from pathlib import Path

from uuid import UUID
from tabpfn_client.client import (
    ServiceClient,
    ClientOptions,
    PredictionResult,
)
import tabpfn_client.constants as constants
from tabpfn_common_utils.utils import Singleton
from tabpfn_client.api_models import (
    FitTaskConfig,
    ClassifierConfig,
    RegressorConfig,
    ThinkingConfig,
    TabPFNSystem,
)
from tabpfn_client.options import get_opts
from tabpfn_client.models import ApiMode

logger = logging.getLogger(__name__)


class ServiceClientWrapper:
    pass


# Singleton class for user authentication
class UserAuthenticationClient(ServiceClientWrapper, Singleton):
    """
    Wrapper of ServiceClient to handle user authentication, including:
    - user registration and login
    - access token caching

    This is implemented as a singleton class with classmethods.
    """

    CACHED_TOKEN_FILE = constants.CACHE_DIR / "config"

    def __new__(cls):
        raise TypeError(
            "This class should not be instantiated. Use classmethods instead."
        )

    @classmethod
    def is_accessible_connection(cls) -> bool:
        return ServiceClient.try_connection()

    @classmethod
    def set_token(cls, access_token: str):
        """Use *access_token* for this process. Does not touch the token cache.

        A token supplied through TABPFN_TOKEN or `set_access_token()` belongs to
        the caller, so we never copy it to disk behind their back. Only
        `interactive_login()` persists, via `persist_token`.
        """
        ServiceClient.authorize(access_token)

    @classmethod
    def persist_token(cls, access_token: str):
        """Cache *access_token* so later runs authenticate without prompting.

        The sole caller is `interactive_login()`: the user explicitly logged in,
        so remembering the result is what they asked for.
        """
        # Mitigate parallel writes by checking if the token is already set to
        # the same value. We'll consider using fcntl if this problem persists.
        try:
            if cls.CACHED_TOKEN_FILE.read_text() == access_token:
                return
        except FileNotFoundError:
            pass

        # Write the new token
        cls.CACHED_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        cls.CACHED_TOKEN_FILE.write_text(access_token)

    @classmethod
    def validate_email(cls, email: str) -> tuple[bool, str]:
        return ServiceClient.validate_email(email)

    @classmethod
    def get_password_policy(cls):
        return ServiceClient.get_password_policy()

    @classmethod
    def send_verification_email(cls, access_token: str) -> tuple[bool, str]:
        return ServiceClient.send_verification_email(access_token)

    @classmethod
    def verify_email(cls, token: str, access_token: str) -> tuple[bool, str]:
        return ServiceClient.verify_email(token, access_token)

    @classmethod
    def set_token_by_registration(
        cls,
        email: str,
        password: str,
        password_confirm: str,
        validation_link: str,
        additional_info: dict,
    ) -> tuple[bool, str, str | None]:
        is_created, message, access_token = ServiceClient.register(
            email, password, password_confirm, validation_link, additional_info
        )
        if access_token is not None:
            # Signing up is an explicit interactive act, so remembering the
            # result is what the user asked for -- same rationale as
            # interactive_login().
            cls.set_token(access_token)
            cls.persist_token(access_token)
        return is_created, message, access_token

    @classmethod
    def resolve_token_with_source(cls) -> tuple[str | None, str | None]:
        """Find an access token without prompting, and say where it came from.

        Resolution order: a token already set on this process (via
        `set_access_token`), then the TABPFN_TOKEN environment variable, then a
        token cached by a previous run. The environment is read here rather than
        at import time so that setting TABPFN_TOKEN after importing the package
        still takes effect.

        The source matters when a token turns out to be bad: only the source
        that produced it may be discarded.
        """
        access_token = ServiceClient.get_access_token()
        if access_token:
            return access_token, "process"

        env_token = os.environ.get("TABPFN_TOKEN")
        if env_token:
            return env_token.strip(), "env"

        if cls.CACHED_TOKEN_FILE.exists():
            cached = cls.CACHED_TOKEN_FILE.read_text().strip()
            if cached:
                return cached, "cache"

        return None, None

    @classmethod
    def resolve_token(cls) -> str | None:
        """Find an access token without prompting, or return None."""
        return cls.resolve_token_with_source()[0]

    @classmethod
    def try_reuse_existing_token(cls) -> tuple[bool, str | None]:
        access_token, source = cls.resolve_token_with_source()
        if access_token is None:
            return False, None

        is_valid = ServiceClient.is_auth_token_outdated(access_token)
        if is_valid is False:
            # Discard only what actually failed. A bad TABPFN_TOKEN must not
            # take the cached token from an earlier login down with it, or
            # unsetting the variable would leave the user with nothing.
            cls._discard_token(source)
            return False, None
        elif is_valid is None:
            return False, access_token

        logger.debug(f"Reusing existing access token? {is_valid}")
        cls.set_token(access_token)

        return True, access_token

    @classmethod
    def reset_cache(cls):
        cls._reset_token()

    @classmethod
    def _discard_token(cls, source: str | None) -> None:
        """Drop the rejected token, leaving the other sources intact."""
        ServiceClient.reset_authorization()
        if source == "cache":
            cls.CACHED_TOKEN_FILE.unlink(missing_ok=True)
        elif source == "env":
            # The variable is set outside the client; here it can only be used
            # or unset, never set.
            get_opts().TABPFN_TOKEN = None

    @classmethod
    def _reset_token(cls):
        ServiceClient.reset_authorization()
        cls.CACHED_TOKEN_FILE.unlink(missing_ok=True)
        # The TABPFN_TOKEN var is always set externally in the environment, in
        # the client it can only be used or unset, never set.
        # Note: we should prefix with the module to make sure the variable is not
        # only mutated in the local module binding.
        get_opts().TABPFN_TOKEN = None

    @classmethod
    def retrieve_greeting_messages(cls):
        return ServiceClient.retrieve_greeting_messages()


class UserDataClient(ServiceClientWrapper, Singleton):
    """
    Wrapper of ServiceClient to handle user data, including:
    - query, or delete user account data
    - query, download, or delete uploaded data
    """

    @classmethod
    def get_data_summary(cls) -> dict:
        try:
            summary = ServiceClient.get_data_summary()
        except RuntimeError as e:
            logging.error(f"Failed to get data summary: {e}")
            raise e

        return summary

    @classmethod
    def download_all_data(cls, save_dir: Path = Path(".")) -> Path:
        try:
            saved_path = ServiceClient.download_all_data(save_dir)
        except RuntimeError as e:
            logging.error(f"Failed to download data: {e}")
            raise e

        if saved_path is None:
            raise RuntimeError("Failed to download data.")

        logging.info(f"Data saved to {saved_path}")
        return saved_path

    @classmethod
    def delete_dataset(cls, dataset_uid: str) -> list[str]:
        try:
            deleted_datasets = ServiceClient.delete_dataset(dataset_uid)
        except RuntimeError as e:
            logging.error(f"Failed to delete dataset: {e}")
            raise e

        logging.info(f"Deleted datasets: {deleted_datasets}")

        return deleted_datasets

    @classmethod
    def delete_all_datasets(cls) -> list[str]:
        try:
            deleted_datasets = ServiceClient.delete_all_datasets()
        except RuntimeError as e:
            logging.error(f"Failed to delete all datasets: {e}")
            raise e

        logging.info(f"Deleted datasets: {deleted_datasets}")

        return deleted_datasets

    @classmethod
    def delete_user_account(cls):
        # local import to avoid circular import
        from tabpfn_client.prompt_agent import PromptAgent

        if not PromptAgent.confirm_user_account_deletion():
            logger.info("Account deletion cancelled — confirmation phrase not entered.")
            return

        try:
            ServiceClient.delete_user_account()
        except RuntimeError as e:
            logging.error(f"Failed to delete user account: {e}")
            raise e

        PromptAgent.prompt_account_deleted()


class InferenceClient(ServiceClientWrapper, Singleton):
    """
    Wrapper of ServiceClient to handle inference, including:
    - fitting
    - prediction
    """

    def __new__(cls):
        raise TypeError(
            "This class should not be instantiated. Use classmethods instead."
        )

    @classmethod
    def fit(
        cls,
        X,
        y,
        task_config: FitTaskConfig,
        tabpfn_systems: list[TabPFNSystem],
        thinking_config: ThinkingConfig | None,
        api_mode: ApiMode,
        client_options: ClientOptions | None,
        description: str | None,
    ) -> UUID:
        return ServiceClient.fit(
            X,
            y,
            task_config=task_config,
            tabpfn_systems=tabpfn_systems,
            thinking_config=thinking_config,
            api_mode=api_mode,
            client_options=client_options,
            description=description,
        )

    @classmethod
    def predict(
        cls,
        X,
        fitted_train_set_id: UUID,
        task_config: ClassifierConfig | RegressorConfig,
        client_options: ClientOptions | None = None,
    ) -> PredictionResult:
        return ServiceClient.predict(
            x_test=X,
            fitted_train_set_id=fitted_train_set_id,
            task_config=task_config,
            client_options=client_options,
        )
