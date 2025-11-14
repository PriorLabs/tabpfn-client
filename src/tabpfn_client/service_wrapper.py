#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import logging
import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from tabpfn_client.client import ServiceClient, ModelType
from tabpfn_client.constants import CACHE_DIR
from tabpfn_common_utils.utils import Singleton

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

    CACHED_TOKEN_FILE = CACHE_DIR / "config"

    def __new__(self):
        raise TypeError(
            "This class should not be instantiated. Use classmethods instead."
        )

    @classmethod
    def is_accessible_connection(cls) -> bool:
        return ServiceClient.try_connection()

    @classmethod
    def set_token(cls, access_token: str):
        ServiceClient.authorize(access_token)

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
        is_valid, message = ServiceClient.validate_email(email)
        return is_valid, message

    @classmethod
    def set_token_by_registration(
        cls,
        email: str,
        password: str,
        password_confirm: str,
        validation_link: str,
        additional_info: dict,
    ) -> tuple[bool, str]:
        is_created, message, access_token = ServiceClient.register(
            email, password, password_confirm, validation_link, additional_info
        )
        if access_token is not None:
            cls.set_token(access_token)
        return is_created, message, access_token

    @classmethod
    def set_token_by_login(cls, email: str, password: str) -> tuple[bool, str]:
        access_token, message, status_code = ServiceClient.login(email, password)

        if access_token is None:
            return False, message, status_code

        cls.set_token(access_token)
        return access_token, message, status_code

    @classmethod
    def try_reuse_existing_token(cls) -> tuple[bool, str or None]:
        if ServiceClient.get_access_token() is None:
            if not cls.CACHED_TOKEN_FILE.exists():
                return False, None

            access_token = cls.CACHED_TOKEN_FILE.read_text()

        else:
            access_token = ServiceClient.get_access_token()

        is_valid = ServiceClient.is_auth_token_outdated(access_token)
        if is_valid is False:
            cls._reset_token()
            return False, None
        elif is_valid is None:
            return False, access_token

        logger.debug(f"Reusing existing access token? {is_valid}")
        cls.set_token(access_token)

        return True, access_token

    @classmethod
    def get_password_policy(cls):
        return ServiceClient.get_password_policy()

    @classmethod
    def reset_cache(cls):
        cls._reset_token()

    @classmethod
    def _reset_token(cls):
        ServiceClient.reset_authorization()
        cls.CACHED_TOKEN_FILE.unlink(missing_ok=True)

    @classmethod
    def retrieve_greeting_messages(cls):
        return ServiceClient.retrieve_greeting_messages()

    @classmethod
    def send_reset_password_email(cls, email: str) -> tuple[bool, str]:
        sent, message = ServiceClient.send_reset_password_email(email)
        return sent, message

    @classmethod
    def send_verification_email(cls, access_token: str) -> tuple[bool, str]:
        sent, message = ServiceClient.send_verification_email(access_token)
        return sent, message

    @classmethod
    def verify_email(cls, token: str, access_token: str) -> tuple[bool, str]:
        verified, message = ServiceClient.verify_email(token, access_token)
        return verified, message

    @classmethod
    def try_browser_login(cls) -> tuple[bool, str]:
        """Try to authenticate using browser-based login"""

        # Allow disabling browser login via environment variables
        # - TABPFN_NO_BROWSER=1/true/yes/on
        # - TABPFN_CLIENT_NO_BROWSER=1/true/yes/on (backward-compatible alias)
        def _truthy(val: str) -> bool:
            return bool(val) and val.strip().lower() in {"1", "true", "yes", "on"}

        if _truthy(os.environ.get("TABPFN_NO_BROWSER")) or _truthy(
            os.environ.get("TABPFN_CLIENT_NO_BROWSER")
        ):
            return False, "Browser login disabled by environment variable"

        success, token_or_message = ServiceClient.try_browser_login()
        if success:
            cls.set_token(token_or_message)
        return success, token_or_message


class UserDataClient(ServiceClientWrapper, Singleton):
    """
    Wrapper of ServiceClient to handle user data, including:
    - query, or delete user account data
    - query, download, or delete uploaded data
    """

    @classmethod
    def get_data_summary(cls) -> {}:
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

        confirm_pass = PromptAgent.prompt_confirm_password_for_user_account_deletion()
        try:
            ServiceClient.delete_user_account(confirm_pass)
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

    def __new__(self):
        raise TypeError(
            "This class should not be instantiated. Use classmethods instead."
        )

    @classmethod
    def fit(
        cls,
        X,
        y,
        model_type: ModelType = ModelType.TABPFN,
        tabpfn_config=None,
        task: Optional[Literal["classification", "regression"]] = None,
        description: str = "",
    ) -> str:
        return ServiceClient.fit(
            X,
            y,
            tabpfn_config=tabpfn_config,
            model_type=model_type,
            task=task,
            description=description,
        )

    @classmethod
    def predict(
        cls,
        X,
        task: Literal["classification", "regression"],
        train_set_uid: str,
        model_type: ModelType = ModelType.TABPFN,
        tabpfn_config=None,
        predict_params=None,
        X_train=None,
        y_train=None,
    ) -> dict[str, np.ndarray]:
        return ServiceClient.predict(
            train_set_uid=train_set_uid,
            x_test=X,
            tabpfn_config=tabpfn_config,
            predict_params=predict_params,
            task=task,
            X_train=X_train,
            y_train=y_train,
            model_type=model_type,
        )
