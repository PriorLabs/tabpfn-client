#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""Client settings read from environment variables."""

from pydantic_settings import BaseSettings


class Options(BaseSettings):
    """Client settings, each overridable by the environment variable of the same name.

    Attributes:
        TABPFN_TOKEN: Access token used for authentication.
        TABPFN_CLIENT_API_URL: Override for the server URL.
        TABPFN_CLIENT_MAX_THREAD_PER_UPLOAD: Maximum threads used per upload.
        TABPFN_CLIENT_TIMEOUT: Default request timeout in seconds.
        TABPFN_CLIENT_UPLOAD_TIMEOUT: Upload timeout in seconds.
        TABPFN_CLIENT_FORCE_REUPLOAD: Whether to re-upload datasets the server
            already holds.
        TABPFN_CLIENT_DEDUP_DATASETS: Whether to deduplicate dataset uploads.
        TABPFN_CLIENT_ASYNC_USE_ABOVE_TRAINSET_SIZE: Train-set size in bytes above
            which the asynchronous API is used. Overridden by the server default
            unless set in the environment.
        TABPFN_CLIENT_ASYNC_POLL_TIMEOUT: Timeout in seconds when polling an
            asynchronous job. Overridden by the server default unless set in the
            environment.
    """

    TABPFN_TOKEN: str | None = None
    TABPFN_CLIENT_API_URL: str | None = None
    TABPFN_CLIENT_MAX_THREAD_PER_UPLOAD: int = 8
    TABPFN_CLIENT_TIMEOUT: float = 900.0
    TABPFN_CLIENT_UPLOAD_TIMEOUT: float = 7200.0  # 2 hours
    TABPFN_CLIENT_FORCE_REUPLOAD: bool = False
    TABPFN_CLIENT_DEDUP_DATASETS: bool = True

    # Overriden by server defaults
    # Environment > server default > client default
    TABPFN_CLIENT_ASYNC_USE_ABOVE_TRAINSET_SIZE: int = 50 * 1024 * 1024
    TABPFN_CLIENT_ASYNC_POLL_TIMEOUT: float = 7200.0


_opts: Options | None = None


# TODO(refactor): Some opts are used in the http client which is initialized as a class
# variable of the ServiceClient singleton. These cannot be updated after importing.
def reload_opts() -> None:
    """Re-read the options from the environment."""
    global _opts
    _opts = Options()


def get_opts() -> Options:
    """Return the current options, reading them from the environment on first use."""
    # Constructed on first use rather than at import time, so that environment
    # variables set after `import tabpfn_client` are still picked up.
    global _opts
    if _opts is None:
        _opts = Options()
    return _opts
