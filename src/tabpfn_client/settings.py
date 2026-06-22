#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    TABPFN_CLIENT_API_URL: str = ""
    TABPFN_CLIENT_MAX_THREAD_PER_UPLOAD: int = 8
    TABPFN_CLIENT_TIMEOUT: float = 900.0
    TABPFN_CLIENT_UPLOAD_TIMEOUT: float = 7200.0
    TABPFN_CLIENT_POLL_INTERVAL: float = 5.0
    TABPFN_CLIENT_POLL_TIMEOUT: float = 7200.0  # 2 hours
    TABPFN_CLIENT_CI_MODE: bool = False
    TABPFN_CLIENT_FORCE_ASYNC: bool = False
    TABPFN_CLIENT_FORCE_REUPLOAD: bool = False
    TABPFN_CLIENT_DEDUP_DATASETS: bool = True
