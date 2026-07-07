#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""Isolate process-global client state between unit tests.

Several tests mutate the module-global ``options._opts`` (e.g. via
``set_access_token`` / ``ServiceClient.authorize``) or the ``TABPFN_TOKEN``
environment variable. Without isolation that state leaks into later tests and
makes ordering-dependent assertions (such as ``test_reload_opts``) fail. Snapshot
both before each test and restore afterwards so tests cannot pollute one another.
"""

import os

import pytest

import tabpfn_client.options as options


@pytest.fixture(autouse=True)
def _restore_global_client_state():  # pyright: ignore[reportUnusedFunction]
    saved_opts = options._opts.model_copy(deep=True)
    saved_token = os.environ.get("TABPFN_TOKEN")
    try:
        yield
    finally:
        options._opts = saved_opts
        if saved_token is None:
            os.environ.pop("TABPFN_TOKEN", None)
        else:
            os.environ["TABPFN_TOKEN"] = saved_token
