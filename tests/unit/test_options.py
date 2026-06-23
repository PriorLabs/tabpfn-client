import os
from tabpfn_client.options import reload_opts


def test_reload_opts():
    from tabpfn_client.options import get_opts
    from tabpfn_client import set_access_token

    assert get_opts().TABPFN_TOKEN is None

    set_access_token("dummy_token")
    assert get_opts().TABPFN_TOKEN == "dummy_token"

    os.environ["TABPFN_TOKEN"] = "dummy_token_from_env"
    reload_opts()
    assert get_opts().TABPFN_TOKEN == "dummy_token_from_env"
