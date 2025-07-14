import respx
from contextlib import AbstractContextManager

from tabpfn_client.client import SERVER_CONFIG


class MockTabPFNServer(AbstractContextManager):
    def __init__(self):
        self.server_config = SERVER_CONFIG
        self.endpoints = self.server_config.endpoints
        self.base_url = f"{self.server_config.protocol}://{self.server_config.host}:{self.server_config.port}"
        self.router = None

    def __enter__(self):
        self.router = respx.mock(base_url=self.base_url, assert_all_called=True)
        self.router.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.router.stop()


def with_mock_server():
    def decorator(func):
        def wrapper(test_class, *args, **kwargs):
            with MockTabPFNServer() as mock_server:
                return func(test_class, mock_server, *args, **kwargs)

        return wrapper

    return decorator
