import unittest
from pathlib import Path

import yaml

import tabpfn_client


class TestServerConfig(unittest.TestCase):
    def setUp(self):
        # Get the path to the config file relative to the test file
        package_dir = Path(tabpfn_client.__file__).parent
        config_path = package_dir / "server_config.yaml"

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def test_host_configuration(self):
        expected_host = "api.priorlabs.ai"
        self.assertEqual(
            self.config["host"], expected_host, f"Host should be {expected_host}"
        )
