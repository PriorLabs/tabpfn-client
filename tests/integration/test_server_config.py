import unittest
import yaml
import os


class TestServerConfig(unittest.TestCase):
    def setUp(self):
        # Get the path to the config file relative to the test file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "..", "..", "server_config.yaml")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def test_host_configuration(self):
        expected_host = "tabpfn-server-wjedmz7r5a-ez.a.run.app"
        self.assertEqual(
            self.config["host"], expected_host, f"Host should be {expected_host}"
        )
