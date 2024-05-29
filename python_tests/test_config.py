import unittest
import json
import os

CONFIG_FILE = "configurations/config.json"


def load_config(config_path=CONFIG_FILE):
    with open(config_path, "r") as f:
        return json.load(f)


def save_config(config, config_path=CONFIG_FILE):
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


config = load_config()


class TestConfiguration(unittest.TestCase):
    def setUp(self):
        self.config_path = "test_config.json"  # Test configuration file
        self.test_data = {"batch_size": 32, "block_size": 128}
        with open(self.config_path, "w") as f:
            json.dump(self.test_data, f)

    def test_load_config(self):
        # Test loading the configuration
        loaded_config = load_config(self.config_path)
        self.assertEqual(loaded_config, self.test_data)

    def test_save_config(self):
        # Test saving the configuration
        new_config = {"batch_size": 64, "block_size": 256}
        save_config(new_config, self.config_path)
        with open(self.config_path, "r") as f:
            config = json.load(f)
        self.assertEqual(config, new_config)

    def tearDown(self):
        os.remove(self.config_path)


if __name__ == "__main__":
    unittest.main()
