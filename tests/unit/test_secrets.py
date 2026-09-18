"""Unit tests for training/config/secrets.py in LemGendary Model Training Suite."""

import os
import unittest
from pathlib import Path

from training.config.secrets import load_secrets, get_secret, _mask_secret
from training.utils.paths import get_project_root


class TestConfigSecrets(unittest.TestCase):
    """Tests dotfile secrets loading, masking, and environment injection."""

    def test_mask_secret(self) -> None:
        self.assertEqual(_mask_secret(""), "")
        self.assertEqual(_mask_secret("short"), "***")
        self.assertEqual(_mask_secret("ghp_1234567890abcdef"), "ghp...def")

    def test_load_secrets_returns_dict(self) -> None:
        summary = load_secrets()
        self.assertIsInstance(summary, dict)
        self.assertIn("GITHUB_PAT", summary)
        self.assertIn("GOOGLE_DRIVE", summary)
        self.assertIn("KAGGLE_KEY", summary)

    def test_get_secret_default(self) -> None:
        val = get_secret("NON_EXISTENT_LEMGENDARY_SECRET_KEY", default="fallback_val")
        self.assertEqual(val, "fallback_val")

    def test_get_secret_with_env(self) -> None:
        os.environ["TEST_SECRET_LEMGENDARY"] = "my_secret_token_123"
        try:
            val = get_secret("TEST_SECRET_LEMGENDARY")
            self.assertEqual(val, "my_secret_token_123")
        finally:
            del os.environ["TEST_SECRET_LEMGENDARY"]


if __name__ == "__main__":
    unittest.main()
