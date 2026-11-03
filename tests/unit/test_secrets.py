"""Unit tests for training/config/secrets.py in LemGendary Model Training Suite."""

import os
import unittest

from training.config.secrets import _mask_secret, get_secret, load_secrets


class TestConfigSecrets(unittest.TestCase):
    """Tests dotfile secrets loading, masking, and environment injection."""

    def test_mask_secret(self) -> None:
        """Verify secret masking for empty, short and long values."""
        self.assertEqual(_mask_secret(""), "")
        self.assertEqual(_mask_secret("short"), "***")
        self.assertEqual(_mask_secret("ghp_1234567890abcdef"), "ghp...def")

    def test_load_secrets_returns_dict(self) -> None:
        """Verify load_secrets returns a summary dict of known keys."""
        summary = load_secrets()
        self.assertIsInstance(summary, dict)
        self.assertIn("GITHUB_PAT", summary)
        self.assertIn("GOOGLE_DRIVE", summary)
        self.assertIn("KAGGLE_KEY", summary)

    def test_get_secret_default(self) -> None:
        """Verify get_secret falls back to the default."""
        val = get_secret("NON_EXISTENT_LEMGENDARY_SECRET_KEY", default="fallback_val")
        self.assertEqual(val, "fallback_val")

    def test_get_secret_with_env(self) -> None:
        """Verify get_secret reads from the environment."""
        os.environ["TEST_SECRET_LEMGENDARY"] = "my_secret_token_123"
        try:
            val = get_secret("TEST_SECRET_LEMGENDARY")
            self.assertEqual(val, "my_secret_token_123")
        finally:
            del os.environ["TEST_SECRET_LEMGENDARY"]


if __name__ == "__main__":
    unittest.main()
