"""Configuration and secrets subsystem for LemGendary Model Training Suite."""

from training.config.secrets import get_secret, load_secrets

__all__ = ["get_secret", "load_secrets"]
