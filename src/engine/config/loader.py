"""Load config from YAML + .env, validate via Pydantic, return frozen AppConfig."""

import logging
from pathlib import Path

import yaml
from dotenv import dotenv_values

from .schema import AppConfig, SecretsConfig

logger = logging.getLogger(__name__)

DEFAULT_YAML = Path(__file__).parent / "settings.yaml"
DEFAULT_ENV = Path(__file__).parents[3] / ".env"  # project root


def load_config(
    yaml_path: Path = DEFAULT_YAML,
    env_path: Path = DEFAULT_ENV,
) -> AppConfig:
    """Load and validate configuration.

    1. Read YAML for strategy parameters
    2. Read .env for secrets (API keys, tokens)
    3. Validate everything via Pydantic
    4. Return frozen (immutable) AppConfig
    """
    # Load YAML
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    # Load .env secrets
    secrets_data = {}
    if env_path.exists():
        env = dotenv_values(env_path)
        secrets_data = {
            "binance_api_key": env.get("BINANCE_API_KEY", ""),
            "binance_api_secret": env.get("BINANCE_API_SECRET", ""),
            "telegram_bot_token": env.get("TELEGRAM_BOT_TOKEN", ""),
            "telegram_chat_id": env.get("TELEGRAM_CHAT_ID", ""),
        }

    raw["secrets"] = secrets_data

    # Validate and freeze
    config = AppConfig(**raw)

    logger.info(
        "Config loaded | hash=%s | mode=%s",
        config.config_hash(),
        config.execution.mode,
    )

    return config
