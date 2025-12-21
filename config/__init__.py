"""
Configuration module for the trading system.
Provides centralized access to all configuration parameters.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from exceptions import ConfigurationError


class Config:
    """
    Configuration manager that loads settings from YAML files.

    Usage:
        config = Config()                    # Load default config
        config = Config("config/prod.yaml")  # Load specific config

        pair = config.get("data.pair")
        k = config.get("similarity.k", default=100)
    """

    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: Optional[str] = None):
        """Singleton pattern - only one config instance."""
        if cls._instance is None or config_path is not None:
            cls._instance = super().__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    def _load(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            # Default config path relative to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError("config_file", f"Invalid YAML syntax: {e}")

        if self._config is None:
            raise ConfigurationError("config_file", "Config file is empty")

        # Override database URL from environment if set
        env_db_url = os.getenv("DATABASE_URL")
        if env_db_url:
            if "data" not in self._config:
                self._config["data"] = {}
            self._config["data"]["database_url"] = env_db_url

    def validate(self) -> List[str]:
        """
        Validate configuration values.

        Returns:
            List of validation errors (empty if valid).

        Raises:
            ConfigurationError: If critical validation fails.
        """
        errors = []

        # Required fields
        required_fields = [
            ("data.pair", str),
            ("data.timeframe", str),
            ("data.database_url", str),
        ]

        for field, expected_type in required_fields:
            value = self.get(field)
            if value is None:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(value, expected_type):
                errors.append(f"Invalid type for {field}: expected {expected_type.__name__}")

        # Validate database URL format
        db_url = self.get("data.database_url")
        if db_url and not self._is_valid_db_url(db_url):
            errors.append(f"Invalid database URL format: {db_url}")

        # Validate date formats
        for date_field in ["data.start_date", "data.end_date"]:
            date_val = self.get(date_field)
            if date_val and not self._is_valid_date(date_val):
                errors.append(f"Invalid date format for {date_field}: {date_val} (expected YYYY-MM-DD)")

        # Validate numeric ranges
        range_validations = [
            ("decision.capital", 0, None, "must be positive"),
            ("decision.risk_per_trade", 0, 1, "must be between 0 and 1"),
            ("decision.max_leverage", 0, None, "must be positive"),
            ("similarity.k", 1, 10000, "must be between 1 and 10000"),
            ("normalization.window", 1, None, "must be positive"),
            ("regime.high_vol_threshold", 0, 1, "must be between 0 and 1"),
            ("regime.low_vol_threshold", 0, 1, "must be between 0 and 1"),
        ]

        for field, min_val, max_val, msg in range_validations:
            value = self.get(field)
            if value is not None:
                if min_val is not None and value <= min_val:
                    errors.append(f"{field}: {msg} (got {value})")
                if max_val is not None and value >= max_val:
                    errors.append(f"{field}: {msg} (got {value})")

        if errors:
            error_list = "\n  - ".join(errors)
            raise ConfigurationError(
                "validation",
                f"Configuration validation failed:\n  - {error_list}"
            )

        return errors

    @staticmethod
    def _is_valid_db_url(url: str) -> bool:
        """Check if database URL has valid format."""
        # Basic PostgreSQL URL pattern
        pattern = r'^postgresql(\+\w+)?://[\w\-\.]+.*'
        return bool(re.match(pattern, url, re.IGNORECASE))

    @staticmethod
    def _is_valid_date(date_str: str) -> bool:
        """Check if date string is valid ISO format."""
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value using dot notation.

        Args:
            key: Dot-separated path (e.g., "data.pair", "similarity.k")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire config section as a dictionary."""
        return self._config.get(section, {})

    @property
    def data(self) -> Dict[str, Any]:
        """Quick access to data section."""
        return self._config.get("data", {})

    @property
    def features(self) -> Dict[str, Any]:
        """Quick access to features section."""
        return self._config.get("features", {})

    @property
    def regime(self) -> Dict[str, Any]:
        """Quick access to regime section."""
        return self._config.get("regime", {})

    @property
    def similarity(self) -> Dict[str, Any]:
        """Quick access to similarity section."""
        return self._config.get("similarity", {})

    @property
    def decision(self) -> Dict[str, Any]:
        """Quick access to decision section."""
        return self._config.get("decision", {})

    @property
    def paths(self) -> Dict[str, Any]:
        """Quick access to paths section."""
        return self._config.get("paths", {})

    def get_output_path(self, data_type: str, pair: str, timeframe: str) -> Path:
        """
        Get the full output path for a data file.

        Args:
            data_type: One of 'state_vectors', 'regimes', 'outcomes'
            pair: Trading pair (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '1m')

        Returns:
            Full path to the parquet file
        """
        base_dir = Path(self.get("paths.data_dir", "data"))
        subdir = self.get(f"paths.{data_type}_dir", data_type)
        filename = f"{pair}_{timeframe}_{data_type.rstrip('s')}.parquet"

        return base_dir / subdir / filename

    def __repr__(self) -> str:
        return f"Config(sections={list(self._config.keys())})"


# Global config instance for easy importing
def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global config instance."""
    return Config(config_path)
