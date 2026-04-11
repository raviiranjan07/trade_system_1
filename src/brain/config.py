"""Brain Pipeline — Config loader

Loads YAML config with inheritance support.
Experiment configs inherit from base and override specific settings.
"""

import yaml
from pathlib import Path
from copy import deepcopy


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(config_path: str) -> dict:
    """Load config with inheritance.

    If config has 'inherit' key, load base config first then override.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Handle inheritance
    if "inherit" in config:
        base_path = config_path.parent / config.pop("inherit")
        base_config = load_config(str(base_path))
        config = deep_merge(base_config, config)

    return config


def get_enabled_features(config: dict) -> list:
    """Get list of enabled feature names from config."""
    features = []
    feat_cfg = config.get("features", {})

    # Price action
    if feat_cfg.get("price_action", {}).get("enabled", False):
        pa = feat_cfg["price_action"]
        if pa.get("roc"):
            features.append("roc")
        if pa.get("rsi7"):
            features.append("rsi7")
        if pa.get("range_position"):
            features.append("range_position")

    # Market structure (S/R)
    if feat_cfg.get("market_structure", {}).get("enabled", False):
        features.extend([
            "support_range_low", "support_range_high",
            "resistance_range_low", "resistance_range_high",
            "zone_width",
            "support_retest", "resistance_retest",
            "distance_to_support", "distance_to_resistance",
            "recovery_up", "recovery_down",
            "window_low", "window_high",
        ])

    # Volatility
    if feat_cfg.get("volatility", {}).get("enabled", False):
        vol = feat_cfg["volatility"]
        if vol.get("atr_pct"):
            features.append("atr_pct")
        if vol.get("atr_percentile"):
            features.append("atr_percentile")
        if vol.get("ema_separation"):
            features.append("ema_separation")

    # Bar characteristics
    if feat_cfg.get("bar_characteristics", {}).get("enabled", False):
        bc = feat_cfg["bar_characteristics"]
        if bc.get("range_bps"):
            features.append("range_bps")
        if bc.get("body_bps"):
            features.append("body_bps")
        if bc.get("dist_from_high20_pct"):
            features.append("dist_from_high20_pct")

    # Activity
    if feat_cfg.get("activity", {}).get("enabled", False):
        act = feat_cfg["activity"]
        if act.get("volume_ratio"):
            features.append("volume_ratio")

    return features


def validate_config(config: dict) -> list:
    """Validate config and return list of warnings."""
    warnings = []

    # Check required sections
    for section in ["ingestion", "features", "labels", "dataset", "training"]:
        if section not in config:
            warnings.append(f"Missing required section: {section}")

    # Check date ranges don't overlap
    split = config.get("dataset", {}).get("split", {})
    if "train" in split and "val" in split and "test" in split:
        train_end = split["train"][1]
        val_start = split["val"][0]
        val_end = split["val"][1]
        test_start = split["test"][0]
        if train_end >= val_start:
            warnings.append(f"Train end ({train_end}) overlaps with val start ({val_start})")
        if val_end >= test_start:
            warnings.append(f"Val end ({val_end}) overlaps with test start ({test_start})")

    # Check at least one feature category is enabled
    feat_cfg = config.get("features", {})
    any_enabled = any(
        feat_cfg.get(cat, {}).get("enabled", False)
        for cat in ["price_action", "market_structure", "volatility", "bar_characteristics", "activity"]
    )
    if not any_enabled:
        warnings.append("No feature categories enabled")

    return warnings
