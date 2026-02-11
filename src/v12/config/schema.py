"""Pydantic models for V1.3 configuration. All models are frozen (immutable after load)."""

import hashlib
import json
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class StrategyConfig(BaseModel):
    """RSI signal generation parameters. Source: EXP-001 (V1 original)."""
    model_config = ConfigDict(frozen=True)

    rsi_period: int = Field(ge=2, le=50)
    rsi_oversold: float = Field(ge=1, le=50)
    rsi_overbought: float = Field(ge=50, le=99)
    sma_period: int = Field(ge=50, le=500)


class LongFilters(BaseModel):
    """LONG-side entry filters. Source: EXP-006 (WHEN analysis applied)."""
    model_config = ConfigDict(frozen=True)

    atr_period: int = Field(ge=5, le=50)
    atr_rolling_window: int = Field(ge=50, le=500)
    atr_percentile_min: float = Field(ge=0, le=100)
    ema_short: int = Field(ge=5, le=100)
    ema_long: int = Field(ge=50, le=500)
    ema_separation_min: float = Field(ge=0, le=10)


class ShortFilters(BaseModel):
    """SHORT-side entry filters. Source: EXP-007 (all filters REJECTED)."""
    model_config = ConfigDict(frozen=True)

    # Explicitly empty — SHORT is robust in all conditions
    # Do NOT add filters here without running EXP-007 again
    enabled: bool = False


class BearLongFilters(BaseModel):
    """BEAR_LONG entry filters. Source: EXP-013 (counter-trend LONG in bear)."""
    model_config = ConfigDict(frozen=True)

    rsi_threshold: float = Field(ge=1, le=20)
    ema_separation_min: float = Field(ge=0, le=10)


class BullShortFilters(BaseModel):
    """BULL_SHORT entry filters. Source: EXP-013 (counter-trend SHORT in bull)."""
    model_config = ConfigDict(frozen=True)

    rsi_threshold: float = Field(ge=80, le=99)
    atr_percentile_min: float = Field(ge=0, le=100)
    ema_separation_min: float = Field(ge=0, le=10)


class ExitConfig(BaseModel):
    """Exit mechanics. Source: EXP-001 (V1), EXP-009 (time exit validated)."""
    model_config = ConfigDict(frozen=True)

    long_trailing_stop_bps: float = Field(ge=5, le=100)
    short_trailing_stop_bps: float = Field(ge=5, le=100)
    max_bars: int = Field(ge=1, le=50)
    tighten_after_bar: int = Field(ge=1, le=50)          # tighten trailing stop after this bar
    tight_trailing_stop_bps: float = Field(ge=1, le=100)  # tighter stop value


class ReentryConfig(BaseModel):
    """Re-entry after trailing stop. Source: EXP-012 (Config A+RE)."""
    model_config = ConfigDict(frozen=True)

    enabled: bool
    max_reentries: int = Field(ge=0, le=5)
    cooldown_bars: int = Field(ge=0, le=20)


class ExecutionConfig(BaseModel):
    """Execution parameters for live/paper trading."""
    model_config = ConfigDict(frozen=True)

    mode: str = Field(pattern=r"^(paper|live)$")
    leverage: int = Field(ge=1, le=125)
    position_size_usd: float = Field(ge=1)


class SecretsConfig(BaseModel):
    """Secrets loaded from .env — never committed to git."""
    model_config = ConfigDict(frozen=True)

    binance_api_key: str = ""
    binance_api_secret: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""


class AppConfig(BaseModel):
    """Root configuration combining all sub-configs."""
    model_config = ConfigDict(frozen=True)

    strategy: StrategyConfig
    long_filters: LongFilters
    short_filters: ShortFilters
    bear_long_filters: BearLongFilters
    bull_short_filters: BullShortFilters
    exit: ExitConfig
    reentry: ReentryConfig
    execution: ExecutionConfig
    secrets: SecretsConfig = SecretsConfig()

    def config_hash(self) -> str:
        """Deterministic hash of all non-secret parameters for reproducibility."""
        data = self.model_dump(exclude={"secrets", "execution"})
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
