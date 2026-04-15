"""Signal quality scorer based on validated conditions from L1-EXP-004.

Conditions found on TRAIN (2020-2023), validated on OOS (2024-2025).
Only conditions that are bad/strong in BOTH periods are included.
"""
from dataclasses import dataclass, field


@dataclass
class SignalQuality:
    """Quality assessment for a trade signal."""
    score: float       # 0.0 (worst) to 1.0 (best)
    tier: str          # "STRONG", "NORMAL", "WEAK"
    reasons: list = field(default_factory=list)


# === VALIDATED BAD CONDITIONS (from EXP-004) ===
# These were bad in TRAIN AND confirmed bad in OOS.

def is_monday_long(trade: dict) -> bool:
    """Monday LONG: 12% win on OOS, -18.2 avg bps."""
    return trade.get('entry_dow') == 0 and trade.get('direction') == 'LONG'


def is_low_atr(trade: dict, threshold: float = 20) -> bool:
    """Low ATR (<20 percentile): weak in both periods."""
    return trade.get('atr_pctl', 50) < threshold


def is_low_ema(trade: dict, threshold: float = 0.3) -> bool:
    """Low EMA separation (<0.3%): weak in both periods."""
    return trade.get('ema_sep', 0.5) < threshold


def is_v12_long_monday(trade: dict) -> bool:
    """V12_LONG + Monday: 13% win on OOS, -18.5 avg bps."""
    return (trade.get('signal_type') == 'V12_LONG' and
            trade.get('entry_dow') == 0)


# === VALIDATED STRONG CONDITIONS (from EXP-004) ===

def is_high_atr(trade: dict, threshold: float = 70) -> bool:
    """High ATR (>70 percentile): strong in both periods."""
    return trade.get('atr_pctl', 50) > threshold


def is_high_ema(trade: dict, threshold: float = 1.0) -> bool:
    """High EMA separation (>1.0%): strong in both periods."""
    return trade.get('ema_sep', 0.5) > threshold


# === DEFAULT CONDITION SETS ===

DEFAULT_BAD_CONDITIONS = [
    {'fn': is_monday_long, 'label': 'monday_long', 'penalty': 1.0},
    {'fn': is_low_atr, 'label': 'low_atr', 'penalty': 0.5},
    {'fn': is_low_ema, 'label': 'low_ema', 'penalty': 0.5},
    {'fn': is_v12_long_monday, 'label': 'v12_long_monday', 'penalty': 0.5},
]

DEFAULT_STRONG_CONDITIONS = [
    {'fn': is_high_atr, 'label': 'high_atr', 'bonus': 0.5},
    {'fn': is_high_ema, 'label': 'high_ema', 'bonus': 0.5},
]


def score_signal(
    trade: dict,
    bad_conditions: list = None,
    strong_conditions: list = None,
) -> SignalQuality:
    """Score a signal based on validated conditions.

    Args:
        trade: Enriched trade dict with market conditions.
        bad_conditions: List of {fn, label, penalty} dicts.
        strong_conditions: List of {fn, label, bonus} dicts.

    Returns:
        SignalQuality with score, tier, and reasons.
    """
    if bad_conditions is None:
        bad_conditions = DEFAULT_BAD_CONDITIONS
    if strong_conditions is None:
        strong_conditions = DEFAULT_STRONG_CONDITIONS

    score = 0.5  # Start neutral
    reasons = []

    # Check bad conditions
    for cond in bad_conditions:
        if cond['fn'](trade):
            score -= cond['penalty']
            reasons.append(cond['label'])

    # Check strong conditions
    for cond in strong_conditions:
        if cond['fn'](trade):
            score += cond['bonus']
            reasons.append(f"+{cond['label']}")

    # Clamp score
    score = max(0.0, min(1.0, score))

    # Determine tier
    if score < 0.3:
        tier = "WEAK"
    elif score > 0.7:
        tier = "STRONG"
    else:
        tier = "NORMAL"

    return SignalQuality(score=score, tier=tier, reasons=reasons)
