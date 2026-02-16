"""L2-002: Signal Expansion Framework

Complete framework: state vectors -> regimes -> gate + signal -> trade simulation -> metrics.

Usage:
    PYTHONPATH=src python experiments/layer2_regime/L2-002/framework.py

Architecture:
    1. Compute STATE VECTOR (16 features on each bar)
    2. Derive REGIME from magnitude features
    3. Check DIRECTION SIGNAL within regime
    4. ENTRY when regime gate passes + signal fires
    5. EXIT via V1.3.2 mechanics (trailing stop, tighten, time exit)
    6. REPORT standardized metrics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "L2-001"))

import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from v12.config.constants import FEES_BPS
from L2_001_feature_revalidation import compute_all_features, load_feature_config

FRAMEWORK_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
OUT_DIR = Path("experiments/layer2_regime/L2-002")

TRAIN_START, TRAIN_END = "2020-01-01", "2023-12-31"
OOS_START, OOS_END = "2024-01-01", "2025-12-31"


# =============================================================================
# 1. STATE VECTOR — computed on every bar
# =============================================================================

# Magnitude features (Gate — ranked by L2-001b)
MAGNITUDE_FEATURES = [
    "atr_pct",              # Rank 1: 4.01x (TOP)
    "range_bps",            # Rank 2: 3.37x (TOP)
    "dist_from_high20_pct", # Rank 3: 2.24x (MID)
    "ema_separation",       # Rank 4: 2.21x (MID)
    "body_bps",             # Rank 5: 2.16x (MID)
    "atr_percentile",       # Rank 6: 1.70x (MID)
    "hour_utc",             # Rank 7: 1.22x (WEAK)
    "volume_ratio",         # Rank 8: 1.20x (WEAK)
]

# Direction features (Signal — ranked by L2-001b, deduplicated)
DIRECTION_FEATURES = [
    "ema9_dist_pct",   # Rank 1: -8.4pp (represents 9 MA features)
    "rsi7",            # Rank 2: -8.4pp (represents rsi, rsi21, bb_position)
    "roc5",            # Rank 3: -7.8pp (unique)
    "range_position",  # Rank 4: -7.6pp (unique)
    "momentum5",       # Rank 5: -6.7pp (unique)
    "roc10",           # Rank 6: -6.4pp (unique)
    "momentum10",      # Rank 7: -6.1pp (unique)
    "roc20",           # Rank 8: -5.1pp (unique)
]


# =============================================================================
# 2. REGIME — derived from magnitude features
# =============================================================================

@dataclass
class RegimeConfig:
    """Define regime thresholds. Derived from magnitude feature quartiles."""
    # ATR threshold (percentile-based: computed from data)
    atr_pct_threshold: float = 0.0       # set from data
    # EMA separation threshold
    ema_sep_threshold: float = 0.0       # set from data
    # Labels
    name: str = ""


class RegimeClassifier:
    """Classifies each bar into a regime based on magnitude features.

    Reads feature_x, feature_y, and split_method from centralized config.yaml.
    """

    def __init__(self, df: pd.DataFrame, cfg: dict = None):
        """Compute thresholds from data using config-driven parameters.

        Args:
            df: TRAIN dataframe to compute thresholds from
            cfg: regime config dict with keys: feature_x, feature_y, split_method
                 If None, loads from centralized config.yaml
        """
        if cfg is None:
            with open(FRAMEWORK_CONFIG_PATH) as f:
                full_cfg = yaml.safe_load(f)
            cfg = full_cfg["regime"]

        self.feature_x = cfg["feature_x"]
        self.feature_y = cfg["feature_y"]
        self.split_method = cfg["split_method"]

        # Compute split thresholds based on method
        self.x_threshold = self._compute_threshold(df[self.feature_x], self.split_method)
        self.y_threshold = self._compute_threshold(df[self.feature_y], self.split_method)

        # Also compute quartile thresholds for flexible gating
        self.thresholds = {}
        for feat in MAGNITUDE_FEATURES:
            if feat in df.columns:
                vals = df[feat].dropna()
                self.thresholds[feat] = {
                    "q25": vals.quantile(0.25),
                    "q50": vals.quantile(0.50),
                    "q75": vals.quantile(0.75),
                    "mean": vals.mean(),
                    "std": vals.std(),
                }

    @staticmethod
    def _compute_threshold(series: pd.Series, method: str) -> float:
        """Compute split threshold from data using specified method."""
        vals = series.dropna()
        if method == "median":
            return vals.median()
        elif method == "q25":
            return vals.quantile(0.25)
        elif method == "q75":
            return vals.quantile(0.75)
        else:
            # Fixed numeric value
            return float(method)

    def classify(self, df: pd.DataFrame) -> pd.Series:
        """Assign regime label to each bar.

        4 regimes based on feature_x × feature_y:
            ACTIVE_TREND: high X + high Y
            ACTIVE_FLAT:  high X + low Y
            QUIET_TREND:  low X  + high Y
            QUIET_FLAT:   low X  + low Y
        """
        high_x = df[self.feature_x] >= self.x_threshold
        high_y = df[self.feature_y] >= self.y_threshold

        regime = pd.Series("QUIET_FLAT", index=df.index)
        regime[high_x & high_y] = "ACTIVE_TREND"
        regime[high_x & ~high_y] = "ACTIVE_FLAT"
        regime[~high_x & high_y] = "QUIET_TREND"
        regime[~high_x & ~high_y] = "QUIET_FLAT"

        return regime

    def print_thresholds(self):
        """Print computed thresholds for transparency."""
        print(f"\n--- Regime Thresholds (from TRAIN data) ---")
        print(f"  X axis: {self.feature_x:<20} split={self.split_method}  threshold={self.x_threshold:.4f}")
        print(f"  Y axis: {self.feature_y:<20} split={self.split_method}  threshold={self.y_threshold:.4f}")
        print(f"\n--- Feature Quartiles ---")
        for feat, t in self.thresholds.items():
            print(f"  {feat:<22} Q25={t['q25']:.4f}  Q50={t['q50']:.4f}  Q75={t['q75']:.4f}")


# =============================================================================
# 3. GATE — magnitude-based entry filter
# =============================================================================

@dataclass
class GateConfig:
    """Define which magnitude features to use as gate and their thresholds.

    Each gate is: feature >= threshold (percentile-based).
    Multiple gates = AND logic (all must pass).
    """
    gates: List[Tuple[str, str, float]] = field(default_factory=list)
    # Each tuple: (feature_name, operator, threshold)
    # operator: ">=", "<=", ">", "<", "in_regime"
    # Example: ("atr_pct", ">=", "q50") means atr_pct >= median

    name: str = "unnamed"
    gate_type: str = "BASELINE"  # Category tag: BASELINE, REGIME, MARKET, MARKET+ENTRY, etc.


def apply_gate(df: pd.DataFrame, gate_config: GateConfig,
               classifier: RegimeClassifier) -> pd.Series:
    """Returns boolean mask: True = gate passes on this bar."""
    mask = pd.Series(True, index=df.index)

    for feat, op, threshold in gate_config.gates:
        if feat == "regime":
            # Special: filter by regime name
            regimes = classifier.classify(df)
            if op == "==":
                mask &= (regimes == threshold)
            elif op == "in":
                mask &= regimes.isin(threshold)
        else:
            # Resolve threshold from TRAIN-computed quartiles (no data leak)
            if isinstance(threshold, str) and threshold.startswith("q"):
                val = classifier.thresholds[feat][threshold]
            else:
                val = float(threshold)

            if op == ">=":
                mask &= (df[feat] >= val)
            elif op == "<=":
                mask &= (df[feat] <= val)
            elif op == ">":
                mask &= (df[feat] > val)
            elif op == "<":
                mask &= (df[feat] < val)

    return mask


def generate_gates(gate_cfg: dict) -> List[GateConfig]:
    """Auto-generate all feature configurations across 3 gate categories.

    Gate categories (by purpose):
    - MARKET: "Is the environment tradeable?"
    - ENTRY: "Is THIS bar strong enough?"
    - STRUCTURE: "Is price positioned well?"

    Creates configurations:
    - NO_GATE (baseline control group)
    - Regime configs (pre-built MARKET configurations), tagged as MARKET
    - All 1, 2, 3-feature combos from categories, tagged with gate type
      e.g. MARKET, MARKET+ENTRY, MARKET+ENTRY+STRUCTURE
    """
    from itertools import combinations

    gates = []

    # 1. Baseline — always included
    gates.append(GateConfig(name="NO_GATE", gates=[], gate_type="BASELINE"))

    # 2. Regime configs — pre-built MARKET configurations (ATR × EMA sep)
    regime_cfg = gate_cfg.get("regimes", {})
    for regime_name in regime_cfg.get("individual", []):
        gates.append(GateConfig(
            name=regime_name,
            gates=[("regime", "==", regime_name)],
            gate_type="MARKET",
        ))
    for combo in regime_cfg.get("combos", []):
        gates.append(GateConfig(
            name=combo["name"],
            gates=[("regime", "in", combo["regimes"])],
            gate_type="MARKET",
        ))

    # 3. Feature gates from categories
    #    Expand each category's features into individual conditions with category tag
    individual_conditions = []
    categories_cfg = gate_cfg.get("categories", {})

    for cat_name, cat_cfg in categories_cfg.items():
        for feat_cfg in cat_cfg.get("features", []):
            feature = feat_cfg["feature"]
            operator = feat_cfg["operator"]
            for threshold in feat_cfg["thresholds"]:
                short_name = feature.upper().replace("_PCT", "").replace("_BPS", "")
                short_name = short_name.replace("DIST_FROM_HIGH20", "DHIGH20")
                short_name = short_name.replace("EMA_SEPARATION", "EMA_SEP")
                short_name = short_name.replace("ATR_PERCENTILE", "ATR_PCTL")
                short_name = short_name.replace("VOLUME_RATIO", "VOL_RATIO")
                short_name = short_name.replace("HOUR_UTC", "HOUR")
                label = f"{short_name}_{threshold}"
                individual_conditions.append({
                    "label": label,
                    "category": cat_name,
                    "condition": (feature, operator, threshold),
                })

    max_cond = gate_cfg.get("max_conditions", 3)

    # Generate all combo sizes: 1, 2, ..., max_conditions
    for size in range(1, max_cond + 1):
        for combo in combinations(individual_conditions, size):
            # Skip combos that use the same feature twice (different thresholds)
            features_used = [c["condition"][0] for c in combo]
            if len(features_used) != len(set(features_used)):
                continue

            name = "+".join(c["label"] for c in combo)
            conditions = [c["condition"] for c in combo]

            # Build gate_type from categories
            cat_list = sorted(set(c["category"] for c in combo))
            gate_type = "+".join(cat_list)

            gates.append(GateConfig(name=name, gates=conditions, gate_type=gate_type))

    # Count by type
    type_counts = {}
    for g in gates:
        type_counts[g.gate_type] = type_counts.get(g.gate_type, 0) + 1

    print(f"\n--- Auto-generated {len(gates)} gate configurations ---")
    print(f"  Gate categories: BASELINE, MARKET, ENTRY, STRUCTURE")
    print(f"  Feature configurations tested across categories:")
    for gt, count in sorted(type_counts.items()):
        print(f"    {gt}: {count}")

    return gates


# =============================================================================
# 4. SIGNAL GATE — direction-based entry trigger ("Which direction?")
# =============================================================================

@dataclass
class SignalConfig:
    """Define direction signal: one or more feature conditions (AND logic).

    Each condition: (feature, long_threshold, short_threshold)
    - LONG fires when ALL features <= their long_threshold
    - SHORT fires when ALL features >= their short_threshold
    """
    conditions: List[Tuple[str, float, float]] = field(default_factory=list)
    # Each tuple: (feature_name, long_threshold, short_threshold)
    name: str = "unnamed"
    signal_type: str = ""  # Category tag: MOMENTUM, REVERSION, RANGE, MOMENTUM+REVERSION, etc.


def generate_signals(signal_cfg: dict) -> List[SignalConfig]:
    """Auto-generate all signal configurations (1, 2, 3-feature combos).

    Signal gate categories (by purpose):
    - MOMENTUM: "Is the move exhausted?" (rsi7, roc5)
    - REVERSION: "Has price deviated too far?" (ema9_dist_pct)
    - RANGE: "Is price at an extreme?" (range_position)

    Combo = AND logic: ALL conditions must be true for signal to fire.
    """
    from itertools import combinations

    # Expand each category's features into individual conditions with category tag
    individual_conditions = []
    categories_cfg = signal_cfg.get("categories", {})

    for cat_name, cat_cfg in categories_cfg.items():
        for feat_cfg in cat_cfg.get("features", []):
            feature = feat_cfg["feature"]
            long_thresholds = feat_cfg["long_thresholds"]
            short_thresholds = feat_cfg["short_thresholds"]

            for long_t, short_t in zip(long_thresholds, short_thresholds):
                short_name = feature.upper().replace("_DIST_PCT", "D").replace("_PCT", "")
                short_name = short_name.replace("RANGE_POSITION", "RPOS")
                label = f"{short_name}_{long_t}_{short_t}"

                individual_conditions.append({
                    "label": label,
                    "category": cat_name,
                    "feature": feature,
                    "long_t": float(long_t),
                    "short_t": float(short_t),
                })

    max_cond = signal_cfg.get("max_conditions", 3)
    signals = []

    # Generate all combo sizes: 1, 2, ..., max_conditions
    for size in range(1, max_cond + 1):
        for combo in combinations(individual_conditions, size):
            # Skip combos that use the same feature twice (different thresholds)
            features_used = [c["feature"] for c in combo]
            if len(features_used) != len(set(features_used)):
                continue

            name = "+".join(c["label"] for c in combo)
            conditions = [(c["feature"], c["long_t"], c["short_t"]) for c in combo]

            # Build signal_type from categories
            cat_list = sorted(set(c["category"] for c in combo))
            signal_type = "+".join(cat_list)

            signals.append(SignalConfig(
                name=name,
                conditions=conditions,
                signal_type=signal_type,
            ))

    # Count by type
    type_counts = {}
    for s in signals:
        type_counts[s.signal_type] = type_counts.get(s.signal_type, 0) + 1

    print(f"\n--- Auto-generated {len(signals)} signal configurations ---")
    print(f"  Signal categories: MOMENTUM, REVERSION, RANGE")
    print(f"  Configurations per category:")
    for st, count in sorted(type_counts.items()):
        print(f"    {st}: {count}")

    return signals


def detect_signals(df: pd.DataFrame, signal_config: SignalConfig,
                   gate_mask: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Returns (long_signal, short_signal) boolean masks.

    Signal only fires where gate is True.
    Multi-condition signals use AND logic: ALL conditions must pass.
    """
    long_signal = gate_mask.copy()
    short_signal = gate_mask.copy()

    for feature, long_t, short_t in signal_config.conditions:
        feat_vals = df[feature]
        long_signal = long_signal & (feat_vals <= long_t)
        short_signal = short_signal & (feat_vals >= short_t)

    return long_signal, short_signal


# =============================================================================
# 5. EXIT STRATEGY — "How do we manage the trade after entry?"
# =============================================================================

@dataclass
class ExitConfig:
    """Exit strategy parameters. One config = one precomputation."""
    trailing_stop_bps: float = 20.0     # Same for LONG and SHORT
    max_bars: int = 10
    tighten: bool = True
    tighten_after_bar: int = 5
    tight_stop_bps: float = 8.0
    name: str = "unnamed"
    exit_type: str = ""  # Category tag: PROTECTION+TIMING+ADAPTATION


def generate_exits(exit_cfg: dict) -> List[ExitConfig]:
    """Auto-generate all exit configurations from category-based config.

    Exit strategy categories (by purpose):
    - PROTECTION: "How much loss can we tolerate?" (trailing stop size)
    - TIMING: "How long do we hold?" (max bars)
    - ADAPTATION: "Does exit change mid-trade?" (tighten or not)

    Grid: pick one from each category, combine into full exit config.
    """
    from itertools import product

    categories = exit_cfg.get("categories", {})
    protection = categories.get("PROTECTION", {}).get("trailing_stop_bps", [20])
    timing = categories.get("TIMING", {}).get("max_bars", [10])
    adaptation = categories.get("ADAPTATION", {}).get("configs", [{"tighten": False}])

    exits = []
    for ts_bps, max_b, adapt in product(protection, timing, adaptation):
        tighten = adapt.get("tighten", False)
        if tighten:
            t_after = adapt["tighten_after_bar"]
            t_bps = adapt["tight_stop_bps"]
            # Skip if tighten_after_bar >= max_bars (makes no sense)
            if t_after >= max_b:
                continue
            name = f"TS{ts_bps}_BAR{max_b}_T{t_after}_{t_bps}"
            exit_type = "PROTECTION+TIMING+ADAPTATION"
        else:
            t_after = 999  # effectively disabled
            t_bps = 0
            name = f"TS{ts_bps}_BAR{max_b}_NOTIGHT"
            exit_type = "PROTECTION+TIMING"

        exits.append(ExitConfig(
            trailing_stop_bps=float(ts_bps),
            max_bars=int(max_b),
            tighten=tighten,
            tighten_after_bar=int(t_after),
            tight_stop_bps=float(t_bps),
            name=name,
            exit_type=exit_type,
        ))

    # Count by type
    type_counts = {}
    for e in exits:
        type_counts[e.exit_type] = type_counts.get(e.exit_type, 0) + 1

    print(f"\n--- Auto-generated {len(exits)} exit configurations ---")
    print(f"  Exit categories: PROTECTION, TIMING, ADAPTATION")
    print(f"  Configurations per type:")
    for et, count in sorted(type_counts.items()):
        print(f"    {et}: {count}")

    return exits



# =============================================================================
# 6. PRE-COMPUTE TRADE RESULTS (optimization: compute once, lookup many)
# =============================================================================

def precompute_all_trades(df: pd.DataFrame, exit_config: ExitConfig):
    """Simulate LONG and SHORT trade from EVERY bar. Returns arrays indexed by bar.

    This is the key optimization: compute once, then for each gate+signal combo
    just filter which bars fired and look up their pre-computed results.
    """
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)

    # Pre-allocate result arrays
    long_net_bps = np.full(n, np.nan)
    long_mfe = np.full(n, np.nan)
    long_exit_bar = np.full(n, -1, dtype=int)
    long_exit_reason = np.empty(n, dtype=object)
    long_bars_held = np.full(n, 0, dtype=int)

    short_net_bps = np.full(n, np.nan)
    short_mfe = np.full(n, np.nan)
    short_exit_bar = np.full(n, -1, dtype=int)
    short_exit_reason = np.empty(n, dtype=object)
    short_bars_held = np.full(n, 0, dtype=int)

    for i in range(n - exit_config.max_bars - 2):
        entry_price = opens[i + 1]
        if entry_price <= 0 or np.isnan(entry_price):
            continue

        # Simulate LONG
        hp_l = 0.0
        for bar in range(1, exit_config.max_bars + 1):
            idx = i + 1 + bar
            if idx >= n:
                break
            h, l, c = highs[idx], lows[idx], closes[idx]
            bar_mfe_l = (h - entry_price) / entry_price * 10000
            bar_pnl_l = (c - entry_price) / entry_price * 10000
            hp_l = max(hp_l, bar_mfe_l)

            active_ts = exit_config.tight_stop_bps if (exit_config.tighten and bar > exit_config.tighten_after_bar) else exit_config.trailing_stop_bps
            dd = hp_l - bar_pnl_l
            if dd >= active_ts and hp_l > 0:
                gross = hp_l - active_ts
                long_net_bps[i] = gross - FEES_BPS
                long_mfe[i] = hp_l
                long_exit_bar[i] = idx
                long_exit_reason[i] = "TS"
                long_bars_held[i] = bar
                break
            if bar >= exit_config.max_bars:
                long_net_bps[i] = bar_pnl_l - FEES_BPS
                long_mfe[i] = hp_l
                long_exit_bar[i] = idx
                long_exit_reason[i] = "TIME"
                long_bars_held[i] = bar
                break

        # Simulate SHORT
        hp_s = 0.0
        for bar in range(1, exit_config.max_bars + 1):
            idx = i + 1 + bar
            if idx >= n:
                break
            h, l, c = highs[idx], lows[idx], closes[idx]
            bar_mfe_s = (entry_price - l) / entry_price * 10000
            bar_pnl_s = (entry_price - c) / entry_price * 10000
            hp_s = max(hp_s, bar_mfe_s)

            active_ts = exit_config.tight_stop_bps if (exit_config.tighten and bar > exit_config.tighten_after_bar) else exit_config.trailing_stop_bps
            dd = hp_s - bar_pnl_s
            if dd >= active_ts and hp_s > 0:
                gross = hp_s - active_ts
                short_net_bps[i] = gross - FEES_BPS
                short_mfe[i] = hp_s
                short_exit_bar[i] = idx
                short_exit_reason[i] = "TS"
                short_bars_held[i] = bar
                break
            if bar >= exit_config.max_bars:
                short_net_bps[i] = bar_pnl_s - FEES_BPS
                short_mfe[i] = hp_s
                short_exit_bar[i] = idx
                short_exit_reason[i] = "TIME"
                short_bars_held[i] = bar
                break

    return {
        "long_net_bps": long_net_bps,
        "long_mfe": long_mfe,
        "long_exit_bar": long_exit_bar,
        "long_exit_reason": long_exit_reason,
        "long_bars_held": long_bars_held,
        "short_net_bps": short_net_bps,
        "short_mfe": short_mfe,
        "short_exit_bar": short_exit_bar,
        "short_exit_reason": short_exit_reason,
        "short_bars_held": short_bars_held,
    }


# =============================================================================
# 7. BACKTEST ENGINE — fast lookup from pre-computed results
# =============================================================================

@dataclass
class BacktestResult:
    """Standardized output for every combination test."""
    config_name: str
    gate_name: str
    signal_name: str
    exit_name: str = ""
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_trades: int = 0
    n_long: int = 0
    n_short: int = 0
    win_rate: float = 0.0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    total_bps: float = 0.0
    long_bps: float = 0.0
    short_bps: float = 0.0
    profit_factor: float = 0.0
    long_pf: float = 0.0
    short_pf: float = 0.0
    avg_bps: float = 0.0
    max_dd_bps: float = 0.0


def compute_gate_decisions(df: pd.DataFrame, gate_config: GateConfig,
                           signal_config: SignalConfig, precomputed: dict,
                           classifier: RegimeClassifier) -> pd.DataFrame:
    """Compute what the gate blocked and what would have happened.

    For each signal bar, records:
    - gate_pass: True/False (did the gate let it through?)
    - direction: LONG/SHORT
    - net_bps: what would have happened (from precomputed)
    - would_win: was the blocked trade a winner?

    This lets us evaluate: is the gate filtering the RIGHT bars?
    """
    # Get gate mask
    gate_mask = apply_gate(df, gate_config, classifier)

    # Get ALL signals (without gate filtering) — what would fire if no gate
    all_long = pd.Series(True, index=df.index)
    all_short = pd.Series(True, index=df.index)
    for feature, long_t, short_t in signal_config.conditions:
        feat_vals = df[feature]
        all_long = all_long & (feat_vals <= long_t)
        all_short = all_short & (feat_vals >= short_t)

    rows = []

    # LONG signals
    for i in np.where(all_long.values)[0]:
        net = precomputed["long_net_bps"][i]
        if np.isnan(net):
            continue
        rows.append({
            "bar_idx": i,
            "timestamp": df.index[i],
            "direction": "LONG",
            "gate_pass": bool(gate_mask.iloc[i]),
            "net_bps": net,
            "mfe_bps": precomputed["long_mfe"][i],
            "exit_reason": precomputed["long_exit_reason"][i],
            "would_win": net > 0,
        })

    # SHORT signals
    for i in np.where(all_short.values)[0]:
        net = precomputed["short_net_bps"][i]
        if np.isnan(net):
            continue
        rows.append({
            "bar_idx": i,
            "timestamp": df.index[i],
            "direction": "SHORT",
            "gate_pass": bool(gate_mask.iloc[i]),
            "net_bps": net,
            "mfe_bps": precomputed["short_mfe"][i],
            "exit_reason": precomputed["short_exit_reason"][i],
            "would_win": net > 0,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def summarize_gate_decisions(decisions: pd.DataFrame, gate_name: str,
                             signal_name: str) -> dict:
    """Summarize gate decision quality: what did it block and was that correct?"""
    if decisions.empty:
        return {}

    passed = decisions[decisions["gate_pass"]]
    blocked = decisions[~decisions["gate_pass"]]

    summary = {
        "gate": gate_name,
        "signal": signal_name,
        "total_signals": len(decisions),
        "passed": len(passed),
        "blocked": len(blocked),
        "passed_winners": int(passed["would_win"].sum()) if len(passed) > 0 else 0,
        "passed_losers": int((~passed["would_win"]).sum()) if len(passed) > 0 else 0,
        "passed_bps": passed["net_bps"].sum() if len(passed) > 0 else 0.0,
        "blocked_winners": int(blocked["would_win"].sum()) if len(blocked) > 0 else 0,
        "blocked_losers": int((~blocked["would_win"]).sum()) if len(blocked) > 0 else 0,
        "blocked_bps": blocked["net_bps"].sum() if len(blocked) > 0 else 0.0,
    }

    # Gate accuracy: % of blocked trades that were actually losers (correctly blocked)
    if len(blocked) > 0:
        summary["block_accuracy"] = summary["blocked_losers"] / len(blocked) * 100
    else:
        summary["block_accuracy"] = 0.0

    # Net benefit: bps saved by blocking losers minus bps lost by blocking winners
    blocked_loser_bps = abs(blocked[~blocked["would_win"]]["net_bps"].sum()) if len(blocked) > 0 else 0
    blocked_winner_bps = blocked[blocked["would_win"]]["net_bps"].sum() if len(blocked) > 0 else 0
    summary["net_benefit_bps"] = blocked_loser_bps - blocked_winner_bps

    return summary


def run_backtest_fast(gate_mask_arr: np.ndarray,
                      long_signal_arr: np.ndarray, short_signal_arr: np.ndarray,
                      precomputed: dict,
                      gate_name: str, signal_name: str,
                      exit_name: str = "") -> BacktestResult:
    """Ultra-fast backtest: takes pre-computed boolean arrays, no DataFrame ops.

    Use this in the hot loop. Pre-compute gate_mask and signal masks ONCE each,
    then call this with boolean AND for every gate x signal combination.
    """
    # Combined masks: signal fires only where gate passes
    long_combined = gate_mask_arr & long_signal_arr
    short_combined = gate_mask_arr & short_signal_arr

    long_idx = np.where(long_combined)[0]
    short_idx = np.where(short_combined)[0]

    # Merge and sort
    all_signals = np.empty(len(long_idx) + len(short_idx), dtype=[('idx', 'i4'), ('dir', 'i1')])
    all_signals['idx'][:len(long_idx)] = long_idx
    all_signals['dir'][:len(long_idx)] = 1  # LONG
    all_signals['idx'][len(long_idx):] = short_idx
    all_signals['dir'][len(long_idx):] = 0  # SHORT
    all_signals.sort(order='idx')

    # Collect trades
    trade_nets = []
    trade_dirs = []
    in_trade_until = -1

    l_net = precomputed["long_net_bps"]
    l_exit = precomputed["long_exit_bar"]
    s_net = precomputed["short_net_bps"]
    s_exit = precomputed["short_exit_bar"]

    for j in range(len(all_signals)):
        bar_idx = int(all_signals['idx'][j])
        if bar_idx <= in_trade_until:
            continue
        is_long = all_signals['dir'][j] == 1
        net = l_net[bar_idx] if is_long else s_net[bar_idx]
        if np.isnan(net):
            continue
        exit_bar = l_exit[bar_idx] if is_long else s_exit[bar_idx]
        trade_nets.append(net)
        trade_dirs.append(1 if is_long else 0)
        in_trade_until = exit_bar

    # Build result
    if exit_name:
        config_name = f"{gate_name}__{signal_name}__{exit_name}"
    else:
        config_name = f"{gate_name}__{signal_name}"

    br = BacktestResult(
        config_name=config_name,
        gate_name=gate_name,
        signal_name=signal_name,
        exit_name=exit_name,
    )

    if not trade_nets:
        return br

    nets = np.array(trade_nets)
    dirs = np.array(trade_dirs)

    br.n_trades = len(nets)
    br.total_bps = nets.sum()
    br.avg_bps = nets.mean()
    br.win_rate = (nets > 0).mean() * 100

    winners_sum = nets[nets > 0].sum()
    losers_sum = abs(nets[nets <= 0].sum())
    br.profit_factor = winners_sum / losers_sum if losers_sum > 0 else float("inf")

    cumsum = nets.cumsum()
    running_max = np.maximum.accumulate(cumsum)
    br.max_dd_bps = (cumsum - running_max).min()

    long_mask = dirs == 1
    short_mask = dirs == 0
    br.n_long = int(long_mask.sum())
    br.n_short = int(short_mask.sum())

    if br.n_long > 0:
        l_nets = nets[long_mask]
        br.long_bps = l_nets.sum()
        br.long_win_rate = (l_nets > 0).mean() * 100
        lw = l_nets[l_nets > 0].sum()
        ll = abs(l_nets[l_nets <= 0].sum())
        br.long_pf = lw / ll if ll > 0 else float("inf")

    if br.n_short > 0:
        s_nets = nets[short_mask]
        br.short_bps = s_nets.sum()
        br.short_win_rate = (s_nets > 0).mean() * 100
        sw = s_nets[s_nets > 0].sum()
        sl = abs(s_nets[s_nets <= 0].sum())
        br.short_pf = sw / sl if sl > 0 else float("inf")

    return br


def run_backtest(df: pd.DataFrame, gate_config: GateConfig,
                 signal_config: SignalConfig, precomputed: dict,
                 classifier: RegimeClassifier,
                 exit_name: str = "") -> BacktestResult:
    """Fast backtest using pre-computed trade results."""

    n = len(df)

    # Step 1-2: Apply gate
    gate_mask = apply_gate(df, gate_config, classifier)

    # Step 3: Detect signals within gate
    long_signals, short_signals = detect_signals(df, signal_config, gate_mask)

    # Step 4: Collect trades (prevent overlapping)
    long_idx = np.where(long_signals.values)[0]
    short_idx = np.where(short_signals.values)[0]

    all_signals = [(i, "LONG") for i in long_idx] + [(i, "SHORT") for i in short_idx]
    all_signals.sort(key=lambda x: x[0])

    trades = []
    in_trade_until = -1

    for bar_idx, direction in all_signals:
        if bar_idx <= in_trade_until:
            continue

        if direction == "LONG":
            net = precomputed["long_net_bps"][bar_idx]
            mfe = precomputed["long_mfe"][bar_idx]
            exit_bar = precomputed["long_exit_bar"][bar_idx]
            exit_reason = precomputed["long_exit_reason"][bar_idx]
            bars_held = precomputed["long_bars_held"][bar_idx]
        else:
            net = precomputed["short_net_bps"][bar_idx]
            mfe = precomputed["short_mfe"][bar_idx]
            exit_bar = precomputed["short_exit_bar"][bar_idx]
            exit_reason = precomputed["short_exit_reason"][bar_idx]
            bars_held = precomputed["short_bars_held"][bar_idx]

        if np.isnan(net):
            continue

        trades.append({
            "entry_bar": bar_idx,
            "exit_bar": exit_bar,
            "bars_held": bars_held,
            "direction": direction,
            "net_bps": net,
            "mfe_bps": mfe,
            "exit_reason": exit_reason,
        })
        in_trade_until = exit_bar

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # Step 5: Compute metrics
    if exit_name:
        config_name = f"{gate_config.name}__{signal_config.name}__{exit_name}"
    else:
        config_name = f"{gate_config.name}__{signal_config.name}"
    br = BacktestResult(
        config_name=config_name,
        gate_name=gate_config.name,
        signal_name=signal_config.name,
        exit_name=exit_name,
        trades=trades_df,
    )

    if trades_df.empty:
        return br

    br.n_trades = len(trades_df)
    br.total_bps = trades_df["net_bps"].sum()
    br.avg_bps = trades_df["net_bps"].mean()
    br.win_rate = (trades_df["net_bps"] > 0).mean() * 100

    winners = trades_df[trades_df["net_bps"] > 0]["net_bps"].sum()
    losers = abs(trades_df[trades_df["net_bps"] <= 0]["net_bps"].sum())
    br.profit_factor = winners / losers if losers > 0 else float("inf")

    cumsum = trades_df["net_bps"].cumsum()
    running_max = cumsum.cummax()
    drawdown = cumsum - running_max
    br.max_dd_bps = drawdown.min()

    longs = trades_df[trades_df["direction"] == "LONG"]
    shorts = trades_df[trades_df["direction"] == "SHORT"]

    br.n_long = len(longs)
    br.n_short = len(shorts)

    if len(longs) > 0:
        br.long_bps = longs["net_bps"].sum()
        br.long_win_rate = (longs["net_bps"] > 0).mean() * 100
        lw = longs[longs["net_bps"] > 0]["net_bps"].sum()
        ll = abs(longs[longs["net_bps"] <= 0]["net_bps"].sum())
        br.long_pf = lw / ll if ll > 0 else float("inf")

    if len(shorts) > 0:
        br.short_bps = shorts["net_bps"].sum()
        br.short_win_rate = (shorts["net_bps"] > 0).mean() * 100
        sw = shorts[shorts["net_bps"] > 0]["net_bps"].sum()
        sl = abs(shorts[shorts["net_bps"] <= 0]["net_bps"].sum())
        br.short_pf = sw / sl if sl > 0 else float("inf")

    return br


# =============================================================================
# 7. REPORTING — standardized output
# =============================================================================

def print_result(br: BacktestResult, label: str = ""):
    """Print standardized metrics for one backtest result."""
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}{br.config_name}")
    print(f"  Trades: {br.n_trades} (L:{br.n_long} S:{br.n_short})")
    print(f"  Win:    {br.win_rate:.1f}% (L:{br.long_win_rate:.1f}% S:{br.short_win_rate:.1f}%)")
    print(f"  Total:  {br.total_bps:+.0f} bps (L:{br.long_bps:+.0f} S:{br.short_bps:+.0f})")
    print(f"  PF:     {br.profit_factor:.2f} (L:{br.long_pf:.2f} S:{br.short_pf:.2f})")
    print(f"  Avg:    {br.avg_bps:+.1f} bps/trade")
    print(f"  MaxDD:  {br.max_dd_bps:.0f} bps")


def print_comparison_table(results: List[BacktestResult], label: str = ""):
    """Print comparison table across multiple configs."""
    print(f"\n{'='*100}")
    print(f"COMPARISON: {label}")
    print(f"{'='*100}")
    print(f"{'Config':<40} {'Trades':>6} {'Win%':>6} {'Total':>8} {'PF':>6} "
          f"{'L':>4} {'L Win%':>7} {'L bps':>7} {'S':>4} {'S Win%':>7} {'S bps':>7} {'DD':>7}")
    print("-" * 100)

    for br in sorted(results, key=lambda x: x.total_bps, reverse=True):
        print(f"{br.config_name:<40} {br.n_trades:>6} {br.win_rate:>5.1f}% {br.total_bps:>+7.0f} "
              f"{br.profit_factor:>5.2f} "
              f"{br.n_long:>4} {br.long_win_rate:>6.1f}% {br.long_bps:>+6.0f} "
              f"{br.n_short:>4} {br.short_win_rate:>6.1f}% {br.short_bps:>+6.0f} "
              f"{br.max_dd_bps:>6.0f}")


def per_year_breakdown(trades_df: pd.DataFrame, df_full: pd.DataFrame, config_name: str):
    """Print per-year breakdown of trades."""
    if trades_df.empty:
        print(f"  No trades for {config_name}")
        return

    # Map entry_bar index to datetime
    trades_df = trades_df.copy()
    trades_df["date"] = df_full.index[trades_df["entry_bar"].values.astype(int)]
    trades_df["year"] = trades_df["date"].dt.year

    print(f"\n--- Per-Year: {config_name} ---")
    print(f"{'Year':>6} {'Trades':>7} {'Win%':>6} {'Total':>8} {'PF':>6} {'Avg':>7}")
    print("-" * 45)

    for year in sorted(trades_df["year"].unique()):
        yt = trades_df[trades_df["year"] == year]
        n = len(yt)
        wr = (yt["net_bps"] > 0).mean() * 100
        total = yt["net_bps"].sum()
        avg = yt["net_bps"].mean()
        w = yt[yt["net_bps"] > 0]["net_bps"].sum()
        l = abs(yt[yt["net_bps"] <= 0]["net_bps"].sum())
        pf = w / l if l > 0 else float("inf")
        print(f"{year:>6} {n:>7} {wr:>5.1f}% {total:>+7.0f} {pf:>5.2f} {avg:>+6.1f}")


# =============================================================================
# 8. REGIME PROFILING — understand the space before testing
# =============================================================================

def profile_regimes(df: pd.DataFrame, classifier: RegimeClassifier, label: str = ""):
    """Profile each regime: bar count, MFE distribution, V1.3.2 trade overlap."""
    regimes = classifier.classify(df)
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    print(f"\n{'='*80}")
    print(f"REGIME PROFILE ({label})")
    print(f"{'='*80}")

    print(f"\n{'Regime':<16} {'Bars':>7} {'%':>6} | {'Med MFE Long':>13} {'Med MFE Short':>14} {'Med Best':>9}")
    print("-" * 70)

    for regime_name in ["ACTIVE_TREND", "ACTIVE_FLAT", "QUIET_TREND", "QUIET_FLAT"]:
        mask = regimes == regime_name
        count = mask.sum()
        pct = count / len(df) * 100

        # Sample MFE for this regime
        indices = np.where(mask.values)[0]
        indices = indices[indices < n - 11]  # need 10 bars forward
        if len(indices) > 3000:
            rng = np.random.RandomState(42)
            indices = rng.choice(indices, 3000, replace=False)

        long_mfes = []
        short_mfes = []
        for i in indices:
            entry = opens[i + 1]
            if entry <= 0 or np.isnan(entry):
                continue
            lmfe = 0.0
            smfe = 0.0
            for j in range(1, 11):
                idx = i + 1 + j
                if idx >= n:
                    break
                lmfe = max(lmfe, (highs[idx] - entry) / entry * 10000)
                smfe = max(smfe, (entry - lows[idx]) / entry * 10000)
            long_mfes.append(lmfe)
            short_mfes.append(smfe)

        med_l = np.median(long_mfes) if long_mfes else 0
        med_s = np.median(short_mfes) if short_mfes else 0
        med_best = np.median([max(l, s) for l, s in zip(long_mfes, short_mfes)]) if long_mfes else 0

        print(f"{regime_name:<16} {count:>7} {pct:>5.1f}% | {med_l:>12.1f} {med_s:>13.1f} {med_best:>8.1f}")


# =============================================================================
# MAIN — Profile + First Combination Tests
# =============================================================================

def main():
    print("=" * 80)
    print("L2-002: Signal Expansion Framework")
    print("State Vectors -> Regimes -> Gate + Signal -> Trade -> Metrics")
    print("=" * 80)

    # Load centralized config
    with open(FRAMEWORK_CONFIG_PATH) as f:
        full_cfg = yaml.safe_load(f)
    data_cfg = full_cfg["data"]

    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(data_cfg["path"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    print(f"Total bars: {len(df)}")

    # Compute state vector (all features) using centralized config
    print("Computing state vectors (features on every bar)...")
    df = compute_all_features(df, cfg=full_cfg)

    train = df[data_cfg["train_start"]:data_cfg["train_end"]].copy()
    oos = df[data_cfg["oos_start"]:data_cfg["oos_end"]].copy()
    print(f"TRAIN: {len(train)} bars | OOS: {len(oos)} bars")

    # Build regime classifier from TRAIN (reads regime config)
    print("\nBuilding regime classifier from TRAIN data...")
    classifier = RegimeClassifier(train, cfg=full_cfg["regime"])
    classifier.print_thresholds()

    # Profile regimes
    profile_regimes(train, classifier, "TRAIN")
    profile_regimes(oos, classifier, "OOS")

    # =====================================================================
    # AUTO-GENERATE ALL CONFIGURATIONS
    # =====================================================================

    # Exit strategies (PROTECTION x TIMING x ADAPTATION grid)
    exits = generate_exits(full_cfg["exit_strategy"])

    # Magnitude gates (MARKET, ENTRY, STRUCTURE categories + combos)
    gates = generate_gates(full_cfg["magnitude_gate"])

    # Signal gates (MOMENTUM, REVERSION, RANGE categories + combos)
    signals = generate_signals(full_cfg["signal_gate"])

    total_combos = len(exits) * len(gates) * len(signals)
    print(f"\nTotal combinations: {len(exits)} exits x {len(gates)} gates x {len(signals)} signals = {total_combos:,}")

    # Build type lookups
    gate_type_map = {g.name: g.gate_type for g in gates}
    signal_type_map = {s.name: s.signal_type for s in signals}
    exit_type_map = {e.name: e.exit_type for e in exits}

    # =====================================================================
    # PHASE 1: TRAIN SCAN — precompute per exit, run all gate x signal
    # Memory-efficient: precompute one exit at a time, then discard
    # =====================================================================
    print(f"\n\n{'='*80}")
    print("PHASE 1: TRAIN SCAN")
    print("For each exit config: precompute trades, then test all gate x signal combos")
    print(f"{'='*80}")

    # Pre-compute gate masks and signal masks ONCE (huge speedup)
    print("\nPre-computing gate masks (1x per gate)...")
    gate_masks = {}
    for gate in gates:
        mask = apply_gate(train, gate, classifier)
        gate_masks[gate.name] = mask.values
    print(f"  {len(gate_masks)} gate masks cached")

    print("Pre-computing signal masks (1x per signal, unfiltered)...")
    signal_long_masks = {}
    signal_short_masks = {}
    for signal in signals:
        long_m = pd.Series(True, index=train.index)
        short_m = pd.Series(True, index=train.index)
        for feature, long_t, short_t in signal.conditions:
            feat_vals = train[feature]
            long_m = long_m & (feat_vals <= long_t)
            short_m = short_m & (feat_vals >= short_t)
        signal_long_masks[signal.name] = long_m.values
        signal_short_masks[signal.name] = short_m.values
    print(f"  {len(signal_long_masks)} signal masks cached")

    all_train_results = []  # List of (BacktestResult, exit_config_name)
    combo_count = 0

    for ei, exit_config in enumerate(exits):
        print(f"\n[Exit {ei+1}/{len(exits)}] {exit_config.name} — precomputing TRAIN...", end=" ", flush=True)
        precomputed_train = precompute_all_trades(train, exit_config)
        batch_count = 0

        for gate in gates:
            g_mask = gate_masks[gate.name]
            for signal in signals:
                br = run_backtest_fast(
                    g_mask, signal_long_masks[signal.name], signal_short_masks[signal.name],
                    precomputed_train, gate.name, signal.name, exit_name=exit_config.name,
                )
                all_train_results.append((br, exit_config.name))
                batch_count += 1

        combo_count += batch_count
        print(f"{batch_count} combos done (total: {combo_count:,}/{total_combos:,})")
        del precomputed_train  # free memory

    print(f"\nTRAIN scan complete: {len(all_train_results):,} combinations tested")

    # =====================================================================
    # PHASE 2: FILTER INTERESTING
    # =====================================================================
    interesting = [(br, en) for br, en in all_train_results if br.profit_factor > 1.5 and br.n_trades >= 50]
    print(f"\nInteresting configs (PF>1.5, trades>=50): {len(interesting):,}")

    if interesting:
        # Show top 50 by total_bps
        top_interesting = sorted(interesting, key=lambda x: x[0].total_bps, reverse=True)[:50]
        print_comparison_table([br for br, _ in top_interesting], "TOP 50 INTERESTING — TRAIN")

    # =====================================================================
    # PHASE 3: OOS VALIDATION — only for interesting configs
    # Group by exit config for efficient precomputation
    # =====================================================================
    oos_results = []
    if interesting:
        print(f"\n\n{'='*80}")
        print("PHASE 3: OOS VALIDATION — Interesting configs only")
        print(f"{'='*80}")

        from collections import defaultdict
        exit_groups = defaultdict(list)
        for br, en in interesting:
            exit_groups[en].append(br)

        for exit_name, group in exit_groups.items():
            ec = next(e for e in exits if e.name == exit_name)
            print(f"\n[OOS] Precomputing exit: {exit_name} ({len(group)} combos)...")
            precomputed_oos = precompute_all_trades(oos, ec)

            for train_br in group:
                gate = next(g for g in gates if g.name == train_br.gate_name)
                signal = next(s for s in signals if s.name == train_br.signal_name)
                oos_br = run_backtest(oos, gate, signal, precomputed_oos, classifier, exit_name=exit_name)
                oos_results.append((oos_br, train_br))

                per_year_breakdown(oos_br.trades, oos, oos_br.config_name)

            del precomputed_oos

        # Print OOS comparison (top 50 by OOS total_bps)
        oos_sorted = sorted(oos_results, key=lambda x: x[0].total_bps, reverse=True)[:50]
        print_comparison_table([br for br, _ in oos_sorted], "TOP 50 — OOS")

        # Side-by-side TRAIN vs OOS (top 50 by OOS performance)
        print(f"\n\n{'='*80}")
        print("TRAIN vs OOS COMPARISON (top 50 by OOS bps)")
        print(f"{'='*80}")
        print(f"{'Config':<55} {'TRAIN bps':>10} {'TRAIN PF':>9} {'OOS bps':>9} {'OOS PF':>8} {'OOS trades':>11}")
        print("-" * 105)
        for oos_br, train_br in oos_sorted:
            print(f"{oos_br.config_name:<55} {train_br.total_bps:>+9.0f} {train_br.profit_factor:>8.2f} "
                  f"{oos_br.total_bps:>+8.0f} {oos_br.profit_factor:>7.2f} {oos_br.n_trades:>11}")

    # =====================================================================
    # SAVE RESULTS CSV — with exit_type and exit_name columns
    # =====================================================================
    rows = []
    for br, en in all_train_results:
        rows.append({
            "config": br.config_name, "gate": br.gate_name, "gate_type": gate_type_map.get(br.gate_name, ""),
            "signal": br.signal_name, "signal_type": signal_type_map.get(br.signal_name, ""),
            "exit": en, "exit_type": exit_type_map.get(en, ""),
            "dataset": "TRAIN", "trades": br.n_trades, "long": br.n_long, "short": br.n_short,
            "win_pct": br.win_rate, "total_bps": br.total_bps, "pf": br.profit_factor,
            "long_bps": br.long_bps, "short_bps": br.short_bps, "avg_bps": br.avg_bps,
            "max_dd": br.max_dd_bps,
        })
    if oos_results:
        for oos_br, train_br in oos_results:
            rows.append({
                "config": oos_br.config_name, "gate": oos_br.gate_name,
                "gate_type": gate_type_map.get(oos_br.gate_name, ""),
                "signal": oos_br.signal_name, "signal_type": signal_type_map.get(oos_br.signal_name, ""),
                "exit": oos_br.exit_name, "exit_type": exit_type_map.get(oos_br.exit_name, ""),
                "dataset": "OOS", "trades": oos_br.n_trades, "long": oos_br.n_long, "short": oos_br.n_short,
                "win_pct": oos_br.win_rate, "total_bps": oos_br.total_bps, "pf": oos_br.profit_factor,
                "long_bps": oos_br.long_bps, "short_bps": oos_br.short_bps, "avg_bps": oos_br.avg_bps,
                "max_dd": oos_br.max_dd_bps,
            })
    pd.DataFrame(rows).to_csv(OUT_DIR / "L2_002_results.csv", index=False)
    print(f"\nResults saved to {OUT_DIR}/L2_002_results.csv ({len(rows):,} rows)")

    # =====================================================================
    # GATE DECISION LOG — sampled exits (running all would be too slow)
    # =====================================================================
    print(f"\n\n{'='*100}")
    print("GATE DECISION ANALYSIS — TRAIN (sampled exits)")
    print(f"{'='*100}")

    # Sample 3 representative exits: first, middle, last
    if len(exits) > 3:
        sample_exits = [exits[0], exits[len(exits)//2], exits[-1]]
    else:
        sample_exits = exits

    gate_decision_rows = []
    for ec in sample_exits:
        print(f"\n[Gate Decisions] Exit: {ec.name} — precomputing...", end=" ", flush=True)
        precomputed_train = precompute_all_trades(train, ec)

        for gate in gates:
            if gate.name == "NO_GATE":
                continue
            for signal in signals:
                decisions = compute_gate_decisions(train, gate, signal, precomputed_train, classifier)
                if decisions.empty:
                    continue
                summary = summarize_gate_decisions(decisions, gate.name, signal.name)
                summary["gate_type"] = gate.gate_type
                summary["signal_type"] = signal.signal_type
                summary["exit"] = ec.name
                summary["exit_type"] = ec.exit_type
                gate_decision_rows.append(summary)

        del precomputed_train
        print(f"done")

    if gate_decision_rows:
        gd_df = pd.DataFrame(gate_decision_rows)
        gd_df.to_csv(OUT_DIR / "L2_002_gate_decisions.csv", index=False)
        print(f"\nGate decisions saved to {OUT_DIR}/L2_002_gate_decisions.csv ({len(gd_df):,} rows)")


if __name__ == "__main__":
    np.random.seed(42)
    main()
