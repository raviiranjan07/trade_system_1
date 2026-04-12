"""
EXP-12: Cooldown after EARLY_CUT, with and without reverse.

Test:
A) Cooldown only: after EARLY_CUT, block new entries for X bars
B) Cooldown + Reverse: after EARLY_CUT, reverse position, then cooldown

Cooldown values: 5, 7, 10 bars (from cut bar, so total = cut_bar + cooldown)
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "src")
from v12.backtest import run_backtest
from v12.config.loader import load_config
from v12.strategy import Direction, SignalType

FEES = 8


def main():
    print("Loading data...")
    config = load_config()

    # Load 15-min data
    df_15m = pd.read_parquet("data/raw/BTCUSDT_15m_ohlcv.parquet")
    df_15m.index = pd.to_datetime(df_15m.index).tz_localize(None)

    from v12.strategy import V12Strategy
    from v12.ml_signal import MLSignalGenerator
    from pathlib import Path

    strategy = V12Strategy(config)
    df_15m = strategy.compute_indicators(df_15m)
    test = df_15m["2024-01-01":"2025-12-31"]

    # Generate all signals
    signals = strategy.generate_signals(test)
    ml_gen = MLSignalGenerator(
        model_path=Path("models/direction_v15/direction_model.onnx"),
        scaler_path=Path("models/direction_v15/scaler.npz"),
    )
    if ml_gen.loaded:
        test_ml = ml_gen.compute_features_from_df(test.copy())
        ml_signals = ml_gen.generate_signals(test_ml)
        signals.extend(ml_signals)

    signal_map = {}
    for s in signals:
        if s.bar_index not in signal_map:
            signal_map[s.bar_index] = s
        else:
            existing = signal_map[s.bar_index]
            if existing.signal_type.value.startswith("ML") and not s.signal_type.value.startswith("ML"):
                signal_map[s.bar_index] = s

    n = len(test)
    highs = test["high"].values
    lows = test["low"].values
    closes = test["close"].values
    opens = test["open"].values
    times = test.index
    bull = test["bull_market"].values
    bear = test["bear_market"].values

    print(f"Bars: {n}, Signals: {len(signal_map)}")

    def regime_ok(sig_type, idx):
        if sig_type in (SignalType.ML_LONG, SignalType.ML_SHORT):
            return True
        if sig_type in (SignalType.V12_LONG, SignalType.BULL_SHORT):
            return bull[idx]
        return bear[idx]

    def sim_trade_v2(entry_idx, entry_price, direction, ts_init):
        """Simulate V2 exit rules. Returns (net_bps, exit_reason, exit_bar_idx, mfe)."""
        peak = 0
        be_active = False
        for bar in range(1, 11):
            idx = entry_idx + bar
            if idx >= n:
                break
            h, l, c = highs[idx], lows[idx], closes[idx]
            if direction == "LONG":
                mfe = (h - entry_price) / entry_price * 10000
                pnl = (c - entry_price) / entry_price * 10000
            else:
                mfe = (entry_price - l) / entry_price * 10000
                pnl = (entry_price - c) / entry_price * 10000
            if mfe > peak:
                peak = mfe
            if peak >= 15:
                be_active = True

            ats = 6 if bar > 4 else ts_init
            dd = peak - pnl
            if dd >= ats and peak > 0:
                exit_gross = peak - ats
                if be_active and exit_gross < 9:
                    exit_gross = 9
                reason = "TIGHT_TS" if bar > 4 else "TS"
                return exit_gross - FEES, reason, idx, peak
            if be_active and pnl <= 9:
                return 9 - FEES, "BE", idx, peak
            if bar == 3 and peak < 3:
                return pnl - FEES, "EARLY", idx, peak
            if bar == 4 and peak < 5:
                return pnl - FEES, "EARLY", idx, peak
            if bar >= 10:
                return pnl - FEES, "TIME", idx, peak
        # Fallback
        last_idx = min(entry_idx + 10, n - 1)
        if direction == "LONG":
            pnl = (closes[last_idx] - entry_price) / entry_price * 10000
        else:
            pnl = (entry_price - closes[last_idx]) / entry_price * 10000
        return pnl - FEES, "TIME", last_idx, peak

    def run_backtest_custom(cooldown_bars=0, reverse=False):
        """Run full backtest with V2 exit + cooldown + optional reverse."""
        trades = []
        blocked_until = -1
        i = 0

        while i < n:
            # Skip if blocked by cooldown
            if i < blocked_until:
                i += 1
                continue

            # Check for signal
            if i in signal_map:
                sig = signal_map[i]
                if regime_ok(sig.signal_type, i) and i + 1 < n:
                    entry_price = opens[i + 1]
                    direction = sig.direction.value
                    ts_init = 20 if direction == "LONG" else 30

                    net, reason, exit_idx, mfe = sim_trade_v2(i + 1, entry_price, direction, ts_init)
                    trades.append({
                        "entry_time": times[i + 1],
                        "direction": direction,
                        "sig_type": sig.signal_type.value,
                        "net": net,
                        "reason": reason,
                        "exit_idx": exit_idx,
                        "mfe": mfe,
                    })

                    if reason == "EARLY" and cooldown_bars > 0:
                        # Apply cooldown
                        blocked_until = exit_idx + cooldown_bars

                        # Optional reverse
                        if reverse:
                            rev_dir = "SHORT" if direction == "LONG" else "LONG"
                            rev_ts = 20 if rev_dir == "LONG" else 30
                            rev_entry_idx = exit_idx + 1
                            if rev_entry_idx < n:
                                rev_entry_price = opens[rev_entry_idx]
                                rev_net, rev_reason, rev_exit_idx, rev_mfe = sim_trade_v2(
                                    rev_entry_idx, rev_entry_price, rev_dir, rev_ts
                                )
                                trades.append({
                                    "entry_time": times[rev_entry_idx],
                                    "direction": rev_dir,
                                    "sig_type": "REVERSE",
                                    "net": rev_net,
                                    "reason": "REV_" + rev_reason,
                                    "exit_idx": rev_exit_idx,
                                    "mfe": rev_mfe,
                                })
                                # Block until reverse exits + cooldown
                                blocked_until = max(blocked_until, rev_exit_idx + 1)

                    # Skip to after exit
                    i = exit_idx + 1
                    continue

            i += 1

        return pd.DataFrame(trades)

    # Run all configs
    print(f"\n{'='*110}")
    print(f" COOLDOWN + REVERSE TEST")
    print(f"{'='*110}")
    print(f"\n{'Config':>40s}  {'#':>5s} {'Win%':>6s} {'Net bps':>10s} {'PF':>6s} {'MaxDD':>8s} {'2024':>8s} {'2025':>8s} {'Early#':>7s}")

    def print_row(label, df_trades):
        if len(df_trades) == 0:
            print(f"{label:>40s}  no trades")
            return
        r = df_trades["net"].values
        wins = (r > 0).sum()
        total = r.sum()
        gw = r[r > 0].sum()
        gl = abs(r[r <= 0].sum())
        pf = gw / gl if gl > 0 else 999
        cum = np.cumsum(r)
        pk = np.maximum.accumulate(cum)
        mdd = (pk - cum).max()
        df_trades["year"] = df_trades["entry_time"].apply(lambda x: x.year)
        r24 = df_trades[df_trades["year"] == 2024]["net"].sum()
        r25 = df_trades[df_trades["year"] == 2025]["net"].sum()
        early_count = (df_trades["reason"] == "EARLY").sum()
        print(f"{label:>40s}  {len(r):5d} {wins/len(r)*100:5.1f}% {total:+10.1f} {pf:6.2f} {-mdd:+8.1f} {r24:+8.0f} {r25:+8.0f} {early_count:7d}")

    # No cooldown (current broken behavior)
    df_no_cd = run_backtest_custom(cooldown_bars=0, reverse=False)
    print_row("V2 no cooldown (current)", df_no_cd)

    # Cooldown only (no reverse)
    for cd in [5, 7, 10]:
        df_cd = run_backtest_custom(cooldown_bars=cd, reverse=False)
        print_row(f"V2 + {cd}-bar cooldown", df_cd)

    # Cooldown + reverse
    for cd in [5, 7, 10]:
        df_rev = run_backtest_custom(cooldown_bars=cd, reverse=True)
        print_row(f"V2 + {cd}-bar cooldown + reverse", df_rev)

    # Detail: best config breakdown
    print(f"\n--- Best config detail ---")
    for cd in [7]:
        for rev in [False, True]:
            label = f"CD={cd}" + (" +REV" if rev else "")
            df_detail = run_backtest_custom(cooldown_bars=cd, reverse=rev)
            print(f"\n  {label}:")
            for reason in sorted(df_detail["reason"].unique()):
                sub = df_detail[df_detail["reason"] == reason]
                w = (sub["net"] > 0).sum()
                print(f"    {reason:>12s}: {len(sub):4d}t ({w:4d}W)  net={sub['net'].sum():+9.1f}  avg={sub['net'].mean():+7.1f}")


if __name__ == "__main__":
    main()
