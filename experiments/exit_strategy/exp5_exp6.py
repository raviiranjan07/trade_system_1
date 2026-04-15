"""
EXP-5: Early loser exit (15-min bars)
  - Bar 3: if MFE < 5 bps, exit at close
  - Bar 4: if MFE < 8 bps, exit at close
  - Everything else = baseline

EXP-6: Asymmetric 1-min loser cut
  - Winners (close PnL > 0): baseline 15-min rules
  - Losers (close PnL < 0): if 45 min passed AND MFE < 5 bps, cut
  - Everything else = baseline

Tested on ML-only and all V1.5.
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import gc
import sys
sys.path.insert(0, "src")

from engine.backtest import run_backtest
from engine.config.loader import load_config

FEES = 8


def run_on_entries(entries, idx_1m, highs_1m, lows_1m, closes_1m):
    """Run EXP-5 and EXP-6 on given entries. Returns DataFrame."""
    results = []

    for entry in entries:
        entry_time = entry["entry_time"]
        entry_price = entry["entry_price"]
        direction = entry["direction"]
        ts_init = entry["ts_init"]

        row = dict(entry)

        # === EXP-5: Early loser exit on 15-min bars ===
        # We simulate bar-by-bar using the real backtest's 15-min logic
        # but add early exit rules at bar 3 and bar 4
        # We need 15-min OHLC — get from 1-min data by resampling
        et = np.datetime64(entry_time)
        mt = et + np.timedelta64(150, "m")
        s = np.searchsorted(idx_1m, et, side="left")
        e = np.searchsorted(idx_1m, mt, side="right")
        sl_t = idx_1m[s:e]
        sl_h = highs_1m[s:e]
        sl_l = lows_1m[s:e]
        sl_c = closes_1m[s:e]

        # Build 15-min bars from 1-min data
        bars_15m = []
        for bar_num in range(1, 11):
            bar_start = et + np.timedelta64((bar_num - 1) * 15, "m")
            bar_end = et + np.timedelta64(bar_num * 15, "m")
            mask = (sl_t >= bar_start) & (sl_t < bar_end)
            if not mask.any():
                break
            bar_h = sl_h[mask].max()
            bar_l = sl_l[mask].min()
            # Close = last 1-min close in this 15-min window
            bar_c = sl_c[mask][-1]
            bars_15m.append((bar_num, bar_h, bar_l, bar_c))

        # EXP-5
        peak5 = 0.0
        exp5_gross = 0
        exp5_reason = ""
        exp5_bar = 0
        for bar_num, h, l, c in bars_15m:
            if direction == "LONG":
                mfe = (h - entry_price) / entry_price * 10000
                pnl = (c - entry_price) / entry_price * 10000
            else:
                mfe = (entry_price - l) / entry_price * 10000
                pnl = (entry_price - c) / entry_price * 10000
            if mfe > peak5:
                peak5 = mfe

            # Baseline trailing stop
            ats = 8 if bar_num > 5 else ts_init
            dd = peak5 - pnl
            if dd >= ats and peak5 > 0:
                exp5_gross = peak5 - ats
                exp5_reason = "TIGHT_TS" if bar_num > 5 else "TRAILING_STOP"
                exp5_bar = bar_num
                break

            # EXP-5 early exit rules
            if bar_num == 3 and peak5 < 5:
                exp5_gross = pnl
                exp5_reason = "EARLY_CUT_3"
                exp5_bar = bar_num
                break
            if bar_num == 4 and peak5 < 8:
                exp5_gross = pnl
                exp5_reason = "EARLY_CUT_4"
                exp5_bar = bar_num
                break

            # Time exit
            if bar_num >= 10:
                exp5_gross = pnl
                exp5_reason = "TIME_EXIT"
                exp5_bar = bar_num
                break

        if exp5_reason == "":
            # Didn't exit in 10 bars (shouldn't happen if we have 10 bars)
            if bars_15m:
                _, _, _, last_c = bars_15m[-1]
                if direction == "LONG":
                    exp5_gross = (last_c - entry_price) / entry_price * 10000
                else:
                    exp5_gross = (entry_price - last_c) / entry_price * 10000
                exp5_reason = "TIME_EXIT"
                exp5_bar = bars_15m[-1][0]

        row["exp5_gross"] = exp5_gross
        row["exp5_net"] = exp5_gross - FEES
        row["exp5_reason"] = exp5_reason
        row["exp5_bar"] = exp5_bar

        # === EXP-6: Asymmetric 1-min loser cut ===
        peak6 = 0.0
        exp6_gross = 0
        exp6_reason = ""
        exp6_bar = 0
        loser_cut_checked = False

        for bar_num, h, l, c in bars_15m:
            if direction == "LONG":
                mfe = (h - entry_price) / entry_price * 10000
                pnl = (c - entry_price) / entry_price * 10000
            else:
                mfe = (entry_price - l) / entry_price * 10000
                pnl = (entry_price - c) / entry_price * 10000
            if mfe > peak6:
                peak6 = mfe

            # Baseline trailing stop
            ats = 8 if bar_num > 5 else ts_init
            dd = peak6 - pnl
            if dd >= ats and peak6 > 0:
                exp6_gross = peak6 - ats
                exp6_reason = "TIGHT_TS" if bar_num > 5 else "TRAILING_STOP"
                exp6_bar = bar_num
                break

            # EXP-6: After bar 3, if trade is losing AND MFE < 5, drill into 1-min
            if bar_num >= 3 and pnl < 0 and peak6 < 5 and not loser_cut_checked:
                loser_cut_checked = True
                # Check 1-min data from bar 3 onwards
                cut_start = et + np.timedelta64(45, "m")  # 45 min = 3 bars
                cut_mask = sl_t >= cut_start
                if cut_mask.any():
                    # Look at 1-min bars — if MFE still < 5 at any point after 45 min
                    # AND close PnL is negative, cut it
                    cut_times = sl_t[cut_mask]
                    cut_closes = sl_c[cut_mask]
                    for k in range(len(cut_times)):
                        ct = cut_times[k]
                        cc = cut_closes[k]
                        if direction == "LONG":
                            cpnl = (cc - entry_price) / entry_price * 10000
                        else:
                            cpnl = (entry_price - cc) / entry_price * 10000
                        mins = (ct - et) / np.timedelta64(1, "m")
                        # Cut on next 15-min boundary after 45 min if still losing
                        if mins >= 45 and mins % 15 < 1.5 and cpnl < 0 and peak6 < 5:
                            exp6_gross = cpnl
                            exp6_reason = "ASYM_CUT"
                            exp6_bar = int(mins / 15) + 1
                            break
                    if exp6_reason == "ASYM_CUT":
                        break

            # Time exit
            if bar_num >= 10:
                exp6_gross = pnl
                exp6_reason = "TIME_EXIT"
                exp6_bar = bar_num
                break

        if exp6_reason == "":
            if bars_15m:
                _, _, _, last_c = bars_15m[-1]
                if direction == "LONG":
                    exp6_gross = (last_c - entry_price) / entry_price * 10000
                else:
                    exp6_gross = (entry_price - last_c) / entry_price * 10000
                exp6_reason = "TIME_EXIT"
                exp6_bar = bars_15m[-1][0]

        row["exp6_gross"] = exp6_gross
        row["exp6_net"] = exp6_gross - FEES
        row["exp6_reason"] = exp6_reason
        row["exp6_bar"] = exp6_bar

        results.append(row)

    return pd.DataFrame(results)


def print_stats(label, r):
    """Print stats for a result array."""
    wins = (r > 0).sum()
    total = r.sum()
    gw = r[r > 0].sum()
    gl = abs(r[r <= 0].sum())
    pf = gw / gl if gl > 0 else 999
    cum = np.cumsum(r)
    pk = np.maximum.accumulate(cum)
    mdd = (pk - cum).max()
    return len(r), wins, total, pf, mdd


def print_analysis(df, label):
    """Print analysis."""
    df["year"] = df["entry_time"].apply(lambda x: x.year)

    print(f"\n{'='*80}")
    print(f" {label}")
    print(f"{'='*80}")

    print(f"\n{'Config':>25s}  {'#':>5s} {'Win%':>6s} {'Net bps':>10s} {'PF':>6s} {'MaxDD':>8s} {'Avg/T':>7s} {'2024':>8s} {'2025':>8s}")

    for col_label, col in [("BASELINE (V1.3)", "real_net"), ("EXP-5 (early cut)", "exp5_net"), ("EXP-6 (asym 1-min)", "exp6_net")]:
        r = df[col].values
        n, wins, total, pf, mdd = print_stats(col_label, r)
        r24 = df[df["year"] == 2024][col].sum()
        r25 = df[df["year"] == 2025][col].sum()
        print(f"{col_label:>25s}  {n:5d} {wins/n*100:5.1f}% {total:+10.1f} {pf:6.2f} {-mdd:+8.1f} {total/n:+7.1f} {r24:+8.0f} {r25:+8.0f}")

    # By signal type
    print(f"\n  By Signal Type:")
    for sig in sorted(df["sig_type"].unique()):
        sub = df[df["sig_type"] == sig]
        bl = sub["real_net"].sum()
        e5 = sub["exp5_net"].sum()
        e6 = sub["exp6_net"].sum()
        print(f"    {sig:>12s} ({len(sub):4d}t): BL={bl:+9.1f}  EXP5={e5:+9.1f}  EXP6={e6:+9.1f}")

    # By exit reason — EXP-5
    print(f"\n  EXP-5 exit reasons:")
    for reason in sorted(df["exp5_reason"].unique()):
        sub = df[df["exp5_reason"] == reason]
        bl = sub["real_net"].sum()
        e5 = sub["exp5_net"].sum()
        w = (sub["exp5_net"] > 0).sum()
        print(f"    {reason:>15s}: {len(sub):4d}t ({w:4d}W)  BL={bl:+9.1f}  EXP5={e5:+9.1f}  diff={e5-bl:+.1f}")

    # By exit reason — EXP-6
    print(f"\n  EXP-6 exit reasons:")
    for reason in sorted(df["exp6_reason"].unique()):
        sub = df[df["exp6_reason"] == reason]
        bl = sub["real_net"].sum()
        e6 = sub["exp6_net"].sum()
        w = (sub["exp6_net"] > 0).sum()
        print(f"    {reason:>15s}: {len(sub):4d}t ({w:4d}W)  BL={bl:+9.1f}  EXP6={e6:+9.1f}  diff={e6-bl:+.1f}")

    # Trade-by-trade comparison
    print(f"\n  Trade-by-trade vs baseline:")
    for col_label, col in [("EXP-5", "exp5_net"), ("EXP-6", "exp6_net")]:
        diff = df[col] - df["real_net"]
        better = (diff > 0.5).sum()
        worse = (diff < -0.5).sum()
        same = len(diff) - better - worse
        total_b = diff[diff > 0.5].sum()
        total_w = diff[diff < -0.5].sum()
        print(f"    {col_label}: Better {better:4d}t(+{total_b:.0f}bps), Worse {worse:4d}t({total_w:.0f}bps), Same {same:4d}t")

    # What happened to baseline TIME_EXIT trades?
    print(f"\n  Baseline TIME_EXIT trades ({(df['real_exit']=='TIME_EXIT').sum()}t):")
    te = df[df["real_exit"] == "TIME_EXIT"]
    bl_te = te["real_net"].sum()
    e5_te = te["exp5_net"].sum()
    e6_te = te["exp6_net"].sum()
    print(f"    BL total:  {bl_te:+.1f} bps")
    print(f"    EXP5 total: {e5_te:+.1f} bps (saved {e5_te-bl_te:+.1f})")
    print(f"    EXP6 total: {e6_te:+.1f} bps (saved {e6_te-bl_te:+.1f})")


def main():
    print("Loading data...")
    config = load_config()

    all_trades = run_backtest(config)
    oos = [t for t in all_trades if t.entry_time >= pd.Timestamp("2024-01-01")]
    print(f"V1.5 OOS: {len(oos)} trades")

    all_entries = []
    for t in oos:
        ts_init = 20 if t.direction == "LONG" else 30
        all_entries.append({
            "entry_time": t.entry_time,
            "entry_price": t.entry_price,
            "direction": t.direction,
            "ts_init": ts_init,
            "sig_type": t.signal_type,
            "real_gross": t.gross_profit_bps,
            "real_net": t.net_profit_bps,
            "real_exit": t.exit_reason,
            "real_bar": t.exit_bar,
            "real_mfe": t.mfe_bps,
        })

    ml_entries = [e for e in all_entries if e["sig_type"] in ("ML_LONG", "ML_SHORT")]
    print(f"ML only: {len(ml_entries)} trades")
    print(f"All V1.5: {len(all_entries)} trades")

    # Load 1-min data
    gc.collect()
    pf_file = pq.ParquetFile("data/raw/BTCUSDT_1m_ohlcv.parquet")
    chunks = []
    for batch in pf_file.iter_batches(batch_size=100_000):
        chunk = batch.to_pandas()
        if chunk.index.tz is not None:
            chunk.index = chunk.index.tz_convert(None)
        chunk = chunk[chunk.index >= "2024-01-01"]
        if len(chunk) > 0:
            chunks.append(chunk[["high", "low", "close"]])
    df_1m = pd.concat(chunks)
    del chunks
    gc.collect()
    idx_1m = df_1m.index.values
    highs_1m = df_1m["high"].values
    lows_1m = df_1m["low"].values
    closes_1m = df_1m["close"].values
    print(f"1-min bars: {len(df_1m):,}")

    # TEST 1: ML only
    print("\nRunning on ML signals...")
    df_ml = run_on_entries(ml_entries, idx_1m, highs_1m, lows_1m, closes_1m)
    print_analysis(df_ml, "TEST 1: ML SIGNALS ONLY (ML_LONG + ML_SHORT)")

    # TEST 2: All V1.5
    print("\nRunning on ALL V1.5 signals...")
    df_all = run_on_entries(all_entries, idx_1m, highs_1m, lows_1m, closes_1m)
    print_analysis(df_all, "TEST 2: ALL V1.5 SIGNALS (V12 + ML)")


if __name__ == "__main__":
    main()
