"""
Exit Strategy Experiments on V1.5 trades.

Test 1: ML signals only (ML_LONG, ML_SHORT)
Test 2: All V1.5 signals (V12 + ML) — 1,680 trades

EXP-4 rules:
- Losing -> baseline (15-min close, 20/30 TS, bar 5 tighten to 8, bar 10 exit)
- Profitable AND MFE >= threshold -> 1-min high/low with 8 bps TS
- Thresholds tested: 25, 35, 40, 50, 60 bps
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import gc
import sys
sys.path.insert(0, "src")

from v12.backtest import run_backtest
from v12.config.loader import load_config

FEES = 8


def run_exp4_on_entries(entries, idx_1m, highs_1m, lows_1m, closes_1m, thresholds, tight_ts=8):
    """Run EXP-4 at multiple MFE thresholds on given entries."""
    results = []
    for entry in entries:
        entry_time = entry["entry_time"]
        entry_price = entry["entry_price"]
        direction = entry["direction"]
        ts_init = entry["ts_init"]

        row = dict(entry)

        et = np.datetime64(entry_time)
        mt = et + np.timedelta64(150, "m")
        s = np.searchsorted(idx_1m, et, side="left")
        e = np.searchsorted(idx_1m, mt, side="right")
        sl_t = idx_1m[s:e]
        sl_h = highs_1m[s:e]
        sl_l = lows_1m[s:e]
        sl_c = closes_1m[s:e]

        for thresh in thresholds:
            peak = 0.0
            trade_prof = False
            mfe_crossed = False
            exited = False
            result_gross = 0
            result_reason = ""

            for j in range(len(sl_t)):
                t = sl_t[j]
                h, l, c = sl_h[j], sl_l[j], sl_c[j]
                if direction == "LONG":
                    bb = (h - entry_price) / entry_price * 10000
                    cp = (c - entry_price) / entry_price * 10000
                    wp = (l - entry_price) / entry_price * 10000
                else:
                    bb = (entry_price - l) / entry_price * 10000
                    cp = (entry_price - c) / entry_price * 10000
                    wp = (entry_price - h) / entry_price * 10000
                if bb > peak:
                    peak = bb
                if peak >= thresh:
                    mfe_crossed = True
                mins = (t - et) / np.timedelta64(1, "m")
                b15 = mins / 15
                is15 = mins > 0 and mins % 15 < 1.5

                # Baseline until profitable AND MFE >= threshold
                if not trade_prof or not mfe_crossed:
                    if not trade_prof and is15 and cp > 0:
                        trade_prof = True
                    if is15:
                        ats = 8 if b15 > 5 else ts_init
                        dd = peak - cp
                        if dd >= ats and peak > 0:
                            result_gross = peak - ats
                            result_reason = "TS_BL"
                            exited = True
                            break
                        if b15 >= 10:
                            result_gross = cp
                            result_reason = "TIME"
                            exited = True
                            break
                    continue

                # 1-min high/low with tight trailing stop
                dd = peak - wp
                if dd >= tight_ts and peak > 0:
                    result_gross = peak - tight_ts
                    result_reason = "TS_1M"
                    exited = True
                    break
                if b15 >= 10 and is15:
                    result_gross = cp
                    result_reason = "TIME"
                    exited = True
                    break

            if not exited and len(sl_c) > 0:
                if direction == "LONG":
                    result_gross = (sl_c[-1] - entry_price) / entry_price * 10000
                else:
                    result_gross = (entry_price - sl_c[-1]) / entry_price * 10000
                result_reason = "END"

            row[f"e4_{thresh}_gross"] = result_gross
            row[f"e4_{thresh}_net"] = result_gross - FEES
            row[f"e4_{thresh}_reason"] = result_reason

        results.append(row)

    return pd.DataFrame(results)


def print_analysis(df, label, thresholds=None):
    """Print analysis for a set of trades."""
    df["year"] = df["entry_time"].apply(lambda x: x.year)
    if thresholds is None:
        thresholds = sorted([int(c.split("_")[1]) for c in df.columns if c.startswith("e4_") and c.endswith("_net")])

    print(f"\n{'='*80}")
    print(f" {label}")
    print(f"{'='*80}")

    # Summary
    print(f"\n{'Config':>20s}  {'#':>5s} {'Win%':>6s} {'Net bps':>10s} {'PF':>6s} {'MaxDD':>8s} {'Avg/T':>7s} {'2024':>8s} {'2025':>8s}")

    for col_label, col in [("BASELINE", "real_net")] + [(f"EXP-4 MFE>={t}", f"e4_{t}_net") for t in thresholds]:
        r = df[col].values
        wins = (r > 0).sum()
        total = r.sum()
        gw = r[r > 0].sum()
        gl = abs(r[r <= 0].sum())
        pf_val = gw / gl if gl > 0 else 999
        cum = np.cumsum(r)
        pk = np.maximum.accumulate(cum)
        mdd = (pk - cum).max()
        r24 = df[df["year"] == 2024][col].sum()
        r25 = df[df["year"] == 2025][col].sum()
        print(f"{col_label:>20s}  {len(r):5d} {wins/len(r)*100:5.1f}% {total:+10.1f} {pf_val:6.2f} {-mdd:+8.1f} {total/len(r):+7.1f} {r24:+8.0f} {r25:+8.0f}")

    # By signal type
    print(f"\n  By Signal Type:")
    for sig in sorted(df["sig_type"].unique()):
        sub = df[df["sig_type"] == sig]
        bl = sub["real_net"].sum()
        line = f"    {sig:>12s} ({len(sub):4d}t): BL={bl:+9.1f}"
        for thresh in thresholds:
            e4 = sub[f"e4_{thresh}_net"].sum()
            line += f"  MFE{thresh}={e4:+9.1f}"
        print(line)

    # By MFE bucket
    print(f"\n  By Baseline MFE:")
    header = f"    {'MFE':>10s} {'#':>5s} {'BL net':>10s}"
    for t in thresholds:
        header += f" {'E4@'+str(t):>10s} {'Diff':>8s}"
    print(header)
    for lo, hi, lbl in [(0,15,"0-15"),(15,30,"15-30"),(30,50,"30-50"),(50,80,"50-80"),(80,150,"80-150"),(150,999,"150+")]:
        sub = df[(df["real_mfe"] >= lo) & (df["real_mfe"] < hi)]
        if len(sub) == 0:
            continue
        bl = sub["real_net"].sum()
        line = f"    {lbl:>10s} {len(sub):5d} {bl:+10.1f}"
        for t in thresholds:
            e4 = sub[f"e4_{t}_net"].sum()
            line += f" {e4:+10.1f} {e4-bl:+8.1f}"
        print(line)

    # Trade-by-trade
    print(f"\n  Trade-by-trade vs baseline:")
    for thresh in thresholds:
        col = f"e4_{thresh}_net"
        diff = df[col] - df["real_net"]
        better = (diff > 0.5).sum()
        worse = (diff < -0.5).sum()
        same = len(diff) - better - worse
        print(f"    MFE>={thresh:2d}: Better {better:4d}t, Worse {worse:4d}t, Same {same:4d}t")


def main():
    print("Loading data...")
    config = load_config()

    # Real V1.5 backtest
    all_trades = run_backtest(config)
    oos = [t for t in all_trades if t.entry_time >= pd.Timestamp("2024-01-01")]
    print(f"V1.5 OOS: {len(oos)} trades")

    # Build entries
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

    # Test MFE>=35 with 12bps and 15bps pullback
    thresholds = [35]

    for tight_ts in [12, 15]:
        # ML only
        print(f"\nRunning EXP-4 MFE>=35, tight_ts={tight_ts}bps on ML signals...")
        df_ml = run_exp4_on_entries(ml_entries, idx_1m, highs_1m, lows_1m, closes_1m, thresholds, tight_ts=tight_ts)
        print_analysis(df_ml, f"ML ONLY — EXP-4 MFE>=35, pullback={tight_ts}bps")

        # All V1.5
        print(f"\nRunning EXP-4 MFE>=35, tight_ts={tight_ts}bps on ALL V1.5...")
        df_all = run_exp4_on_entries(all_entries, idx_1m, highs_1m, lows_1m, closes_1m, thresholds, tight_ts=tight_ts)
        print_analysis(df_all, f"ALL V1.5 — EXP-4 MFE>=35, pullback={tight_ts}bps")


if __name__ == "__main__":
    main()
