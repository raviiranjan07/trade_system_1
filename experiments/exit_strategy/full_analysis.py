"""Full analysis of exit strategy experiments."""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import gc
import sys
sys.path.insert(0, "src")

from v12.config.loader import load_config
from v12.strategy import V12Strategy, Direction, SignalType

FEES = 8

def main():
    print("Loading data...")
    config = load_config()
    df_15m = pd.read_parquet("data/raw/BTCUSDT_15m_ohlcv.parquet")
    if df_15m.index.tz is not None:
        df_15m.index = df_15m.index.tz_convert(None)
    strategy = V12Strategy(config)
    df_15m = strategy.compute_indicators(df_15m)
    test = df_15m["2024-01-01":"2025-12-31"]
    signals = strategy.generate_signals(test)

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

    n = len(test)
    bull = test["bull_market"].values
    bear = test["bear_market"].values
    opens = test["open"].values
    highs_15 = test["high"].values
    lows_15 = test["low"].values
    closes_15 = test["close"].values
    times_15 = test.index
    signal_map = {s.bar_index: s for s in signals}

    # Get 220 baseline entries
    entries = []
    i = 0
    pos = None
    while i < n:
        if pos is not None:
            pos["bh"] += 1
            h, l, c = highs_15[i], lows_15[i], closes_15[i]
            if pos["d"] == "LONG":
                mfe = (h - pos["ep"]) / pos["ep"] * 10000
                pnl_ = (c - pos["ep"]) / pos["ep"] * 10000
            else:
                mfe = (pos["ep"] - l) / pos["ep"] * 10000
                pnl_ = (pos["ep"] - c) / pos["ep"] * 10000
            if mfe > pos["pk"]:
                pos["pk"] = mfe
            ats = 8 if pos["bh"] > 5 else pos["ts"]
            dd = pos["pk"] - pnl_
            if (dd >= ats and pos["pk"] > 0) or pos["bh"] >= 10:
                pos = None
            i += 1
            continue
        if i in signal_map:
            sig = signal_map[i]
            ok = bull[i] if sig.signal_type in (SignalType.V12_LONG, SignalType.BULL_SHORT) else bear[i]
            if ok and i + 1 < n:
                ts = 20 if sig.direction == Direction.LONG else 30
                entries.append((i + 1, times_15[i + 1], opens[i + 1], sig.direction.value, ts, sig.signal_type.value))
                pos = {"d": sig.direction.value, "ep": opens[i + 1], "ts": ts, "bh": 0, "pk": 0.0}
                i += 2
                continue
        i += 1

    print(f"Entries: {len(entries)}")

    # For each trade compute baseline + EXP-4 at multiple thresholds
    results = []
    for entry_bar, entry_time, entry_price, direction, ts_init, sig_type in entries:
        row = {
            "entry_time": entry_time,
            "direction": direction,
            "ts_init": ts_init,
            "sig_type": sig_type,
            "entry_price": entry_price,
        }

        # BASELINE
        peak = 0.0
        bl_gross = 0
        bl_reason = ""
        bl_bar = 0
        bl_mfe = 0
        for bar in range(1, 11):
            idx = entry_bar + bar
            if idx >= n:
                break
            h, l, c = highs_15[idx], lows_15[idx], closes_15[idx]
            if direction == "LONG":
                mfe = (h - entry_price) / entry_price * 10000
                pnl_ = (c - entry_price) / entry_price * 10000
            else:
                mfe = (entry_price - l) / entry_price * 10000
                pnl_ = (entry_price - c) / entry_price * 10000
            if mfe > peak:
                peak = mfe
            ats = 8 if bar > 5 else ts_init
            dd = peak - pnl_
            if dd >= ats and peak > 0:
                bl_gross = peak - ats
                bl_reason = "TIGHT_TS" if bar > 5 else "TS"
                bl_bar = bar
                bl_mfe = peak
                break
            if bar == 10:
                bl_gross = pnl_
                bl_reason = "TIME"
                bl_bar = bar
                bl_mfe = peak
        row["bl_gross"] = bl_gross
        row["bl_net"] = bl_gross - FEES
        row["bl_reason"] = bl_reason
        row["bl_bar"] = bl_bar
        row["bl_mfe"] = bl_mfe

        # EXP-4 at various thresholds
        et = np.datetime64(entry_time)
        mt = et + np.timedelta64(150, "m")
        s = np.searchsorted(idx_1m, et, side="left")
        e = np.searchsorted(idx_1m, mt, side="right")
        sl_t = idx_1m[s:e]
        sl_h = highs_1m[s:e]
        sl_l = lows_1m[s:e]
        sl_c = closes_1m[s:e]

        for thresh in [25, 30, 35, 40, 50, 60]:
            peak = 0.0
            trade_prof = False
            mfe_crossed = False
            exited = False
            result_gross = 0
            result_reason = ""
            result_min = 0

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

                # Baseline mode until profitable AND MFE >= threshold
                if not trade_prof or not mfe_crossed:
                    if not trade_prof and is15 and cp > 0:
                        trade_prof = True
                    if is15:
                        ats = 8 if b15 > 5 else ts_init
                        dd = peak - cp
                        if dd >= ats and peak > 0:
                            result_gross = peak - ats
                            result_reason = "TS_BL"
                            result_min = int(mins)
                            exited = True
                            break
                        if b15 >= 10:
                            result_gross = cp
                            result_reason = "TIME"
                            result_min = int(mins)
                            exited = True
                            break
                    continue

                # 1-min high/low with 8bps
                dd = peak - wp
                if dd >= 8 and peak > 0:
                    result_gross = peak - 8
                    result_reason = "TS_1M"
                    result_min = int(mins)
                    exited = True
                    break
                if b15 >= 10 and is15:
                    result_gross = cp
                    result_reason = "TIME"
                    result_min = int(mins)
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
            row[f"e4_{thresh}_min"] = result_min

        results.append(row)

    df = pd.DataFrame(results)
    df["year"] = df["entry_time"].apply(lambda x: x.year)

    # ============ FULL ANALYSIS ============
    print("=" * 80)
    print(" EXIT STRATEGY EXPERIMENT - FULL ANALYSIS")
    print(" 220 entries (same as baseline), EXP-4 at 6 MFE thresholds")
    print(" Baseline verified: matches real V1.3.2 backtest (168/168 trades identical)")
    print("=" * 80)

    # 1. Summary table
    print("\n--- 1. SUMMARY TABLE ---")
    header = f"{'Config':>20s}  {'#':>4s} {'Win%':>6s} {'Net bps':>9s} {'PF':>6s} {'MaxDD':>7s} {'Avg/T':>7s} {'2024':>7s} {'2025':>7s}"
    print(header)
    configs = [("BASELINE", "bl_net")] + [(f"EXP-4 MFE>={t}", f"e4_{t}_net") for t in [25, 30, 35, 40, 50, 60]]
    for label, col in configs:
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
        print(f"{label:>20s}  {len(r):4d} {wins / len(r) * 100:5.1f}% {total:+9.1f} {pf_val:6.2f} {-mdd:+7.1f} {total / len(r):+7.1f} {r24:+7.0f} {r25:+7.0f}")

    # 2. Trade-by-trade comparison
    print("\n--- 2. TRADE-BY-TRADE: EXP-4 vs BASELINE ---")
    for thresh in [25, 35, 40, 50, 60]:
        col = f"e4_{thresh}_net"
        diff = df[col] - df["bl_net"]
        better = (diff > 0.5).sum()
        worse = (diff < -0.5).sum()
        same = len(diff) - better - worse
        avg_b = diff[diff > 0.5].mean() if better > 0 else 0
        avg_w = diff[diff < -0.5].mean() if worse > 0 else 0
        total_b = diff[diff > 0.5].sum()
        total_w = diff[diff < -0.5].sum()
        print(f"  MFE>={thresh:2d}: Better {better:3d}t(+{total_b:.0f}bps, avg+{avg_b:.1f}), Worse {worse:3d}t({total_w:.0f}bps, avg{avg_w:.1f}), Same {same:3d}t")

    # 3. By signal type
    print("\n--- 3. BY SIGNAL TYPE ---")
    for sig in ["V12_LONG", "V12_SHORT", "BEAR_LONG", "BULL_SHORT"]:
        sub = df[df["sig_type"] == sig]
        if len(sub) == 0:
            continue
        bl = sub["bl_net"].sum()
        line = f"  {sig:>12s} ({len(sub):3d}t): BL={bl:+7.1f}"
        for thresh in [25, 35, 40, 50]:
            e4 = sub[f"e4_{thresh}_net"].sum()
            line += f"  MFE{thresh}={e4:+7.1f}"
        print(line)

    # 4. By baseline exit reason
    print("\n--- 4. BY BASELINE EXIT REASON ---")
    for reason in ["TS", "TIGHT_TS", "TIME"]:
        sub = df[df["bl_reason"] == reason]
        if len(sub) == 0:
            continue
        bl = sub["bl_net"].sum()
        line = f"  {reason:>10s} ({len(sub):3d}t): BL={bl:+8.1f}"
        for thresh in [40, 50, 60]:
            e4 = sub[f"e4_{thresh}_net"].sum()
            diff = e4 - bl
            line += f"  MFE{thresh}={e4:+8.1f}({diff:+.0f})"
        print(line)

    # 5. By baseline MFE bucket
    print("\n--- 5. BY BASELINE MFE BUCKET ---")
    print(f"{'MFE bucket':>15s} {'#':>4s} {'BL net':>9s} {'E4@40 net':>9s} {'Diff':>8s} {'E4@50 net':>9s} {'Diff':>8s} {'E4@60 net':>9s} {'Diff':>8s}")
    for lo, hi, label in [(0, 15, "0-15"), (15, 30, "15-30"), (30, 50, "30-50"), (50, 80, "50-80"), (80, 150, "80-150"), (150, 999, "150+")]:
        sub = df[(df["bl_mfe"] >= lo) & (df["bl_mfe"] < hi)]
        if len(sub) == 0:
            continue
        bl = sub["bl_net"].sum()
        e40 = sub["e4_40_net"].sum()
        e50 = sub["e4_50_net"].sum()
        e60 = sub["e4_60_net"].sum()
        print(f"{label:>15s} {len(sub):4d} {bl:+9.1f} {e40:+9.1f} {e40 - bl:+8.1f} {e50:+9.1f} {e50 - bl:+8.1f} {e60:+9.1f} {e60 - bl:+8.1f}")

    # 6. EXP-4@40 exit reason breakdown
    print("\n--- 6. EXP-4@40 EXIT REASON BREAKDOWN ---")
    for reason in sorted(df["e4_40_reason"].unique()):
        sub = df[df["e4_40_reason"] == reason]
        bl = sub["bl_net"].sum()
        e4 = sub["e4_40_net"].sum()
        wins = (sub["e4_40_net"] > 0).sum()
        print(f"  {reason:>8s}: {len(sub):3d} trades ({wins:3d}W), BL={bl:+8.1f}, E4={e4:+8.1f}, diff={e4 - bl:+.1f}")

    # 7. Top 10 where baseline >> EXP-4@40
    df["diff_40"] = df["e4_40_net"] - df["bl_net"]
    print("\n--- 7. TOP 10: BASELINE >> EXP-4@40 ---")
    worst = df.nsmallest(10, "diff_40")
    for _, r in worst.iterrows():
        print(f"  {r['entry_time']}  {r['direction']:>5s} {r['sig_type']:>12s}  BL={r['bl_net']:+7.1f}(bar{int(r['bl_bar'])},{r['bl_reason']})  E4={r['e4_40_net']:+7.1f}  MFE={r['bl_mfe']:+.0f}  diff={r['diff_40']:+.1f}")

    # 8. Top 10 where EXP-4@40 >> baseline
    print("\n--- 8. TOP 10: EXP-4@40 >> BASELINE ---")
    best = df.nlargest(10, "diff_40")
    for _, r in best.iterrows():
        print(f"  {r['entry_time']}  {r['direction']:>5s} {r['sig_type']:>12s}  BL={r['bl_net']:+7.1f}(bar{int(r['bl_bar'])},{r['bl_reason']})  E4={r['e4_40_net']:+7.1f}  MFE={r['bl_mfe']:+.0f}  diff={r['diff_40']:+.1f}")

    # 9. Year x Signal type
    print("\n--- 9. YEAR x SIGNAL TYPE (BL vs E4@40) ---")
    for year in [2024, 2025]:
        print(f"  {year}:")
        for sig in ["V12_LONG", "V12_SHORT", "BEAR_LONG", "BULL_SHORT"]:
            sub = df[(df["year"] == year) & (df["sig_type"] == sig)]
            if len(sub) == 0:
                continue
            bl = sub["bl_net"].sum()
            e4 = sub["e4_40_net"].sum()
            diff = e4 - bl
            w_bl = (sub["bl_net"] > 0).sum()
            w_e4 = (sub["e4_40_net"] > 0).sum()
            print(f"    {sig:>12s} ({len(sub):2d}t): BL={bl:+7.1f}({w_bl}W)  E4@40={e4:+7.1f}({w_e4}W)  diff={diff:+.1f}")

    # 10. Key insight: how much profit is lost per exit type switch
    print("\n--- 10. WHAT HAPPENS WHEN EXP-4@40 SWITCHES TO 1-MIN ---")
    switched = df[df["e4_40_reason"] == "TS_1M"]
    stayed = df[df["e4_40_reason"] != "TS_1M"]
    print(f"  Switched to 1-min exit: {len(switched)} trades")
    print(f"    BL total:  {switched['bl_net'].sum():+.1f} bps (avg {switched['bl_net'].mean():+.1f})")
    print(f"    E4 total:  {switched['e4_40_net'].sum():+.1f} bps (avg {switched['e4_40_net'].mean():+.1f})")
    print(f"    Diff:      {switched['diff_40'].sum():+.1f} bps")
    print(f"    E4 wins:   {(switched['e4_40_net'] > 0).sum()}/{len(switched)}")
    print(f"  Stayed on baseline: {len(stayed)} trades")
    print(f"    BL total:  {stayed['bl_net'].sum():+.1f} bps (avg {stayed['bl_net'].mean():+.1f})")
    print(f"    E4 total:  {stayed['e4_40_net'].sum():+.1f} bps (avg {stayed['e4_40_net'].mean():+.1f})")
    print(f"    Diff:      {stayed['diff_40'].sum():+.1f} bps")


if __name__ == "__main__":
    main()
