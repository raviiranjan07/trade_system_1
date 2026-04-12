"""Deep dive into EXP-5 to find further improvements."""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import gc
import sys
sys.path.insert(0, "src")
from v12.backtest import run_backtest
from v12.config.loader import load_config

FEES = 8


def build_15m_bars(et, ep, direction, sl_t, sl_h, sl_l, sl_c):
    """Build 15-min bars from 1-min data for a trade."""
    bars = []
    peak = 0
    for bn in range(1, 11):
        bs = et + np.timedelta64((bn - 1) * 15, "m")
        be = et + np.timedelta64(bn * 15, "m")
        mask = (sl_t >= bs) & (sl_t < be)
        if not mask.any():
            break
        h = sl_h[mask].max()
        l = sl_l[mask].min()
        c = sl_c[mask][-1]
        if direction == "LONG":
            mfe = (h - ep) / ep * 10000
            pnl = (c - ep) / ep * 10000
        else:
            mfe = (ep - l) / ep * 10000
            pnl = (ep - c) / ep * 10000
        if mfe > peak:
            peak = mfe
        bars.append({"bar": bn, "peak": peak, "pnl": pnl, "mfe_bar": mfe})
    return bars


def sim_exp5(bars, ts_init, bar3_thresh=5, bar4_thresh=8):
    """Simulate EXP-5 with given thresholds."""
    peak = 0
    for b in bars:
        bn = b["bar"]
        peak = b["peak"]
        pnl = b["pnl"]
        ats = 8 if bn > 5 else ts_init
        dd = peak - pnl
        if dd >= ats and peak > 0:
            return peak - ats - FEES, "TS", bn
        if bn == 3 and peak < bar3_thresh:
            return pnl - FEES, "EARLY3", bn
        if bn == 4 and peak < bar4_thresh:
            return pnl - FEES, "EARLY4", bn
        if bn >= 10:
            return pnl - FEES, "TIME", bn
    if bars:
        return bars[-1]["pnl"] - FEES, "TIME", bars[-1]["bar"]
    return -FEES, "NONE", 0


def main():
    print("Loading data...")
    config = load_config()
    trades = run_backtest(config)
    oos = [t for t in trades if t.entry_time >= pd.Timestamp("2024-01-01")]

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

    # Pre-build all trade data
    all_data = []
    for t in oos:
        et = np.datetime64(t.entry_time)
        mt = et + np.timedelta64(150, "m")
        s = np.searchsorted(idx_1m, et, side="left")
        e = np.searchsorted(idx_1m, mt, side="right")
        bars = build_15m_bars(
            et, t.entry_price, t.direction,
            idx_1m[s:e], highs_1m[s:e], lows_1m[s:e], closes_1m[s:e]
        )
        ts_init = 20 if t.direction == "LONG" else 30
        all_data.append({
            "entry_time": t.entry_time,
            "direction": t.direction,
            "sig_type": t.signal_type,
            "ts_init": ts_init,
            "real_net": t.net_profit_bps,
            "real_exit": t.exit_reason,
            "real_mfe": t.mfe_bps,
            "bars": bars,
        })

    print(f"Trades: {len(all_data)}")

    # ============================================================
    print("\n" + "=" * 80)
    print(" 1. TRADES CUT BY EXP-5 THAT WERE WINNERS IN BASELINE")
    print("=" * 80)

    cut_won = []
    cut_lost = []
    for d in all_data:
        net5, reason, _ = sim_exp5(d["bars"], d["ts_init"])
        if reason in ("EARLY3", "EARLY4"):
            if d["real_net"] > 0:
                cut_won.append(d)
            else:
                cut_lost.append(d)

    print(f"  Cut and was BL winner: {len(cut_won)} trades, {sum(d['real_net'] for d in cut_won):+.1f} bps sacrificed")
    print(f"  Cut and was BL loser:  {len(cut_lost)} trades, {sum(d['real_net'] for d in cut_lost):+.1f} bps saved")
    if cut_won:
        print(f"\n  Winners we sacrificed (top 15):")
        cut_won.sort(key=lambda x: x["real_net"], reverse=True)
        for d in cut_won[:15]:
            b3_mfe = d["bars"][2]["peak"] if len(d["bars"]) >= 3 else 0
            print(f"    {d['entry_time']}  {d['direction']:>5s} {d['sig_type']:>12s}  BL={d['real_net']:+7.1f}  MFE@bar3={b3_mfe:.1f}  final_MFE={d['real_mfe']:.1f}")

    # ============================================================
    print("\n" + "=" * 80)
    print(" 2. EXP-5 REMAINING LOSERS BREAKDOWN")
    print("=" * 80)

    losers = []
    for d in all_data:
        net5, reason5, bar5 = sim_exp5(d["bars"], d["ts_init"])
        if net5 <= 0:
            final_mfe = d["bars"][-1]["peak"] if d["bars"] else 0
            losers.append({
                "time": d["entry_time"], "dir": d["direction"], "sig": d["sig_type"],
                "net5": net5, "reason5": reason5, "bar5": bar5,
                "mfe": final_mfe, "bl_net": d["real_net"],
            })

    df_l = pd.DataFrame(losers)
    print(f"  Total losers: {len(df_l)}, {df_l['net5'].sum():+.1f} bps")

    print(f"\n  By exit reason:")
    for r in sorted(df_l["reason5"].unique()):
        sub = df_l[df_l["reason5"] == r]
        print(f"    {r:>8s}: {len(sub):4d}t, {sub['net5'].sum():+8.1f} bps, avg {sub['net5'].mean():+.1f}")

    print(f"\n  By MFE bucket:")
    for lo, hi, lbl in [(0, 5, "0-5"), (5, 10, "5-10"), (10, 20, "10-20"), (20, 30, "20-30"), (30, 999, "30+")]:
        sub = df_l[(df_l["mfe"] >= lo) & (df_l["mfe"] < hi)]
        if len(sub) == 0:
            continue
        print(f"    MFE {lbl:>5s}: {len(sub):4d}t, {sub['net5'].sum():+8.1f} bps, avg {sub['net5'].mean():+.1f}")

    print(f"\n  TS losers (went right, not far enough):")
    ts_l = df_l[df_l["reason5"] == "TS"]
    print(f"    Count: {len(ts_l)}, total: {ts_l['net5'].sum():+.1f} bps")
    if len(ts_l) > 0:
        print(f"    MFE percentiles: 25th={ts_l['mfe'].quantile(0.25):.1f}, 50th={ts_l['mfe'].quantile(0.5):.1f}, 75th={ts_l['mfe'].quantile(0.75):.1f}")
        print(f"    Net bps percentiles: 25th={ts_l['net5'].quantile(0.25):.1f}, 50th={ts_l['net5'].quantile(0.5):.1f}, 75th={ts_l['net5'].quantile(0.75):.1f}")

    # ============================================================
    print("\n" + "=" * 80)
    print(" 3. SENSITIVITY: DIFFERENT EARLY CUT THRESHOLDS")
    print("=" * 80)

    baseline_net = sum(d["real_net"] for d in all_data)
    print(f"  Baseline: {baseline_net:+.1f} bps")
    print(f"\n  {'Bar3<':>8s} {'Bar4<':>8s} {'Net bps':>10s} {'vs BL':>8s} {'Win%':>6s} {'#Cut':>5s}")

    best_net = 0
    best_config = ""
    for b3 in [3, 5, 8, 10, 12, 15]:
        for b4 in [5, 8, 10, 12, 15, 20]:
            total = 0
            wins = 0
            cuts = 0
            for d in all_data:
                net5, reason5, _ = sim_exp5(d["bars"], d["ts_init"], bar3_thresh=b3, bar4_thresh=b4)
                total += net5
                if net5 > 0:
                    wins += 1
                if reason5 in ("EARLY3", "EARLY4"):
                    cuts += 1
            diff = total - baseline_net
            if total > baseline_net:
                print(f"  {b3:8d} {b4:8d} {total:+10.1f} {diff:+8.1f} {wins/len(all_data)*100:5.1f}% {cuts:5d}")
                if total > best_net:
                    best_net = total
                    best_config = f"Bar3<{b3}, Bar4<{b4}"

    print(f"\n  Best: {best_config} = {best_net:+.1f} bps (+{best_net - baseline_net:.1f} vs BL)")

    # ============================================================
    print("\n" + "=" * 80)
    print(" 4. WHAT IF WE ADD BAR 2 AND BAR 5 CUTS?")
    print("=" * 80)

    print(f"\n  {'B2<':>5s} {'B3<':>5s} {'B4<':>5s} {'B5<':>5s} {'Net bps':>10s} {'vs BL':>8s} {'Win%':>6s}")
    for b2 in [0, 3, 5]:  # 0 = no bar 2 cut
        for b3 in [3, 5, 8]:
            for b4 in [5, 8, 12]:
                for b5 in [0, 8, 10, 12]:  # 0 = no bar 5 cut
                    total = 0
                    wins = 0
                    for d in all_data:
                        peak = 0
                        exited = False
                        net = 0
                        for b in d["bars"]:
                            bn = b["bar"]
                            peak = b["peak"]
                            pnl = b["pnl"]
                            ats = 8 if bn > 5 else d["ts_init"]
                            dd = peak - pnl
                            if dd >= ats and peak > 0:
                                net = peak - ats - FEES
                                exited = True
                                break
                            if bn == 2 and b2 > 0 and peak < b2:
                                net = pnl - FEES
                                exited = True
                                break
                            if bn == 3 and peak < b3:
                                net = pnl - FEES
                                exited = True
                                break
                            if bn == 4 and peak < b4:
                                net = pnl - FEES
                                exited = True
                                break
                            if bn == 5 and b5 > 0 and peak < b5:
                                net = pnl - FEES
                                exited = True
                                break
                            if bn >= 10:
                                net = pnl - FEES
                                exited = True
                                break
                        if not exited and d["bars"]:
                            net = d["bars"][-1]["pnl"] - FEES
                        total += net
                        if net > 0:
                            wins += 1
                    diff = total - baseline_net
                    if total > best_net:
                        b2s = f"{b2}" if b2 > 0 else "-"
                        b5s = f"{b5}" if b5 > 0 else "-"
                        print(f"  {b2s:>5s} {b3:5d} {b4:5d} {b5s:>5s} {total:+10.1f} {diff:+8.1f} {wins/len(all_data)*100:5.1f}%  ** NEW BEST")
                        best_net = total
                        best_config = f"B2<{b2s} B3<{b3} B4<{b4} B5<{b5s}"

    print(f"\n  Overall best: {best_config} = {best_net:+.1f} bps (+{best_net - baseline_net:.1f} vs BL)")


if __name__ == "__main__":
    main()
