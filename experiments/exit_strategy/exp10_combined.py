"""
EXP-10: EXP-5 V2 + 1-min monitoring when profitable AND MFE >= threshold.

Uses 15-min bars for baseline mode (matches exp8 exactly).
Switches to 1-min when conditions met.
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

    # Build trade data with both 15-min bars and 1-min slices
    all_data = []
    for t in oos:
        et = np.datetime64(t.entry_time)
        mt = et + np.timedelta64(150, "m")
        s = np.searchsorted(idx_1m, et, side="left")
        e = np.searchsorted(idx_1m, mt, side="right")
        sl_t = idx_1m[s:e]
        sl_h = highs_1m[s:e]
        sl_l = lows_1m[s:e]
        sl_c = closes_1m[s:e]

        ep = t.entry_price
        d = t.direction
        ts_init = 20 if d == "LONG" else 30

        # Build 15-min bars (same as exp8)
        bars_15m = []
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
            if d == "LONG":
                mfe = (h - ep) / ep * 10000
                pnl = (c - ep) / ep * 10000
            else:
                mfe = (ep - l) / ep * 10000
                pnl = (ep - c) / ep * 10000
            if mfe > peak:
                peak = mfe
            bars_15m.append({"bar": bn, "peak": peak, "pnl": pnl})

        all_data.append({
            "entry_time": t.entry_time,
            "entry_price": ep,
            "direction": d,
            "sig_type": t.signal_type,
            "ts_init": ts_init,
            "real_net": t.net_profit_bps,
            "bars_15m": bars_15m,
            "sl_t": sl_t,
            "sl_h": sl_h,
            "sl_l": sl_l,
            "sl_c": sl_c,
            "et": et,
        })

    n = len(all_data)
    bl_total = sum(d["real_net"] for d in all_data)
    print(f"Trades: {n}, Baseline: {bl_total:+.1f} bps")

    def sim_exp5_v2(bars, ts_init):
        """Pure EXP-5 V2 on 15-min bars (matches exp8 exactly)."""
        be_active = False
        for b in bars:
            bn = b["bar"]
            peak = b["peak"]
            pnl = b["pnl"]
            if peak >= 15:
                be_active = True
            ats = 6 if bn > 4 else ts_init
            dd = peak - pnl
            if dd >= ats and peak > 0:
                exit_gross = peak - ats
                if be_active and exit_gross < 9:
                    exit_gross = 9
                return exit_gross - FEES
            if be_active and pnl <= 9:
                return 9 - FEES
            if bn == 3 and peak < 3:
                return pnl - FEES
            if bn == 4 and peak < 5:
                return pnl - FEES
            if bn >= 10:
                return pnl - FEES
        if bars:
            return bars[-1]["pnl"] - FEES
        return -FEES

    def sim_exp10(data, mfe_switch):
        """EXP-5 V2 on 15-min bars, switch to 1-min when profitable AND MFE >= threshold."""
        bars = data["bars_15m"]
        ts_init = data["ts_init"]
        ep = data["entry_price"]
        d = data["direction"]
        et = data["et"]

        be_active = False
        trade_profitable = False

        for b in bars:
            bn = b["bar"]
            peak = b["peak"]
            pnl = b["pnl"]

            if peak >= 15:
                be_active = True
            if pnl > 0:
                trade_profitable = True

            # Check if we should switch to 1-min from this bar onwards
            if mfe_switch > 0 and peak >= mfe_switch and trade_profitable:
                # Switch to 1-min monitoring from current bar's start
                bar_start = et + np.timedelta64((bn - 1) * 15, "m")
                sl_t = data["sl_t"]
                sl_h = data["sl_h"]
                sl_l = data["sl_l"]
                sl_c = data["sl_c"]

                # Process remaining time on 1-min
                for j in range(len(sl_t)):
                    t = sl_t[j]
                    if t < bar_start:
                        continue
                    h, l, c = sl_h[j], sl_l[j], sl_c[j]
                    if d == "LONG":
                        bb = (h - ep) / ep * 10000
                        wp = (l - ep) / ep * 10000
                        cp = (c - ep) / ep * 10000
                    else:
                        bb = (ep - l) / ep * 10000
                        wp = (ep - h) / ep * 10000
                        cp = (ep - c) / ep * 10000
                    if bb > peak:
                        peak = bb
                    if peak >= 15:
                        be_active = True

                    mins = (t - et) / np.timedelta64(1, "m")
                    cur_bar = int(mins / 15) + 1

                    # 1-min high/low monitoring with 6 bps TS
                    dd = peak - wp
                    if dd >= 6 and peak > 0:
                        exit_gross = peak - 6
                        if be_active and exit_gross < 9:
                            exit_gross = 9
                        return exit_gross - FEES

                    # BE lock on 1-min
                    if be_active and wp <= 9:
                        return 9 - FEES

                    # Time exit at bar 10 (check at 15-min boundaries)
                    if cur_bar >= 10 and mins >= 149:
                        return cp - FEES

                # Fallback
                if len(sl_c) > 0:
                    if d == "LONG":
                        return (sl_c[-1] - ep) / ep * 10000 - FEES
                    else:
                        return (ep - sl_c[-1]) / ep * 10000 - FEES
                return -FEES

            # Normal EXP-5 V2 logic on 15-min
            ats = 6 if bn > 4 else ts_init
            dd = peak - pnl
            if dd >= ats and peak > 0:
                exit_gross = peak - ats
                if be_active and exit_gross < 9:
                    exit_gross = 9
                return exit_gross - FEES
            if be_active and pnl <= 9:
                return 9 - FEES
            if bn == 3 and peak < 3:
                return pnl - FEES
            if bn == 4 and peak < 5:
                return pnl - FEES
            if bn >= 10:
                return pnl - FEES

        if bars:
            return bars[-1]["pnl"] - FEES
        return -FEES

    # Print results
    print(f"\n{'='*100}")
    print(f" EXP-10: EXP-5 V2 + 1-MIN MONITORING WHEN PROFITABLE")
    print(f"{'='*100}")
    print(f"\n{'Config':>35s}  {'#':>5s} {'Win%':>6s} {'Net bps':>10s} {'PF':>6s} {'MaxDD':>8s} {'2024':>8s} {'2025':>8s} {'vs BL':>8s}")

    def print_row(label, results):
        r = np.array(results)
        wins = (r > 0).sum()
        total = r.sum()
        gw = r[r > 0].sum()
        gl = abs(r[r <= 0].sum())
        pf = gw / gl if gl > 0 else 999
        cum = np.cumsum(r)
        pk = np.maximum.accumulate(cum)
        mdd = (pk - cum).max()
        r24 = sum(results[i] for i, d in enumerate(all_data) if d["entry_time"].year == 2024)
        r25 = sum(results[i] for i, d in enumerate(all_data) if d["entry_time"].year == 2025)
        print(f"{label:>35s}  {len(r):5d} {wins/len(r)*100:5.1f}% {total:+10.1f} {pf:6.2f} {-mdd:+8.1f} {r24:+8.0f} {r25:+8.0f} {total-bl_total:+8.1f}")

    # Baseline
    print_row("BASELINE (V1.3)", [d["real_net"] for d in all_data])

    # EXP-5 V2 (verify matches +33,062)
    exp5_results = [sim_exp5_v2(d["bars_15m"], d["ts_init"]) for d in all_data]
    exp5_total = sum(exp5_results)
    print_row("EXP-5 V2 (frozen)", exp5_results)

    # EXP-10 with different MFE thresholds
    for mfe in [25, 35, 40]:
        results = [sim_exp10(d, mfe_switch=mfe) for d in all_data]
        label = f"EXP-10 MFE>={mfe} (1-min 6bps)"
        print_row(label, results)

    # Trade-by-trade comparison
    print(f"\n  --- Trade-by-trade vs EXP-5 V2 ---")
    for mfe in [25, 35, 40]:
        results = [sim_exp10(d, mfe_switch=mfe) for d in all_data]
        diff = [results[i] - exp5_results[i] for i in range(n)]
        better = sum(1 for d in diff if d > 0.5)
        worse = sum(1 for d in diff if d < -0.5)
        same = n - better - worse
        print(f"    MFE>={mfe}: Better {better:4d}t, Worse {worse:4d}t, Same {same:4d}t, net diff {sum(diff):+.1f}")


if __name__ == "__main__":
    main()
