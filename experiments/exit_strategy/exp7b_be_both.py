"""
Breakeven lock test on BOTH EXP-5 and EXP-6.
Best config from EXP-7: BE@9 gross, activate when MFE >= 15.
Test all combos to compare.
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

    # Pre-build trade data
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

        # Build 15-min bars
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

        # For EXP-6: find 1-min data at 45-min mark for asymmetric cut
        exp6_cut_pnl = None  # PnL at first 15-min boundary >= 45 min where losing + MFE < 5
        exp6_cut_bar = None
        if len(bars_15m) >= 3:
            # Check bar 3 onwards on 15-min boundaries
            for b in bars_15m[2:]:  # bar 3+
                if b["pnl"] < 0 and b["peak"] < 5:
                    exp6_cut_pnl = b["pnl"]
                    exp6_cut_bar = b["bar"]
                    break

        all_data.append({
            "entry_time": t.entry_time,
            "direction": d,
            "sig_type": t.signal_type,
            "ts_init": ts_init,
            "real_net": t.net_profit_bps,
            "bars": bars_15m,
            "exp6_cut_pnl": exp6_cut_pnl,
            "exp6_cut_bar": exp6_cut_bar,
        })

    n = len(all_data)
    bl_total = sum(d["real_net"] for d in all_data)
    print(f"Trades: {n}, Baseline: {bl_total:+.1f} bps")

    def sim_exp5_be(data, be_activation=0, be_floor=0):
        """EXP-5 (Bar3<3, Bar4<5) + breakeven lock."""
        be_active = False
        for b in data["bars"]:
            bn = b["bar"]
            peak = b["peak"]
            pnl = b["pnl"]
            if be_activation > 0 and peak >= be_activation:
                be_active = True
            ats = 8 if bn > 5 else data["ts_init"]
            dd = peak - pnl
            if dd >= ats and peak > 0:
                exit_gross = peak - ats
                if be_active and exit_gross < be_floor:
                    exit_gross = be_floor
                return exit_gross - FEES
            if be_active and pnl <= be_floor:
                return be_floor - FEES
            if bn == 3 and peak < 3:
                return pnl - FEES
            if bn == 4 and peak < 5:
                return pnl - FEES
            if bn >= 10:
                return pnl - FEES
        if data["bars"]:
            return data["bars"][-1]["pnl"] - FEES
        return -FEES

    def sim_exp6_be(data, be_activation=0, be_floor=0):
        """EXP-6 (asymmetric 1-min loser cut) + breakeven lock."""
        be_active = False
        cut_done = False
        for b in data["bars"]:
            bn = b["bar"]
            peak = b["peak"]
            pnl = b["pnl"]
            if be_activation > 0 and peak >= be_activation:
                be_active = True
            ats = 8 if bn > 5 else data["ts_init"]
            dd = peak - pnl
            if dd >= ats and peak > 0:
                exit_gross = peak - ats
                if be_active and exit_gross < be_floor:
                    exit_gross = be_floor
                return exit_gross - FEES
            if be_active and pnl <= be_floor:
                return be_floor - FEES
            # EXP-6 asymmetric cut: at bar 3+ if losing and MFE < 5
            if not cut_done and bn >= 3 and data["exp6_cut_pnl"] is not None and bn == data["exp6_cut_bar"]:
                cut_done = True
                return data["exp6_cut_pnl"] - FEES
            if bn >= 10:
                return pnl - FEES
        if data["bars"]:
            return data["bars"][-1]["pnl"] - FEES
        return -FEES

    def calc_stats(results):
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
        return len(r), wins, total, pf, mdd, r24, r25

    # Print all results
    print(f"\n{'='*100}")
    print(f" BREAKEVEN LOCK: EXP-5 vs EXP-6")
    print(f"{'='*100}")
    print(f"\n{'Config':>40s}  {'#':>5s} {'Win%':>6s} {'Net bps':>10s} {'PF':>6s} {'MaxDD':>8s} {'2024':>8s} {'2025':>8s} {'vs BL':>8s}")

    def print_row(label, results):
        n, wins, total, pf, mdd, r24, r25 = calc_stats(results)
        print(f"{label:>40s}  {n:5d} {wins/n*100:5.1f}% {total:+10.1f} {pf:6.2f} {-mdd:+8.1f} {r24:+8.0f} {r25:+8.0f} {total-bl_total:+8.1f}")

    # Baseline
    print_row("BASELINE (V1.3)", [d["real_net"] for d in all_data])

    # EXP-5 and EXP-6 without breakeven
    exp5_no_be = [sim_exp5_be(d) for d in all_data]
    exp6_no_be = [sim_exp6_be(d) for d in all_data]
    print_row("EXP-5 (no BE)", exp5_no_be)
    print_row("EXP-6 (no BE)", exp6_no_be)

    print()
    for be_floor in [0, 9]:
        for act in [15, 20, 25, 30]:
            exp5_r = [sim_exp5_be(d, be_activation=act, be_floor=be_floor) for d in all_data]
            exp6_r = [sim_exp6_be(d, be_activation=act, be_floor=be_floor) for d in all_data]
            print_row(f"EXP-5 + BE@{be_floor}g act>={act}", exp5_r)
            print_row(f"EXP-6 + BE@{be_floor}g act>={act}", exp6_r)
        print()


if __name__ == "__main__":
    main()
