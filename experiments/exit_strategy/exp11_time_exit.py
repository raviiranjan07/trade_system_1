"""
EXP-11: Time exit sensitivity on top of EXP-5 V2 frozen config.
Test max bars: 7, 8, 9, 10, 12, 15.
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

    # Build 15-min bars (up to 15 bars = 225 min)
    all_data = []
    for t in oos:
        et = np.datetime64(t.entry_time)
        mt = et + np.timedelta64(225, "m")  # 15 bars max
        s = np.searchsorted(idx_1m, et, side="left")
        e = np.searchsorted(idx_1m, mt, side="right")
        sl_t = idx_1m[s:e]
        sl_h = highs_1m[s:e]
        sl_l = lows_1m[s:e]
        sl_c = closes_1m[s:e]

        ep = t.entry_price
        d = t.direction
        ts_init = 20 if d == "LONG" else 30

        bars = []
        peak = 0
        for bn in range(1, 16):  # up to 15 bars
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
            bars.append({"bar": bn, "peak": peak, "pnl": pnl})

        all_data.append({
            "entry_time": t.entry_time,
            "direction": d,
            "ts_init": ts_init,
            "real_net": t.net_profit_bps,
            "bars": bars,
        })

    n = len(all_data)
    bl_total = sum(d["real_net"] for d in all_data)
    print(f"Trades: {n}, Baseline: {bl_total:+.1f} bps")

    def sim_trade(bars, ts_init, max_bars=10):
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
                return exit_gross - FEES, "TS"
            if be_active and pnl <= 9:
                return 9 - FEES, "BE"
            if bn == 3 and peak < 3:
                return pnl - FEES, "EARLY3"
            if bn == 4 and peak < 5:
                return pnl - FEES, "EARLY4"
            if bn >= max_bars:
                return pnl - FEES, "TIME"
        if bars:
            return bars[-1]["pnl"] - FEES, "TIME"
        return -FEES, "NONE"

    print(f"\n{'='*100}")
    print(f" TIME EXIT SENSITIVITY (on top of EXP-5 V2)")
    print(f"{'='*100}")
    print(f"\n{'Config':>20s}  {'#':>5s} {'Win%':>6s} {'Net bps':>10s} {'PF':>6s} {'MaxDD':>8s} {'2024':>8s} {'2025':>8s} {'vs BL':>8s} {'TIME#':>6s} {'TIME bps':>9s}")

    def print_row(label, results, reasons):
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
        time_count = sum(1 for re in reasons if re == "TIME")
        time_bps = sum(results[i] for i in range(len(reasons)) if reasons[i] == "TIME")
        print(f"{label:>20s}  {len(r):5d} {wins/len(r)*100:5.1f}% {total:+10.1f} {pf:6.2f} {-mdd:+8.1f} {r24:+8.0f} {r25:+8.0f} {total-bl_total:+8.1f} {time_count:6d} {time_bps:+9.1f}")

    # Baseline
    print_row("BASELINE (V1.3)", [d["real_net"] for d in all_data], ["" for _ in all_data])

    for max_bars in [7, 8, 9, 10, 12, 15]:
        results = []
        reasons = []
        for d in all_data:
            net, reason = sim_trade(d["bars"], d["ts_init"], max_bars=max_bars)
            results.append(net)
            reasons.append(reason)
        label = f"Max bar {max_bars}"
        print_row(label, results, reasons)


if __name__ == "__main__":
    main()
