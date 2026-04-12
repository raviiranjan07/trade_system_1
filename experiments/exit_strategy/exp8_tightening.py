"""
EXP-8: Tightening sensitivity on top of EXP-5 + BE@9g act>=15.
Test tighten_bar (4,5,6,7) x tighten_bps (6,8,10,12,15).
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

    # Pre-build 15-min bars
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

    def sim_trade(data, tighten_bar=5, tighten_bps=8, be_activation=15, be_floor=9):
        """EXP-5 + BE@9 act>=15 + configurable tightening."""
        be_active = False
        for b in data["bars"]:
            bn = b["bar"]
            peak = b["peak"]
            pnl = b["pnl"]

            if peak >= be_activation:
                be_active = True

            # Tightening
            if bn > tighten_bar:
                ats = tighten_bps
            else:
                ats = data["ts_init"]

            # Trailing stop
            dd = peak - pnl
            if dd >= ats and peak > 0:
                exit_gross = peak - ats
                if be_active and exit_gross < be_floor:
                    exit_gross = be_floor
                return exit_gross - FEES

            # Breakeven lock
            if be_active and pnl <= be_floor:
                return be_floor - FEES

            # Early cuts
            if bn == 3 and peak < 3:
                return pnl - FEES
            if bn == 4 and peak < 5:
                return pnl - FEES

            # Time exit
            if bn >= 10:
                return pnl - FEES

        if data["bars"]:
            return data["bars"][-1]["pnl"] - FEES
        return -FEES

    # Print header
    print(f"\n{'='*100}")
    print(f" TIGHTENING SENSITIVITY (on top of EXP-5 + BE@9g act>=15)")
    print(f"{'='*100}")
    print(f"\n{'Config':>25s}  {'#':>5s} {'Win%':>6s} {'Net bps':>10s} {'PF':>6s} {'MaxDD':>8s} {'2024':>8s} {'2025':>8s} {'vs BL':>8s}")

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
        print(f"{label:>25s}  {len(r):5d} {wins/len(r)*100:5.1f}% {total:+10.1f} {pf:6.2f} {-mdd:+8.1f} {r24:+8.0f} {r25:+8.0f} {total-bl_total:+8.1f}")

    # Baseline and current best
    print_row("BASELINE (V1.3)", [d["real_net"] for d in all_data])
    current = [sim_trade(d) for d in all_data]
    print_row("Current (bar5, 8bps)", current)

    # Grid search
    print()
    best_total = sum(current)
    best_label = "Current"

    for tighten_bar in [4, 5, 6, 7]:
        for tighten_bps in [6, 8, 10, 12, 15]:
            results = [sim_trade(d, tighten_bar=tighten_bar, tighten_bps=tighten_bps) for d in all_data]
            label = f"bar{tighten_bar}, {tighten_bps}bps"
            total = sum(results)
            print_row(label, results)
            if total > best_total:
                best_total = total
                best_label = label

    print(f"\n  Best: {best_label} = {best_total:+.1f} bps (+{best_total-bl_total:.1f} vs BL)")

    # Also test: no tightening at all
    print(f"\n  --- No tightening (keep 20/30 full duration) ---")
    results_no = [sim_trade(d, tighten_bar=99, tighten_bps=99) for d in all_data]
    print_row("No tightening", results_no)


if __name__ == "__main__":
    main()
