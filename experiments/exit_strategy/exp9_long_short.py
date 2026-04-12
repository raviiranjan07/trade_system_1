"""
EXP-9: LONG vs SHORT separate early cut thresholds.
On top of best config: BE@9g act>=15, tighten bar4 to 6bps.

Test separate Bar3/Bar4 thresholds for LONG and SHORT.
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
            "sig_type": t.signal_type,
            "ts_init": ts_init,
            "real_net": t.net_profit_bps,
            "bars": bars,
        })

    n = len(all_data)
    bl_total = sum(d["real_net"] for d in all_data)
    n_long = sum(1 for d in all_data if d["direction"] == "LONG")
    n_short = sum(1 for d in all_data if d["direction"] == "SHORT")
    print(f"Trades: {n} (LONG: {n_long}, SHORT: {n_short}), Baseline: {bl_total:+.1f} bps")

    def sim_trade(data, long_b3=3, long_b4=5, short_b3=3, short_b4=5):
        """Best config + separate LONG/SHORT early cuts."""
        be_active = False
        is_long = data["direction"] == "LONG"
        b3_thresh = long_b3 if is_long else short_b3
        b4_thresh = long_b4 if is_long else short_b4

        for b in data["bars"]:
            bn = b["bar"]
            peak = b["peak"]
            pnl = b["pnl"]

            if peak >= 15:
                be_active = True

            # Tighten at bar 4 to 6 bps
            ats = 6 if bn > 4 else data["ts_init"]

            # Trailing stop
            dd = peak - pnl
            if dd >= ats and peak > 0:
                exit_gross = peak - ats
                if be_active and exit_gross < 9:
                    exit_gross = 9
                return exit_gross - FEES

            # Breakeven lock
            if be_active and pnl <= 9:
                return 9 - FEES

            # Early cuts
            if bn == 3 and peak < b3_thresh:
                return pnl - FEES
            if bn == 4 and peak < b4_thresh:
                return pnl - FEES

            # Time exit
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
        return n, wins, total, pf, mdd

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

    # Current best
    print(f"\n{'='*100}")
    print(f" LONG vs SHORT EARLY CUT THRESHOLDS")
    print(f"{'='*100}")
    print(f"\n{'Config':>35s}  {'#':>5s} {'Win%':>6s} {'Net bps':>10s} {'PF':>6s} {'MaxDD':>8s} {'2024':>8s} {'2025':>8s} {'vs BL':>8s}")

    print_row("BASELINE (V1.3)", [d["real_net"] for d in all_data])
    current = [sim_trade(d) for d in all_data]
    print_row("Current (L=S: b3<3,b4<5)", current)
    current_total = sum(current)

    # First: analyze LONG and SHORT separately with current config
    print(f"\n  --- Current config split ---")
    long_results = [(i, sim_trade(d)) for i, d in enumerate(all_data) if d["direction"] == "LONG"]
    short_results = [(i, sim_trade(d)) for i, d in enumerate(all_data) if d["direction"] == "SHORT"]
    long_nets = [r for _, r in long_results]
    short_nets = [r for _, r in short_results]
    long_w = sum(1 for r in long_nets if r > 0)
    short_w = sum(1 for r in short_nets if r > 0)
    print(f"  LONG:  {len(long_nets)}t, {long_w}W ({long_w/len(long_nets)*100:.1f}%), {sum(long_nets):+.1f} bps")
    print(f"  SHORT: {len(short_nets)}t, {short_w}W ({short_w/len(short_nets)*100:.1f}%), {sum(short_nets):+.1f} bps")

    # Grid search: separate LONG and SHORT thresholds
    print(f"\n  --- Grid search ---")
    print(f"  Testing LONG b3/b4 x SHORT b3/b4...")

    best_total = current_total
    best_label = "Current"
    best_config = None

    thresholds = [0, 3, 5, 8, 10]  # 0 = no early cut for that bar

    count = 0
    for lb3 in thresholds:
        for lb4 in thresholds:
            for sb3 in thresholds:
                for sb4 in thresholds:
                    results = [sim_trade(d, long_b3=lb3, long_b4=lb4, short_b3=sb3, short_b4=sb4) for d in all_data]
                    total = sum(results)
                    count += 1
                    if total > best_total:
                        best_total = total
                        best_label = f"L(b3<{lb3},b4<{lb4}) S(b3<{sb3},b4<{sb4})"
                        best_config = (lb3, lb4, sb3, sb4)

    print(f"  Tested {count} combinations")
    print(f"\n  Best: {best_label} = {best_total:+.1f} bps (+{best_total-bl_total:.1f} vs BL, +{best_total-current_total:.1f} vs current)")

    # Print best config details
    if best_config:
        lb3, lb4, sb3, sb4 = best_config
        results = [sim_trade(d, long_b3=lb3, long_b4=lb4, short_b3=sb3, short_b4=sb4) for d in all_data]
        print(f"\n{'Config':>35s}  {'#':>5s} {'Win%':>6s} {'Net bps':>10s} {'PF':>6s} {'MaxDD':>8s} {'2024':>8s} {'2025':>8s} {'vs BL':>8s}")
        print_row("BASELINE", [d["real_net"] for d in all_data])
        print_row("Current (L=S)", current)
        print_row(f"Best: {best_label}", results)

        # LONG/SHORT split of best
        long_r = [results[i] for i, d in enumerate(all_data) if d["direction"] == "LONG"]
        short_r = [results[i] for i, d in enumerate(all_data) if d["direction"] == "SHORT"]
        long_bl = [d["real_net"] for d in all_data if d["direction"] == "LONG"]
        short_bl = [d["real_net"] for d in all_data if d["direction"] == "SHORT"]
        print(f"\n  LONG:  {len(long_r)}t  BL={sum(long_bl):+.1f}  Current={sum(long_nets):+.1f}  Best={sum(long_r):+.1f}  diff={sum(long_r)-sum(long_nets):+.1f}")
        print(f"  SHORT: {len(short_r)}t  BL={sum(short_bl):+.1f}  Current={sum(short_nets):+.1f}  Best={sum(short_r):+.1f}  diff={sum(short_r)-sum(short_nets):+.1f}")

    # Also show top 5 configs
    print(f"\n  --- Top 10 configs ---")
    all_results = []
    for lb3 in thresholds:
        for lb4 in thresholds:
            for sb3 in thresholds:
                for sb4 in thresholds:
                    results = [sim_trade(d, long_b3=lb3, long_b4=lb4, short_b3=sb3, short_b4=sb4) for d in all_data]
                    total = sum(results)
                    all_results.append((total, lb3, lb4, sb3, sb4))

    all_results.sort(reverse=True)
    print(f"  {'L_b3':>5s} {'L_b4':>5s} {'S_b3':>5s} {'S_b4':>5s} {'Net bps':>10s} {'vs BL':>8s} {'vs cur':>8s}")
    for total, lb3, lb4, sb3, sb4 in all_results[:10]:
        print(f"  {lb3:5d} {lb4:5d} {sb3:5d} {sb4:5d} {total:+10.1f} {total-bl_total:+8.1f} {total-current_total:+8.1f}")


if __name__ == "__main__":
    main()
