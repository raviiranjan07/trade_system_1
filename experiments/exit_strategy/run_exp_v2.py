"""
Exit Strategy Experiments V2 — Same 220 entries, different exit logic.

EXP-1: 1-min high/low always. 20/30 TS. Bar 5 tighten to 8. Bar 10 exit.
EXP-2: Baseline until MFE >= 25 bps, then 1-min high/low with 8 bps TS. Bar 10 exit.
EXP-3: Losing -> baseline. Profitable -> EXP-1 rules.
EXP-4: Losing -> baseline. Profitable AND MFE >= 25 -> 1-min high/low with 8 bps TS. Else baseline.
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import gc
import sys
sys.path.insert(0, "src")

from engine.config.loader import load_config
from engine.strategy import V12Strategy, Direction, SignalType

FEES = 8
DATA_15M = "data/raw/BTCUSDT_15m_ohlcv.parquet"
DATA_1M = "data/raw/BTCUSDT_1m_ohlcv.parquet"


def main():
    print("Loading data...")
    config = load_config()
    df_15m = pd.read_parquet(DATA_15M)
    if df_15m.index.tz is not None:
        df_15m.index = df_15m.index.tz_convert(None)

    strategy = V12Strategy(config)
    df_15m = strategy.compute_indicators(df_15m)
    test = df_15m["2024-01-01":"2025-12-31"]
    signals = strategy.generate_signals(test)

    gc.collect()
    pf = pq.ParquetFile(DATA_1M)
    chunks = []
    for batch in pf.iter_batches(batch_size=100_000):
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

    # Get baseline 220 entries (same signal blocking as baseline)
    entries = []
    i = 0
    pos = None
    while i < n:
        if pos is not None:
            pos["bh"] += 1
            h, l, c = highs_15[i], lows_15[i], closes_15[i]
            if pos["d"] == "LONG":
                mfe = (h - pos["ep"]) / pos["ep"] * 10000
                pnl = (c - pos["ep"]) / pos["ep"] * 10000
            else:
                mfe = (pos["ep"] - l) / pos["ep"] * 10000
                pnl = (pos["ep"] - c) / pos["ep"] * 10000
            if mfe > pos["pk"]:
                pos["pk"] = mfe
            ats = 8 if pos["bh"] > 5 else pos["ts"]
            dd = pos["pk"] - pnl
            if (dd >= ats and pos["pk"] > 0) or pos["bh"] >= 10:
                pos = None
            i += 1
            continue
        if i in signal_map:
            sig = signal_map[i]
            if sig.signal_type in (SignalType.V12_LONG, SignalType.BULL_SHORT):
                ok = bull[i]
            else:
                ok = bear[i]
            if ok and i + 1 < n:
                ts = 20 if sig.direction == Direction.LONG else 30
                entries.append((i + 1, times_15[i + 1], opens[i + 1], sig.direction.value, ts))
                pos = {"d": sig.direction.value, "ep": opens[i + 1], "ts": ts, "bh": 0, "pk": 0.0}
                i += 2
                continue
        i += 1

    print(f"Entries: {len(entries)}")

    def run_baseline(entry_bar, entry_price, direction, ts_init):
        peak = 0.0
        for bar in range(1, 11):
            idx = entry_bar + bar
            if idx >= n:
                break
            h, l, c = highs_15[idx], lows_15[idx], closes_15[idx]
            if direction == "LONG":
                mfe = (h - entry_price) / entry_price * 10000
                pnl = (c - entry_price) / entry_price * 10000
            else:
                mfe = (entry_price - l) / entry_price * 10000
                pnl = (entry_price - c) / entry_price * 10000
            if mfe > peak:
                peak = mfe
            ats = 8 if bar > 5 else ts_init
            dd = peak - pnl
            if dd >= ats and peak > 0:
                return peak - ats - FEES
            if bar == 10:
                return pnl - FEES
        return -FEES

    def run_1min_exit(entry_bar, entry_time, entry_price, direction, ts_init, mode, mfe_threshold=25):
        et = np.datetime64(entry_time)
        mt = et + np.timedelta64(150, "m")
        s = np.searchsorted(idx_1m, et, side="left")
        e = np.searchsorted(idx_1m, mt, side="right")
        sl_t = idx_1m[s:e]
        sl_h = highs_1m[s:e]
        sl_l = lows_1m[s:e]
        sl_c = closes_1m[s:e]

        peak = 0.0
        trade_profitable = False
        mfe_crossed = False

        for j in range(len(sl_t)):
            t = sl_t[j]
            h, l, c = sl_h[j], sl_l[j], sl_c[j]

            if direction == "LONG":
                bar_best = (h - entry_price) / entry_price * 10000
                close_pnl = (c - entry_price) / entry_price * 10000
                worst_pnl = (l - entry_price) / entry_price * 10000
            else:
                bar_best = (entry_price - l) / entry_price * 10000
                close_pnl = (entry_price - c) / entry_price * 10000
                worst_pnl = (entry_price - h) / entry_price * 10000

            if bar_best > peak:
                peak = bar_best
            if peak >= mfe_threshold:
                mfe_crossed = True

            mins = (t - et) / np.timedelta64(1, "m")
            b15 = mins / 15
            is15 = mins > 0 and mins % 15 < 1.5

            # Decide: use baseline (15-min close) or 1-min high/low this minute?
            use_baseline = False

            if mode == "EXP-1":
                # Always 1-min high/low
                use_baseline = False

            elif mode == "EXP-2":
                # Baseline until MFE >= 25, then 1-min with 8bps
                use_baseline = not mfe_crossed

            elif mode == "EXP-3":
                # Losing -> baseline. Profitable -> EXP-1 (1-min, bar-count tighten)
                if not trade_profitable:
                    use_baseline = True
                    if is15 and close_pnl > 0:
                        trade_profitable = True
                else:
                    use_baseline = False

            elif mode == "EXP-4":
                # Losing -> baseline. Profitable AND MFE>=25 -> 1-min 8bps. Else baseline.
                if not trade_profitable:
                    use_baseline = True
                    if is15 and close_pnl > 0:
                        trade_profitable = True
                elif not mfe_crossed:
                    use_baseline = True
                else:
                    use_baseline = False

            if use_baseline:
                # Check only at 15-min boundaries
                if is15:
                    ats = 8 if b15 > 5 else ts_init
                    dd = peak - close_pnl
                    if dd >= ats and peak > 0:
                        return peak - ats - FEES
                    if b15 >= 10:
                        return close_pnl - FEES
                continue

            # 1-min high/low monitoring
            if mode in ("EXP-1", "EXP-3"):
                # Bar-count tightening
                ats = 8 if b15 > 5 else ts_init
            elif mode in ("EXP-2", "EXP-4"):
                # Only get here when MFE >= 25 -> always 8 bps
                ats = 8

            dd = peak - worst_pnl
            if dd >= ats and peak > 0:
                return peak - ats - FEES

            # Time exit at bar 10
            if b15 >= 10 and is15:
                return close_pnl - FEES

        # Fallback: use last available close
        if len(sl_c) > 0:
            if direction == "LONG":
                return (sl_c[-1] - entry_price) / entry_price * 10000 - FEES
            else:
                return (entry_price - sl_c[-1]) / entry_price * 10000 - FEES
        return -FEES

    # Run all modes on same 220 entries, with multiple MFE thresholds
    thresholds = [25, 35, 40]

    # Baseline (only once)
    bl_results = []
    for entry_bar, entry_time, entry_price, direction, ts_init in entries:
        bl_results.append(run_baseline(entry_bar, entry_price, direction, ts_init))

    def print_stats(name, r):
        wins = sum(1 for x in r if x > 0)
        total = sum(r)
        gw = sum(x for x in r if x > 0)
        gl = abs(sum(x for x in r if x <= 0))
        pf_val = gw / gl if gl > 0 else 999
        cum = 0; pk = 0; mdd = 0
        for x in r:
            cum += x
            if cum > pk: pk = cum
            dd = pk - cum
            if dd > mdd: mdd = dd
        r24 = [r[i] for i, e in enumerate(entries) if e[1].year == 2024]
        r25 = [r[i] for i, e in enumerate(entries) if e[1].year == 2025]
        better = sum(1 for i in range(len(r)) if r[i] > bl_results[i])
        worse = sum(1 for i in range(len(r)) if r[i] < bl_results[i])
        same = sum(1 for i in range(len(r)) if abs(r[i] - bl_results[i]) < 0.1)
        print(f"  {name:>30s}  {len(r):4d}  {wins/len(r)*100:5.1f}%  {total:+9.1f}  {pf_val:6.2f}  {-mdd:+8.1f}  {total/len(r):+7.1f}  {sum(r24):+8.0f}  {sum(r25):+8.0f}  {better:>3d}/{worse:>3d}/{same:>3d}")

    print(f"\n  {'':>30s}  {'#':>4s}  {'Win%':>6s}  {'Net bps':>9s}  {'PF':>6s}  {'MaxDD':>8s}  {'Avg/T':>7s}  {'2024':>8s}  {'2025':>8s}  {'B/W/S':>11s}")
    print_stats("BASELINE", bl_results)

    for thresh in thresholds:
        print(f"\n  --- MFE threshold: {thresh} bps ({thresh/100:.2f}%) ---")
        for mode in ["EXP-2", "EXP-4"]:
            r = []
            for entry_bar, entry_time, entry_price, direction, ts_init in entries:
                r.append(run_1min_exit(entry_bar, entry_time, entry_price, direction, ts_init, mode, mfe_threshold=thresh))
            label = f"{mode} (MFE>={thresh})"
            print_stats(label, r)

    # Also run EXP-1 and EXP-3 (no MFE threshold dependency)
    print(f"\n  --- No MFE threshold (always 1-min) ---")
    for mode in ["EXP-1", "EXP-3"]:
        r = []
        for entry_bar, entry_time, entry_price, direction, ts_init in entries:
            r.append(run_1min_exit(entry_bar, entry_time, entry_price, direction, ts_init, mode))
        print_stats(mode, r)


if __name__ == "__main__":
    main()
