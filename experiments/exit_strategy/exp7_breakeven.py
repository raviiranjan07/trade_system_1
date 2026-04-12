"""
EXP-7: Breakeven lock on top of EXP-5 (Bar3<3, Bar4<5).

Once MFE crosses activation threshold, trailing stop floor = breakeven level.
- Breakeven @ 0 bps gross (net = -8 after fees)
- Breakeven @ 9 bps gross (net = +1 after fees)
- Activation thresholds: MFE >= 15, 20, 25, 30 bps

Normal rules still apply:
- Bar 3: MFE < 3 → cut
- Bar 4: MFE < 5 → cut
- Trailing stop 20/30 bps
- Bar 5 tighten to 8 bps from peak
- Bar 10 time exit
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


def build_15m_bars(et, ep, direction, sl_t, sl_h, sl_l, sl_c):
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
        bars.append({"bar": bn, "peak": peak, "pnl": pnl})
    return bars


def sim_trade(bars, ts_init, bar3_thresh=3, bar4_thresh=5, be_activation=0, be_floor=0):
    """
    Simulate trade with EXP-5 early cuts + optional breakeven lock.
    be_activation: MFE threshold to activate breakeven lock (0 = disabled)
    be_floor: gross bps floor once activated (0 = breakeven, 9 = cover fees+1bp)
    """
    be_active = False

    for b in bars:
        bn = b["bar"]
        peak = b["peak"]
        pnl = b["pnl"]

        # Check if breakeven lock activates
        if be_activation > 0 and peak >= be_activation:
            be_active = True

        # Determine trailing stop
        if bn > 5:
            active_ts = 8  # tighten after bar 5
        else:
            active_ts = ts_init

        # Normal trailing stop check
        dd = peak - pnl
        if dd >= active_ts and peak > 0:
            exit_gross = peak - active_ts
            # Apply breakeven floor if active
            if be_active and exit_gross < be_floor:
                exit_gross = be_floor
            return exit_gross - FEES, "TS", bn

        # Breakeven floor check: if lock is active and pnl drops to floor
        if be_active and pnl <= be_floor:
            return be_floor - FEES, "BE_LOCK", bn

        # Early cut rules
        if bn == 3 and peak < bar3_thresh:
            return pnl - FEES, "EARLY3", bn
        if bn == 4 and peak < bar4_thresh:
            return pnl - FEES, "EARLY4", bn

        # Time exit
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

    # Pre-build all trade bars
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
            "real_mfe": t.mfe_bps,
            "bars": bars,
        })

    n = len(all_data)
    print(f"Trades: {n}")

    # Baseline (V1.3)
    bl_total = sum(d["real_net"] for d in all_data)

    # EXP-5 best (Bar3<3, Bar4<5, no breakeven)
    exp5_results = []
    for d in all_data:
        net, reason, bar = sim_trade(d["bars"], d["ts_init"])
        exp5_results.append(net)
    exp5_total = sum(exp5_results)

    # Print header
    print(f"\n{'='*90}")
    print(f" BREAKEVEN LOCK TEST (on top of EXP-5: Bar3<3, Bar4<5)")
    print(f"{'='*90}")

    print(f"\n{'Config':>35s}  {'#':>5s} {'Win%':>6s} {'Net bps':>10s} {'PF':>6s} {'MaxDD':>8s} {'vs BL':>8s} {'vs E5':>8s}")

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
        print(f"{label:>35s}  {len(r):5d} {wins/len(r)*100:5.1f}% {total:+10.1f} {pf:6.2f} {-mdd:+8.1f} {total-bl_total:+8.1f} {total-exp5_total:+8.1f}")

    print_row("BASELINE (V1.3)", [d["real_net"] for d in all_data])
    print_row("EXP-5 (Bar3<3, Bar4<5)", exp5_results)

    # Test all combinations
    best_total = exp5_total
    best_label = "EXP-5"
    all_configs = []

    for be_floor in [0, 9]:
        for activation in [15, 20, 25, 30]:
            results = []
            for d in all_data:
                net, reason, bar = sim_trade(
                    d["bars"], d["ts_init"],
                    be_activation=activation, be_floor=be_floor
                )
                results.append(net)
            total = sum(results)
            label = f"BE@{be_floor}g act>={activation}"
            print_row(label, results)
            all_configs.append((label, results, total))
            if total > best_total:
                best_total = total
                best_label = label

    print(f"\n  Best: {best_label} = {best_total:+.1f} bps")

    # Detailed breakdown of best config
    print(f"\n{'='*90}")
    print(f" DETAIL: {best_label}")
    print(f"{'='*90}")

    # Find best config params
    best_floor = 0
    best_act = 15
    for be_floor in [0, 9]:
        for activation in [15, 20, 25, 30]:
            results = []
            for d in all_data:
                net, _, _ = sim_trade(d["bars"], d["ts_init"], be_activation=activation, be_floor=be_floor)
                results.append(net)
            if sum(results) == best_total:
                best_floor = be_floor
                best_act = activation
                break

    # Run best config with details
    detail_results = []
    for d in all_data:
        net, reason, bar = sim_trade(d["bars"], d["ts_init"], be_activation=best_act, be_floor=best_floor)
        detail_results.append({
            "net": net, "reason": reason, "bar": bar,
            "sig_type": d["sig_type"], "direction": d["direction"],
            "real_net": d["real_net"], "mfe": d["real_mfe"],
            "year": d["entry_time"].year,
        })
    df_det = pd.DataFrame(detail_results)

    # By exit reason
    print(f"\n  By exit reason:")
    for reason in sorted(df_det["reason"].unique()):
        sub = df_det[df_det["reason"] == reason]
        w = (sub["net"] > 0).sum()
        bl = sub["real_net"].sum()
        e7 = sub["net"].sum()
        print(f"    {reason:>10s}: {len(sub):4d}t ({w:4d}W)  BL={bl:+9.1f}  E7={e7:+9.1f}  diff={e7-bl:+.1f}")

    # By signal type
    print(f"\n  By signal type:")
    for sig in sorted(df_det["sig_type"].unique()):
        sub = df_det[df_det["sig_type"] == sig]
        bl = sub["real_net"].sum()
        e7 = sub["net"].sum()
        print(f"    {sig:>12s} ({len(sub):4d}t): BL={bl:+9.1f}  E7={e7:+9.1f}  diff={e7-bl:+.1f}")

    # By year
    print(f"\n  By year:")
    for year in [2024, 2025]:
        sub = df_det[df_det["year"] == year]
        bl = sub["real_net"].sum()
        e7 = sub["net"].sum()
        w = (sub["net"] > 0).sum()
        print(f"    {year}: {len(sub)}t {w}W  BL={bl:+.1f}  E7={e7:+.1f}  diff={e7-bl:+.1f}")

    # Winners/losers summary
    winners = df_det[df_det["net"] > 0]
    losers = df_det[df_det["net"] <= 0]
    print(f"\n  Winners: {len(winners)}t, +{winners['net'].sum():.1f} bps")
    print(f"  Losers:  {len(losers)}t, {losers['net'].sum():.1f} bps")

    # How many trades did BE_LOCK save from being losers?
    be_trades = df_det[df_det["reason"] == "BE_LOCK"]
    if len(be_trades) > 0:
        bl_net_of_be = be_trades["real_net"].values
        be_net = be_trades["net"].values
        saved_from_loss = ((bl_net_of_be <= 0) & (be_net > 0)).sum()
        made_worse = ((bl_net_of_be > 0) & (be_net <= bl_net_of_be)).sum()
        print(f"\n  BE_LOCK trades: {len(be_trades)}")
        print(f"    Was BL loser, now winner: {saved_from_loss}")
        print(f"    Was BL winner, made worse: {made_worse}")
        print(f"    BL total: {bl_net_of_be.sum():+.1f}  E7 total: {be_net.sum():+.1f}")


if __name__ == "__main__":
    main()
