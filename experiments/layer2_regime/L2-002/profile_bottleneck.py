"""Profile WHERE time is spent in run_backtest_fast() — find the real bottleneck."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "L2-001"))

import yaml, numpy as np, pandas as pd
from framework import (
    FRAMEWORK_CONFIG_PATH, generate_gates, generate_signals, generate_exits,
    RegimeClassifier, precompute_all_trades, run_backtest_fast, apply_gate,
)
from L2_001_feature_revalidation import compute_all_features

def p(msg):
    print(msg, flush=True)

# ── Setup ──
p("Loading data + features...")
with open(FRAMEWORK_CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
df = pd.read_parquet(cfg["data"]["path"])
df.index = pd.to_datetime(df.index).tz_localize(None)
df = compute_all_features(df, cfg=cfg)
train = df[cfg["data"]["train_start"]:cfg["data"]["train_end"]].copy()
p(f"TRAIN: {len(train)} bars")

classifier = RegimeClassifier(train, cfg=cfg["regime"])
exits = generate_exits(cfg["exit_strategy"])
gates = generate_gates(cfg["magnitude_gate"])
signals = generate_signals(cfg["signal_gate"])

# Pre-compute masks
p("\nPre-computing masks...")
gate_masks = {}
for gate in gates:
    mask = apply_gate(train, gate, classifier)
    gate_masks[gate.name] = mask.values

signal_long_masks = {}
signal_short_masks = {}
for signal in signals:
    long_m = pd.Series(True, index=train.index)
    short_m = pd.Series(True, index=train.index)
    for feature, long_t, short_t in signal.conditions:
        feat_vals = train[feature]
        long_m = long_m & (feat_vals <= long_t)
        short_m = short_m & (feat_vals >= short_t)
    signal_long_masks[signal.name] = long_m.values
    signal_short_masks[signal.name] = short_m.values

# Precompute 1 exit
ec = exits[0]
p(f"\nPrecomputing exit: {ec.name}...")
precomputed = precompute_all_trades(train, ec)

# ═══════════════════════════════════════════════════════════════
# TEST 1: Profile INDIVIDUAL STEPS inside run_backtest_fast
# ═══════════════════════════════════════════════════════════════
p("\n" + "="*70)
p("TEST 1: Profile individual steps of run_backtest_fast()")
p("="*70)

# Pick a typical gate+signal combo
g_mask = gate_masks[gates[0].name]
s_long = signal_long_masks[signals[0].name]
s_short = signal_short_masks[signals[0].name]

N_ITER = 1000
n_bars = len(train)

# Step A: Boolean AND
t0 = time.perf_counter()
for _ in range(N_ITER):
    long_combined = g_mask & s_long
    short_combined = g_mask & s_short
t_and = (time.perf_counter() - t0) / N_ITER * 1000
p(f"  Step A (boolean AND):        {t_and:.3f} ms/combo")

# Step B: np.where
t0 = time.perf_counter()
for _ in range(N_ITER):
    long_idx = np.where(long_combined)[0]
    short_idx = np.where(short_combined)[0]
t_where = (time.perf_counter() - t0) / N_ITER * 1000
p(f"  Step B (np.where x2):        {t_where:.3f} ms/combo")
p(f"    long signals: {len(long_idx)}, short signals: {len(short_idx)}")

# Step C: Merge + Sort
t0 = time.perf_counter()
for _ in range(N_ITER):
    all_signals = np.empty(len(long_idx) + len(short_idx), dtype=[('idx', 'i4'), ('dir', 'i1')])
    all_signals['idx'][:len(long_idx)] = long_idx
    all_signals['dir'][:len(long_idx)] = 1
    all_signals['idx'][len(long_idx):] = short_idx
    all_signals['dir'][len(long_idx):] = 0
    all_signals.sort(order='idx')
t_merge = (time.perf_counter() - t0) / N_ITER * 1000
p(f"  Step C (merge+sort):         {t_merge:.3f} ms/combo  (n={len(all_signals)})")

# Step D: Python overlap loop
l_net = precomputed["long_net_bps"]
l_exit = precomputed["long_exit_bar"]
s_net = precomputed["short_net_bps"]
s_exit = precomputed["short_exit_bar"]

t0 = time.perf_counter()
for _ in range(N_ITER):
    trade_nets = []
    trade_dirs = []
    in_trade_until = -1
    for j in range(len(all_signals)):
        bar_idx = int(all_signals['idx'][j])
        if bar_idx <= in_trade_until:
            continue
        is_long = all_signals['dir'][j] == 1
        net = l_net[bar_idx] if is_long else s_net[bar_idx]
        if np.isnan(net):
            continue
        exit_bar = l_exit[bar_idx] if is_long else s_exit[bar_idx]
        trade_nets.append(net)
        trade_dirs.append(1 if is_long else 0)
        in_trade_until = exit_bar
t_loop = (time.perf_counter() - t0) / N_ITER * 1000
p(f"  Step D (Python overlap loop): {t_loop:.3f} ms/combo  (trades={len(trade_nets)})")

# Step E: Metrics computation
t0 = time.perf_counter()
for _ in range(N_ITER):
    if trade_nets:
        nets = np.array(trade_nets)
        dirs = np.array(trade_dirs)
        n_trades = len(nets)
        total_bps = nets.sum()
        avg_bps = nets.mean()
        win_rate = (nets > 0).mean() * 100
        winners_sum = nets[nets > 0].sum()
        losers_sum = abs(nets[nets <= 0].sum())
        pf = winners_sum / losers_sum if losers_sum > 0 else float("inf")
        cumsum = nets.cumsum()
        running_max = np.maximum.accumulate(cumsum)
        max_dd = (cumsum - running_max).min()
t_metrics = (time.perf_counter() - t0) / N_ITER * 1000
p(f"  Step E (metrics):            {t_metrics:.3f} ms/combo")

total_per_combo = t_and + t_where + t_merge + t_loop + t_metrics
p(f"\n  TOTAL per combo:             {total_per_combo:.3f} ms")
p(f"  Estimated combos/sec:        {1000/total_per_combo:.0f}")
p(f"  Actual dry_run2 result:      43 combos/sec (23ms each)")
p(f"  Discrepancy factor:          {23/total_per_combo:.1f}x")

# Step C-ALT: Faster merge using concatenate + argsort
p(f"\n  --- Step C alternatives ---")
t0 = time.perf_counter()
for _ in range(N_ITER):
    ai = np.concatenate([long_idx, short_idx])
    ad = np.concatenate([np.ones(len(long_idx), dtype=np.int8), np.zeros(len(short_idx), dtype=np.int8)])
    sort_order = np.argsort(ai, kind='mergesort')
    ai_sorted = ai[sort_order]
    ad_sorted = ad[sort_order]
t_merge_alt = (time.perf_counter() - t0) / N_ITER * 1000
p(f"  Step C-ALT (concat+argsort): {t_merge_alt:.3f} ms (vs structured {t_merge:.3f} ms) = {t_merge/t_merge_alt:.1f}x faster")

# ═══════════════════════════════════════════════════════════════
# TEST 2: Full batch timing — 1000 combos
# ═══════════════════════════════════════════════════════════════
p("\n" + "="*70)
p("TEST 2: Batch 1000 combos (real run_backtest_fast)")
p("="*70)

t0 = time.perf_counter()
count = 0
for g in gates[:10]:  # 10 gates × 100 signals = 1000
    g_m = gate_masks[g.name]
    for s in signals[:100]:
        br = run_backtest_fast(
            g_m, signal_long_masks[s.name], signal_short_masks[s.name],
            precomputed, g.name, s.name, exit_name=ec.name,
        )
        count += 1
t_batch = time.perf_counter() - t0
p(f"  {count} combos in {t_batch:.2f}s = {count/t_batch:.0f} combos/sec = {t_batch/count*1000:.2f} ms/combo")

# ═══════════════════════════════════════════════════════════════
# TEST 3: Signal density — how many signals per combo?
# ═══════════════════════════════════════════════════════════════
p("\n" + "="*70)
p("TEST 3: Signal density across combos")
p("="*70)

signal_counts = []
for g in gates[:50]:
    g_m = gate_masks[g.name]
    for s in signals[:50]:
        lc = (g_m & signal_long_masks[s.name]).sum()
        sc = (g_m & signal_short_masks[s.name]).sum()
        signal_counts.append(lc + sc)

sc = np.array(signal_counts)
p(f"  Sample: {len(sc)} combos")
p(f"  Signal fires per combo:")
p(f"    Min:    {sc.min()}")
p(f"    Q25:    {np.percentile(sc, 25):.0f}")
p(f"    Median: {np.median(sc):.0f}")
p(f"    Q75:    {np.percentile(sc, 75):.0f}")
p(f"    Max:    {sc.max()}")
p(f"    Zero:   {(sc == 0).sum()} ({(sc == 0).mean()*100:.1f}%)")
p(f"    <10:    {(sc < 10).sum()} ({(sc < 10).mean()*100:.1f}%)")
p(f"    <50:    {(sc < 50).sum()} ({(sc < 50).mean()*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════
# TEST 4: Can we skip combos with <50 signals early?
# ═══════════════════════════════════════════════════════════════
p("\n" + "="*70)
p("TEST 4: Early skip — how fast is signal count check?")
p("="*70)

t0 = time.perf_counter()
skip_count = 0
run_count = 0
for g in gates[:100]:
    g_m = gate_masks[g.name]
    for s in signals:
        n_fire = int((g_m & signal_long_masks[s.name]).sum() + (g_m & signal_short_masks[s.name]).sum())
        if n_fire < 50:
            skip_count += 1
        else:
            run_count += 1
t_check = time.perf_counter() - t0
total_checked = skip_count + run_count
p(f"  Checked {total_checked} combos in {t_check:.2f}s ({total_checked/t_check:.0f}/sec)")
p(f"  Skip (<50 signals): {skip_count} ({skip_count/total_checked*100:.1f}%)")
p(f"  Run  (>=50 signals): {run_count} ({run_count/total_checked*100:.1f}%)")
p(f"  Time per check: {t_check/total_checked*1000:.3f} ms")

# ═══════════════════════════════════════════════════════════════
# TEST 5: Numba potential — same overlap loop compiled
# ═══════════════════════════════════════════════════════════════
p("\n" + "="*70)
p("TEST 5: Numba JIT potential")
p("="*70)

try:
    from numba import njit

    @njit(cache=True)
    def overlap_loop_numba(signal_indices, signal_dirs,
                           l_net, l_exit, s_net, s_exit):
        """Numba-compiled overlap removal + trade collection."""
        n = len(signal_indices)
        # Pre-allocate max size
        trade_nets = np.empty(n, dtype=np.float64)
        trade_dirs = np.empty(n, dtype=np.int8)
        count = 0
        in_trade_until = -1

        for j in range(n):
            bar_idx = signal_indices[j]
            if bar_idx <= in_trade_until:
                continue
            is_long = signal_dirs[j] == 1
            if is_long:
                net = l_net[bar_idx]
                exit_bar = l_exit[bar_idx]
            else:
                net = s_net[bar_idx]
                exit_bar = s_exit[bar_idx]

            if np.isnan(net):
                continue

            trade_nets[count] = net
            trade_dirs[count] = 1 if is_long else 0
            count += 1
            in_trade_until = exit_bar

        return trade_nets[:count], trade_dirs[:count]

    @njit(cache=True)
    def compute_metrics_numba(nets, dirs):
        """Numba-compiled metrics computation."""
        n = len(nets)
        if n == 0:
            return np.int64(0), 0.0, 0.0, 0.0, 0.0, np.int64(0), np.int64(0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        total_bps = 0.0
        winners_sum = 0.0
        losers_sum = 0.0
        n_win = 0
        n_long = 0
        n_short = 0
        long_bps = 0.0
        short_bps = 0.0
        long_win = 0
        short_win = 0
        long_winners = 0.0
        long_losers = 0.0
        short_winners = 0.0
        short_losers = 0.0

        for i in range(n):
            total_bps += nets[i]
            if nets[i] > 0:
                winners_sum += nets[i]
                n_win += 1
            else:
                losers_sum += abs(nets[i])
            if dirs[i] == 1:
                n_long += 1
                long_bps += nets[i]
                if nets[i] > 0:
                    long_win += 1
                    long_winners += nets[i]
                else:
                    long_losers += abs(nets[i])
            else:
                n_short += 1
                short_bps += nets[i]
                if nets[i] > 0:
                    short_win += 1
                    short_winners += nets[i]
                else:
                    short_losers += abs(nets[i])

        win_rate = n_win / n * 100.0
        pf = winners_sum / losers_sum if losers_sum > 0 else 999.0
        avg_bps = total_bps / n
        long_wr = long_win / n_long * 100.0 if n_long > 0 else 0.0
        short_wr = short_win / n_short * 100.0 if n_short > 0 else 0.0
        long_pf = long_winners / long_losers if long_losers > 0 else 999.0
        short_pf = short_winners / short_losers if short_losers > 0 else 999.0

        # Max drawdown
        cumsum = 0.0
        running_max = 0.0
        max_dd = 0.0
        for i in range(n):
            cumsum += nets[i]
            if cumsum > running_max:
                running_max = cumsum
            dd = cumsum - running_max
            if dd < max_dd:
                max_dd = dd

        return (n, total_bps, win_rate, pf, avg_bps, n_long, n_short,
                long_bps, short_bps, long_wr, short_wr, long_pf, short_pf, max_dd)

    # Warm up JIT
    p("  Warming up numba JIT...")
    # Prepare test data
    long_combined = g_mask & s_long
    short_combined = g_mask & s_short
    long_idx = np.where(long_combined)[0]
    short_idx = np.where(short_combined)[0]

    # Merge into sorted arrays
    all_idx = np.concatenate([long_idx, short_idx])
    all_dir = np.concatenate([np.ones(len(long_idx), dtype=np.int8), np.zeros(len(short_idx), dtype=np.int8)])
    sort_order = np.argsort(all_idx)
    all_idx = all_idx[sort_order]
    all_dir = all_dir[sort_order]

    # First call = compilation
    _ = overlap_loop_numba(all_idx, all_dir, l_net, l_exit.astype(np.int64), s_net, s_exit.astype(np.int64))
    nets_test, dirs_test = overlap_loop_numba(all_idx, all_dir, l_net, l_exit.astype(np.int64), s_net, s_exit.astype(np.int64))
    _ = compute_metrics_numba(nets_test, dirs_test)
    p("  JIT compiled OK")

    # Benchmark numba overlap loop
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        tn, td = overlap_loop_numba(all_idx, all_dir, l_net, l_exit.astype(np.int64), s_net, s_exit.astype(np.int64))
    t_numba_loop = (time.perf_counter() - t0) / N_ITER * 1000
    p(f"  Numba overlap loop:  {t_numba_loop:.3f} ms (vs Python {t_loop:.3f} ms) = {t_loop/t_numba_loop:.0f}x speedup")

    # Benchmark numba metrics
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        _ = compute_metrics_numba(tn, td)
    t_numba_metrics = (time.perf_counter() - t0) / N_ITER * 1000
    p(f"  Numba metrics:       {t_numba_metrics:.3f} ms (vs Python {t_metrics:.3f} ms) = {t_metrics/t_numba_metrics:.0f}x speedup")

    # Full numba pipeline: AND + where + merge + numba_loop + numba_metrics
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        lc = g_mask & s_long
        sc = g_mask & s_short
        li = np.where(lc)[0]
        si = np.where(sc)[0]
        ai = np.concatenate([li, si])
        ad = np.concatenate([np.ones(len(li), dtype=np.int8), np.zeros(len(si), dtype=np.int8)])
        so = np.argsort(ai)
        ai = ai[so]
        ad = ad[so]
        tn, td = overlap_loop_numba(ai, ad, l_net, l_exit.astype(np.int64), s_net, s_exit.astype(np.int64))
        _ = compute_metrics_numba(tn, td)
    t_full_numba = (time.perf_counter() - t0) / N_ITER * 1000
    p(f"\n  Full pipeline (numba): {t_full_numba:.3f} ms/combo = {1000/t_full_numba:.0f} combos/sec")
    p(f"  Full pipeline (orig):  {total_per_combo:.3f} ms/combo = {1000/total_per_combo:.0f} combos/sec")
    p(f"  Speedup: {total_per_combo/t_full_numba:.1f}x")

    # Batch test with numba
    p("\n  Batch 1000 combos with numba pipeline...")
    l_exit_i64 = l_exit.astype(np.int64)
    s_exit_i64 = s_exit.astype(np.int64)

    t0 = time.perf_counter()
    count = 0
    for g in gates[:10]:
        g_m = gate_masks[g.name]
        for s in signals[:100]:
            lc = g_m & signal_long_masks[s.name]
            sc = g_m & signal_short_masks[s.name]
            li = np.where(lc)[0]
            si = np.where(sc)[0]
            if len(li) + len(si) == 0:
                count += 1
                continue
            ai = np.concatenate([li, si])
            ad = np.concatenate([np.ones(len(li), dtype=np.int8), np.zeros(len(si), dtype=np.int8)])
            so = np.argsort(ai)
            tn, td = overlap_loop_numba(ai[so], ad[so], l_net, l_exit_i64, s_net, s_exit_i64)
            if len(tn) > 0:
                _ = compute_metrics_numba(tn, td)
            count += 1
    t_batch_numba = time.perf_counter() - t0
    p(f"  {count} combos in {t_batch_numba:.2f}s = {count/t_batch_numba:.0f} combos/sec")

except ImportError:
    p("  numba not installed — skipping JIT test")
    p("  Install with: pip install numba")

# ═══════════════════════════════════════════════════════════════
# TEST 6: Full scan estimate with optimizations
# ═══════════════════════════════════════════════════════════════
p("\n" + "="*70)
p("TEST 6: FULL SCAN ESTIMATES")
p("="*70)

precompute_time = 0.8  # from dry_run2
n_exits = len(exits)
n_combos_per_exit = len(gates) * len(signals)
total = n_exits * n_combos_per_exit

p(f"  {n_exits} exits × {n_combos_per_exit:,} gate×signal = {total:,} total combos")
p(f"  Precompute per exit: {precompute_time}s")
p(f"")
p(f"  Current (43/sec):     {total/43:.0f}s = {total/43/3600:.1f} hr")

try:
    numba_rate = count / t_batch_numba
    p(f"  Numba ({numba_rate:.0f}/sec):   {total/numba_rate:.0f}s = {total/numba_rate/3600:.1f} hr")
    with_precompute = total / numba_rate + precompute_time * n_exits
    p(f"  Numba + precompute:   {with_precompute:.0f}s = {with_precompute/3600:.1f} hr")
except:
    pass

p("\nDONE")
