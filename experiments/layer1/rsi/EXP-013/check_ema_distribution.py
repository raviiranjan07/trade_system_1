"""Check actual EMA separation distribution at bear oversold signal points."""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")

RSI_PERIOD = 14
SMA_PERIOD = 200
EMA_SHORT = 50
EMA_LONG = 200

df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)

# RSI
delta = df["close"].diff()
gain = delta.where(delta > 0, 0).rolling(window=RSI_PERIOD).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
rs = gain / loss
df["rsi"] = 100 - (100 / (1 + rs))

# SMA200
df["sma"] = df["close"].rolling(SMA_PERIOD).mean()
df["bear_market"] = df["close"] < df["sma"]

# EMA separation
df["ema_short"] = df["close"].ewm(span=EMA_SHORT, adjust=False).mean()
df["ema_long"] = df["close"].ewm(span=EMA_LONG, adjust=False).mean()
df["ema_separation"] = abs(df["ema_short"] - df["ema_long"]) / df["close"] * 100

# ATR percentile
tr = np.maximum(
    df["high"] - df["low"],
    np.maximum(
        abs(df["high"] - df["close"].shift(1)),
        abs(df["low"] - df["close"].shift(1)),
    ),
)
df["atr"] = tr.rolling(14).mean()
atr_bps = df["atr"] / df["close"] * 10000
df["atr_percentile"] = atr_bps.rolling(200).rank(pct=True) * 100

# RSI cross signals
for thresh in [8, 10, 12, 15]:
    oversold = df["rsi"] < thresh
    df[f"cross_{thresh}"] = oversold & ~oversold.shift(1, fill_value=False)

# Filter to OOS
oos = df["2024":"2025"]

print("=" * 70)
print("EMA SEPARATION DISTRIBUTION AT BEAR OVERSOLD SIGNALS (OOS 2024-2025)")
print("=" * 70)

for thresh in [8, 10, 12, 15]:
    signals = oos[oos[f"cross_{thresh}"] & oos["bear_market"]]
    if len(signals) == 0:
        print(f"\nRSI<{thresh}: NO SIGNALS")
        continue

    ema_vals = signals["ema_separation"]
    atr_vals = signals["atr_percentile"]

    print(f"\nRSI<{thresh} + Bear: {len(signals)} signals")
    print(f"  EMA Sep:  min={ema_vals.min():.2f}%  median={ema_vals.median():.2f}%  "
          f"mean={ema_vals.mean():.2f}%  max={ema_vals.max():.2f}%")
    print(f"  ATR Pctl: min={atr_vals.min():.1f}  median={atr_vals.median():.1f}  "
          f"mean={atr_vals.mean():.1f}  max={atr_vals.max():.1f}")

    print(f"\n  EMA Sep percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"    P{p}: {ema_vals.quantile(p/100):.2f}%")

    print(f"\n  EMA Sep buckets:")
    bins = [0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
    for i in range(len(bins)-1):
        count = ((ema_vals >= bins[i]) & (ema_vals < bins[i+1])).sum()
        pct = count / len(ema_vals) * 100
        print(f"    [{bins[i]:.1f} - {bins[i+1]:.1f}%): {count} signals ({pct:.1f}%)")
    count = (ema_vals >= bins[-1]).sum()
    if count > 0:
        print(f"    [>={bins[-1]:.1f}%): {count} signals ({count/len(ema_vals)*100:.1f}%)")

    print(f"\n  ATR Pctl buckets:")
    atr_bins = [0, 10, 25, 40, 50, 60, 75, 90, 100]
    for i in range(len(atr_bins)-1):
        count = ((atr_vals >= atr_bins[i]) & (atr_vals < atr_bins[i+1])).sum()
        pct = count / len(atr_vals) * 100
        print(f"    [{atr_bins[i]} - {atr_bins[i+1]}): {count} signals ({pct:.1f}%)")

# Also check TRAIN period
print(f"\n{'='*70}")
print("TRAIN (2020-2023) FOR COMPARISON")
print(f"{'='*70}")

train = df["2020":"2023"]
for thresh in [10, 12]:
    signals = train[train[f"cross_{thresh}"] & train["bear_market"]]
    if len(signals) == 0:
        continue
    ema_vals = signals["ema_separation"]
    print(f"\nRSI<{thresh} + Bear: {len(signals)} signals")
    print(f"  EMA Sep:  min={ema_vals.min():.2f}%  median={ema_vals.median():.2f}%  "
          f"mean={ema_vals.mean():.2f}%  max={ema_vals.max():.2f}%")
    print(f"\n  EMA Sep percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"    P{p}: {ema_vals.quantile(p/100):.2f}%")
