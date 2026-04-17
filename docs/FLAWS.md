# ML Model Flaws Analysis

**Date:** 2026-04-17
**Models analyzed:** ML V1 (MLP, 10 features) and ML V2 (LSTM+Attention, 32 features)
**Exit rules:** V2 (V1 minus LOCKED_PROFIT)
**Data:** 2024-2025 OOS, BTCUSDT 15m

---

## What Works

1. **Exit system is the edge.** PT_TARGET/PT_LOCK/MID_TRAIL/STOP_LOSS create asymmetric payoffs: winners avg +41 bps, losers capped at -18 bps. This is the entire source of profitability.

2. **WITH_MFE architecture is best quality.** Despite MFE heads being mostly ignored by the direction head (91.7% weight on attended vector), they act as a regularizer. Result: fewer signals but best per-trade quality (PF 1.40 vs 1.22 without MFE, vs 1.24 with asymmetry head).

3. **Dataset is clean.** Class balance 49.5%/50.5% across all splits. No NaN in labels. No look-ahead leakage. Date-based splits (train 2020-2023, val 2024, test 2025).

4. **The system IS profitable.** +3,023 to +10,129 bps depending on configuration. Not broken — just inefficient.

---

## Flaw 1: Model Only Learns One Pattern — Blind Mean Reversion

**Evidence:**
- 803 of 804 ML V2 signals fight the current bar direction (contrarian)
- 802 of 804 signals oppose prior 3-bar momentum
- The model fires LONG whenever recent ROC diffs are negative, SHORT when positive
- This is the ONLY learnable pattern from "how price changed recently"

**Root cause:** All 32 features (V2) and 10 features (V1) are derived from price. The model has no information about WHY price moved — only that it moved. Mean reversion is the optimal strategy given this feature set.

**Impact:** 50% of entries are directionally wrong. The model cannot distinguish "this drop will reverse" from "this drop will continue."

---

## Flaw 2: No Feature Separates Winners from Losers

**Evidence (tested on 804 ML V2 trades, 405 stopped vs 399 winners):**

| Feature | Winners median | Stopped median | Overlap |
|---------|:-:|:-:|:-:|
| RSI7 | 35.5 | 35.1 | 99% |
| Range position 50 | 0.40 | 0.35 | 93% |
| Range position 20 | 0.37 | 0.31 | 98% |
| SMA200 dist % | -1.37 | -1.57 | 89% |
| ROC1 | -33.5 | -35.0 | 98% |
| ROC8 | -58.4 | -79.0 | 96% |
| All 32 V2 diff features | — | — | 89-99% overlap |
| Model probability | 0.612 | 0.612 | identical |
| MFE up prediction | 1.345 | 1.321 | near-identical |
| MFE down prediction | 1.158 | 1.163 | near-identical |
| Attention weights | [0.255, 0.175, ...] | [0.250, 0.168, ...] | identical pattern |
| Direction logit | 0.1413 | 0.1181 | identical |

**Also tested and found no separation:**
- ATR percentile at entry: 40-52% stop rate across all buckets
- EMA separation at entry: 48-53% stop rate across all buckets
- Hour of day: 50-68% stop rate (08:00 UTC worst but small sample)
- Entry bar range: ~50% stop rate regardless of bar size
- S/R zone proximity (25-bar KDE): 48-54% stop rate at all distances
- S/R zone position (at support / dead zone / at resistance): 49-52% all same
- Signal bar magnitude: 48-61% stop rate, no clear pattern

**One weak signal found:**
- MFE predicted ratio (up/down) > 1.2: 48.3% stop rate, +19.2 avg bps
- MFE predicted ratio < 0.8: 58.0% stop rate, +9.5 avg bps
- 10 percentage point gap — but compressed away by h_dir layer

**One directional signal found (earlier analysis, not retested here):**
- LONG in BULL: 26.9% stop rate, +31.2 avg bps (only 52 trades — small sample)
- All other direction x regime combos: ~50% stop rate

---

## Flaw 3: Label-Stop Mismatch

**The label:** "Does price hit +15 bps or -15 bps first within 8 bars?" (symmetric, first-hit)

**The reality:** Trade has a -10 bps stop loss. Entry where price dips to -12 then recovers to +15 is labeled LONG (correct) but produces a -18 bps loss in practice (stopped at -10 before recovery).

**Evidence:**
- 81% of stopped trades have MFE = 0 (never went positive)
- 99% of stops fire on bar 0 (within first 15 minutes)
- Recovery analysis: widening stop to -30 only recovers 47/405 trades past MFE > 10
- Even at -100 bps stop, only 48/405 ever reach MFE > 10

**Impact:** Model is trained to call entries "correct" that lose money under actual trading rules. The training label does not penalize entries that path through -10 bps on the way to +15.

---

## Flaw 4: MFE Auxiliary Task Learns Volatility, Not Direction

**Architecture:** LSTM → Attention → h_mfe_up(128→8) + h_mfe_down(128→8) → concat(128+8+8=144) → h_dir(144→1)

**Loss:** MSE(mfe_up) × 1.0 + MSE(mfe_down) × 1.0 + BCE(direction) × 0.5

**What went wrong:**
1. MFE up and MFE down are both driven by volatility (how much the market is moving), not direction
2. The MFE heads learn "current volatility predicts ~30 bps up AND ~30 bps down" — correct but useless for direction
3. Loss weights give MFE 4× the gradient of direction — LSTM optimizes for volatility prediction
4. h_dir learned to ignore MFE inputs: 91.7% weight on attended vector, 2.6% on MFE up, 5.7% on MFE down
5. MFE down weights are POSITIVE (+0.097) — backwards: more predicted downside → more LONG (reinforces mean reversion)
6. Zeroing MFE inputs only flips 7.4% of direction decisions

**Ablation test (3 runs each, test accuracy averaged):**

| Architecture | Conf Acc | N Signals | BT trades | BT bps | BT PF | BT Stop% | BT DD |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| WITH_MFE (production) | 55.5% | 191 | 722 | +3,023 | 1.40 | 57.9% | -420 |
| NO_MFE | 55.4% | 777 | 3,369 | +8,130 | 1.22 | 59.8% | -1,705 |
| ASYMMETRY (mfe_up - mfe_down) | 57.1% | 237 | 759 | +1,976 | 1.24 | 60.7% | -920 |

**Conclusion:** MFE doesn't improve accuracy but acts as implicit regularizer (best PF). The asymmetry head (mfe_up - mfe_down) was tested as a fix — did not help (worst total bps, highest stop rate).

---

## Flaw 5: 20% of Data Excluded (BOTH + SKIP)

**Direction H8 label distribution:**
- LONG: 83,261 (39.7%)
- SHORT: 84,890 (40.4%)
- BOTH: 37,544 (17.9%) — price hit +15 AND -15 within 8 bars
- SKIP: 4,182 (2.0%) — neither threshold hit

**BOTH bars are the highest-volatility bars** where the biggest moves happen. These are exactly the bars where our exit rules have the highest payoff potential (PT_TARGET fires on big moves). Yet the model never trains on them.

---

## What Has NOT Been Tried

| Approach | What it addresses | Why it might work |
|---|---|---|
| **Asymmetric label (+15/-10)** | Flaw 3 (label-stop mismatch) | Directly teaches "don't enter if path goes through -10." Zero cost, same model. |
| **Volume features** | Flaw 1 & 2 (blind mean reversion, no separation) | First non-price information. Exhaustion (declining volume) vs continuation (increasing volume) is the missing signal. Data exists in OHLCV but unused. |
| **Include BOTH bars** | Flaw 5 (data exclusion) | Recovers 17.9% of training data. Could label by which threshold hit first, or by path. |
| **Multi-timeframe context** | Flaw 1 (no structural awareness) | 1H/4H trend direction tells model "is this reversal aligned with bigger picture?" |
| **RL (entry + exit policy)** | All flaws | Learns from actual P&L, not proxy labels. But requires improved features first — same features = same coin flip. |

---

## Recommended Priority

1. **Asymmetric label** (1 day) — cheapest test, directly addresses Flaw 3
2. **Volume features** (2 days) — first non-price information, addresses Flaw 1 & 2
3. **Include BOTH bars** (1 day) — recover discarded high-value data, addresses Flaw 5
4. **Multi-timeframe** (2 days) — structural context, addresses Flaw 1
5. **RL** (2-3 weeks) — after features improved, optimize full policy

---

## Experiment Tracking

All ablation results logged to MLflow experiment `attention_architecture_ablation`.
Detailed JSON: `experiments/attention_ablation/results/ablation_results.json`
