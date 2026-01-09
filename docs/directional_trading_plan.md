# Direction-Aware Trading Implementation Plan

> **STATUS: ✅ FULLY IMPLEMENTED** (Updated: 2026-01-08)

Objective: make long and short trades use their own outcome stats, risk controls, and sizing while keeping current pipeline structure, with clear steps to implement, validate, and roll out.

## Scope
- Similarity aggregation/output
- Decision logic, TP/SL, and sizing
- Callers (backtester/orchestrator)
- Config alignment and cleanup
- Tests and sanity checks

## Detailed Steps

### Visual overview

```
            +-------------------+
            | Outcome Labeling  |
            | (mfe/mae long/short)
            +---------+---------+
                      |
                      v
             +------------------+
             | SimilarityEngine |
             | find neighbors   |
             +---------+--------+
                       |
        +--------------+---------------+
        |                              |
        v                              v
 +--------------+              +---------------+
 | long stats   |              | short stats   |
 | mean_mfe     |              | mean_mfe      |
 | mean_mae     |              | mean_mae      |
 | expectancy   |              | expectancy    |
 | mae_5pct     |              | mae_5pct      |
 +------+-------+              +-------+-------+
        \                            /
         \                          /
          \                        /
           v                      v
               +----------------+
               | DecisionEngine |
               +-------+--------+
                       |
          +------------+------------+
          |                         |
          v                         v
   +-------------+           +-------------+
   | LONG trade  |           | SHORT trade |
   +-------------+           +-------------+
```

Decision sizing and filters per side:

```
side stats -> filters (status, blocked regime, distance, min_exp, min_mfe)
         -> stop_loss_pct = max(abs(mae_5pct), stop_floor)
         -> take_profit_pct = mean_mfe
         -> position_size = min(capital * risk_per_trade / stop_loss_pct,
                                capital * max_leverage)
```

### 1) SimilarityEngine updates
- Compute both sides from the same neighbor set (no extra search):
  - Long stats: `mean_mfe = neighbors[mfe_long_*].mean()`, `mean_mae`, `expectancy = mean_mfe + mean_mae`, `mae_5pct = mae.quantile(0.05)`.
  - Short stats: same using `mfe_short_*`, `mae_short_*`.
  - Distance stats stay shared.
- Return a structured dict:
  ```python
  {
    "status": "OK",
    "neighbors": ...,
    "distance_mean": ...,
    "distance_max": ...,
    "long":  {"mean_mfe": ..., "mean_mae": ..., "expectancy": ..., "mae_5pct": ...},
    "short": {"mean_mfe": ..., "mean_mae": ..., "expectancy": ..., "mae_5pct": ...},
  }
  ```
- Preserve existing non-OK statuses (`INSUFFICIENT_DATA`, `UNKNOWN_REGIME`, etc.).
- Add a clear warning/fallback if FAISS is requested but unavailable (auto-switch to brute force or raise with guidance).

### 2) DecisionEngine updates
- Make configuration explicit: require callers to supply capital, risk_per_trade, min_expectancy, max_distance, blocked_regimes (no hidden defaults; comes from config.yaml/config_test.yaml).
- Inputs: the structured similarity result and `regime`.
- Filters:
  - Check `status == "OK"`, `regime` not blocked, `distance_mean <= max_distance`.
  - Use side-specific `expectancy` and `mean_mfe` for min_expectancy/min_mfe checks.
- Direction choice:
  - Option A (simple): pick the side with higher expectancy.
  - Option B (current-style): if long mean_mfe > |long mean_mae| choose LONG else compare short side; pick the best expectancy overall.
- TP/SL per side:
  - `take_profit_pct = chosen_side.mean_mfe` (favorable move sign-aware).
  - `stop_loss_pct = max(abs(chosen_side.mae_5pct), stop_floor)`; skip trade if stop is 0/NaN.
- Sizing:
  - `raw_size = capital * risk_per_trade / stop_loss_pct`.
  - Cap: `position_size = min(raw_size, capital * max_leverage)`.
  - If stop invalid, return NO_TRADE with reason.
- Defaults:
  - Align with config: `min_expectancy=0.001`, `max_distance=3.0`, `blocked_regimes=[]` (unless override), add `stop_floor` (e.g., 1e-4) to avoid extreme sizing.
- Output:
  ```python
  {
    "action": "TRADE",
    "direction": "LONG"/"SHORT",
    "position_size": ...,
    "stop_loss_pct": ...,
    "take_profit_pct": ...,
    "expectancy": chosen_side.expectancy,
    "regime": regime,
    "side": "long"/"short"
  }
  ```

### 3) Callers (backtester + orchestrator)
- Handle new similarity result shape; pass directly into DecisionEngine.
- Keep single-horizon per run unless you later add multi-horizon aggregation.
- On non-OK similarity status, keep current NO_TRADE behavior.

### 4) Cleanup
- Remove non-ASCII/corrupted chars in comments/prints/logs (decision_engine, state_store, config comments, orchestrator spinners).
- Ensure config defaults match code defaults; add FAISS fallback message.

### 5) Tests
- SimilarityEngine unit test: synthetic neighbors produce expected long/short aggregates and status.
- DecisionEngine unit tests:
  - Chooses correct side based on expectancy.
  - Applies TP/SL signs correctly for long vs short.
  - Risk sizing respects caps and stop_floor.
  - Filter reasons (negative expectancy, low similarity, blocked regime, invalid stop).
- Integration smoke: one-step backtester or orchestrator call with mocked similarity result for a LONG path and a SHORT path.

### 6) Rollout
- Implement similarity + decision changes with cleanup.
- Update callers.
- Add tests; run unit tests.
- Run a short backtest to inspect long vs short trade counts, P&L, and no-trade reasons.

## Risks / Mitigations
- Over-sizing with tiny stops: enforce stop_floor and leverage caps; skip if stop invalid.
- Behavior drift vs current backtest: document changes and keep an option to revert to legacy sizing/logic if needed.
- Performance: negligible impact if reusing neighbor set; avoid duplicate FAISS/brute-force queries.

---

## Implementation Status

### ✅ Completed Components

#### 1. Outcome Labeling (`src/trade_system/outcomes/outcome_labeler.py`)
**Status: IMPLEMENTED**

Short-side outcomes are computed by inverting long-side logic:
```python
# Lines 61-68
mfe_short = -mae_long   # Short's max profit = Long's max loss (inverted)
mae_short = -mfe_long   # Short's max loss = Long's max profit (inverted)

outcome_df[f"mfe_long_{h}m"] = mfe_long
outcome_df[f"mae_long_{h}m"] = mae_long
outcome_df[f"mfe_short_{h}m"] = mfe_short
outcome_df[f"mae_short_{h}m"] = mae_short
```

**Verified columns in `data/outcomes/BTCUSDT_1m_outcomes.parquet`:**
| Horizon | Long MFE | Long MAE | Short MFE | Short MAE |
|---------|----------|----------|-----------|-----------|
| 2m | `mfe_long_2m` | `mae_long_2m` | `mfe_short_2m` ✅ | `mae_short_2m` ✅ |
| 3m | `mfe_long_3m` | `mae_long_3m` | `mfe_short_3m` ✅ | `mae_short_3m` ✅ |
| 5m | `mfe_long_5m` | `mae_long_5m` | `mfe_short_5m` ✅ | `mae_short_5m` ✅ |
| 10m | `mfe_long_10m` | `mae_long_10m` | `mfe_short_10m` ✅ | `mae_short_10m` ✅ |
| 15m | `mfe_long_15m` | `mae_long_15m` | `mfe_short_15m` ✅ | `mae_short_15m` ✅ |
| 30m | `mfe_long_30m` | `mae_long_30m` | `mfe_short_30m` ✅ | `mae_short_30m` ✅ |

#### 2. SimilarityEngine (`src/trade_system/similarity/similarity_engine.py`)
**Status: IMPLEMENTED**

Returns structured dict with both `long` and `short` stats (lines 530-573):
```python
return {
    "status": "OK",
    "neighbors": len(neighbors),
    "distance_mean": float(distances.mean()),
    "distance_max": float(distances.max()),
    "long": {
        "mean_mfe": float(mfe_long.mean()),
        "mean_mae": float(mae_long.mean()),
        "expectancy": float(mfe_long.mean() + mae_long.mean()),
        "mae_5pct": float(mae_long.quantile(0.05)),
    },
    "short": {
        "mean_mfe": float(mfe_short.mean()),
        "mean_mae": float(mae_short.mean()),
        "expectancy": float(mfe_short.mean() + mae_short.mean()),
        "mae_5pct": float(mae_short.quantile(0.05)),
    },
}
```

**Additional features implemented:**
- NaN/Inf handling for FAISS (line 466-467)
- Safety checks in index building (lines 488-505)
- Fallback to FlatL2 for small datasets (lines 509-513)

#### 3. DecisionEngine (`src/trade_system/decision/decision_engine.py`)
**Status: IMPLEMENTED**

Direction selection based on highest expectancy (lines 66-97):
```python
# Get both sides
long_stats = similarity_result.get("long")
short_stats = similarity_result.get("short")

# Build candidates
candidates = []
if long_stats and long_stats.get("expectancy") is not None:
    candidates.append(("LONG", long_stats))
if short_stats and short_stats.get("expectancy") is not None:
    candidates.append(("SHORT", short_stats))

# Pick best side
side, stats = max(candidates, key=lambda x: x[1].get("expectancy", float("-inf")))
```

**Risk controls per side (lines 99-121):**
```python
stop_pct = max(abs(stats.get("mae_5pct")), self.stop_floor)
take_profit_pct = stats.get("mean_mfe", 0.0)
position_size = min(
    (self.capital * self.risk_per_trade) / stop_pct,
    self.capital * self.max_leverage
)
```

#### 4. Callers (Backtester + Orchestrator)
**Status: IMPLEMENTED**

- `src/trade_system/pipeline/orchestrator.py` - handles new similarity result shape
- `src/trade_system/backtest/backtester.py` - processes LONG/SHORT trades

#### 5. Configuration
**Status: IMPLEMENTED**

Key parameters in `src/trade_system/config/config.yaml`:
```yaml
decision:
  capital: 100
  risk_per_trade: 0.005    # 0.5% risk per trade
  min_expectancy: 0.001    # Minimum expectancy filter
  max_distance: 3.0        # Maximum KNN distance
  blocked_regimes: []      # No regime blocking
  max_leverage: 1.0        # Position cap
  stop_floor: 1e-4         # Minimum stop loss
```

### Data Flow Summary

```
OHLCV Data
    ↓
┌─────────────────────────────────────────┐
│ Outcome Labeler                         │
│ - mfe_long_*m, mae_long_*m              │
│ - mfe_short_*m, mae_short_*m (inverted) │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ SimilarityEngine (KNN/FAISS)            │
│ - Find k=200 nearest neighbors          │
│ - Aggregate long stats from neighbors   │
│ - Aggregate short stats from neighbors  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ DecisionEngine                          │
│ - Compare long.expectancy vs short      │
│ - Pick side with higher expectancy      │
│ - Calculate TP/SL/size for chosen side  │
└─────────────────────────────────────────┘
    ↓
TRADE (LONG or SHORT) or NO_TRADE
```

### Verification Commands

```bash
# Run pipeline with local data
python scripts/run_pipeline.py

# Run backtest
python scripts/run_backtest.py

# Check outcome columns
python -c "import pandas as pd; print(pd.read_parquet('data/outcomes/BTCUSDT_1m_outcomes.parquet').columns.tolist())"
```
