# Adaptive Multi-Horizon Trading Implementation Plan

> **STATUS: PLANNING** (Created: 2026-01-08, Updated: 2026-01-09)

## Executive Summary

Implement **state-based adaptive horizon selection** where the system queries the SAME market state against MULTIPLE horizon outcome datasets, compares MFE across horizons, and picks the horizon with the highest MFE for that specific state.

---

## Core Concept: State-Based Horizon Selection

### The Key Insight

At any given time `t`, the same market state may have **different MFE potential** depending on the prediction horizon. Instead of guessing which horizon is best based on volatility or market conditions, we **directly query historical data** to find which horizon historically worked best for similar states.

### How It Works

```
At time t, current state vector = [rsi, macd, volume, spread, ...]

Query SAME state against SEPARATE horizon outcome datasets:

┌─────────────────────────────────────────────────────────────────┐
│  H=3 Outcome Dataset + FAISS Index                              │
│  Query → Find k similar states → Avg MFE = 1.0%                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  H=5 Outcome Dataset + FAISS Index                              │
│  Query → Find k similar states → Avg MFE = 1.5%   ← WINNER     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  H=10 Outcome Dataset + FAISS Index                             │
│  Query → Find k similar states → Avg MFE = 0.8%                │
└─────────────────────────────────────────────────────────────────┘

→ Select H=5 for this trade (highest MFE for THIS specific state)
```

### Why This Approach is Superior

| Aspect | Volatility-Based (Traditional) | State-Based (Our Approach) |
|--------|-------------------------------|---------------------------|
| Decision basis | General market condition | Specific state lookup |
| Logic | "High vol → short H" (assumption) | "This state → best H" (data) |
| Accuracy | Approximation | Direct measurement |
| Data needed | Volatility indicator | Multiple outcome datasets |

---

## Real-World Example

### Scenario: BTC at 2:45 PM

```
Current State:
├── RSI = 65
├── MACD = bullish crossover
├── Volume = 1.2x average
├── Spread = tight
└── Recent momentum = +0.3%

Query this EXACT state against each horizon:

H=3 Dataset (3-minute outcomes):
├── Found 100 similar historical states
├── Avg MFE = 0.08% (small, price didn't move much in 3 min)
└── Many false signals

H=5 Dataset (5-minute outcomes):
├── Found 100 similar historical states
├── Avg MFE = 0.15% (good momentum continuation)  ← BEST
└── Clear directional move

H=10 Dataset (10-minute outcomes):
├── Found 100 similar historical states
├── Avg MFE = 0.12% (momentum faded, some reversals)
└── Mixed outcomes

DECISION: Use H=5 horizon for this trade
├── TP based on H=5 MFE (0.15%)
├── SL based on H=5 MAE
└── Max holding time = 5 minutes
```

---

## Architecture Overview

### Required Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ADAPTIVE HORIZON SYSTEM                          │
└─────────────────────────────────────────────────────────────────────────┘

DATA LAYER:
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ outcomes_h3.parquet │ outcomes_h5.parquet │ outcomes_h10.parquet │
│ + faiss_index_h3   │ + faiss_index_h5   │ + faiss_index_h10   │
└──────────────────┘  └──────────────────┘  └──────────────────┘

QUERY LAYER:
┌─────────────────────────────────────────────────────────────────────────┐
│                    AdaptiveHorizonEngine                                 │
│                                                                          │
│  - Loads all horizon datasets and FAISS indices                         │
│  - query_all_horizons(state) → returns MFE for each horizon             │
│  - select_best_horizon() → picks highest MFE                            │
└─────────────────────────────────────────────────────────────────────────┘

DECISION LAYER:
┌─────────────────────────────────────────────────────────────────────────┐
│                    DecisionEngine (enhanced)                             │
│                                                                          │
│  - Receives best horizon selection                                       │
│  - Calculates TP/SL based on selected horizon's stats                   │
│  - Executes trade with appropriate parameters                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Steps

### Phase 1: Generate Multiple Outcome Datasets

**Goal:** Create separate outcome files for each horizon.

```python
# generate_multi_horizon_outcomes.py

HORIZONS = [3, 5, 10, 15, 30]

for horizon in HORIZONS:
    print(f"Generating outcomes for H={horizon}...")

    # Run outcome labeler with this horizon
    outcome_df = generate_outcomes(
        ohlcv_path="data/BTCUSDT_1m.parquet",
        horizon=horizon,
        output_path=f"data/outcomes/outcomes_h{horizon}.parquet"
    )

    # Build FAISS index for this horizon
    build_faiss_index(
        outcome_df=outcome_df,
        output_path=f"data/indices/faiss_index_h{horizon}.bin"
    )
```

**Output Structure:**
```
data/
├── outcomes/
│   ├── outcomes_h3.parquet   # MFE/MAE for 3-minute horizon
│   ├── outcomes_h5.parquet   # MFE/MAE for 5-minute horizon
│   ├── outcomes_h10.parquet  # MFE/MAE for 10-minute horizon
│   ├── outcomes_h15.parquet  # MFE/MAE for 15-minute horizon
│   └── outcomes_h30.parquet  # MFE/MAE for 30-minute horizon
└── indices/
    ├── faiss_index_h3.bin
    ├── faiss_index_h5.bin
    ├── faiss_index_h10.bin
    ├── faiss_index_h15.bin
    └── faiss_index_h30.bin
```

---

### Phase 2: Create AdaptiveHorizonEngine

**File:** `src/trade_system/adaptive/adaptive_horizon_engine.py`

```python
import faiss
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


class AdaptiveHorizonEngine:
    """
    Queries the same state against multiple horizon datasets
    and selects the horizon with the highest MFE.
    """

    def __init__(
        self,
        horizons: List[int] = [3, 5, 10],
        outcomes_dir: Path = None,
        indices_dir: Path = None,
        k: int = 100,
        max_distance: float = 4.0
    ):
        self.horizons = horizons
        self.k = k
        self.max_distance = max_distance

        # Load all datasets and indices
        self.outcome_dfs = {}
        self.faiss_indices = {}
        self.state_matrices = {}

        for h in horizons:
            # Load outcome data
            outcome_path = outcomes_dir / f"outcomes_h{h}.parquet"
            self.outcome_dfs[h] = pd.read_parquet(outcome_path)

            # Load FAISS index
            index_path = indices_dir / f"faiss_index_h{h}.bin"
            self.faiss_indices[h] = faiss.read_index(str(index_path))

            # Store state matrix for distance calculations
            state_cols = [c for c in self.outcome_dfs[h].columns if c.startswith('state_')]
            self.state_matrices[h] = self.outcome_dfs[h][state_cols].values.astype('float32')

    def query_all_horizons(
        self,
        current_state: np.ndarray,
        max_timestamp: pd.Timestamp = None
    ) -> Dict[int, Dict]:
        """
        Query the same state against all horizon datasets.

        Args:
            current_state: Current state vector [n_features]
            max_timestamp: Only use historical data before this time

        Returns:
            {
                3: {"avg_mfe": 0.008, "avg_mae": -0.005, "neighbors": 100, "avg_dist": 2.1},
                5: {"avg_mfe": 0.015, "avg_mae": -0.008, "neighbors": 100, "avg_dist": 1.9},
                10: {"avg_mfe": 0.012, "avg_mae": -0.010, "neighbors": 100, "avg_dist": 2.3},
            }
        """
        results = {}

        # Normalize state vector
        state_vector = current_state.reshape(1, -1).astype('float32')
        faiss.normalize_L2(state_vector)

        for h in self.horizons:
            # Query FAISS index for this horizon
            distances, indices = self.faiss_indices[h].search(state_vector, self.k)

            # Filter by max_timestamp if provided
            valid_mask = np.ones(len(indices[0]), dtype=bool)
            if max_timestamp is not None:
                timestamps = self.outcome_dfs[h].index[indices[0]]
                valid_mask = timestamps < max_timestamp

            valid_indices = indices[0][valid_mask]
            valid_distances = distances[0][valid_mask]

            # Filter by max_distance
            distance_mask = valid_distances <= self.max_distance
            valid_indices = valid_indices[distance_mask]
            valid_distances = valid_distances[distance_mask]

            if len(valid_indices) == 0:
                results[h] = {
                    "avg_mfe": 0.0,
                    "avg_mae": 0.0,
                    "neighbors": 0,
                    "avg_dist": float('inf'),
                    "status": "NO_NEIGHBORS"
                }
                continue

            # Get outcome stats for neighbors
            neighbors = self.outcome_dfs[h].iloc[valid_indices]

            results[h] = {
                "avg_mfe": float(neighbors["mfe"].mean()),
                "avg_mae": float(neighbors["mae"].mean()),
                "neighbors": len(valid_indices),
                "avg_dist": float(valid_distances.mean()),
                "expectancy": float(neighbors["mfe"].mean() + neighbors["mae"].mean()),
                "status": "OK"
            }

        return results

    def select_best_horizon(
        self,
        horizon_results: Dict[int, Dict],
        min_mfe: float = 0.001,
        min_neighbors: int = 50
    ) -> Tuple[int, Dict]:
        """
        Select the horizon with the highest MFE that passes filters.

        Args:
            horizon_results: Output from query_all_horizons()
            min_mfe: Minimum MFE to consider
            min_neighbors: Minimum neighbors required

        Returns:
            (best_horizon, stats) or (None, {}) if no valid horizon
        """
        candidates = []

        for h, stats in horizon_results.items():
            if stats.get("status") != "OK":
                continue
            if stats.get("neighbors", 0) < min_neighbors:
                continue
            if stats.get("avg_mfe", 0) < min_mfe:
                continue

            candidates.append((h, stats))

        if not candidates:
            return None, {"reason": "no_qualifying_horizons"}

        # Sort by MFE descending, pick best
        candidates.sort(key=lambda x: x[1]["avg_mfe"], reverse=True)

        return candidates[0]

    def decide(
        self,
        current_state: np.ndarray,
        max_timestamp: pd.Timestamp = None,
        min_mfe: float = 0.001
    ) -> Dict:
        """
        Full decision: query all horizons, select best, return trade params.
        """
        # Step 1: Query all horizons
        horizon_results = self.query_all_horizons(current_state, max_timestamp)

        # Step 2: Select best horizon
        best_h, stats = self.select_best_horizon(horizon_results, min_mfe=min_mfe)

        if best_h is None:
            return {
                "action": "NO_TRADE",
                "reason": stats.get("reason", "no_qualifying_horizons"),
                "horizon_results": horizon_results
            }

        # Step 3: Build trade decision
        return {
            "action": "LONG" if stats["expectancy"] > 0 else "SHORT",
            "horizon": best_h,
            "mfe": stats["avg_mfe"],
            "mae": stats["avg_mae"],
            "expectancy": stats["expectancy"],
            "neighbors": stats["neighbors"],
            "avg_distance": stats["avg_dist"],
            "horizon_results": horizon_results,  # Full comparison data
            "reason": f"Best MFE at H={best_h}"
        }
```

---

### Phase 3: Integrate with Backtester

**File:** `src/trade_system/backtest/backtester.py`

```python
class AdaptiveBacktester:
    """Backtester that uses adaptive horizon selection."""

    def __init__(self, config):
        self.adaptive_engine = AdaptiveHorizonEngine(
            horizons=config.get("horizons", [3, 5, 10]),
            outcomes_dir=Path(config["outcomes_dir"]),
            indices_dir=Path(config["indices_dir"]),
            k=config.get("k", 100),
            max_distance=config.get("max_distance", 4.0)
        )
        self.min_mfe = config.get("min_mfe", 0.002)
        # ... other config

    def _process_bar(self, timestamp, bar):
        # Get current state vector
        current_state = self._extract_state(bar)

        # Query all horizons and select best
        decision = self.adaptive_engine.decide(
            current_state=current_state,
            max_timestamp=timestamp,
            min_mfe=self.min_mfe
        )

        if decision["action"] == "NO_TRADE":
            return

        # Execute trade with selected horizon's parameters
        self._execute_trade(
            side=decision["action"],
            horizon=decision["horizon"],
            mfe=decision["mfe"],
            mae=decision["mae"],
            entry_price=bar["close"],
            timestamp=timestamp
        )
```

---

### Phase 4: Create Grid Search for Adaptive Mode

```python
# run_adaptive_grid_search.py

PARAM_GRID = {
    "horizons": [[3, 5], [3, 5, 10], [5, 10, 15]],  # Horizon combinations
    "k": [50, 100, 150],
    "max_distance": [3.0, 4.0, 5.0],
    "min_mfe": [0.001, 0.0015, 0.002],
    "min_neighbors": [30, 50, 100],
}

# Test each combination
for params in generate_grid(PARAM_GRID):
    engine = AdaptiveHorizonEngine(**params)
    results = run_backtest(engine)
    log_results(params, results)
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STATE-BASED ADAPTIVE HORIZON FLOW                     │
└─────────────────────────────────────────────────────────────────────────┘

                              OHLCV Bar at time t
                                      │
                                      ▼
                      ┌─────────────────────────────┐
                      │    Extract State Vector      │
                      │  [rsi, macd, volume, ...]   │
                      └─────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
           ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
           │  H=3 Index   │  │  H=5 Index   │  │  H=10 Index  │
           │  FAISS Query │  │  FAISS Query │  │  FAISS Query │
           └──────────────┘  └──────────────┘  └──────────────┘
                    │                 │                 │
                    ▼                 ▼                 ▼
           ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
           │ k neighbors  │  │ k neighbors  │  │ k neighbors  │
           │ Avg MFE=0.8% │  │ Avg MFE=1.5% │  │ Avg MFE=1.2% │
           └──────────────┘  └──────────────┘  └──────────────┘
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      │
                                      ▼
                      ┌─────────────────────────────┐
                      │   Compare MFE Across All    │
                      │                             │
                      │   H=3:  0.8%                │
                      │   H=5:  1.5%  ← HIGHEST     │
                      │   H=10: 1.2%                │
                      └─────────────────────────────┘
                                      │
                                      ▼
                      ┌─────────────────────────────┐
                      │      SELECT H=5             │
                      │                             │
                      │   Use H=5 outcomes for:     │
                      │   - TP target (1.5% MFE)    │
                      │   - SL target (from MAE)   │
                      │   - Max hold time (5 min)  │
                      └─────────────────────────────┘
                                      │
                                      ▼
                      ┌─────────────────────────────┐
                      │         EXECUTE TRADE       │
                      │                             │
                      │   Horizon: 5                │
                      │   Direction: LONG           │
                      │   TP: +1.5%                 │
                      │   SL: -0.8%                 │
                      └─────────────────────────────┘
```

---

## Key Differences from Previous Approach

| Aspect | Previous (Single Index) | New (Multi-Index) |
|--------|------------------------|-------------------|
| **Outcome files** | 1 file with all horizons | Separate file per horizon |
| **FAISS indices** | 1 index | Separate index per horizon |
| **Query count** | 1 query | N queries (one per horizon) |
| **Neighbor set** | Same neighbors for all H | Different neighbors per H |
| **MFE source** | Different columns | Different datasets |

### Why Separate Indices?

Each horizon may have **different optimal neighbors** for the same state. The MFE/MAE labels are horizon-specific, so the index should be built on horizon-specific data.

```
Example: State with RSI=70, strong momentum

H=3 neighbors: States where 3-min momentum was high
H=5 neighbors: States where 5-min trend was strong
H=10 neighbors: States where longer moves developed

These may be DIFFERENT historical states!
```

---

## Performance Considerations

### Query Overhead

```
Single horizon (current):
  1 FAISS query: ~2ms

Adaptive (3 horizons):
  3 FAISS queries: ~6ms

Adaptive (5 horizons):
  5 FAISS queries: ~10ms
```

For backtest: acceptable. For live trading: still sub-15ms, fine for 1-min bars.

### Memory Usage

```
Per horizon:
  Outcome DataFrame: ~50MB
  FAISS Index: ~20MB

5 horizons total: ~350MB
```

Acceptable for 16GB system.

---

## Testing Strategy

### Unit Tests

```python
def test_query_returns_all_horizons():
    engine = AdaptiveHorizonEngine(horizons=[3, 5, 10])
    results = engine.query_all_horizons(state_vector)

    assert 3 in results
    assert 5 in results
    assert 10 in results

def test_selects_highest_mfe():
    results = {
        3: {"avg_mfe": 0.008, "neighbors": 100, "status": "OK"},
        5: {"avg_mfe": 0.015, "neighbors": 100, "status": "OK"},  # Highest
        10: {"avg_mfe": 0.012, "neighbors": 100, "status": "OK"},
    }

    best_h, stats = engine.select_best_horizon(results)
    assert best_h == 5

def test_filters_low_mfe():
    results = {
        3: {"avg_mfe": 0.0005, "neighbors": 100, "status": "OK"},  # Below min
        5: {"avg_mfe": 0.002, "neighbors": 100, "status": "OK"},
    }

    best_h, stats = engine.select_best_horizon(results, min_mfe=0.001)
    assert best_h == 5  # Only qualifying option
```

### Integration Tests

```python
def test_backtest_uses_different_horizons():
    """Verify that trades use different horizons based on state."""
    results = run_adaptive_backtest(horizons=[3, 5, 10])

    # Should have variety in selected horizons
    horizon_counts = Counter(t["horizon"] for t in results["trades"])
    assert len(horizon_counts) > 1  # Not all same horizon
```

---

## Implementation Order

1. **Phase 1**: Generate multi-horizon outcome datasets [1 hour]
2. **Phase 2**: Build FAISS indices per horizon [30 min]
3. **Phase 3**: Create AdaptiveHorizonEngine class [2 hours]
4. **Phase 4**: Integration with backtester [1 hour]
5. **Phase 5**: Grid search for adaptive mode [1 hour]
6. **Phase 6**: Unit tests [1 hour]
7. **Phase 7**: Validation backtest [1 hour]

**Total estimated effort: ~7-8 hours**

---

## Expected Benefits

1. **Data-driven horizon selection**: No assumptions, direct historical lookup
2. **State-specific optimization**: Each state gets its optimal horizon
3. **Better trade quality**: Only takes trades where a horizon has good MFE
4. **Flexibility**: Easy to add/remove horizons from the comparison

---

## Sign-Off

- [ ] Plan reviewed
- [ ] Multi-horizon outcome datasets generated
- [ ] FAISS indices built per horizon
- [ ] AdaptiveHorizonEngine implemented
- [ ] Backtester integration complete
- [ ] Tests passing
- [ ] Validation backtest run
- [ ] Grid search results analyzed
