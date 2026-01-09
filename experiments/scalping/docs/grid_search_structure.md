# Grid Search Results - File Structure Plan

> **STATUS: PLANNED** (Created: 2026-01-08)

## Current Structure (Problem)

```
experiments/scalping/grid_search/
├── scalping_BATCH_1_20260108_145921.csv
├── scalping_BATCH_2_20260108_145921.csv
├── scalping_BATCH_FINAL_20260108_022417.csv
└── ... (flat files, no organization)
```

**Issues:**
- No separation by horizon
- No metadata about parameters used
- Hard to compare experiments
- Will get messy with many runs

---

## Proposed Structure

```
experiments/scalping/grid_search/
├── h3/
│   ├── exp_20260108_201451/
│   │   ├── results.csv           # All combination results
│   │   ├── metadata.json         # PARAM_GRID + best params
│   │   └── batches/              # Optional: intermediate batch files
│   │       ├── batch_01.csv
│   │       └── batch_02.csv
│   └── exp_20260109_103022/
│       ├── results.csv
│       └── metadata.json
├── h5/
│   └── exp_20260108_154500/
│       ├── results.csv
│       └── metadata.json
├── h10/
│   └── ...
└── multi_horizon/
    └── exp_20260110_120000/
        ├── results.csv
        └── metadata.json
```

---

## Metadata JSON Format

```json
{
  "experiment": {
    "name": "h3_optimized_v1",
    "horizon": 3,
    "run_date": "2026-01-08 20:14:51",
    "run_id": "exp_20260108_201451"
  },
  "param_grid": {
    "horizon": [3],
    "normalization_window": [180, 300],
    "min_expectancy": [0.0, 0.0001, 0.0003],
    "max_distance": [3.0, 4.0, 5.0, 6.0],
    "k": [100, 150, 200],
    "min_mfe": [0.0005, 0.001, 0.0015, 0.002],
    "max_bars_in_trade": [0, 3, 5],
    "sample_interval": [1, 3],
    "blocked_regimes": [[]]
  },
  "data": {
    "sample_size": 500000,
    "train_ratio": 0.7,
    "data_start": "2024-03-02",
    "data_end": "2024-12-14",
    "pair": "BTCUSDT"
  },
  "execution": {
    "total_combinations": 864,
    "total_batches": 8,
    "batch_size": 108,
    "runtime_seconds": 3600,
    "n_workers": 2
  },
  "results_summary": {
    "profitable_combinations": 45,
    "profitable_pct": 5.2,
    "total_trades": 1250,
    "best_pnl": 2.45
  },
  "best_params": {
    "horizon": 3,
    "normalization_window": 300,
    "min_expectancy": 0.0,
    "max_distance": 4.0,
    "k": 150,
    "min_mfe": 0.001,
    "max_bars_in_trade": 3,
    "sample_interval": 1,
    "blocked_regimes": [],
    "total_pnl": 2.45,
    "win_rate": 68.5,
    "total_trades": 35,
    "profit_factor": 1.85,
    "sharpe": 1.2
  }
}
```

---

## Implementation Changes

### 1. Output Directory Logic

```python
# Current
output_dir = PROJECT_ROOT / "experiments" / "scalping" / "grid_search"

# New
horizon = PARAM_GRID["horizon"][0] if len(PARAM_GRID["horizon"]) == 1 else "multi"
run_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
output_dir = PROJECT_ROOT / "experiments" / "scalping" / "grid_search" / f"h{horizon}" / run_id
```

### 2. Save Metadata Function

```python
def save_metadata(output_dir: Path, param_grid: dict, results_df: pd.DataFrame,
                  runtime: float, data_info: dict):
    """Save experiment metadata as JSON."""

    best_row = results_df.iloc[0]  # Sorted by PnL descending

    metadata = {
        "experiment": {
            "name": f"h{param_grid['horizon'][0]}_grid_search",
            "horizon": param_grid["horizon"][0] if len(param_grid["horizon"]) == 1 else "multi",
            "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": output_dir.name
        },
        "param_grid": param_grid,
        "data": data_info,
        "execution": {
            "total_combinations": len(results_df),
            "runtime_seconds": int(runtime),
            "batch_size": BATCH_SIZE,
        },
        "results_summary": {
            "profitable_combinations": len(results_df[results_df['total_pnl'] > 0]),
            "profitable_pct": round(len(results_df[results_df['total_pnl'] > 0]) / len(results_df) * 100, 1),
            "best_pnl": float(best_row['total_pnl'])
        },
        "best_params": {
            "horizon": int(best_row['horizon']),
            "normalization_window": int(best_row['norm_window']),
            "min_expectancy": float(best_row['min_expectancy']),
            "max_distance": float(best_row['max_distance']),
            "k": int(best_row['k']),
            "min_mfe": float(best_row['min_mfe']),
            "max_bars_in_trade": int(best_row['max_bars_in_trade']),
            "sample_interval": int(best_row['sample_interval']),
            "total_pnl": float(best_row['total_pnl']),
            "win_rate": float(best_row['win_rate']),
            "total_trades": int(best_row['total_trades']),
        }
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
```

### 3. Final Save Logic

```python
# Save final results
results_df = pd.DataFrame(all_results).sort_values("total_pnl", ascending=False)
results_df.to_csv(output_dir / "results.csv", index=False)

# Save metadata
save_metadata(
    output_dir=output_dir,
    param_grid=PARAM_GRID,
    results_df=results_df,
    runtime=total_time,
    data_info={
        "sample_size": SAMPLE_SIZE,
        "train_ratio": train_ratio,
        "pair": pair,
    }
)

print(f"\nResults saved to: {output_dir}")
print(f"  - results.csv")
print(f"  - metadata.json")
```

---

## Usage Examples

### Find Best Parameters for h=3

```python
import json
from pathlib import Path

# Find all h=3 experiments
h3_dir = Path("experiments/scalping/grid_search/h3")
for exp_dir in sorted(h3_dir.iterdir()):
    with open(exp_dir / "metadata.json") as f:
        meta = json.load(f)
    print(f"{exp_dir.name}: PnL=${meta['best_params']['total_pnl']:.2f}")
```

### Compare Experiments

```python
import pandas as pd
import json

experiments = []
for h_dir in Path("experiments/scalping/grid_search").iterdir():
    if h_dir.is_dir():
        for exp_dir in h_dir.iterdir():
            meta_file = exp_dir / "metadata.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                experiments.append({
                    "horizon": meta["experiment"]["horizon"],
                    "run_date": meta["experiment"]["run_date"],
                    "best_pnl": meta["best_params"]["total_pnl"],
                    "profitable_pct": meta["results_summary"]["profitable_pct"],
                })

df = pd.DataFrame(experiments)
print(df.sort_values("best_pnl", ascending=False))
```

---

## Migration Plan

1. Keep existing flat files (don't delete)
2. Implement new structure for future runs
3. Optionally migrate old results with a script

---

## Files to Modify

1. `experiments/scalping/scripts/run_scalping_grid_search_batch.py`
   - Add `import json`
   - Update `output_dir` logic
   - Add `save_metadata()` function
   - Call `save_metadata()` at end of run

2. Create helper script (optional):
   - `experiments/scalping/scripts/analyze_experiments.py`
   - Compare experiments, find best params across runs
