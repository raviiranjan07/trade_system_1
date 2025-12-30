# Project Restructuring Plan

**Status:** PENDING (Execute after all grid search experiments complete)

## Current Structure (Problems)

```
trade_system_1/
├── data/
│   ├── raw/
│   │   └── ohlcv_loader.py      ❌ Code in data folder
│   └── validators/               ❌ Code in data folder (if .py files)
├── backtest/
├── config/
├── decision/
├── features/
├── outcomes/
├── pipeline/
├── regime/
├── similarity/
├── state/
├── tests/
├── run_backtest.py               ⚠️ Scripts in root
├── run_grid_search.py            ⚠️ Scripts in root
├── run_pipeline.py               ⚠️ Scripts in root
├── run_visualizations.py         ⚠️ Scripts in root
└── exceptions.py                 ⚠️ Module in root
```

## Target Structure (Standard)

```
trade_system_1/
│
├── config/                        # Configuration only
│   └── config.yaml
│
├── src/                           # All source code
│   ├── __init__.py
│   ├── exceptions.py              # Moved from root
│   ├── loaders/
│   │   ├── __init__.py
│   │   └── ohlcv_loader.py        # Moved from data/raw/
│   ├── features/                  # Moved
│   ├── state/                     # Moved
│   ├── regime/                    # Moved
│   ├── outcomes/                  # Moved
│   ├── similarity/                # Moved
│   ├── decision/                  # Moved
│   ├── backtest/                  # Moved
│   ├── pipeline/                  # Moved
│   └── utils/                     # New - common utilities
│
├── scripts/                       # Executable scripts
│   ├── run_pipeline.py
│   ├── run_backtest.py
│   ├── run_grid_search.py
│   └── run_visualizations.py
│
├── data/                          # DATA ONLY (no .py files!)
│   ├── ohlcv/                     # Raw OHLCV data
│   ├── state_vectors/             # Processed state vectors
│   ├── outcomes/                  # Outcome labels
│   ├── regimes/                   # Regime labels
│   └── results/
│       └── grid_search/
│           ├── h5/
│           ├── h10/
│           ├── h15/
│           └── h30/
│
├── tests/                         # Test files
│   ├── __init__.py
│   ├── unit/                      # Unit tests
│   └── grid_search/               # Grid search experiments
│       ├── base.py
│       ├── h5/
│       ├── h10/
│       ├── h15/
│       └── h30/
│
├── logs/
├── notebooks/                     # Jupyter notebooks (optional)
├── requirements.txt
├── README.md
├── EXPERIMENTS.txt
└── .env
```

## Step-by-Step Execution Plan

### Phase 1: Create New Structure

```bash
# Create src/ directory structure
mkdir -p src/loaders src/utils

# Create scripts/ directory
mkdir -p scripts
```

### Phase 2: Move Code Files

| From | To |
|------|-----|
| `data/raw/ohlcv_loader.py` | `src/loaders/ohlcv_loader.py` |
| `data/validators/*.py` | `src/validators/` (if exists) |
| `exceptions.py` | `src/exceptions.py` |
| `backtest/` | `src/backtest/` |
| `decision/` | `src/decision/` |
| `features/` | `src/features/` |
| `outcomes/` | `src/outcomes/` |
| `pipeline/` | `src/pipeline/` |
| `regime/` | `src/regime/` |
| `similarity/` | `src/similarity/` |
| `state/` | `src/state/` |
| `run_*.py` | `scripts/run_*.py` |

### Phase 3: Update Imports

All imports need to change from:
```python
from similarity.similarity_engine import SimilarityEngine
from decision.decision_engine import DecisionEngine
```

To:
```python
from src.similarity.similarity_engine import SimilarityEngine
from src.decision.decision_engine import DecisionEngine
```

### Phase 4: Update Config Paths

Update `config/config.yaml` if any paths reference old structure.

### Phase 5: Clean Up Data Directory

Remove any `.py` files from `data/`:
```bash
# Remove code files from data/
rm data/raw/ohlcv_loader.py
rm data/__init__.py
rm -rf data/__pycache__
rm -rf data/validators/  # if contains .py files
```

### Phase 6: Move Results

```bash
# Rename data/grid_search to data/results/grid_search
mkdir -p data/results
mv data/grid_search data/results/
```

### Phase 7: Test Everything

```bash
# Test imports work
python -c "from src.similarity.similarity_engine import SimilarityEngine"

# Test scripts work
python scripts/run_pipeline.py --help
python scripts/run_backtest.py --help
```

## Files That Need Import Updates

1. `scripts/run_pipeline.py`
2. `scripts/run_backtest.py`
3. `scripts/run_grid_search.py`
4. `scripts/run_visualizations.py`
5. `tests/grid_search/base.py`
6. All experiment files in `tests/grid_search/h*/`
7. All module `__init__.py` files

## Estimated Time

- Phase 1-2: 10 minutes (file moves)
- Phase 3-4: 30-45 minutes (import updates)
- Phase 5-7: 15 minutes (cleanup & testing)

**Total: ~1 hour**

## Rollback Plan

If something breaks:
```bash
git checkout .  # Revert all changes
```

Make sure to commit current state before restructuring.

---

**Execute this plan after all grid search experiments (H5, H10, H15, H30) are complete.**
