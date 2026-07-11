# MLOps Usage Guide

> **STATUS (2026-07-12):** the 3-model lineage this guide's examples
> reference was retired in the clean-slate reset. The machinery described
> (runner, protocols, DVC, registry, promotion) is all current — read old
> model names in examples as placeholders for the new architecture.

## Quick Start

### Run a tracked experiment

```python
from mlops.runner import run_experiment
from mlops.evaluation import evaluate_direction_prediction

with run_experiment(
    experiment_name="direction_prediction",
    protocol_name="direction_prediction_v1",
    config_path="configs/base.yaml",
    params={"model": "lstm_attention", "temperature": 0.5},
    model_type="LSTMAttention",
    dataset_version="feature_cache_cleaned_23col",
    primary_metric="test_confident_accuracy",
    notes="Attention temp=0.5 evaluation",
) as run:
    # ... load data, build model, train ...

    metrics = evaluate_direction_prediction(probs, labels, mfe_up, mfe_down)
    run.log_metrics(metrics)
    run.log_artifact("model.pt")
```

### View past runs

**MLflow web UI:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Opens at http://localhost:5000. Browse, sort, filter, compare runs visually.

**Terminal leaderboard:**
```bash
PYTHONPATH=src python scripts/mlops/leaderboard.py --experiment direction_prediction
```
Prints a sorted table of all runs for that experiment.

**Registry CSV:**
```bash
cat experiments/mlops_registry.csv
```
One row per run. Grep-able, Excel-compatible, git-tracked.

---

## Data Structure (Standard MLOps Layout)

```
system_1/
├── data/
│   ├── raw/                               # Source data — never modified
│   │   ├── BTCUSDT_15m_ohlcv.parquet
│   │   └── BTCUSDT_1m_ohlcv.parquet
│   │
│   ├── features/                          # Computed features — ready for training
│   │   ├── direction_prediction/          # Per protocol
│   │   │   ├── feature_cache.parquet      # 23 columns (DVC tracked)
│   │   │   ├── feature_cache.parquet.dvc
│   │   │   ├── labels.parquet             # 30 label columns (DVC tracked)
│   │   │   └── labels.parquet.dvc
│   │   └── sr_bounce_break/               # Per protocol
│   │       ├── stage9/                    # Per dataset version
│   │       ├── every_bar/
│   │       └── stage9a_static/
│   │
│   ├── archive/                           # Old datasets — no longer actively used
│   │   ├── brain_datasets/
│   │   ├── sr_original/
│   │   ├── sr_entry_only/
│   │   └── sr_v2/
│   │
│   └── trades/                            # Trade logs + risk state
│       ├── trades_paper.csv
│       ├── trades_ml_paper.csv
│       ├── risk_state.json
│       └── risk_logs/
│
├── models/                                # Trained model artifacts
│   └── ML_V1/                     # Production V1.5
│       ├── direction_model.onnx
│       ├── direction_model.pt
│       └── scaler.npz
│
├── configs/
│   ├── base.yaml                          # Base pipeline config
│   ├── protocols/                         # Evaluation contracts
│   │   ├── direction_prediction_v1.yaml
│   │   └── sr_bounce_break_v1.yaml
│   ├── data_cards/                        # Dataset documentation
│   │   └── l2_003_feature_cache.yaml
│   └── model_cards/                       # Model documentation
│       ├── ML_V1.yaml
│       ├── lstm_gru_direction.yaml
│       ├── lstm_attention_direction.yaml
│       └── mlp_curriculum_direction.yaml
│
├── src/
│   ├── engine/                               # Trading bot + backtest
│   ├── brain/                             # ML pipeline (zone detection, features)
│   ├── mlops/                             # Experiment tracking
│   │   ├── git.py
│   │   ├── tracking.py
│   │   ├── protocol.py
│   │   ├── evaluation.py
│   │   ├── registry.py
│   │   └── runner.py
│   ├── trade_system/                      # Older trade system
│   └── web/                               # Dashboard
│
├── experiments/
│   ├── mlops_registry.csv                 # Run index (git-tracked)
│   ├── brain/                             # SR bounce/break experiments
│   ├── direction_prediction/              # Direction prediction runs
│   ├── exit_strategy/                     # Exit strategy experiments
│   ├── layer1/                            # Strategy + risk experiments
│   └── layer2/                            # Direction prediction research
│
├── scripts/
│   ├── mlops/                             # MLOps utilities
│   │   ├── leaderboard.py
│   │   └── log_colab_results.py
│   └── colab/                             # Google Colab scripts
│
├── docs/
│   ├── MLOPS.md                           # This file
│   └── MLOPS.md                           # This guide
│
├── mlflow.db                              # MLflow SQLite store (gitignored)
└── .dvc/                                  # DVC config + cache
```

### Key principle

| Directory | What it holds | Rule |
|---|---|---|
| `data/` | ALL data files | No code, no docs |
| `src/` | ALL source code | No data files |
| `experiments/` | Experiment scripts + docs + run results | No data files (moved to data/) |
| `configs/` | Protocols, data cards, model cards, configs | No code, no data |
| `models/` | Trained model artifacts | Only model files |
| `scripts/` | Utility + eval scripts | Grouped by purpose |

---

## DVC (Data Version Control)

### What DVC does

Versions your data files like git versions code. Each dataset gets a hash — if the file changes, DVC detects it and stores both versions.

### Currently tracked files

```
data/features/direction_prediction/feature_cache.parquet  (23 columns, 40 MB)
data/features/direction_prediction/labels.parquet          (30 columns, 32 MB)
```

### Common DVC commands

**Check if data changed:**
```bash
dvc status
```

**Track a new or changed dataset:**
```bash
dvc add data/features/sr_bounce_break/stage9
git add data/features/sr_bounce_break/stage9.dvc
git commit -m "Updated stage9 dataset"
```

**Restore an old dataset version:**
```bash
git checkout <old_commit> -- data/features/direction_prediction/feature_cache.parquet.dvc
dvc checkout
```

**Clean old cached versions:**
```bash
dvc gc --workspace
```

### Where DVC stores data

```
.dvc/cache/files/md5/    ← actual data files stored by hash
```

Each file is hashed (MD5). The `.dvc` pointer file (tiny, git-tracked) records the hash. The actual data (large, gitignored) lives in the cache.

---

## Data Cards

Document what's inside each dataset. Stored in `configs/data_cards/`.

**Current cards:**
- `l2_003_feature_cache.yaml` — direction prediction features + labels

**What a data card contains:**
- Identity (name, version, file paths, DVC hashes)
- Pipeline (how the data was built, step by step)
- Composition (columns, rows, distributions)
- Label details (what each label means, how it was computed)
- Known issues (bugs found, limitations)

**When to create:** every time you create a new dataset or significantly change an existing one.

---

## Model Cards

Document how each model works and what it achieved. Stored in `configs/model_cards/`.

**Current cards:**
- `ML_V1.yaml` — production MLP (deployed)
- `lstm_gru_direction.yaml` — LSTM baseline (rejected)
- `lstm_attention_direction.yaml` — LSTM + Attention (best accuracy)
- `mlp_curriculum_direction.yaml` — curriculum learning (didn't help)

**What a model card contains:**
- Architecture (layers, params, diagram)
- Data usage (which columns from which files)
- Training config (optimizer, lr, batch_size, split)
- Performance (all 26 metrics + backtest)
- What the model learned / cannot learn
- Known issues and caveats

**When to create:** every time you build a new model architecture.

---

## What the Runner Does Automatically

You don't need to do any of this manually — `run_experiment()` handles it:

1. Loads the protocol YAML and validates it
2. Gets git commit, branch, dirty flag
3. Creates a timestamped run folder
4. Snapshots config + protocol + git info into the folder
5. Starts an MLflow run, logs params and git tags
6. Captures stdout/stderr to a log file
7. On finish: validates metrics and artifacts against protocol
8. Writes metrics.json
9. Ends MLflow run
10. Appends a row to the registry CSV
11. Prints a summary

If the run crashes, it still logs everything it can (partial metrics, error traceback) and marks the run as FAILED in both MLflow and the registry.

---

## Protocols

A protocol locks 5 things so every run is comparable:

| Locked | Why |
|---|---|
| Label definition | Every run predicts the same thing |
| Train/val/test split dates | Every run evaluates on the same data |
| Required metrics (exact names) | Every run produces comparable numbers |
| Required artifacts | Nothing is accidentally missing |
| Baseline | You always know what "random" looks like |

Features, model architecture, hyperparameters, and horizon are NOT in the protocol — they vary per run and are logged in the config.

### Current protocols

| Protocol | Problem | Label |
|---|---|---|
| `direction_prediction_v1` | Which direction hits ±15bps first | LONG vs SHORT |
| `sr_bounce_break_v1` | Does price bounce or break at S/R zone | Bounce vs Break |

### Base metrics (must be in EVERY protocol)

Every protocol, regardless of problem type, must include these baseline performance analysis metrics. Copy this list when creating a new protocol, then add problem-specific metrics on top.

**Overall performance:**
```
test_accuracy                  — overall accuracy on test set
test_f1                        — macro F1 score
test_confusion_matrix          — full confusion matrix
n_test                         — number of test samples
class_balance_test             — fraction of positive class in test
baseline_accuracy              — what random/majority would score
delta_vs_baseline              — how much better than baseline
```

**Per-class analysis (replace `{class}` with your class names):**
```
test_precision_{class}         — precision per class
test_recall_{class}            — recall per class
test_f1_{class}                — F1 per class
```

**Required artifact:**
```
metrics.json                   — all metrics in one file
```

**Problem-specific metrics are added on top.** For example:
- Direction prediction adds: confidence thresholds, per-class confident accuracy, MFE/MAE per class, long/short ratio
- S/R bounce/break adds: trading metrics (win rate, profit factor, expected value after fees)

### Adding a new protocol

1. Start with the base metrics listed above
2. Add problem-specific metrics
3. Create `configs/protocols/<name>.yaml`
4. Reference it by name in `run_experiment(protocol_name="<name>")`
5. If needed, add an evaluator function in `src/mlops/evaluation.py`

---

## Standard Evaluators

`evaluation.py` provides evaluator functions that compute all required metrics for a protocol.

### `evaluate_direction_prediction(probs, labels, mfe_up_bps, mfe_down_bps)`

Returns dict with all 26 `direction_prediction_v1` required metrics including:
- Overall + per-class accuracy, precision, recall, F1
- Confidence analysis (per-class confident accuracy, long/short ratio)
- Magnitude (MFE/MAE per class on confident bars)

### `evaluate_sr_bounce_break(preds, labels, mfe_bps=None, mae_bps=None)`

Returns dict with all `sr_bounce_break_v1` required metrics. Pass MFE/MAE arrays to also get trading metrics (win rate, profit factor, expected value after 8bps fees).

---

## Run Object Methods

Inside the `with run_experiment(...) as run:` block:

| Method | What it does |
|---|---|
| `run.log_metric(key, value)` | Log one metric |
| `run.log_metrics(dict)` | Log multiple metrics at once |
| `run.log_artifact(path)` | Copy a file to run's artifacts folder + MLflow |
| `run.log_plot(path)` | Copy a plot to run's plots folder + MLflow |
| `run.set_tag(key, value)` | Set metadata tag on MLflow run |
| `run.set_note(text)` | Set human-readable note for this run |

---

## Maintenance

### Prune old runs
Run folders under `experiments/**/runs/` can grow. Delete old ones you don't need:
```bash
rm -rf experiments/<name>/runs/<old_run_id>
```

### DVC cache
```bash
dvc gc --workspace   # remove old cached versions not in current .dvc files
```

### MLflow database
```bash
# Back up and start fresh if mlflow.db gets large
cp mlflow.db mlflow_backup.db
rm mlflow.db
```

### Registry CSV
Git-tracked. Grows by one row per run. Never needs pruning.
