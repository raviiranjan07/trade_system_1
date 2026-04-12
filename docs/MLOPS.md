# MLOps Usage Guide

## Quick Start

### Run a tracked experiment

```python
from mlops.runner import run_experiment
from mlops.evaluation import evaluate_sr_bounce_break

with run_experiment(
    experiment_name="stage9_sr_advisor",
    protocol_name="sr_bounce_break_v1",
    config_path="experiments/brain/SR/config.yaml",
    params={"model": "conv1d", "lr": 0.001, "horizon": 25},
    primary_metric="test_accuracy",
    notes="Stage 9 baseline — Steps 1+2+3 architecture",
) as run:
    # ... load data, build model, train ...

    metrics = evaluate_sr_bounce_break(preds, labels)
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
PYTHONPATH=src python scripts/leaderboard.py --experiment stage9_sr_advisor
```
Prints a sorted table of all runs for that experiment.

**Registry CSV:**
```bash
cat experiments/mlops_registry.csv
```
One row per run. Grep-able, Excel-compatible, git-tracked.

---

## Folder Layout

```
system_1/
├── mlflow.db                          # MLflow SQLite tracking store (gitignored)
├── configs/
│   └── protocols/                     # Locked protocol YAMLs (git-tracked)
│       ├── direction_prediction_v1.yaml
│       └── sr_bounce_break_v1.yaml
├── src/
│   └── mlops/                         # MLOps Python package
│       ├── __init__.py
│       ├── git.py                     # get_git_info()
│       ├── tracking.py                # MLflow wrapper
│       ├── protocol.py                # protocol loader + validators
│       ├── evaluation.py              # standard metric evaluators
│       ├── registry.py                # CSV registry writer
│       └── runner.py                  # run_experiment() entry point
├── experiments/
│   ├── mlops_registry.csv             # auto-generated run index (git-tracked)
│   └── <experiment_name>/
│       └── runs/                      # per-run folders (gitignored)
│           └── <run_id>/
│               ├── config.yaml        # snapshot of config used
│               ├── protocol.yaml      # snapshot of protocol used
│               ├── git_info.txt       # commit, branch, dirty flag
│               ├── metrics.json       # all logged metrics
│               ├── stdout.log         # captured terminal output
│               ├── error.log          # traceback (only if run failed)
│               ├── plots/             # saved plots
│               └── artifacts/         # saved artifacts (model, etc.)
└── scripts/
    └── leaderboard.py                 # terminal leaderboard
```

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

These 10+ metrics are the minimum. Without them, you can't do basic performance analysis: overall accuracy, per-class breakdown, comparison to baseline, confusion patterns.

**Problem-specific metrics are added on top.** For example:
- Direction prediction adds: confidence thresholds, per-class confident accuracy, MFE/MAE per class, long/short ratio
- S/R bounce/break adds: trading metrics (win rate, profit factor, expected value after fees)

### Adding a new protocol

1. Start with the base metrics listed above
2. Add problem-specific metrics
3. Create `configs/protocols/<name>.yaml` with: name, label_spec, data_split, required_metrics, required_artifacts, baseline
4. Reference it by name in `run_experiment(protocol_name="<name>")`
5. If needed, add an evaluator function in `src/mlops/evaluation.py`

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

### Properties available on run:

| Property | What it is |
|---|---|
| `run.run_id` | Timestamped unique ID |
| `run.run_dir` | Path to this run's folder |
| `run.protocol` | The loaded Protocol object |
| `run.mlflow_run_id` | MLflow's internal run ID |

---

## Standard Evaluators

`evaluation.py` provides evaluator functions that compute all required metrics for a protocol. Use them instead of computing metrics manually.

### `evaluate_sr_bounce_break(preds, labels, mfe_bps=None, mae_bps=None)`

Returns dict with all `sr_bounce_break_v1` required metrics. Pass MFE/MAE arrays to also get trading metrics (win rate, profit factor, expected value after 8bps fees).

---

## Maintenance

### Prune old runs
Run folders under `experiments/**/runs/` can grow. Delete old ones you don't need:
```bash
rm -rf experiments/<name>/runs/<old_run_id>
```
The registry CSV row stays (it's just a record), but the artifacts are gone.

### MLflow database
`mlflow.db` grows with each run. For small models (~674 params), it stays small for hundreds of runs. If it ever gets large:
```bash
# Back up and start fresh
cp mlflow.db mlflow_backup.db
rm mlflow.db
# Next run creates a new database automatically
```

### Registry CSV
Git-tracked. Grows by one row per run. Never needs pruning — it's tiny.
