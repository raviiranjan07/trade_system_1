# MLOps Implementation Plan

**Status:** DRAFT — pending approval
**Approach:** Option 3 — staged adoption of standard MLOps, triggered by real pain
**Owner:** raviranjan
**Last updated:** 2026-04-12

---

## 1. Problem Statement & Goals

### Pain points we're solving

1. **"I can't find the experiment I ran 2 weeks ago."** Old experiments live in ad-hoc folders with inconsistent naming. Some have configs, some don't. Some have results, some don't.
2. **"I can't reproduce the result from 2 weeks ago."** Configs aren't versioned with results. Git commits aren't tagged to runs. Dataset versions aren't tracked. If you change a threshold and rebuild, the old results are unreachable.
3. **"I can't compare runs directly."** Different experiments output different file structures and different metric names, so comparison requires manual work each time.
4. **"I'm losing work."** The SR folder got deleted earlier in this session. Nothing was in git. There was no safety net.

### Goals (in priority order)

1. Every experiment run is **tracked**: params, metrics, artifacts, git commit, config snapshot, timestamp.
2. Every experiment produces **standardized output** conforming to a locked protocol (same file names, same metric keys, same structure).
3. Every run is **reproducible** from its logged state (config + git commit + dataset version).
4. Every experiment's results are **directly comparable** via a leaderboard script.
5. All of the above with **minimum friction** — the runner should be one function call.

### Non-goals (explicit)

- Model deployment to production — not yet.
- Serving models as HTTP APIs — not yet.
- Model drift monitoring — not yet.
- Distributed training orchestration — never for solo research.
- Cloud-hosted MLflow / team collaboration features — not needed.
- Replacing human judgment with automated experiment selection — always a bad idea.

---

## 2. Architecture Overview

### Standard MLOps — the 6 pillars

Industry reference architectures (Google, MLOps Community, Martin Fowler) define roughly 6 pillars:

1. **Experiment tracking** — params, metrics, artifacts, code version per run
2. **Data versioning** — datasets versioned like code, lineage tracked
3. **Pipeline orchestration** — reproducible multi-stage workflows
4. **Model registry** — versioned models with stage tags
5. **Model serving** — deploy models as APIs with rollback
6. **Monitoring** — drift detection, performance tracking in production

### Our adoption strategy: staged (Option 3)

We adopt each pillar only when a real pain point justifies it. Adopting all 6 upfront is a common solo-researcher mistake — weeks of setup for tools that go unused.

| Stage | Pillar | Tool | Trigger | Status |
|---|---|---|---|---|
| 1 | Experiment tracking + protocol layer | MLflow + Python runner | **NOW** — multiple pain points | Planned (this doc) |
| 2 | Data versioning | DVC | When ≥3 meaningful dataset versions exist | Deferred |
| 3 | Pipeline orchestration | Prefect (not Airflow) | When multi-stage pipelines run repeatedly | Deferred |
| 4 | Model registry | MLflow Model Registry (built in) | When multiple models compete for "production" | Deferred |
| 5 | Model serving | MLflow pyfunc / BentoML | When bot actually consumes a model | Deferred |
| 6 | Monitoring | Evidently AI / custom | When a model is in production for weeks | Deferred |

### End state (after all stages)

```
Code (git)
  │
  ▼
Pipeline Runner (Prefect) ─► Dataset (DVC-versioned)
  │                                │
  │                                ▼
  └─► Training ──► Evaluation ──► Metrics, Artifacts (MLflow)
                                   │
                                   ▼
                              Model Registry (MLflow)
                                   │
                                   ▼
                              Serving (Bot / pyfunc)
                                   │
                                   ▼
                              Monitoring (Evidently)
```

Today we build **only the first box and its MLflow output**. Everything else is a future addition.

---

## 3. Stage 1: Experiment Tracking + Protocol Layer (ADOPT NOW)

This is the only stage being implemented in the first pass. Every other stage is roadmap-only in this document.

### 3.1 What Stage 1 solves

- **Pain point 1** (finding old experiments): solved via MLflow UI + registry CSV
- **Pain point 2** (reproducing old results): solved via config snapshots + git commit logging
- **Pain point 3** (comparing runs): solved via the protocol layer — every run produces identical output structure
- **Pain point 4** (losing work): partially solved — runs live under `experiments/**/runs/` which is gitignored but visible; MLflow keeps a separate store. The remaining gap (accidental file deletion) requires the user to commit important files to git, which MLOps can't force.

### 3.2 Key concept — the protocol layer

A **protocol** defines everything that stays fixed for a given problem:
- What is predicted (the label)
- How data is split (train/val/test date ranges)
- What metrics are computed
- What files are produced

A **config** defines everything that varies per run:
- Features used
- Model architecture
- Training hyperparameters
- Random seed

The protocol is **locked**. Configs **vary**. The runner enforces the protocol so every run produces identical outputs.

**Why this matters:** without a protocol, "standard output" is just a promise. With a protocol, the runner fails loudly if a run doesn't produce the expected files or metric keys.

### 3.3 Components

```
src/mlops/
  __init__.py
  git_utils.py         # get_git_info() — commit, branch, dirty flag
  tracking.py          # thin MLflow wrapper (so MLflow is swappable later)
  protocol.py          # Protocol dataclass + loader + validator
  evaluation.py        # standard evaluators per protocol
  registry.py          # append row to registry.csv atomically
  runner.py            # run_experiment() context manager — main entry point
  errors.py            # ProtocolViolationError, RunnerError

experiments/
  protocols/           # one YAML per protocol (locked, versioned in git)
    sr_bounce_break_v1.yaml
    strategy_backtest_v1.yaml
    layer2_direction_v1.yaml
    feature_validation_v1.yaml
  <experiment_name>/
    config.yaml        # the config being tested (varies)
    runs/              # per-run artifacts (gitignored)
      <run_id>/
        config.yaml
        protocol.yaml
        git_info.txt
        stdout.log
        metrics.json
        predictions.parquet
        plots/
        artifacts/
          model.pt
  registry.csv         # master index (git-tracked)

mlruns/                # MLflow tracking store (gitignored)
scripts/
  leaderboard.py       # aggregate metrics.json across runs
  prune_old_runs.py    # cleanup utility

docs/
  MLOPS.md             # usage guide (written alongside Stage 1 code)
  MLOPS_PLAN.md        # this document
```

### 3.4 File-level spec for Stage 1

#### `src/mlops/git_utils.py`
```
get_git_info() -> dict:
  returns {
    'commit': str (7-char short hash),
    'commit_full': str (40-char full hash),
    'branch': str,
    'dirty': bool,
    'last_msg': str (one line)
  }
  gracefully handles "not a git repo" case
```

#### `src/mlops/tracking.py`
Thin wrapper over MLflow. Functions:
```
init(tracking_uri='./mlruns')
start_run(experiment_name, run_name=None) -> run_id
log_params(dict)
log_metric(key, value, step=None)
log_metrics(dict, step=None)
log_artifact(path)
set_tag(key, value)
end_run(status='FINISHED')
```

Reason for the wrapper: if MLflow is ever replaced, experiments don't need to change — only this file does.

#### `src/mlops/protocol.py`
```python
@dataclass(frozen=True)
class Protocol:
    name: str                         # e.g., "sr_bounce_break_v1"
    version: int
    description: str
    label_spec: dict                  # how the label is computed
    data_split: dict                  # train/val/test date ranges
    required_metrics: list[str]       # must be in metrics.json
    optional_metrics: list[str]
    required_artifacts: list[str]     # must exist in runs/<id>/
    baseline: str                     # name of baseline function
    tags: dict

def load_protocol(path: str) -> Protocol
def validate_metrics(metrics: dict, protocol: Protocol) -> None  # raises on violation
def validate_artifacts(run_dir: Path, protocol: Protocol) -> None
```

#### `src/mlops/evaluation.py`
Standard evaluators — one function per protocol that takes `(predictions, labels, protocol)` and returns a `metrics_dict` with all required keys. Example:
```
evaluate_sr_bounce_break(preds, labels, zone_width_bps, protocol) -> dict
  returns:
    test_accuracy, test_precision_bounce, test_precision_break,
    test_recall_bounce, test_recall_break, test_f1,
    test_confusion_matrix (2x2),
    val_accuracy, train_accuracy,
    baseline_accuracy (50% for balanced),
    delta_vs_baseline,
    # trading metrics
    signal_win_rate, expected_value_bps, profit_factor,
    # plots (paths to saved PNGs)
    confusion_matrix_plot, roc_curve_plot, calibration_plot
```

#### `src/mlops/registry.py`
```
@dataclass
class RegistryRow:
    run_id, experiment_name, protocol_name, status,
    start_time, duration_s,
    git_commit, git_dirty, git_branch,
    config_path, artifacts_dir,
    primary_metric_name, primary_metric_value,
    notes

def append_run(row: RegistryRow, path='experiments/registry.csv')
  # creates file with header if not exists
  # file locking for concurrent safety
```

#### `src/mlops/runner.py`
The main entry point. Context manager:

```python
with run_experiment(
    experiment_name="stage9_sr_advisor",
    protocol_name="sr_bounce_break_v1",
    config_path="experiments/brain/SR/config.yaml",
    params={"model": "conv1d", "lr": 0.001},
    primary_metric="test_accuracy",
    notes="First Stage 9 run — baseline",
) as run:
    # run code here
    run.log_metric("test_accuracy", 0.547)
    run.log_artifact("model.pt")
    run.log_predictions(predictions_df)  # validated against protocol
```

**On entry:**
1. Load protocol from `experiments/protocols/<protocol_name>.yaml`
2. Validate config against protocol (all required fields present)
3. Generate run_id: `YYYY-MM-DD_HHMMSS_<6char>`
4. Create folder `experiments/<experiment_name>/runs/<run_id>/`
5. Snapshot config file, protocol file, git info into the folder
6. Start MLflow run; log params and git info as tags
7. Tee stdout/stderr to `stdout.log`
8. Return `Run` object

**On exit (normal):**
1. Validate all required metrics and artifacts exist
2. Write `metrics.json` from accumulated metrics
3. Finalize MLflow run with status='FINISHED'
4. Append registry row
5. Print summary: run_id, folder, primary metric, delta vs baseline

**On exit (exception):**
1. Finalize MLflow run with status='FAILED'
2. Write `error.log` with traceback
3. Append registry row with status='FAILED'
4. Re-raise

### 3.5 The first protocol — `sr_bounce_break_v1`

This is the protocol for Stage 9 (S/R bounce/break prediction).

```yaml
name: sr_bounce_break_v1
version: 1
description: |
  Binary classification: bounce vs break at S/R zone touch.
  Uses 15-min BTC data with hybrid KDE zones.

label_spec:
  type: binary
  classes: [break, bounce]
  horizon_bars: 25
  threshold_bps: 15
  direction_normalized: true  # support: bounce=UP, resistance: bounce=DOWN

data_split:
  train: ["2020-01-01", "2022-11-30"]
  val:   ["2023-01-01", "2023-11-30"]
  test:  ["2024-01-01", "2025-12-31"]
  gap_months: 1

required_metrics:
  # ML metrics
  - test_accuracy
  - test_precision_bounce
  - test_precision_break
  - test_recall_bounce
  - test_recall_break
  - test_f1
  - test_confusion_matrix
  - val_accuracy
  - train_accuracy
  - baseline_accuracy
  - delta_vs_baseline
  # Trading metrics
  - signal_win_rate
  - expected_value_bps_after_fees
  - profit_factor
  - avg_winner_bps
  - avg_loser_bps
  # Per-class size
  - n_train
  - n_val
  - n_test
  - class_balance_train
  - class_balance_test

required_artifacts:
  - metrics.json
  - predictions.parquet
  - plots/confusion_matrix.png
  - plots/roc_curve.png
  - plots/calibration.png
  - artifacts/model.pt

baseline:
  name: always_predict_majority
  description: Predict the majority class from train. Expected ~50% on balanced test.

tags:
  stage: 9
  problem: sr_bounce_break
  timeframe: 15m
  asset: BTCUSDT
```

### 3.6 Other protocols (Option c scope)

Per your choice of option (c), here are **sketches** of protocols for the other problem areas in `experiments/`. Each will be implemented as a YAML file under `experiments/protocols/` when its respective experiment is next run — not all at once.

#### `strategy_backtest_v1` — for V12, V13, V1.3.x backtests
```
Label: trade outcome (win/loss) + size in bps
Split: OOS = 2024-2025 (consistent with memory)
Required metrics:
  - total_bps, profit_factor, win_rate, n_trades
  - max_drawdown_bps, max_drawdown_duration_bars
  - sharpe_15m, sharpe_daily
  - avg_winner_bps, avg_loser_bps, largest_loss_bps
  - per_year_bps (2024, 2025)
  - per_quarter_bps (8 quarters)
  - per_signal_type breakdown (V12_LONG, V12_SHORT, BEAR_LONG, BULL_SHORT)
Required artifacts:
  - trades.parquet (full trade log)
  - equity_curve.png
  - drawdown_plot.png
  - per_quarter_bars.png
  - signal_type_breakdown.csv
Baseline:
  V1.2 = +3,250 bps, PF 2.47, 56.9% win (as frozen benchmark)
```

#### `layer2_direction_v1` — for Layer 2 ML models
```
Label: next-N-bar direction (up/down), configurable N
Split: 2020-2022 train, 2023 val, 2024-2025 test
Required metrics:
  - test_accuracy, test_precision, test_recall, test_f1
  - test_auc, test_log_loss
  - calibration_score
  - per_decile_accuracy
  - feature_importance (top 20)
Required artifacts:
  - predictions.parquet (bar, pred, proba, label)
  - feature_importance.csv
  - plots/calibration.png, plots/per_decile_acc.png
Baseline:
  50% random + "always predict majority" + prior L2 best (57-58%)
```

#### `feature_validation_v1` — for feature validation / WHAT analysis
```
Label: generic (depends on what's being validated)
Split: train 2020-2022, OOS 2024-2025
Required metrics:
  - feature_separation_pct (bounce vs break)
  - univariate_auc
  - combination_score (top 10 2-feature combinations)
Required artifacts:
  - feature_stats.csv
  - plots/feature_distribution.png
  - plots/separation_bar.png
Baseline: 0% separation (null hypothesis)
```

#### `exit_strategy_v1` — for exit strategy experiments
```
Label: trade outcome given exit rules
Split: OOS = 2024-2025
Required metrics:
  - total_bps, pf, win_rate, max_dd
  - avg_hold_bars
  - loss_conversion_rate (losers that became winners)
  - exit_type_breakdown (TRAILING_STOP, TIME_EXIT, EXIT_V2, etc.)
Required artifacts:
  - trades_comparison.parquet (baseline vs new)
  - plots/dd_comparison.png
Baseline: V1.3 exit rules
```

These four protocols + the Stage 9 protocol cover all current experiment categories in `experiments/`. If a new category appears (e.g., multi-asset), we add a new protocol YAML at that time.

**Important:** only `sr_bounce_break_v1` will be written and usable in the Stage 1 first pass. The other protocols are sketched here so we agree on the shape, but will be fully written when their respective experiments are next run.

### 3.7 The registry CSV schema (locked)

```csv
run_id,experiment_name,protocol_name,status,start_time,duration_s,
git_commit,git_dirty,git_branch,config_path,artifacts_dir,
primary_metric_name,primary_metric_value,notes
```

Schema is locked. Changing columns later requires migrating the file.

### 3.8 Example usage (after Stage 1 is built)

```python
# experiments/brain/SR/train_stage9.py
from mlops.runner import run_experiment
from mlops.evaluation import evaluate_sr_bounce_break

with run_experiment(
    experiment_name="stage9_sr_advisor",
    protocol_name="sr_bounce_break_v1",
    config_path="experiments/brain/SR/config_stage9.yaml",
    params={
        "model": "hierarchical_conv1d",
        "strict_threshold": 0.10,
        "history_n": 10,
        "lr": 0.001,
        "batch_size": 128,
    },
    primary_metric="test_accuracy",
    notes="Stage 9 baseline — Steps 1+2+3 architecture",
) as run:
    # ... load data, build model, train ...
    preds = model(X_test)

    # The evaluator computes ALL required metrics per protocol
    metrics = evaluate_sr_bounce_break(preds, y_test, zone_widths, run.protocol)

    run.log_metrics(metrics)
    run.log_artifact("model.pt")
    run.log_predictions(preds_df)  # validated against protocol schema
```

On completion:
```
Run stage9_sr_advisor/runs/2026-04-12_143022_a1b2c3 — FINISHED
Protocol: sr_bounce_break_v1
Primary: test_accuracy = 0.547  (delta_vs_baseline = +0.047)
Folder: experiments/stage9_sr_advisor/runs/2026-04-12_143022_a1b2c3
MLflow UI: http://localhost:5000/#/experiments/1/runs/<id>
```

### 3.9 Dependencies

One new package:
```
pip install mlflow
```

No other new dependencies in Stage 1.

### 3.10 Out of scope for Stage 1 (explicit)

- DVC / data versioning
- Prefect / Airflow / pipeline orchestration
- Model registry stages (Staging/Production tags)
- Model serving
- Monitoring / drift detection
- Cloud-hosted MLflow
- Hydra / complex parameter sweeps
- Distributed training
- Writing any protocols beyond `sr_bounce_break_v1`
- Running Stage 9 through the new system (that's the third pass)

### 3.11 Estimated effort

~1 day of implementation + ~half day of testing and documentation. ~600-800 lines of Python across the `src/mlops/` package plus YAML protocol and docs.

---

## 4. Stage 2: Data Versioning (DVC) — Roadmap Only

**Trigger to adopt:** when the repo has ≥3 meaningful dataset versions and tracking them by folder name becomes unreliable. (We're already close — Stage 9 was rebuilt twice with different thresholds. One more rebuild and this becomes worth adopting.)

**What it adds:**
- Dataset files tracked by hash, not filename
- Every run logs which dataset version it used
- Old runs stay reproducible with old datasets
- Remote store for datasets (can be a local folder, no cloud required)

**Tool:** DVC (industry standard, git-native, local-first)

**Integration with Stage 1:**
- The runner adds `dataset_version` to logged params and registry
- The evaluator reads dataset metadata to confirm the right version was loaded
- MLflow tags include `dvc.dataset_hash`

**Estimated effort when adopted:** ~half day setup + ongoing habit of `dvc add data/new_dataset.npz` before commits.

**Explicitly NOT adopted today:** DVC has setup overhead (remote store, init, ignore patterns) that's not worth it until we actually need to retrieve old dataset versions.

---

## 5. Stage 3: Pipeline Orchestration (Prefect) — Roadmap Only

**Trigger to adopt:** when a single experiment involves 3+ dependent scripts run in sequence, manually, more than 2-3 times.

**What it adds:**
- A `flow` (Prefect term for a pipeline) chains stages: `build → train → evaluate → log`
- Each stage is a Python function with declared inputs/outputs
- Automatic caching: skip stages whose inputs haven't changed
- Parallel execution when stages are independent
- Retry logic for flaky stages
- Local web UI for monitoring flow runs

**Tool:** Prefect (NOT Airflow)

**Why not Airflow:**
- Airflow requires a database, scheduler daemon, webserver — massive setup
- Airflow is designed for production pipelines with scheduled daily jobs
- Prefect is designed for data science workflows: install with pip, write Python functions, run immediately
- Prefect has a local UI (`prefect server start`) with no database setup required
- For solo research, Prefect is ~10x less setup for equivalent functionality

**Integration with Stage 1:**
- The `run_experiment()` context manager is used **inside** a Prefect task
- MLflow run is still the unit of tracking; Prefect is the unit of orchestration
- No conflict between the two

**Estimated effort when adopted:** ~1-2 days to install, write first flow, refactor existing scripts.

---

## 6. Stage 4: Model Registry — Roadmap Only

**Trigger to adopt:** when there are 3+ trained models competing for "best for production" and they need to be tracked with lifecycle stages.

**What it adds:**
- Named model versions: `stage9_sr_advisor v1, v2, v3, ...`
- Lifecycle tags: `None`, `Staging`, `Production`, `Archived`
- One-line API to load the current production model: `mlflow.pyfunc.load_model("models:/stage9_sr_advisor/Production")`
- Model lineage: click a model in the UI, see the exact run that produced it

**Tool:** MLflow Model Registry (built into MLflow — zero new dependencies)

**Integration with Stage 1:**
- The runner already logs models as artifacts; Stage 4 adds a `register_model=True` flag
- When set, the model is automatically registered and tagged

**Estimated effort when adopted:** ~1 hour. Nothing to install, mostly just using the feature that's already in MLflow.

---

## 7. Stage 5: Model Serving — Roadmap Only

**Trigger to adopt:** when a trained model needs to be consumed by something else (the V12 bot, a dashboard, an API).

**What it adds:**
- Load a model by name/version, not a file path
- Swap models without changing consumer code
- Input validation: model rejects inputs that don't match its expected schema
- Two deployment styles:
  - **In-process:** `mlflow.pyfunc.load_model()` — the bot imports and calls the model as a Python function
  - **HTTP endpoint:** `mlflow models serve -m models:/stage9/Production` — the bot POSTs JSON to a local HTTP server

**Tool:** MLflow `pyfunc` (simplest) or BentoML (if we outgrow pyfunc)

**Integration with Stage 1:**
- The runner logs models using `mlflow.pytorch.log_model()` which produces a loadable artifact
- Stage 5 just enables consumption — nothing changes in Stage 1 itself

**Estimated effort when adopted:** ~half day for in-process; ~1 day if we need HTTP serving.

---

## 8. Stage 6: Monitoring — Roadmap Only

**Trigger to adopt:** when a model has been running in production (i.e., feeding the bot) for at least 2-3 weeks, and we need to know if its accuracy is degrading.

**What it adds:**
- Log predictions + actuals as they happen
- Scheduled job to compute:
  - Input feature distribution vs training distribution (data drift)
  - Prediction distribution vs expected (prediction drift)
  - Actual accuracy vs training accuracy (concept drift)
- Alerts when any metric crosses a threshold

**Tool:** Evidently AI (free, open source, good reports) or custom scripts writing to MLflow metrics

**Integration with Stage 1:**
- Monitoring reads from the MLflow run that registered the production model — so the training-time distributions are available to compare against
- Monitoring is its own flow (Stage 3 Prefect task) that runs on a schedule

**Estimated effort when adopted:** ~1-2 days for first monitoring pipeline.

---

## 9. Deferred / Not in Scope

These are things that would come up in a discussion about MLOps but are explicitly **not** being adopted:

| Item | Why not |
|---|---|
| **Airflow** | Overkill for solo research. Prefect (Stage 3) is the right tool. |
| **Kubeflow / Kubernetes** | Requires cluster setup. Irrelevant for a single-machine research workflow. |
| **Cloud MLflow (Databricks, etc.)** | Local MLflow is sufficient. No team collaboration need. |
| **Hydra / complex config sweeps** | Simple YAML + a Python loop is enough. Hydra adds learning curve without clear value yet. |
| **Distributed training** | Models are small (hundreds to thousands of params). Single-GPU is more than enough. |
| **Automated hyperparameter tuning (Optuna, Ray Tune)** | Manual experimentation with clear intent is currently more valuable than random/bayesian search. Revisit when hand-tuning hits diminishing returns. |
| **Feature store (Feast, Tecton)** | Useful when the same features are consumed by many models. Not our situation. |
| **CI/CD for models (GitHub Actions training)** | No need to train on push; research is manual. |
| **A/B testing infrastructure** | Not yet serving models to users. |
| **Data quality monitoring (Great Expectations)** | Useful eventually, not now. Evidently (Stage 6) covers what matters first. |

---

## 10. Trigger Conditions — Summary

| Stage | Adopt when |
|---|---|
| 1 | NOW (already justified by current pain) |
| 2 (DVC) | 3+ dataset versions exist and tracking by folder name fails |
| 3 (Prefect) | Running multi-script pipelines 3+ times manually |
| 4 (Model Registry) | 3+ model candidates for "production" |
| 5 (Serving) | A model needs to be consumed by the bot |
| 6 (Monitoring) | A production model has been running for weeks |

No stage is adopted on a schedule — only when the trigger fires.

---

## 11. Implementation Order (Once Approved)

1. **Plan review** (this document) → approval or revision
2. **Discuss Stage 1 in detail** — file-level walkthrough, any changes
3. **Write Stage 1 code** — ~600-800 lines across `src/mlops/`
4. **Review Stage 1 code** — file by file before commit
5. **Install MLflow** — `pip install mlflow` (you run, not me)
6. **Test the runner** — with a throwaway experiment, confirm everything works
7. **Commit Stage 1 to git** — with a clear message
8. **Retroactive cleanup** (Option A, as agreed) — separate plan, separate review
9. **Run Stage 9 through the new system** — becomes the first "real" tracked experiment

---

## 12. Open Questions for Discussion

Before code is written, these need explicit answers:

1. **Protocol location** — `experiments/protocols/` (proposed) or elsewhere?
2. **Registry location** — `experiments/registry.csv` (existing file — need to check if its schema conflicts with the new one)
3. **MLflow UI access** — how do you want to launch it? Manual `mlflow ui` command, or auto-start alongside the first run?
4. **Trading metrics in the SR protocol** — should `expected_value_bps_after_fees` assume 8bps round-trip fees (our standing value) or be configurable per run?
5. **Predictions storage format** — Parquet (proposed, fast, typed) or CSV (human-readable but slow)?
6. **What is the "primary metric" for Stage 9** — test_accuracy, or delta_vs_baseline (more meaningful but derived)?
7. **How aggressive should the protocol be** — raise hard errors on missing metrics, or log warnings? My proposal: hard errors, because the whole point is enforcement.

---

## 13. Glossary

- **Run** — a single execution of an experiment (one training + evaluation)
- **Experiment** — a named collection of runs (e.g., `stage9_sr_advisor`)
- **Protocol** — locked rules defining the problem (label, splits, metrics)
- **Config** — per-run parameters that vary (features, model, hyperparameters)
- **Artifact** — a file produced by a run (model, plot, prediction file)
- **Registry** — the CSV index of all runs across all experiments
- **MLflow store** — the on-disk database MLflow uses (`./mlruns/`)
- **Primary metric** — the single number used to rank a run (configurable per experiment)
- **Baseline** — the reference score the run is compared against (e.g., "always predict majority")

---

## Status

**Next step:** your review. Once approved, we discuss Stage 1 in detail and I write the code.
