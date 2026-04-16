# Retrain Pipeline — Plan & Implementation Record

**Goal:** Replace the ad-hoc training process with a proper MLOps pipeline
runnable via a single command. Honest evaluation, no data leakage, mechanical
verification of every retrain.

**Why we needed this:**
- Old `feature_cache.parquet` builder was lost → couldn't retrain on new data
- Old training had data leakage (random split, scaler fit on full dataset) →
  reported +18,207 bps OOS was inflated ~9×; honest OOS is +1,283 bps
- No reproducibility, versioning, or safety checks before promotion

---

## Current architecture (as of 2026-04-16)

### Commands

```bash
# List available pipelines
python scripts/mlops/run_pipeline.py --list

# End-to-end test (full pipeline, cleans up MLflow versions it created)
python scripts/mlops/run_pipeline.py ml_v1 --test

# Production retraining (runs only stale stages)
python scripts/mlops/run_pipeline.py ml_v1

# Force full rebuild, keeps the new version (rare)
python scripts/mlops/run_pipeline.py ml_v1 --force
```

### Pipeline (6 DVC stages)

```
1. build_features     raw 15m OHLCV -> feature_cache.parquet  (23 cols)
2. build_labels       features     -> labels.parquet  (30 cols)
3. train_mlp_v15      features+labels -> ML_V1_staging/ + MLflow @staging
                      also writes training_manifest.json
4. verify_ml_v1       reads manifest, runs 11 generic checks
                      (split disjoint, scaler train-only, hashes, registry...)
5. backtest_staging   ML_V1_staging + 1m ticks -> metrics.json + trades.parquet
                      V1 exits fire via pm.on_tick() (not just on_bar)
6. verify_backtest    invariant checks on every trade
                      (STOP_LOSS net ~-18, PT_TARGET mfe>=80, etc.)
```

Any stage fails → DVC halts → downstream doesn't run → no bad model promoted.

### Namespaces

- Trading engine: `src/engine/` (was `src/v12/`)
- MLflow registered model: `ML_V1` (was `direction_v15`)
- Signal generator class: `MLV1` (was `DirectionV15`)

### Files (what the pipeline reads/writes)

| Role | File |
|---|---|
| Pipeline manifest | `configs/pipelines.yaml` |
| Tunable params | `configs/params.yaml` (single source of truth) |
| Pipeline DAG | `dvc.yaml` |
| Feature builder | `src/engine/build_features.py` |
| Label builder | `experiments/layer2/L2-003/stage_3/L2_003_stage3_labels.py` |
| Trainer | `src/engine/ml_train.py` |
| Signal generator (inference) | `src/engine/signals/ml_v1.py` |
| Backtest engine | `src/engine/backtest.py` |
| Training verifier | `src/mlops/verify.py` (generic, manifest-driven) |
| Backtest verifier | `scripts/mlops/verify_backtest.py` (V1 invariants) |
| Pipeline wrapper | `scripts/mlops/run_pipeline.py` |

---

## Implementation phases — DONE

### Phase 1 — build_features (DONE, commit 6e696b9)
- 23-column feature builder from raw 15m OHLCV
- Cross-platform (no PYTHONPATH prefix)
- DVC stage wired

### Phase 2 — honest training (DONE, commit 66a5d88)
- Date-based split: train 2020-23 / val 2024 / test 2025
- Scaler fit on train only
- Auto-registers as `ML_V1` @staging
- Writes `models/ML_V1_staging/` (NOT `models/ML_V1/` — staging doesn't touch live)

### Phase 3 — honest backtest (refactored)
- Replaces old `compare_models.py` with `backtest_staging.py`
- Single-model focus (no unfair comparison to leaky v1)
- **Critical fix:** V1 exits require 1m tick feed via `pm.on_tick()`, not just
  `pm.on_bar()`. Original backtest had `exit_version="v3"` set but V1 logic never
  fired. Fixed by loading 1m data and walking ticks within each 15m bar.

### Phase 4 — parameterization (params.yaml)
- Structure: global (exit, backtest) + per-model (ml_v1.*)
- DVC `params:` key per stage tracks relevant sections only
- Changing `exit.version` marks backtest stale; changing `ml_v1.training.lr`
  marks training stale
- **Threshold fix:** inference thresholds were duplicated (hardcoded in signal
  generator, different values in params.yaml). Now single source of truth in
  `params.yaml -> ml_v1.inference`.

### Phase 5 — generic verification framework
- **Training verifier** (`src/mlops/verify.py`) — manifest-based, model-agnostic.
  Training scripts write `training_manifest.json` declaring split method,
  scaler policy, data hashes, MLflow registration. Verifier reads manifest and
  runs generic checks. Any future model adds a manifest writer; verifier stays
  unchanged.
- **Backtest verifier** (`scripts/mlops/verify_backtest.py`) — per-trade
  invariant checks against V1 exit rules. Each exit_reason has mathematical
  properties that must hold (e.g., STOP_LOSS trades must have net in [-30..-10]).
  All 242 trades of 2025 OOS backtest currently pass.

### Phase 6 — multi-model pipeline wrapper
- `configs/pipelines.yaml` declares each model's stage list
- `scripts/mlops/run_pipeline.py` runs a named pipeline
- `--test` mode runs full pipeline, then restores MLflow registry to pre-test
  state (no version pollution from testing)

### Phase 7 — namespace cleanup
- `src/v12/` -> `src/engine/` (267 subs across 85 files)
- `direction_v15` -> `ML_V1`, `DirectionV15` -> `MLV1` (116 subs across 37 files)
- Signal type strings (`V12_LONG`, etc.) preserved as internal vocabulary

---

## Bugs caught during implementation

| Bug | How found | Fix |
|---|---|---|
| Leaky training (random split, scaler on full) | User audit | Date-based split, scaler fit on train only |
| V1 exits configured but never firing | Exit-reason distribution showed only NO_ZONE/TIME_EXIT | Load 1m ticks, feed via on_tick in backtest |
| Inference thresholds hardcoded, diverged from params | Code inspection | Single source of truth in params.yaml |
| Ghost files in git (models deleted from disk, still tracked) | `git mv` failure | `git rm` the stale tracking entries |
| `PYTHONPATH=src` syntax doesn't work on Windows cmd | DVC subprocess failed | Use `sys.path.insert(0, src_dir)` in scripts |
| UTF-8 emojis crash cp1252-encoded subprocess output | torch.onnx emoji caused UnicodeEncodeError | Set `PYTHONIOENCODING=utf-8` in subprocess env |

---

## Honest backtest numbers (ML_V1 @staging, 2025 OOS, V1 exits)

| Metric | Value |
|---|---|
| Trades | 242 |
| Win % | 42.6 |
| Total bps | +1,283 |
| PF | 1.56 |
| Max DD | -312 bps |

Per exit reason (all V1 invariants verified):
- PT_TARGET: 22 trades, all wins, +94 bps avg
- MID_TRAIL: 47 trades, all wins, +22 bps avg
- PT_LOCK: 7 trades, all wins, +52 bps avg
- LOCKED_PROFIT: 40 trades, mixed, +2 bps avg
- STOP_LOSS: 125 trades, all losses, -18 bps avg (exactly as declared)
- TIME_EXIT: 1 trade, -11 bps

Compare to leaky v1's reported +18,207 bps — honest is 9× smaller. Audit was correct.

---

## Multi-model pipeline strategy

| Shop size | Pattern |
|---|---|
| Small (<10 eng) | One `dvc.yaml`, frozen flags, wrapper script — **what we have** |
| Mid (10-100 eng) | One `dvc.yaml` per model + shared `data/dvc.yaml` |
| Large (100+) | Orchestrator (Airflow/Kubeflow) + feature store (Feast/Tecton) |

### Phase A — now (1-2 models): wrapper script — DONE
- `configs/pipelines.yaml` declares each model's stage list
- `scripts/mlops/run_pipeline.py` runs by name
- Adding a model = 5 lines of yaml + its own DVC stages

### Phase B — when hitting 3+ models: split dvc.yaml per model
```
pipelines/
  data/dvc.yaml          # shared build_features + build_labels
  ml_v1/dvc.yaml
  ml_v2_attention/dvc.yaml
  ml_v3_xgboost/dvc.yaml
```
Migration effort: ~2 hours when triggered. **Do not build preemptively.**

### Phase C — never (unless scaling to a team platform)

---

## Out of scope (deferred to Stage 3 / Stage 4)

- Live model monitoring / drift detection
- Auto-promotion (always manual gate here — correct for trading real money)
- A/B testing / shadow mode
- Feature store
- CI/CD integration of pipeline into GitHub Actions
- Attention model (ML V2) retraining pipeline — still Colab-only

---

## Capability matrix — what the pipeline supports today

### Supported workflows (use now)

| Goal | Command | Notes |
|---|---|---|
| End-to-end test for ml_v1 | `python scripts/mlops/run_pipeline.py ml_v1 --test` | Cleans up MLflow versions created |
| Production retrain | `python scripts/mlops/run_pipeline.py ml_v1` | Runs only stale stages |
| Force full rebuild | `python scripts/mlops/run_pipeline.py ml_v1 --force` | Keeps new MLflow version |
| Train only | `dvc repro train_mlp_v15` | Stops after training |
| Backtest only | `dvc repro backtest_staging` | Uses current trained model |
| Change exit rules (v1 vs v2) | Edit `exit.version` in params.yaml, `dvc repro backtest_staging` | Only backtest re-runs |
| Change training hyperparams | Edit `ml_v1.training` in params.yaml, `dvc repro` | train + backtest re-run |
| Change labels (same model/exit) | Edit label builder script, `dvc repro` | Full cascade: labels -> train -> backtest |
| Change features (same model/exit) | Edit feature builder script, `dvc repro` | Full cascade |
| Change inference thresholds | Edit `ml_v1.inference` in params.yaml, `dvc repro backtest_staging` | Only backtest re-runs |

### Not-yet-wired (needs work before it runs)

| Goal | Effort | What to add |
|---|---|---|
| Backtest across multiple models (ml_v1 + attention) | ~30 min | Generalize `backtest_staging.py` to take `--model` arg; add per-model backtest stages |
| Test a new exit strategy (e.g. v3) | ~1 hour | Add exit rules in code; add V3 invariants in verify_backtest.py |
| Promotion: staging -> production + alias swap | ~20 min | Add a `promote.py` helper that copies files and swaps MLflow alias |
| Attention model pipeline (Colab-trained) | ~2-3 hr | Write train script, add DVC stages, add pipelines.yaml entry |
| All-models aggregate pipeline | ~15 min after per-model wiring | Add `all_models` entry in pipelines.yaml |

### Known limitations of current pipeline

1. **Single-model backtest only.** `backtest_staging.py` hardcodes `models/ML_V1_staging/` as the staging dir. To backtest Attention or another model, need a `--model <name>` arg that reads a per-model staging dir. Applies to `backtest_staging.py` and the DVC stage output path.
2. **Exit version invariants are V1-only.** `verify_backtest.py` encodes V1 rules (STOP_LOSS = -10, PT_TARGET = 80, etc.). Switching `exit.version: v2` and running verify_backtest will fail. For V2 testing, either skip verifier or add V2 invariant branch.
3. **Backtest reports get overwritten.** `backtest_staging.json` and `backtest_staging_trades.parquet` are replaced each run. MLflow has model version history but no corresponding backtest JSONs. Fix: log metrics back to MLflow run, or archive per-run JSONs.
4. **No multi-model comparison stage.** To compare ML_V1 vs Attention head-to-head, you'd need both backtested on the same window, then a diff script. Not built.
5. **Manifest writer and verifier are loosely coupled.** Adding a new model requires writing a correct manifest (schema_version, split.method, scaler.fit_on, feature_recipe.compute_fn). No schema enforcement — typos silently make verifier skip checks.
6. **Inference thresholds in params.yaml are ML_V1-specific.** `ml_v1.inference.*`. Each new model needs its own section and its own signal-generator code that reads from there.

### Workflow patterns (decision tree)

```
What changed?
├─ Tunable parameter (hyperparam, threshold, window)
│   └── Edit configs/params.yaml -> dvc repro
│
├─ Code (feature builder, label builder, training logic)
│   └── Edit script -> dvc repro (cascades automatically)
│
├─ Data (new raw bars added)
│   └── dvc add data/raw/<file> -> dvc repro (full cascade)
│
├─ Exit strategy (new ruleset)
│   └── Add exit method + schema + settings -> flip params.yaml
│       If new invariants needed: extend verify_backtest.py
│
├─ New model architecture
│   └── New training script + DVC stages + pipelines.yaml entry
│       Generic verifier works as-is if manifest is correct
│
├─ Card/documentation only
│   └── Edit card -> git commit (no pipeline rerun)
│
└─ Runtime config default (schema.py, settings.yaml)
    └── Edit + dvc repro
```

---

## How to add a new model (checklist)

1. Write the training script (e.g. `src/engine/train_xgboost.py`) that:
   - Reads features + labels
   - Uses date-based split from `params.yaml`
   - Fits scaler/preprocessor on train only
   - Writes artifacts to `models/<MODEL_NAME>_staging/`
   - Writes `training_manifest.json` (same schema as ML_V1)
   - Auto-registers to MLflow as `<MODEL_NAME>` @staging

2. Add to `configs/params.yaml` a per-model section with inference + training
   + split.

3. Add DVC stages to `dvc.yaml`:
   - `train_<model>` with `params:` tracking per-model section
   - `verify_<model>` running `python src/mlops/verify.py models/<NAME>_staging`
   - `backtest_<model>` (if needed — may share logic with backtest_staging)

4. Declare the pipeline in `configs/pipelines.yaml`:
   ```yaml
   my_new_model:
     description: What it does
     stages: [build_features, build_labels, train_<name>, verify_<name>, ...]
   ```

5. Run `python scripts/mlops/run_pipeline.py my_new_model --test` to validate
   end-to-end without polluting the registry.

**What you do NOT need to change:**
- `src/mlops/verify.py` (generic, reads the manifest)
- `scripts/mlops/verify_backtest.py` (if the model uses V1 exits)
- `scripts/mlops/run_pipeline.py` (driven by the yaml manifest)

---

## Known gaps / future work

1. **Backtest reports overwritten on each retrain.** Only latest on disk. MLflow
   has all model versions but not corresponding backtest JSONs. Should log
   backtest metrics back to the MLflow run, or archive per-run JSONs.
2. **Attention model still has leakage.** Same split bug in
   `scripts/colab/train_attention_production.py`. Needs same honest retrain.
3. **No automated promotion.** Human must manually copy staging files to
   production and swap MLflow alias. Correct for live money but could be a
   one-liner helper.
4. **V12 signal performance.** V12_SHORT loses -357 bps in 2025. Not a pipeline
   gap — a strategy signal gap. Worth investigating separately.

---

_Last updated: 2026-04-16_
_Current status: Phases 1-7 complete, end-to-end tested via `--test` mode,
awaiting commit._
