# ALPHA — Trading Stack

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)
[![Status](https://img.shields.io/badge/status-skeleton-orange)]()

A 15-minute BTCUSDT trading system: ML signal models on a shared exit
engine, an adaptive risk layer, and a real-time dashboard.

> **Min profitable move = 12 bps.** Fees = 8 bps round-trip (limit orders).
> Anything under 12 bps is noise. See [AGENTS.md](AGENTS.md) for the full rule set.

---

## Status — clean slate (2026-07-12)

The 2026 H1 model lineage (ML V1 MLP, ML V2 Attention, ML V3 exit-aware —
and the earlier V1.4 rule strategy) was **retired and wiped**: model files,
MLflow history, feature cache, reports. The codebase is now a
**model-agnostic skeleton** — every pipeline stage, registry, and track
slot is an empty, marked template awaiting the new first architecture.

- **The bot cannot start** until a new model is trained and promoted.
- Old lineage: git history + [archive/](archive/).
- Kept: raw OHLCV data ([data/raw/](data/raw/), through 2025-12-30),
  the [experiments/](experiments/) research ledger, all machinery below.

| Layer | Status |
|---|---|
| Layer 0 — Foundation (rule strategy) | RETIRED — findings locked in docs/ |
| Layer 1 — Risk Management | DONE, integrated ([src/engine/risk/](src/engine/risk/)) |
| Layer 2 — Direction (ML) | RESET — new first architecture in design |
| Layer 3–10 | NOT STARTED — [docs/PROJECT_VISION.md](docs/PROJECT_VISION.md) |

### Adding the new model — the touchpoints (each is a marked template)

```
training:  architectures.ARCHITECTURES → train.MODEL_SPECS + TASKS
config:    configs/params.yaml block + configs/protocols/<name>.yaml
pipeline:  dvc.yaml stage chain + configs/pipelines.yaml
research:  engine/signals/<adapter>.py + backtest.ML_GENERATORS
live bot:  orchestrator.TRACK_METAS + build_tracks spec
frontend:  model roster in src/web/frontend (currently hardcodes the old 3)
```

Known day-one decisions: pick ONE `range_position` variant + ONE ATR
window ([feature_lib.py](src/engine/signals/feature_lib.py) docstring),
and build the missing data-ingestion stage (raw data is 6+ months stale).

---

## Architecture

```
   BTCUSDT 15m + 1m OHLCV (data/raw/*.parquet)
        │
        ├── training pipeline (dvc repro)
        │     build_features → build_(exit_)labels → train → verify → backtest
        │     tracked by MLflow (runs) + protocols (metric gates) + manifests
        │     └── models/<NAME>_staging  ──[human promote]──►  models/<NAME>
        │
        └── live bot (engine.bot)
              orchestrator → one StrategyTrack per model
                shared: feed, indicators, dashboard bridge
                isolated: wallet, health, risk sizing, position manager
              exits: V1/V2 rules in position_manager.py (tick-level)
              dashboard: FastAPI + WebSocket + React (:8080)
```

Dependency rule: `training/` and `research/` import `engine`; `engine`
imports neither.

---

## Prerequisites

- **Python 3.10+**, **Git**, **DVC**
- **Node.js 18+** (only if rebuilding the React dashboard)
- System tested on Windows 10 (PowerShell 5.1); Git Bash works.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt      # runtime (bot needs onnxruntime only)
pip install dvc mlflow               # pipeline + tracking
pip install -e .                     # optional, editable imports
```

Training stack (PyTorch etc.) is intentionally not in `requirements.txt` —
install manually when training.

## Configuration

| File | Purpose |
|---|---|
| [configs/params.yaml](configs/params.yaml) | per-model training/split/features/inference (template inside) |
| [src/engine/config/settings.yaml](src/engine/config/settings.yaml) | shared indicator windows, exit params, execution mode |
| [src/engine/risk/config.yaml](src/engine/risk/config.yaml) | Risk Layer 1 sizing/health |
| [configs/protocols/](configs/protocols/) | pre-registered experiment contracts (enforced by mlops runner) |
| [configs/schemas/backtest_report.yaml](configs/schemas/backtest_report.yaml) | backtest report schema |
| `.env` | secrets (gitignored) |

**Default execution mode is `paper`.**

## Quick Start (once a model is registered)

```powershell
$env:PYTHONPATH="src"; python -m engine.bot                       # bot + dashboard
$env:PYTHONPATH="src"; python -m training.train --model <key>     # train
$env:PYTHONPATH="src"; python -m research.backtest --model <key> --exit-version v2 `
    --start 2025-01-01 --end 2025-12-31                           # honest backtest
dvc repro                                                         # full pipeline
python scripts/mlops/promote.py <NAME>                            # staging → production
```

---

## Exit engine

Exit rules live in [src/engine/position_manager.py](src/engine/position_manager.py)
(V1 and V2 variants; spec: [configs/strategy_cards/](configs/strategy_cards/)).
They also generate **exit-aware labels**: `build_exit_labels` replays the
real position manager over 1m ticks so labels encode actual trade outcomes,
not endpoint direction — the strongest idea of the retired lineage, kept.

## Risk Layer 1

[src/engine/risk/](src/engine/risk/) — adaptive sizer (`wallet × risk% /
worst_loss`), account health (drawdown, streaks, recent win rate), 5-check
preflight (bot refuses to start on failure), per-track decision logging,
state persistence. Worst-loss baseline from
`experiments/layer1/L1R-001/train_stats.json`.

## MLOps

- **DVC** ([dvc.yaml](dvc.yaml)) — 3 shared data stages + per-model chain template.
  Full retired pipeline: [archive/dvc_3model_pipeline.yaml](archive/dvc_3model_pipeline.yaml).
- **MLflow** — sqlite `mlflow.db` (created on first run) + `mlruns/` artifacts;
  model registry with `@staging` alias set by `training.train`.
- **Protocols** — [src/mlops/runner.py](src/mlops/runner.py) validates every
  run against its declared protocol (required metrics/artifacts) or fails it.
- **Manifests** — every trained model gets `training_manifest.json`
  (git commit, data MD5s, MLflow run id) checked by [src/mlops/verify.py](src/mlops/verify.py).
- **Promotion** — [scripts/mlops/promote.py](scripts/mlops/promote.py):
  gates + atomic copy + alias swap + [docs/PROMOTION_LOG.md](docs/PROMOTION_LOG.md).

---

## Project Layout

```
system_1/
├── src/
│   ├── engine/           # PRODUCTION: bot, orchestrator, tracks, PM, risk, signals/
│   ├── training/         # train.py entry point, architectures, utils, label builders
│   ├── research/         # backtest, sweeps, compare, risk harnesses
│   ├── mlops/            # runner, protocol, tracking, registry, verify, report
│   ├── web/              # FastAPI + WebSocket + React dashboard
│   └── brain/            # SR/zones research module (separate pipeline)
├── configs/              # params, pipelines, protocols, cards, schemas
├── models/               # created by training: <NAME>_staging → <NAME>
├── data/                 # raw/ (OHLCV), features/ + reports/ (pipeline outputs)
├── experiments/          # research ledger (EXP-001…, layer1/, layer2/, registry.csv)
├── archive/              # retired model/data cards + old pipeline reference
├── scripts/mlops/        # promote, run_pipeline, backtest_staging, leaderboard
├── docs/                 # vision, analyses, MLOps guide, promotion log
├── dvc.yaml              # pipeline graph (skeleton)
└── AGENTS.md             # development rules — READ FIRST (CLAUDE.md points here)
```

## Testing

```powershell
$env:PYTHONPATH="src"; pytest tests/test_risk_unit.py -v          # risk unit tests
$env:PYTHONPATH="src"; pytest src/research/risk_validation/ -v    # MC/stress harnesses
```

Pipeline-level testing = verify gates + honest backtest + paper trading
(see AGENTS.md "CURRENT PHASE").

## Data Split

- **Train:** 2020-01-01 → 2023-12-31
- **Val:** 2024-01-01 → 2024-12-31
- **Test (OOS):** 2025-01-01 → 2025-12-31

Never say "2024 data" — always "2024–2025 data" or "test data".

## Documentation

| Doc | Topic |
|---|---|
| [docs/PROJECT_VISION.md](docs/PROJECT_VISION.md) | layered roadmap |
| [docs/MLOPS.md](docs/MLOPS.md) | MLOps architecture |
| [docs/WHAT_analysis.md](docs/WHAT_analysis.md) / [docs/WHEN_analysis.md](docs/WHEN_analysis.md) | locked market findings |
| [docs/PROMOTION_LOG.md](docs/PROMOTION_LOG.md) | model promotion history |
| [AGENTS.md](AGENTS.md) | development rules + decision protocol (canonical) |

## License

MIT — see [pyproject.toml](pyproject.toml). Internal project; not currently distributed.
