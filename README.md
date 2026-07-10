# ALPHA — Trading Stack

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)
[![Status](https://img.shields.io/badge/status-internal-orange)]()

A 15-minute BTCUSDT trading system. Four parallel models (one rule-based + three ML) share a single exit engine, an adaptive risk layer, and a real-time dashboard. Trained on 2020–2023, tested out-of-sample on 2024–2025.

> **Min profitable move = 12 bps.** Fees = 8 bps round-trip (limit orders). Anything under 12 bps is noise. See [AGENTS.md](AGENTS.md) for the full rule set.

---

## Table of Contents

- [Status](#status)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Models](#models)
- [Exit V2](#exit-v2-production-frozen-2026-03-29)
- [Risk Layer 1](#risk-layer-1)
- [MLOps](#mlops)
- [Project Layout](#project-layout)
- [Testing](#testing)
- [Data Split](#data-split)
- [Documentation](#documentation)
- [License](#license)

---

## Status

| Layer | Status | Notes |
|---|---|---|
| Layer 0 — Foundation (V1.3.2) | DONE | V1 rule strategy locked |
| Layer 1 — Risk Management | DONE, integrated | [src/engine/risk/](src/engine/risk/) |
| Layer 2 — Direction (ML) | PARTIAL | V1, V2 Attention, V3 — all live (57–58% ceiling) |
| Layer 2 — Regime Detection | NOT STARTED | original HMM/clustering plan unbuilt |
| Layer 3–10 | NOT STARTED | see [docs/PROJECT_VISION.md](docs/PROJECT_VISION.md) |

Live in the bot today: **4 models, all on Exit V2.**

---

## Architecture

```
                         ┌──────────────────────────────┐
                         │  BTCUSDT 15m + 1m OHLCV      │
                         │  (data/raw/*.parquet)        │
                         └──────────────┬───────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
       ┌─────────────┐         ┌────────────────┐         ┌──────────────┐
       │  V1.4       │         │  ML V1 / V2 /  │         │  Feature     │
       │  Rule-based │         │  V3 (signals/) │         │  cache +     │
       │  signals    │         │  ONNX runtime  │         │  labels      │
       └──────┬──────┘         └────────┬───────┘         └──────────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
                ┌──────────────────────┐
                │  Risk Layer 1        │
                │  (size, health,      │
                │   preflight)         │
                └──────────┬───────────┘
                           ▼
                ┌──────────────────────┐
                │  Position Manager    │
                │  Exit V2             │
                │  (early cut, BE      │
                │   lock, tighten,     │
                │   time exit)         │
                └──────────┬───────────┘
                           ▼
              ┌────────────┴────────────┐
              ▼                         ▼
       ┌─────────────┐          ┌──────────────────┐
       │  Paper /    │          │  Dashboard       │
       │  Live exec  │          │  (FastAPI + WS   │
       │  (Binance)  │          │   + React)       │
       └─────────────┘          └──────────────────┘
```

---

## Prerequisites

- **Python 3.10+**
- **Git + Git LFS** (large parquet/model files are LFS-tracked)
- **DVC** for the data/model pipeline
- **Node.js 18+** (only if rebuilding the React dashboard)
- **Docker** (optional — bot can run containerized)
- ~5 GB free disk for raw + feature data, more for `mlruns/`

System tested on Windows 10 (PowerShell 5.1). Bash via WSL or Git Bash also works.

---

## Installation

```powershell
# 1. Clone (with LFS)
git clone <repo-url>
cd system_1
git lfs pull

# 2. Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install runtime + research deps
pip install -r requirements.txt
pip install scipy matplotlib seaborn scikit-learn tqdm pyarrow xgboost mlflow numba
pip install dvc

# 4. (Optional) install the package in editable mode for imports
pip install -e .

# 5. Pull DVC-tracked data and models
dvc pull
```

> **Note:** training stack (PyTorch, etc.) is intentionally not in `requirements.txt` — the bot only needs `onnxruntime` for inference. Install training deps manually when retraining.

---

## Configuration

| File | Purpose |
|---|---|
| [configs/params.yaml](configs/params.yaml) | Per-model thresholds, training, split, sweep grid |
| [src/engine/config/settings.yaml](src/engine/config/settings.yaml) | V1.4 strategy filters, exit V1 fallback params, re-entry, execution mode |
| [src/engine/risk/config.yaml](src/engine/risk/config.yaml) | Risk Layer 1 (worst-loss baseline, sizing %, health thresholds) |
| [configs/strategy_cards/exit_v2.yaml](configs/strategy_cards/exit_v2.yaml) | Exit V2 spec (production, frozen) |
| [configs/data_cards/](configs/data_cards/) | Dataset specs (15m, 1m OHLCV, label sets) |
| [configs/model_cards/](configs/model_cards/) | Model architecture cards |
| [configs/protocols/](configs/protocols/) | Training/eval protocols |
| [configs/schemas/backtest_report.yaml](configs/schemas/backtest_report.yaml) | Backtest report schema (v2.0) |
| `.env` | Secrets — `DATABASE_URL`, exchange API keys (gitignored) |

**Default execution mode is `paper`.** Set `execution.mode: live` in [settings.yaml](src/engine/config/settings.yaml) only after you understand the risk module and have real keys configured.

---

## Quick Start

```powershell
# Bot (paper trading by default)
$env:PYTHONPATH="src"; python -m engine.bot

# Backtest a single model independently
$env:PYTHONPATH="src"; python -m research.backtest --model ml_v3 --independent `
    --exit-version v2 --start 2025-01-01 --end 2025-12-31

# V1.4 only (rule-based, no ML)
$env:PYTHONPATH="src"; python -m research.backtest --v14-only --exit-version v2 `
    --start 2024-01-01 --end 2025-12-31

# Full pipeline: features → train → verify → backtest all → compare
dvc repro

# Only the comparison step (after individual backtests are current)
dvc repro compare_all

# Threshold sweep on a single model
$env:PYTHONPATH="src"; python -m research.sweep_thresholds --model ml_v3

# Hyperparameter training sweep (frozen — manual trigger)
dvc repro sweep_training_ml_v3
```

The dashboard auto-starts with the bot — open the URL printed at startup. WebSocket pushes wallet, drawdown, account-health multiplier, and per-model state in real time.

---

## Models

| Model | Type | Confidence (L / S) | Source |
|---|---|---|---|
| **V1.4** | Rule-based: RSI + SMA200 + counter-trend levels | n/a | [src/engine/strategy.py](src/engine/strategy.py) |
| **ML V1** | MLP direction predictor | 0.60 / 0.58 | [src/engine/signals/ml_v1.py](src/engine/signals/ml_v1.py) |
| **ML V2 Attention** | LSTM + Attention direction | 0.60 / 0.58 | [src/engine/signals/direction_attention.py](src/engine/signals/direction_attention.py) |
| **ML V3** | Exit-aware labels + snapshot features | 0.40 / 0.40 | [src/engine/signals/ml_v3.py](src/engine/signals/ml_v3.py) |

Thresholds and exit version are set in [configs/params.yaml](configs/params.yaml). ML models load from `models/<MODEL>/` (production) or `models/<MODEL>_staging/` (candidate from DVC).

V1.4 signal types: `V12_LONG`, `V12_SHORT` (RSI cross-based) + `BEAR_LONG`, `BULL_SHORT` (level-based counter-trend from EXP-014). Pure V1.4 OOS 2024–2025 with V1 exits: 239 trades, 51.0 % win, +1,343 bps, PF 1.67, DD −330. V2 exit A/B in `memory/v14_exit_comparison.md`.

---

## Exit V2 (production, frozen 2026-03-29)

Applied identically to all four models. Spec: [configs/strategy_cards/exit_v2.yaml](configs/strategy_cards/exit_v2.yaml). Logic: [src/engine/position_manager.py](src/engine/position_manager.py).

```
Bar 1–2:  Wide trailing stop (LONG 20 / SHORT 30 bps)
Bar 3:    MFE < 3 bps?   → EARLY CUT
Bar 4:    MFE < 5 bps?   → EARLY CUT
Bar 4+:   Trailing stop tightens to 6 bps
Anytime:  MFE ≥ 15 bps?  → BE lock floor at 9 bps gross (+1 bps net)
Bar 10:   Still open?    → TIME EXIT
```

OOS 2024–2025 on V1.5 signals (1,680 trades): **+33,062 bps, PF 4.66, DD −276 bps.** V1 exits comparison: +30,312 bps, PF 2.21, DD −774.

---

## Risk Layer 1

[src/engine/risk/](src/engine/risk/). Adaptive position sizer (`wallet × risk% / worst_loss`) plus account-health monitor (drawdown from peak, consecutive losses, recent win rate). Preflight runs 5 startup checks; the bot refuses to start if any fail. Every TRADE/SKIP decision is logged to `data/risk_logs/decisions.csv`. State persists across restarts via `risk_state.json`.

At $5 wallet: exchange minimum (0.001 BTC) dominates — risk parameters are placeholders until the account reaches ~$173+. Worst-loss baseline `worst_loss_bps=865` comes from training stats.

---

## MLOps

- **DVC** ([dvc.yaml](dvc.yaml)) — pipeline graph: features → labels → train → verify → backtest (V1 + V2 exits) → compare.
- **MLflow** — local tracking DB at [mlflow.db](mlflow.db), artifacts under [mlruns/](mlruns/). Threshold sweeps and backtest metrics logged with scope tags.
- **Verification gates** — [src/mlops/verify.py](src/mlops/verify.py) writes `data/reports/verification_<model>.json`; backtest stage depends on it.
- **Report schema v2.0** — [configs/schemas/backtest_report.yaml](configs/schemas/backtest_report.yaml). All reports validated.
- **Promotion log** — [docs/PROMOTION_LOG.md](docs/PROMOTION_LOG.md).

---

## Project Layout

```
system_1/
├── src/
│   ├── engine/              # PRODUCTION trading engine
│   │   ├── bot.py              # live/paper bot (entrypoint)
│   │   ├── backtest.py         # backtester CLI
│   │   ├── strategy.py         # V1.4 rule signals
│   │   ├── position_manager.py # Exit V2 logic
│   │   ├── ml_train.py         # ML V1 training
│   │   ├── train_attention.py  # ML V2 training
│   │   ├── train_v3.py         # ML V3 training (exit-aware)
│   │   ├── build_features.py
│   │   ├── build_exit_labels.py
│   │   ├── sweep_thresholds.py
│   │   ├── sweep_training.py
│   │   ├── compare_models.py
│   │   ├── monitoring.py       # expected vs actual + daily drift
│   │   ├── signals/            # per-model signal adapters
│   │   ├── risk/               # Layer 1 — sizing, health, preflight
│   │   └── config/             # settings.yaml + schema + loader
│   ├── mlops/               # MLflow tracking, DVC integration, gates
│   ├── web/                 # FastAPI backend + React frontend
│   ├── brain/               # SR / zones research module
│   └── trade_system/        # LEGACY (old KNN/state-vector — not used)
│
├── configs/                 # params, data/model/strategy cards, schemas
├── models/                  # per-model production + staging artifacts
├── data/
│   ├── raw/                 # OHLCV parquet (LFS)
│   ├── features/            # feature_cache, labels, exit-aware labels
│   ├── reports/             # backtest.json + trades.parquet per model
│   └── risk_logs/           # decisions.csv (every TRADE/SKIP)
├── experiments/             # research (layer2/, exit_strategy/, etc.)
├── docs/                    # planning + analysis
├── scripts/                 # one-off CLIs, analysis, colab notebooks
├── dvc.yaml                 # pipeline graph
├── pyproject.toml
├── requirements.txt
├── Dockerfile
└── AGENTS.md                # development rules — READ FIRST (CLAUDE.md points here)
```

---

## Testing

```powershell
# Unit tests for the risk module (only first-class test suite)
$env:PYTHONPATH="src"; pytest tests/test_risk_unit.py -v

# Stress + Monte Carlo + failure-mode research harnesses (slower)
$env:PYTHONPATH="src"; pytest src/research/risk_validation/ -v
```

`experiments/` contains historical research scripts named `test_*.py` — those are exploratory backtests, **not** unit tests, and may be slow / require data files.

Code style: `black` (line-length 100) + `isort`. Not currently enforced in CI.

---

## Data Split

Memorize this — never say "2024 data" without qualifying:

- **Train:** 2020-01-01 → 2023-12-31
- **Val:**   2024-01-01 → 2024-12-31
- **Test (OOS):** 2025-01-01 → 2025-12-31

Always say "2024–2025 data" or "test data". See [AGENTS.md](AGENTS.md) for full data discipline rules.

---

## Documentation

| Doc | Topic |
|---|---|
| [docs/PROJECT_VISION.md](docs/PROJECT_VISION.md) | Layered roadmap |
| [docs/MLOPS.md](docs/MLOPS.md) | MLOps architecture |
| [docs/WHAT_analysis.md](docs/WHAT_analysis.md) | "What happens in the market" findings |
| [docs/WHEN_analysis.md](docs/WHEN_analysis.md) | "When do outcomes happen" findings |
| [docs/FLAWS.md](docs/FLAWS.md) | Known limitations |
| [docs/SCALPING_REQUIREMENTS.md](docs/SCALPING_REQUIREMENTS.md) | Scalping constraints |
| [docs/PROMOTION_LOG.md](docs/PROMOTION_LOG.md) | Model promotion history |
| [AGENTS.md](AGENTS.md) | Development rules + decision protocol (canonical; CLAUDE.md points here) |

---

## License

MIT — see [pyproject.toml](pyproject.toml). Internal project; not currently distributed.
