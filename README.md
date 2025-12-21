Database (OHLCV)
   ↓
DB Fetch Layer
   ↓
Data Validation & Ordering
   ↓
Feature Computation
   ↓
Normalization
   ↓
Market State Vector
   ↓
Store State Vectors



# 🧠 Trading Agent — State-Driven Quant Trading System

A **state-based quantitative trading system** designed to trade **only when historical market conditions show statistical edge**.  
This system avoids prediction, emotion, and overtrading by relying on **market memory, regimes, and expectancy**.

---

## 🎯 Core Philosophy

> **We do not predict price.  
We recognize market states and act only when history supports an asymmetric edge.**

Key principles:
- Markets are probabilistic, not deterministic
- Capital preservation comes first
- Fewer high-quality trades > frequent trades
- Structure protects survival
- Decisions are statistics-driven, not indicator-driven

---

## 🏗️ High-Level Architecture

# 🧠 Trading Agent — State-Driven Quant Trading System

A **state-based quantitative trading system** designed to trade **only when historical market conditions show statistical edge**.  
This system avoids prediction, emotion, and overtrading by relying on **market memory, regimes, and expectancy**.

---

## 🎯 Core Philosophy

> **We do not predict price.  
We recognize market states and act only when history supports an asymmetric edge.**

Key principles:
- Markets are probabilistic, not deterministic
- Capital preservation comes first
- Fewer high-quality trades > frequent trades
- Structure protects survival
- Decisions are statistics-driven, not indicator-driven

---

## 🏗️ High-Level Architecture

# 🧠 Trading Agent — State-Driven Quant Trading System

A **state-based quantitative trading system** designed to trade **only when historical market conditions show statistical edge**.  
This system avoids prediction, emotion, and overtrading by relying on **market memory, regimes, and expectancy**.

---

## 🎯 Core Philosophy

> **We do not predict price.  
We recognize market states and act only when history supports an asymmetric edge.**

Key principles:
- Markets are probabilistic, not deterministic
- Capital preservation comes first
- Fewer high-quality trades > frequent trades
- Structure protects survival
- Decisions are statistics-driven, not indicator-driven

---

## 🏗️ High-Level Architecture

PostgreSQL / TimescaleDB (1m OHLCV)
↓
Market State Vector Engine
↓
Regime Detection
↓
Outcome Labeling (MFE / MAE)
↓
Similarity Search (Market Memory)
↓
Decision Engine (Expected Value)
↓
Risk & Exit Management
↓
Backtest / Live Execution


---

## 📁 Project Structure

trading_agent/
│
├── data/
│ ├── raw/
│ │ └── ohlcv_loader.py # DB fetch layer
│ └── validators/
│ └── data_integrity.py # OHLCV validation
│
├── features/ # Deterministic feature computation
│ ├── trend.py
│ ├── momentum.py
│ ├── volatility.py
│ ├── volume.py
│ └── location.py
│
├── state/ # Market State Vector Engine (CORE)
│ ├── state_schema.py
│ ├── normalizer.py
│ ├── state_builder.py
│ └── run_state_pipeline.py
│
├── decision/ # (Planned) Decision logic
├── outcomes/ # (Planned) MFE / MAE labeling
├── regime/ # (Planned) Regime detection
├── similarity/ # (Planned) KNN similarity engine
├── risk/ # (Planned) Risk & exit management
├── backtest/ # (Planned) Walk-forward backtesting
├── live/ # (Planned) Live trading loop
│
├── requirements.txt
├── README.md
└── .env # DATABASE_URL


---

## 🧠 What Is a Market State Vector?

A **Market State Vector** is a compact, normalized numerical representation of the market at a specific moment.

It replaces:
- Indicators
- Chart patterns
- Human interpretation

With:
- Stable
- Comparable
- Regime-aware representations

### Example State Dimensions


---

## 🧠 What Is a Market State Vector?

A **Market State Vector** is a compact, normalized numerical representation of the market at a specific moment.

It replaces:
- Indicators
- Chart patterns
- Human interpretation

With:
- Stable
- Comparable
- Regime-aware representations

### Example State Dimensions


---

## 🧠 What Is a Market State Vector?

A **Market State Vector** is a compact, normalized numerical representation of the market at a specific moment.

It replaces:
- Indicators
- Chart patterns
- Human interpretation

With:
- Stable
- Comparable
- Regime-aware representations

### Example State Dimensions

[
ema50_slope_z,
ema200_slope_z,
trend_alignment,
return_5m_z,
return_15m_z,
rsi_z,
atr_percentile,
volume_z,
vwap_distance_z,
range_position
]


Each minute produces **one state vector**.  
Over years, this becomes **market memory**.

---

## 🗄️ Data Requirements

- 1-minute OHLCV data
- Stored in PostgreSQL / TimescaleDB
- Example schema:

```sql
ohlcv_data (
    time TIMESTAMP,
    pair TEXT,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume FLOAT,
    num_trades INT
)

Each minute produces **one state vector**.  
Over years, this becomes **market memory**.

---

## 🗄️ Data Requirements

- 1-minute OHLCV data
- Stored in PostgreSQL / TimescaleDB
- Example schema:

```sql
ohlcv_data (
    time TIMESTAMP,
    pair TEXT,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume FLOAT,
    num_trades INT
)

---

If you want next:
- **Regime detection implementation**
- **Outcome labeling (MFE / MAE)**
- **Architecture diagram**
- **Docstrings across codebase**

Just tell me 👍
