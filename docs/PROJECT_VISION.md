# INTELLIGENT TRADING SYSTEM — Full Vision & Architecture

> Last updated: 2026-02-07

---

## 1. MISSION

Build the smartest and most intelligent trading system possible.

Not just a bot that follows rules. An **intelligent agent** that reads markets, makes decisions, adapts to conditions, manages risk, and learns from experience — using every relevant concept from ML, RL, quantum computing, information theory, game theory, and beyond.

**Starting point:** $10, BTCUSDT, Binance Futures
**Foundation:** V1.3.2 strategy (proven edge, 220 trades, PF 3.46, +5,267 bps OOS)
**Ambition:** R&D everything. Test everything. Keep what works. Kill what doesn't.

---

## 2. THE PROBLEM

Human trading weaknesses:
- Emotions (fear, greed, FOMO, revenge trading)
- Overtrading (taking bad setups)
- Poor risk management (wrong size, no stops)
- Wrong entries/exits (early, late, no plan)

**The system must decide:**
1. **WHEN** to go LONG
2. **WHEN** to go SHORT
3. **WHEN** to EXIT
4. **HOW MUCH** to risk
5. **WHEN** to sit out entirely

---

## 3. CURRENT STATE (V1.3.2)

### What We Have

| Component | Status | Details |
|-----------|--------|---------|
| Data | 15-min BTCUSDT candles 2020-2025 | TimescaleDB + Parquet |
| V12_LONG | RSI crosses below 20 + bull (price>SMA200) + ATR>=25 + EMA>=0.5% | Filtered LONG [EXP-006] |
| V12_SHORT | RSI crosses above 80 + bear (price<SMA200) | Unfiltered SHORT [EXP-007] |
| BEAR_LONG | RSI < 10 (level) + bear + EMA>=1.0% | Level-based counter-trend LONG [EXP-014] |
| BULL_SHORT | RSI > 90 (level) + bull + ATR>=60 + EMA>=1.0% | Level-based counter-trend SHORT [EXP-014] |
| Exit | LONG: 20 bps trailing stop, SHORT: 30 bps trailing stop | |
| Time tightening | After bar 5: trailing stop tightens to 8 bps | V1.3.1 improvement |
| Time exit | Bar 10 (2.5 hours) | Protects portfolio |
| Backtester | Custom Python | Full OOS validation |
| Live bot | Paper trading 24/7 | WebSocket + Binance Futures |
| Dashboard | Real-time web UI | Charts, alerts, analytics, drawings |

### V1.3.2 Performance (OOS 2024-2025)

| Metric | Value |
|--------|-------|
| Total trades | 220 |
| Win rate | 60.0% |
| Total net profit | +5,267 bps |
| Profit factor | 3.46 |
| Max drawdown | -192 bps |
| Config hash | 874ffca20d4a |

### Evolution Path

| Version | Trades | PF | Total bps | What Changed |
|---------|--------|-----|-----------|-------------|
| V1.0 | 266 | 1.97 | +3,024 | RSI + SMA200 baseline |
| V1.2 | 202 | 2.47 | +3,250 | LONG filters (ATR + EMA) |
| V1.3 | 262 | 3.00 | +4,438 | Counter-trend signals (BEAR_LONG, BULL_SHORT) |
| V1.3.1 | 211 | 3.34 | +4,915 | Time-based tightening (8bps after bar 5) |
| V1.3.2 | 220 | 3.46 | +5,267 | Level-based counter-trend entry (BEAR_LONG, BULL_SHORT) |

### What V1.3.2 Is NOT

- Not adaptive (same rules in all conditions)
- Not intelligent (no learning, no memory)
- No position sizing (fixed size — no Kelly criterion)
- No drawdown circuit breaker
- No adaptive leverage
- Single strategy family only (RSI-based)
- No regime awareness beyond SMA200 binary
- No self-monitoring or degradation detection
- No execution optimization
- Uses only price data (ignores order book, sentiment, on-chain, macro)

---

## 4. TARGET STATE — THE INTELLIGENT SYSTEM

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    LLM REASONING LAYER                       │
│          Claude API — analysis, hypotheses, journal          │
├──────────────────────────────────────────────────────────────┤
│                    AGENT (Brain)                             │
│      RL Policy + Bayesian Beliefs + Quantum ML Layer         │
├──────────────┬────────────────┬──────────────────────────────┤
│   Regime     │   Strategy     │      Risk Manager            │
│   Detector   │   Selector     │      Kelly + Circuit Breaker │
│   HMM +      │   Bandit +     │      Drawdown Control        │
│   Clustering │   Ensemble     │      Adaptive Leverage        │
├──────────────┴────────────────┴──────────────────────────────┤
│                    STRATEGIES                                │
│   V1.3.2 │ Volume Spike │ Microstructure │ Sentiment │ Future│
├──────────────────────────────────────────────────────────────┤
│                    FEATURE ENGINE                            │
│   Price │ Order Book │ On-chain │ Sentiment │ Macro          │
│   Transformers │ Entropy │ Fractals │ NLP                    │
├──────────────────────────────────────────────────────────────┤
│                    EXECUTION ENGINE                          │
│   Order Management │ Smart Execution │ Latency Optimization  │
├──────────────────────────────────────────────────────────────┤
│                    DATA LAYER                                │
│   OHLCV │ Tick │ Order Book │ News │ On-chain │ Macro        │
├──────────────────────────────────────────────────────────────┤
│                    MONITORING & SAFETY                       │
│   Kill Switch │ Degradation │ Alerts │ P&L Dashboard         │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. PROBLEMS & CONCEPTS MAPPING

Every concept must solve a **real problem**. No technology for technology's sake.

### Problem 1: When to Enter (Smarter Entry)

| Concept | How It Helps |
|---------|-------------|
| Reinforcement Learning | Agent learns optimal entry timing from experience |
| Bayesian Inference | Update belief about signal quality in real-time |
| Information Theory (Entropy) | Measure if current price action is signal or noise |
| Transformers | Pattern recognition across time series |
| Order Book Imbalance | Short-term directional pressure |
| Sentiment Analysis (NLP) | Is the crowd positioned for or against us? |

### Problem 2: When to Exit (Smarter Exit)

| Concept | How It Helps |
|---------|-------------|
| Optimal Stopping Theory | Mathematically optimal exit point |
| RL Agent | Learns when to hold vs exit from experience |
| Kalman Filter | Estimate true price vs noise, better trailing |
| Fractal Analysis | Multi-timeframe trend strength for hold/exit decision |

### Problem 3: How Much to Risk (Position Sizing)

| Concept | How It Helps |
|---------|-------------|
| Kelly Criterion | Mathematically optimal bet size given edge and odds |
| Bayesian Updating | Adjust confidence per trade based on conditions |
| Drawdown Control | Reduce size during losing streaks |
| CVaR Optimization | Manage tail risk (worst-case scenarios) |
| Adaptive Leverage | More leverage when confident, less when uncertain |

### Problem 4: What Market Are We In (Regime Detection)

| Concept | How It Helps |
|---------|-------------|
| Hidden Markov Models (HMM) | Detect hidden market states (trending, ranging, volatile, quiet) |
| Unsupervised Clustering | Find natural market regimes from data |
| Change-Point Detection | Know the exact moment regime shifts |
| Lyapunov Exponents | Measure market predictability in real-time |
| Entropy Measures | Is the market orderly or chaotic right now? |

### Problem 5: Is Our Strategy Still Working (Self-Monitoring)

| Concept | How It Helps |
|---------|-------------|
| CUSUM / Statistical Process Control | Detect strategy degradation early |
| Online Learning | Adapt parameters as market changes |
| Meta-Learning | Learn WHEN to trust the model vs be skeptical |
| Anomaly Detection (Autoencoders) | Detect unusual market states the model hasn't seen |

### Problem 6: Multiple Strategies (Strategy Selection)

| Concept | How It Helps |
|---------|-------------|
| Multi-Armed Bandits | Explore vs exploit — try new strategies vs stick with winners |
| Portfolio Theory | Allocate capital between strategies optimally |
| Ensemble Methods | Combine weak edges into strong ones |
| Genetic Algorithms | Evolve new strategies from existing ones |

### Problem 7: Execution (Smart Order Placement)

| Concept | How It Helps |
|---------|-------------|
| Market Microstructure | Understand order book, spread, liquidity |
| Control Theory (PID) | Smooth position management |
| Optimal Execution (TWAP/VWAP) | Minimize market impact |
| Latency Optimization | Fast execution when it matters |

### Problem 8: Understanding the World (Data Intelligence)

| Concept | How It Helps |
|---------|-------------|
| NLP / Sentiment Analysis | Parse news, social media for market-moving events |
| On-Chain Analytics | Whale movements, exchange flows, funding rates |
| Macro Correlation | BTC relationship with DXY, S&P500, Gold, Fed decisions |
| Causal Inference | Does news CAUSE moves or just correlate? |
| Graph Neural Networks | Multi-asset relationship modeling |

---

## 6. TECHNOLOGY STACK

### Core Languages

| Language | Purpose |
|----------|---------|
| **Python** | ML, research, backtesting, prototyping, agent logic |
| **Rust** | Execution engine, low-latency components (future) |

### Data Layer

| Purpose | Technology |
|---------|-----------|
| Historical OHLCV | Parquet files, TimescaleDB |
| Real-time data feed | WebSocket (Binance), CCXT |
| Feature store | Redis (real-time), DuckDB (analysis) |
| Data pipeline | Pandas, Polars (faster), NumPy |
| Tick data | Custom collector + TimescaleDB |
| Alternative data | APIs (Glassnode, CryptoQuant, Twitter, NewsAPI) |

### ML / AI Stack

| Concept | Library |
|---------|---------|
| Classical ML | scikit-learn |
| Deep Learning | PyTorch |
| Reinforcement Learning | Stable-Baselines3, Gymnasium |
| Transformers | PyTorch, HuggingFace |
| Time Series Models | pytorch-forecasting (Temporal Fusion Transformer) |
| LSTM / GRU | PyTorch |
| GANs (synthetic data) | PyTorch |
| Autoencoders (anomaly) | PyTorch |
| Hidden Markov Models | hmmlearn, pomegranate |
| Bayesian Methods | PyMC, NumPyro |
| Time Series Stats | statsmodels, tsfresh |
| Optimization | Optuna (hyperparams), scipy |
| Genetic Algorithms | DEAP |
| Signal Processing | scipy.signal, filterpy (Kalman) |
| NLP / Sentiment | HuggingFace Transformers, spaCy |
| Graph Neural Networks | PyTorch Geometric |

### Quantum Computing Stack

| Purpose | Technology | Cost |
|---------|-----------|------|
| Quantum ML | PennyLane (integrates with PyTorch) | Free |
| Quantum circuits | Qiskit (IBM) | Free |
| Quantum simulation | Cirq (Google) | Free |
| Real quantum hardware | IBM Quantum free tier | Free |
| Cloud quantum | Amazon Braket | Pay per use |

**Quantum applications in our system:**

| Problem | Quantum Approach |
|---------|-----------------|
| Strategy parameter optimization | QAOA (Quantum Approximate Optimization Algorithm) |
| Regime detection | Quantum kernel methods for classification |
| Feature selection | Quantum-enhanced sampling |
| Risk simulation | Quantum Monte Carlo (faster rare event detection) |
| Portfolio allocation | Quantum annealing for multi-strategy optimization |

**Hybrid quantum-classical pipeline:**
```
Classical Features → Quantum Circuit → Classical NN → Decision
  (RSI, ATR, etc.)   (PennyLane)      (PyTorch)     (LONG/SHORT/WAIT)
```

### Agent Architecture

| Component | Technology |
|-----------|-----------|
| Agent framework | Custom Python (state machine + decision engine) |
| LLM reasoning layer | Claude API (meta-analysis, reporting, hypotheses) |
| Event system | asyncio + custom event bus |
| Strategy orchestration | Custom multi-armed bandit / portfolio allocator |
| State management | Custom + Redis |

### Exchange / Execution

| Purpose | Technology |
|---------|-----------|
| Exchange API | python-binance, CCXT |
| Order management | Custom OMS (Order Management System) |
| Paper trading | Custom simulator |
| Multi-exchange | CCXT abstraction layer |

### Monitoring / Ops

| Purpose | Technology |
|---------|-----------|
| Logging | Python logging + structured logs (JSON) |
| Metrics dashboard | Grafana + Prometheus |
| Alerting | Telegram bot API |
| Strategy health | Custom CUSUM / degradation detection |
| P&L tracking | Custom + database |

### Research / Experimentation

| Purpose | Technology |
|---------|-----------|
| Notebooks | Jupyter |
| Experiment tracking | MLflow |
| Visualization | Matplotlib, Plotly |
| Backtesting | Custom engine (existing) |
| Version control | Git |
| Walk-forward validation | Custom |
| Monte Carlo testing | NumPy |

### Infrastructure

| Purpose | Technology | Cost |
|---------|-----------|------|
| Run the bot | VPS or local machine | $5-10/month |
| Scheduling | APScheduler or cron | Free |
| Config management | YAML + Pydantic | Free |
| Database | SQLite (local), PostgreSQL (VPS) | Free |
| Message queue | Redis pub/sub | Free |

### Security & Safety

| Component | Purpose |
|-----------|---------|
| Kill switch | Emergency stop — auto-shutdown on critical failure |
| API key encryption | Protect exchange credentials (env vars, vault) |
| Max drawdown circuit breaker | Auto-stop trading if cumulative loss exceeds threshold |
| Position limits | Never risk more than X% of capital per trade |
| Rate limiting | Don't spam exchange API |
| Audit log | Record every decision for review |

---

## 7. DATA SOURCES

### Currently Using

| Source | Data | Status |
|--------|------|--------|
| Binance | 15-min OHLCV candles | ACTIVE |

### To Add

| Source | Data | API | Priority |
|--------|------|-----|----------|
| Binance | Funding rate | REST/WS | HIGH |
| Binance | Open interest | REST/WS | HIGH |
| Binance | Order book (depth) | WebSocket | HIGH |
| Binance | Liquidation data | WebSocket | HIGH |
| Binance | Trade stream (tick) | WebSocket | MEDIUM |
| CryptoQuant | Exchange inflow/outflow | REST | MEDIUM |
| Glassnode | On-chain metrics | REST | MEDIUM |
| Twitter/X | Social sentiment | REST | MEDIUM |
| Reddit | r/bitcoin, r/cryptocurrency | REST | LOW |
| Alternative.me | Fear & Greed Index | REST | MEDIUM |
| Google Trends | Search interest | REST | LOW |
| Yahoo Finance | S&P500, Gold, DXY | REST | MEDIUM |
| FRED | Fed funds rate, CPI, macro | REST | LOW |
| NewsAPI | Crypto news headlines | REST | MEDIUM |

---

## 8. ADVANCED CONCEPTS — R&D AREAS

### Chaos Theory & Complexity

| Concept | Application |
|---------|-------------|
| Fractal analysis | Self-similar patterns across timeframes — is current move a fractal of larger move? |
| Lyapunov exponents | Measure market predictability — trade only when predictable |
| Strange attractors | Identify price attractor levels from dynamics |
| Entropy (Shannon) | Quantify market disorder — low entropy = predictable, high = chaotic |

### Game Theory

| Concept | Application |
|---------|-------------|
| Adversarial thinking | Model market as opponent trying to take your money |
| Nash equilibria | Understand stable strategies that survive against all others |
| Auction theory | Optimal bid/ask placement |
| Mechanism design | Design order placement that accounts for other participants |

### Information Theory

| Concept | Application |
|---------|-------------|
| Mutual information | Feature selection — which indicators actually carry information? |
| Transfer entropy | Causal relationships — does BTC dominance CAUSE altcoin moves? |
| Fisher information | Measure how much information each bar carries about future |
| Kolmogorov complexity | Is this price pattern compressible (structured) or random? |

### Behavioral Finance

| Concept | Application |
|---------|-------------|
| Herding detection | Is the crowd all on one side? (contrarian signal) |
| Disposition effect | Retail holds losers, sells winners — exploit this |
| Anchoring | Price levels that act as psychological anchors |
| Sentiment cycles | Fear → capitulation → hope → greed → euphoria → repeat |

### Network Science

| Concept | Application |
|---------|-------------|
| Correlation networks | Dynamic correlation between BTC and other assets |
| Centrality measures | Which asset leads, which follows? |
| Community detection | Groups of assets that move together |
| Contagion models | How does a crash in one asset spread? |

### Simulation & Synthetic Data

| Concept | Application |
|---------|-------------|
| GANs | Generate synthetic market scenarios for stress testing |
| Monte Carlo | Simulate thousands of strategy runs to estimate risk |
| Agent-based modeling | Simulate a market with many agents to understand dynamics |
| Scenario analysis | What happens to our strategy during a 2020-style crash? |

---

## 9. RESEARCH LAYERS (BUILD ORDER)

Build in layers. Each layer = R&D project. Test rigorously. Keep what works.

```
LAYER 0: V1.3.2 Foundation           [DONE +++]
    └── 4 signal types: V12_LONG, V12_SHORT, BEAR_LONG, BULL_SHORT
    └── V12 signals: cross-based (RSI crosses threshold)
    └── Counter-trend signals: level-based (RSI in extreme zone) [V1.3.2]
    └── Time-based tightening (8bps after bar 5) [V1.3.1]
    └── 220 trades, PF 3.46, +5,267 bps OOS
    └── Live paper trading bot + web dashboard
    └── Alerts, analytics, PnL calendar, drawing tools

LAYER 1: Risk Management             [NEXT]
    └── Kelly criterion position sizing
    └── Drawdown circuit breaker
    └── Adaptive leverage
    └── Handle known edge cases:
        - Cascading losses: level-based entry can chain losing trades when price keeps falling
        - RSI deceleration bounce: RSI bounces above/below threshold while price continues against us
        - Circuit breaker should catch both (consecutive loss limit or max DD per window)
    └── This keeps $10 alive

LAYER 2: Regime Detection             [...]
    └── HMM for market state classification
    └── Change-point detection
    └── Entropy measures
    └── Tell the agent WHAT market we're in

LAYER 3: Adaptive Exit                [...]
    └── RL agent for exit decisions
    └── Optimal stopping theory
    └── Kalman filter for noise removal
    └── Smarter than fixed trailing stop

LAYER 4: Multi-Strategy               [...]
    └── Add Volume Spike strategy
    └── Add microstructure strategy
    └── Multi-armed bandit for selection
    └── Portfolio allocation

LAYER 5: Data Intelligence            [...]
    └── Order book / funding rate features
    └── Sentiment analysis (NLP)
    └── On-chain analytics
    └── Macro correlation

LAYER 6: Deep Learning Models         [...]
    └── Transformers for time series
    └── LSTM for sequence modeling
    └── Autoencoders for anomaly detection
    └── GANs for synthetic data

LAYER 7: Quantum ML Layer             [...]
    └── PennyLane hybrid circuits
    └── Quantum kernel methods
    └── QAOA for optimization
    └── Benchmark vs classical methods

LAYER 8: Self-Monitoring              [...]
    └── CUSUM for strategy degradation
    └── Online learning / parameter adaptation
    └── Meta-learning (when to trust model)

LAYER 9: LLM Reasoning               [...]
    └── Claude API for market interpretation
    └── Hypothesis generation
    └── Trade journaling and explanation
    └── Anomaly reasoning

LAYER 10: Full Agent                  [...]
    └── Orchestrates all layers
    └── RL policy over entire system
    └── Bayesian belief network
    └── Truly intelligent decision making
```

---

## 10. ECONOMICS & CONSTRAINTS

| Parameter | Value |
|-----------|-------|
| Starting capital | $10 |
| Exchange | Binance Futures (USDT perpetuals) |
| Asset | BTCUSDT |
| Fees | 8 bps round-trip (limit orders) |
| Minimum profitable move | 12 bps (Rule #1) |
| Leverage range | 1x-20x (adaptive) |
| Timeframe | 15-minute (primary), multi-timeframe (future) |

### Realistic Growth Path

| Milestone | Capital | Leverage | Monthly Return | Timeline |
|-----------|---------|----------|----------------|----------|
| Start | $10 | 10-20x | ~15-30% | Month 0 |
| Survive | $50 | 10-15x | ~15-25% | ~4 months |
| Grow | $200 | 5-10x | ~10-20% | ~8 months |
| Scale | $1,000 | 5x | ~8-15% | ~14 months |
| Compound | $5,000+ | 3-5x | ~5-10% | ~20 months |

> As capital grows, leverage decreases, returns moderate, but dollar amounts increase.

---

## 11. VALIDATED FINDINGS (LOCKED)

These findings are proven and should not be re-tested:

| Finding | Source | Status |
|---------|--------|--------|
| Direction is 50/50 at all horizons | WHAT analysis | LOCKED |
| Random entry has no edge (0/432 profitable) | WHAT analysis | LOCKED |
| RSI oversold works for LONG (62.6% accuracy) | WHAT analysis | LOCKED |
| RSI overbought does NOT cause reversal alone | WHAT analysis | LOCKED |
| V1 SHORT = relief rally exhaustion (trend continuation) | Option D investigation | LOCKED |
| EMA bounce: 0/2,880 profitable after fees | V8/V10 scripts | LOCKED |
| ATR is #1 filter for Case 1 prediction | WHEN analysis | LOCKED |
| V1.2 LONG filters improve PF 1.47 → 2.29 | EXP-006 | LOCKED |
| SHORT is robust in ALL conditions — no filtering needed | EXP-007 | LOCKED |
| Range support works as filter, NOT as entry signal | EXP-008 | LOCKED |
| TIME_EXIT protects portfolio — do NOT remove | EXP-009 | LOCKED |
| Re-entry after TS works: +2,609 bps, PF 2.96 | EXP-005 extended | VALIDATED |
| Hold longer does NOT work — worse than re-entry | EXP-005 extended | VALIDATED |
| Counter-trend signals profitable (BEAR_LONG PF 3.00, BULL_SHORT PF 2.16) | EXP-013 | LOCKED |
| SHORT filtering REJECTED — all 7 configs reduce profit | EXP-007 | LOCKED |
| Re-entry disabled = cleaner risk (PF 2.09→3.00, DD -511→-270) | EXP-013 | LOCKED |
| Time tightening (8bps after bar 5) improves all metrics | V1.3.1 | LOCKED |
| EMA crossover too noisy on 15min (50/50, PF 1.13-1.47) | EXP-010 | LOCKED |
| Level-based > cross-based for counter-trend at extreme RSI | EXP-014 | LOCKED |
| RSI saturates at extremes (<10): price moves decouple from RSI | EXP-014 | LOCKED |

---

## 12. EXPERIMENT HISTORY

| ID | What | Result | Status |
|----|------|--------|--------|
| EXP-001 | RSI + MA baseline | RSI oversold/overbought confirmed as signals | Complete |
| EXP-002 | RSI + MA trend filter | Design flaw — below MAs actually better for LONG | Complete |
| EXP-003 | RSI failure deep dive | 10/11 SHORT failures in bull market, SMA200 catches 91% | Complete |
| EXP-004 | MA regime filter comparison | SMA200 confirmed best — balance of volume + quality | Complete |
| EXP-005 | Exit strategy analysis | TS=20 LONG, TS=30 SHORT optimal. Re-entry works. | Complete |
| EXP-006 | V1.2 backtest (LONG filters) | ATR + EMA sep filters: PF 1.47 → 2.29 LONG | Complete |
| EXP-007 | SHORT filtering | REJECTED — all filters reduce SHORT profit | Complete |
| EXP-008 | Range support strategy | WEAK — PF 1.27, too many signals, SHORT side loses | Complete |
| EXP-009 | Remove TIME_EXIT | REJECTED — returns worse, catastrophic trades emerge | Complete |
| EXP-010 | EMA crossover strategy | WEAK — 50/50 direction, thin edge, 2025 degrades | Complete |
| EXP-011 | 4 new strategy screen | Volume Spike promising: +15,946 bps, PF 1.43 | Needs deeper work |
| EXP-012 | Combined V1.2 + Volume Spike + Re-entry | Config A+RE chosen: 289t, +4,182 bps, PF 1.99 | Complete |
| EXP-013 | Counter-trend signals (BEAR_LONG, BULL_SHORT) | ACCEPTED → V1.3. RE disabled = best risk profile | Complete |
| V1.3.1 | Time-based tightening | 8bps stop after bar 5: PF 3.00→3.34, DD -270→-192 | Complete |
| EXP-014 | Level-based counter-trend entry | Level > cross for BEAR_LONG/BULL_SHORT: +352 bps, 9 new trades (7/9 win) | Complete |
| V1.3.2 | Level-based counter-trend (from EXP-014) | PF 3.34→3.46, +4,915→+5,267 bps, same DD -192 | Complete |

---

## 13. KEY PRINCIPLES

1. **Economics first** — if the math doesn't work after fees, nothing else matters
2. **Edge before intelligence** — the system must have a real edge; intelligence amplifies it
3. **Test everything, assume nothing** — all parameters come from data, not assumptions
4. **Simple rules survive** — complexity must earn its place with measurable improvement
5. **Kill what doesn't work** — no attachment to ideas; data decides
6. **R&D is the path** — learn, test, build, iterate. Every concept is worth exploring.
7. **Safety first** — kill switches, circuit breakers, position limits. Survival before profit.

---

## 14. FILES & LOCATIONS

| What | Path |
|------|------|
| Project root | `system_1/` |
| Strategy V1 setup | `experiments/rsi/TRADING_SETUP_V1.md` |
| V1 backtest | `experiments/rsi/backtest_v1.py` |
| V1.2 backtest | `experiments/rsi/EXP-006/backtest_v12.py` |
| V1.3.2 bot code | `src/v12/` (config/, strategy.py, position_manager.py, backtest.py, bot.py) |
| Web dashboard | `src/web/` (server.py, state.py, frontend/) |
| WHAT analysis | `docs/WHAT_analysis.md` |
| WHEN analysis | `docs/WHEN_analysis.md` |
| Experiment registry | `experiments/registry.csv` |
| This document | `docs/PROJECT_VISION.md` |
| Core rules | `CLAUDE.md` |

---

> This is a living document. Updated as we build, learn, and evolve the system.
