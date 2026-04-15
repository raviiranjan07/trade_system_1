# V1.5 Trading System Notes

## System Overview
V1.5 = V1.4 (rule-based) + ML direction model (neural network)

Two independent signal sources running in parallel with separate wallets and trade logs.

## V1.4 Signals (Rule-Based, Unchanged)
- V12_LONG: RSI crosses below 20 + bull + ATR>=25 + EMA>=0.5%
- V12_SHORT: RSI crosses above 80 + bear
- BEAR_LONG: RSI < 10 level + bear + EMA>=1.0%
- BULL_SHORT: RSI > 90 level + bull + ATR>=60 + EMA>=1.0%

## ML Signals (Neural Network, NEW in V1.5)
- ML_LONG: MLP probability > 0.60
- ML_SHORT: MLP probability < 0.35 (confidence > 0.65)

### ML Model Details
- Architecture: MLP 10 -> 128 -> 128 -> 1 (binary classification)
- Features: roc1, roc2, roc3, roc4, roc5, roc6, roc7, roc8, range_position (50-bar), rsi7
- Training: 2020-2025 data, random 90/10 split, 155K train bars
- Val accuracy: 52.8% overall, 62.4% on confident bars (5% of data)
- Thresholds: LONG at prob>0.60, SHORT at prob<0.35 (asymmetric — SHORT needs higher confidence)

### Why Asymmetric Thresholds
- ML_LONG at 0.60: profitable in backtest (+7,035 bps, 73.6% win)
- ML_SHORT at 0.35 (not 0.40): SHORT predictions fail in bull market at lower confidence
- Backtest showed SHORT at 0.40 threshold lost -4,056 bps; at 0.35 only 6 trades, manageable

### Feature Importance
- roc1 (23%) + roc2 (22%) = 45% — short-term price velocity dominates
- range_position (19%) — where price is in 50-bar range
- rsi7 (7%) — momentum oscillator
- roc3-roc8 (3-8%) — supporting velocity features

## Exit Rules (Same for All Signal Types)
- LONG: 20 bps trailing stop
- SHORT: 30 bps trailing stop
- Time exit: bar 10
- Tightening: after bar 5, trailing stop tightens to 8 bps

## Backtest Results (2024-2025)

### Combined V1.5
- 439 trades, 66.7% win, +11,096 bps, PF 2.44, DD -639

### By Signal Type
| Signal | Trades | Win% | Net bps | PF |
|--------|--------|------|---------|-----|
| ML_LONG | 220 | 73.6% | +7,035 | 2.39 |
| V12_SHORT | 117 | 59.8% | +1,846 | 2.70 |
| V12_LONG | 59 | 54.2% | +1,242 | 2.86 |
| BULL_SHORT | 20 | 60.0% | +934 | 8.71 |
| BEAR_LONG | 17 | 76.5% | +590 | 9.64 |
| ML_SHORT | 6 | 66.7% | -551 | 0.25 |

### V1.4 vs V1.5
| System | Trades | Win% | Net bps | PF |
|--------|--------|------|---------|-----|
| V1.4 alone | 220 | 60.0% | +5,267 | 3.46 |
| V1.5 combined | 439 | 66.7% | +11,096 | 2.44 |

### Signal Overlap
- Only 8 bars overlap between V1.4 and ML (1.2%)
- All 8 overlap bars agree on direction
- Systems are almost completely independent

## File Structure
```
src/engine/
  strategy.py        -- V1.4 signals + ML_LONG/ML_SHORT enum
  ml_signal.py       -- ML signal generator class
  ml_train.py        -- Model training script (PYTHONPATH=src python -m engine.ml_train)
  ml_model/
    direction_model.pt  -- Trained MLP weights
    scaler.npz          -- Feature normalization (mean/std from 2020-2025)
  bot.py             -- Live bot with V1.4 + ML running in parallel
  backtest.py        -- Combined V1.4 + ML backtest

data/v12_trades/
  trades_paper.csv       -- V1.4 paper trades
  trades_ml_paper.csv    -- ML paper trades (separate)
  risk_state.json        -- V1.4 wallet state
  risk_state_ml.json     -- ML wallet state (separate)

data/risk_logs/
  decisions.csv          -- V1.4 risk decisions
  ml/decisions.csv       -- ML risk decisions (separate)
```

## Key Findings from L2-003 Research

1. Direction is only predictable at extreme conditions (~2-5% of bars)
2. MLP > LSTM for indicator-based features (indicators already encode temporal history)
3. LSTM memory adds nothing — weight decay blocks gate learning; without it, gates learn to shut down memory
4. roc1 + roc2 + range_position = 64% of directional signal
5. Model predicts magnitude well but direction weakly (52-53% overall, 60-63% on confident bars)
6. ML_LONG works well, ML_SHORT struggles in bull markets
7. 95.7% of bars have a twin with opposite direction — most bars are inherently unpredictable

## Monitoring Checklist (Paper Trading)
- [ ] Compare V1.4 wallet vs ML wallet over time
- [ ] Track ML_LONG win rate — should be >55% to be useful
- [ ] Track ML_SHORT — monitor if it should be disabled
- [ ] Check if model needs retraining after 3-6 months
- [ ] Monitor confident prediction rate — should stay around 5%

## Next Release TODO
- [ ] Add liquidation price calculation to bot + dashboard
  - At 0.002 BTC, $7 wallet, 20x leverage → liq is only ~498 bps away
  - Need to show liq price in logs and dashboard for safety
  - Formula: liq_price = entry ± (wallet / position_value) for LONG/SHORT
- [ ] Fix dashboard "IN POSITION: ?" — show correct position direction for ML trades
