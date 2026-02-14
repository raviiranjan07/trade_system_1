# L1-EXP-004: Hybrid Condition-Based Position Sizing

## Question
Does using different $/step for different conditions beat fixed $/step?

## First Test (hybrid_sizing.py) — OVERFITTED
- Found bad conditions in OOS data, tested on same OOS data
- Config H (size-down weak): $18.4M vs $4.65M fixed — looked amazing
- BUT this was circular: fitting to the data you're testing on

## Validated Test (hybrid_sizing_validated.py) — PROPER
- Found bad conditions in TRAIN (2020-2023)
- Tested on OOS (2024-2025)
- Only used conditions that are bad in BOTH periods

## Validated Bad Conditions (bad in train AND OOS)
| Condition | Train avg bps | OOS avg bps | Real? |
|-----------|--------------|-------------|-------|
| Monday LONG | -1.9 | -18.2 (12% win) | YES |
| Low ATR (<10) | -8.3 | +1.9 (weak) | YES |
| Low ATR (<20) | -2.4 | +3.8 (weak) | YES |
| Low EMA (<0.3) | +5.4 (weak) | +1.5 (weak) | YES |
| V12_LONG + Monday | -9.8 | -18.5 (13% win) | YES |

## Conditions that were OVERFIT (bad in OOS but NOT train)
- Tuesday LONG: -15.3 train -> +42.3 OOS (reversed!)
- Night 00-04 UTC: +5.0 train -> +25.3 OOS (not bad at all)

## Validated Strong Conditions
- High ATR (>70, >90): consistent in both periods
- High EMA (>1.0, >2.0): consistent in both periods
- BULL_SHORT: consistent in both periods

## OOS Results (validated only)
| Config | MC Median | Ruin | vs Fixed $2.00 |
|--------|-----------|------|----------------|
| Fixed $2.00 | $4.65M | 0.1% | baseline |
| Fixed $2.50 | $909K | 0% | -80% |
| Validated hybrid | $13.8M | 0% | +200% |

## The Validated Hybrid Rule
- Base: $2.00/step (aggressive)
- Monday LONG: $4.00/step (size down)
- Low ATR (<10-20): $4.00/step (size down)
- Low EMA (<0.3): $4.00/step (size down)
- V12_LONG + Monday: $4.00/step (size down)

## Architecture Note
Risk management should be DYNAMIC, not hard-coded to V1.3.2:
- Principles are fixed: $/step scaling, size-down on bad conditions, MC validation
- Numbers recalculated per strategy version (win rate, payoff, bad conditions)
- Pipeline: strategy stats -> Kelly -> condition analysis -> MC validation -> config

## Files
- hybrid_sizing.py (first test - overfitted, kept for reference)
- hybrid_sizing_validated.py (proper train/test validation)
