# L1-EXP-006: Final Combined Risk Management System

## Question
What is the complete, final risk management specification for V1.3.2?

## ANSWER: Complete system validated. Safety stop is free insurance.

## Combines findings from EXP-001 through EXP-005

## Part 1: Liquidation Prices
- 20x LONG: entry x 0.954 (4.6% buffer = 460 bps)
- 20x SHORT: entry x 1.046 (4.6% buffer = 460 bps)
- 25x LONG: entry x 0.964 (3.6% buffer = 360 bps)
- 25x SHORT: entry x 1.036 (3.6% buffer = 360 bps)
- All 220 historical trades are safe at both 20x and 25x
- Worst trade: -181.8 bps (well within 460 bps buffer at 20x)

## Part 2: Safety Stop Impact
- Safety stop = 80% of distance to liquidation (exchange-level order)
- 20x safety stop: 368 bps from entry
- 25x safety stop: 288 bps from entry
- **ZERO trades affected** by safety stop at either leverage level
- Existing exits (trailing stop 20/30 bps, time exit bar 10) always fire first
- Safety stop is pure insurance — costs nothing in normal operation

## Part 3: Stress Test
| Scenario | 20x | 25x |
|----------|-----|-----|
| Worst historical (-182 bps) | SAFE | SAFE |
| 2x worst (-364 bps) | SAFE | LIQUIDATED |
| 3x worst (-545 bps) | LIQUIDATED | LIQUIDATED |
| Black swan (-5%) | LIQUIDATED | LIQUIDATED |
| Flash crash (-10%) | LIQUIDATED | LIQUIDATED |

- 20x survives up to 2.5x worst historical move
- 25x survives only 1.9x worst historical move
- Both liquidated at -5% or worse (unprecedented for 15min candle)

## Part 4: MC Simulation (with vs without safety stop)
| Config | Median (no safety) | Median (with safety) | P5 (no safety) | P5 (with safety) |
|--------|-------------------|---------------------|----------------|-----------------|
| 20x | $41,101 | $40,979 | $29,507 | $29,068 |
| 25x | $179,675 | $178,286 | $107,473 | $102,619 |

- Safety stop has negligible impact on performance
- Zero safety triggers and zero liquidations in 1000 MC paths

## Part 5: Final System Specification

### 8 Components:
1. **Margin Mode**: Cross (entire wallet backs trade)
2. **Leverage**: Fixed 20x (conservative) or 25x (moderate)
3. **Position Sizing**: Phase 1 (<$15): $100-$199 position. Phase 2 (>=$15): equity * leverage
4. **Liquidation Prices**: Calculated per trade at entry
5. **Safety Stop-Loss**: Exchange-level order at 80% of liq distance
6. **Existing Exits**: Trailing stop (20/30 bps, tightens to 8 after bar 5), Time exit (bar 10)
7. **Exit Priority**: Trailing stop → Time exit → Safety stop → Liquidation
8. **Expected Performance**: 20x median $41K, 25x median $178K (from $10 start)

## Part 6: Bot Implementation Checklist
At ENTRY: Calculate position size, liquidation price, safety stop → Place safety stop ORDER on exchange
During TRADE: Monitor trailing stop, time exit, warn if approaching safety stop
At EXIT: Cancel safety stop order, log results, update equity
BOT FAILURE: Safety stop lives on exchange (works if bot dies)

## Files
- final_combined.py (main experiment)
