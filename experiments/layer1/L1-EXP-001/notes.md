# L1-EXP-001: Fixed Position Baseline

## Question
What happens with fixed BTC qty at 125x cross margin on V1.3.2 trades?

## Setup (CORRECT Binance Setup)
- Leverage: 125x (FIXED, never changes)
- Margin mode: Cross (full wallet backs every trade)
- Starting wallet: $10
- Min qty: 0.001 BTC, step: 0.001 BTC
- Min notional: $100 (DYNAMIC based on BTC price)
- Margin per trade: ~$0.80-$1.06 (position / 125)

## Key Findings

### FINDING 1: Leverage is IRRELEVANT
- Leverage setting = 125x always. It just sets margin requirement.
- The ONLY variable is position size (how many 0.001 BTC)
- At 125x, margin = position / 125 (tiny fraction of wallet)
- Full wallet backs the trade in cross margin

### FINDING 2: Fixed 1x qty (0.001 BTC) = safe but linear
- Position: ~$100-$130 (depends on BTC price)
- Final: $10 -> $82 (+724%)
- Max DD: 14.5%
- MC Ruin: 0%
- Growth is LINEAR (no compounding) because position stays constant

### FINDING 3: Liquidation is far away at small qty
| Qty | Position | Liq Buffer |
|-----|----------|------------|
| 0.001 | $100 | 9600 bps |
| 0.002 | $200 | 4600 bps |
| 0.003 | $300 | 2933 bps |
| 0.004 | $400 | 2100 bps |
| 0.005 | $500 | 1600 bps |

Worst trade = -182 bps. All sizes safe from liquidation.

### FINDING 4: Old vs New setup
- OLD (EXP-001 original): 20x leverage, $170 min notional, position = equity * 20
  - Result: $56K, 18.6% ruin
  - Problem: position scales with wallet (compounding) but ruin risk is high
- NEW (125x): Fixed qty, ~$100 position, margin ~$0.80
  - Result: $82 (1x fixed), 0% ruin
  - No compounding = safe but slow growth

### FINDING 5: Compounding requires SCALING position
- Fixed qty = linear growth (same $ profit per trade regardless of wallet)
- To compound: increase qty as wallet grows
- This is EXP-002's job: find the right scaling rule

## Verdict
**BASELINE ESTABLISHED.** Fixed 1x qty at 125x cross is safe (0% ruin) but only reaches $82. Need position scaling (EXP-002) for compounding growth.

## Files
- fixed_leverage_baseline.py (old 20x test)
- fixed_leverage_125x.py (new 125x test)
