# L1-EXP-002: Position Sizing (BTC Qty Scaling)

## Question
How many 0.001 BTC to trade per signal, and how to scale as wallet grows?

## Setup
- Leverage: 125x (FIXED, never changes)
- Margin mode: Cross
- Starting wallet: $10
- Position size: multiples of 0.001 BTC
- Variable: qty scaling rule as wallet grows

## The Rule
**"For every $X in your wallet, trade 0.001 BTC"**

Formula: `qty = floor(wallet / $/step) * 0.001 BTC`

Example ($2.50/step):
- $10 wallet: floor(10/2.5) = 4 -> 0.004 BTC
- $25 wallet: floor(25/2.5) = 10 -> 0.010 BTC
- $100 wallet: floor(100/2.5) = 40 -> 0.040 BTC

Position scales up with wallet (compounding). Position scales down after losses (protection).

## Key Findings

### FINDING 1: No liquidation risk at any tested size
- Worst trade: -182 bps
- Smallest buffer (5x qty = 0.005 BTC): 1600 bps
- All sizes survive worst trade. Risk is DRAWDOWN, not liquidation.

### FINDING 2: $/step controls risk vs growth
| $/step | Qty@$10 | Worst Trade Loss | MC Ruin | MC Max DD | MC Median Final |
|--------|---------|------------------|---------|-----------|-----------------|
| $1.50 | 0.007 | ~95% of wallet | 95% | - | DEAD |
| $2.00 | 0.005 | 86% of wallet | 0.1% | 79% | $4,650,000 |
| $2.50 | 0.004 | 69% of wallet | 0% | 68% | $909,000 |
| $3.00 | 0.003 | 52% of wallet | 0% | ~58% | $93,000 |
| $5.00 | 0.002 | 35% of wallet | 0% | ~40% | ~$5,000 |

### FINDING 3: Sharp cliff between $1.75 and $2.00
- $1.75/step: 95% ruin (too aggressive)
- $2.00/step: 0.1% ruin (barely safe)
- Small change in position size = life or death

### FINDING 4: Compounding creates exponential differences
- $2.00 vs $2.50 is just 1 extra 0.001 BTC at $10 wallet
- But over 220 trades: $4.65M vs $909K (5x difference!)
- Small position change compounds into massive final difference

## Options (DECISION PENDING)
1. **Aggressive**: $2.00/step - 0.1% ruin, $4.65M median
2. **Conservative**: $2.50/step - 0% ruin, $909K median
3. **Safe**: $3.00/step - 0% ruin, $93K median

## How It Works in Practice
1. Check wallet balance
2. qty = floor(wallet / $/step) * 0.001 BTC
3. Enforce minimum 0.001 BTC
4. Place order at 125x leverage
5. Margin auto-calculated by Binance (position / 125)
6. Full wallet backs trade (cross margin)

## Files
- position_sizing.py (old, superseded)
- position_sizing_v2.py (old, leverage-based framing)
- position_sizing_125x.py (margin% framing)
- qty_sizing.py (CURRENT - pure BTC qty framing)
