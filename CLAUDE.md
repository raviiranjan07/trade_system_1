# Trading System Development Rules

## SYSTEM VISION

**Problem we're solving:**
- Human trading weaknesses: emotions, overtrading, poor risk management, wrong entries/exits
- System should be smart enough to read market situations and handle accordingly
- Decide: when to go LONG, when to go SHORT, when to EXIT




## RULE #1: MINIMUM PROFITABLE MOVE = 12bp

**Any move < 12bp is NOISE - cannot profit from it.**

- Fees: 8 bps (round-trip, limit orders)
- 12bp target - 8bp fees = 4bp net profit (minimum worthwhile)
- 8bp target = 0 net profit (structurally impossible)
- 10bp target = 2bp net profit (barely worth it)

**Implications:**
- Noise threshold: 12bp (not 10bp)
- Minimum target: 12bp+
- Tradeable targets: 15bp, 25bp (not 8bp)

---

## ECONOMICS FIRST (Before Any Code)

1. **MWNM (Minimum Worthwhile Net Move)**
   - Fees: 8 bps (round-trip, limit orders)
   - Slippage: 0 bps (limit orders)
   - **MWNM = 8 bps**
   - **Minimum profitable target = 12 bps** (Rule #1)

2. **BTC Minimum Price Move Rule**
   - Break-even: 8 bps (0.08%) - just covers fees
   - **Minimum profitable move: 12 bps (0.12%)**
   - Example: Entry $90,000 → need $90,108+ to profit ($108 move = 12 bps)

3. **Break-even Win Rate**
   - Formula: `break_even = stop / (target + stop)`


## CURRENT PHASE: Foundation Work

**Objective:** Define boundaries and rules from data analysis

**Process:**
1. WHAT - What happens in the market? (price distribution, outcomes)
2. WHEN - When do different outcomes happen? (conditions, timing)
3. RULES - Set boundaries based on findings
4. SYSTEM - Build system using rules

**Stay focused:** Before any analysis, ask:
> "Does this help define WHAT, WHEN, or RULES?"
> If no, skip it.

---

## DECISION-MAKING PROTOCOL

**ALWAYS ask for clarification when:**

1. **Stuck on approach** - Multiple valid paths, unclear which to take
   - Example: "Should I analyze by time windows or by MAE thresholds?"
   - DON'T guess - ASK

2. **Ambiguous requirements** - Task can be interpreted multiple ways
   - Example: "Complete analysis" - does this mean all horizons? All targets?
   - DON'T assume - ASK

3. **Trade-offs exist** - Different approaches have pros/cons
   - Example: "Quick summary vs detailed tables - which do you prefer?"
   - DON'T decide alone - ASK

4. **Human expertise needed** - Domain knowledge or judgment required
   - Example: "Is 50bp drawdown acceptable for 15bp target?"
   - DON'T invent rules - ASK

5. **Major direction change** - About to pivot approach or methodology
   - Example: "Should we abandon this analysis and try a different angle?"
   - DON'T change course without approval - ASK

6. **Resource-intensive task** - Will take significant time/compute
   - Example: "This will analyze 77 combinations and take 10 minutes - proceed?"
   - DON'T start without confirmation - ASK

7. **Conflicting instructions** - Previous guidance conflicts with current request
   - Example: "Earlier you said X, now you're asking for Y - clarify?"
   - DON'T resolve conflict alone - ASK

**NEVER:**
- Make up trading rules without data backing
- Guess at user's intent when unclear
- Proceed with major changes without approval
- Assume "obvious" answers - markets are complex
- **Update documentation without user approval** - ALWAYS show findings to user FIRST, get explicit approval, THEN add to docs
- **Jump to implementation without asking** - ALWAYS give options first, let user choose

**BEFORE ANY IMPLEMENTATION:**
1. Present OPTIONS (not just one suggestion)
2. Wait for user to CHOOSE
3. Only then proceed with chosen option
4. If user asks "what should we do?" → Give options, don't decide

**DATA-DRIVEN PRINCIPLE:**
- Do NOT assume features or thresholds
- ALL parameters must come FROM the data
- If something is assumed, flag it and ask user first

**DATA SPLIT (MEMORIZE THIS):**
- Train data: 2020-2023
- Test data (OOS): 2024-2025 (NOT just "2024"!)
- NEVER say "2024 data" - ALWAYS say "2024-2025 data" or "test data"

**Format for asking:**
```
I need clarification on: [specific issue]

Option A: [approach 1]
Option B: [approach 2]

Which should I proceed with?




experiments/
├── registry.csv           # Master log of ALL experiments
├── ema/
│   ├── EXP-001/          # Each experiment in own folder
│   │   ├── config.yaml   # Parameters used
│   │   ├── results.csv   # Trades data
│   │   ├── metrics.json  # Performance metrics
│   │   ├── notes.md      # What we learned
│   │   └── plots/        # Visualizations