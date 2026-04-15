# How Models Learn — Deep Explanation

---

## Part 1: What IS a Model?

A model is nothing magical. It is a **mathematical function** that maps inputs to outputs.

```
f(input) = output
```

Simple example you already know:

```
f(temperature) = "wear jacket or not"

if temperature < 15°C → wear jacket
if temperature >= 15°C → don't wear jacket
```

That is a model. It takes a number (temperature) and outputs a decision.

Our trading brain does the same thing:

```
f(11,178 market numbers) = LONG / SHORT / SKIP
```

The only difference: instead of 1 input number, we have 11,178. Instead of you writing the rules manually, the model FINDS the rules from data.

---

## Part 2: What is Training?

Training = showing the model examples WITH the correct answers, so it can find the rules itself.

**Analogy: Teaching a child to recognize cats**

```
Show photo 1 → "this is a CAT"
Show photo 2 → "this is NOT a cat (it's a dog)"
Show photo 3 → "this is a CAT"
...
Show 10,000 photos → child learns what makes a cat

Now show new photo → child says "CAT" without being told
```

The child learned the rules (pointy ears, whiskers, fur pattern) from examples.

Our model does the same with BTC bars:

```
Show Bar 1 → [11,178 numbers] → "this is LONG"
Show Bar 2 → [11,178 numbers] → "this is SKIP"
Show Bar 3 → [11,178 numbers] → "this is SHORT"
...
Show 140,000 bars → model learns what makes LONG/SHORT/SKIP

Now show new bar → model says "LONG" without being told
```

---

## Part 3: How Does It Actually Learn? (The Loop)

**The model starts knowing NOTHING.**

It has internal numbers (called parameters/weights) that are initialized randomly.
These internal numbers control what decision the model makes.

### The Training Loop:

**Step 1: Make a prediction**

```
Bar 1: [t_ATR=0.89, t_range=55, t96_ATR=0.23, ...]
Model's internal numbers say: SKIP (random guess at start)
Actual label: LONG
```

**Step 2: Measure the error**

```
Model said: SKIP (0% LONG, 0% SHORT, 100% SKIP)
Reality:    LONG (100% LONG)

Error = how far off was the prediction?
      = large error (completely wrong)
```

The error is measured as a NUMBER (loss function).

Example: Binary Cross Entropy loss = -log(probability assigned to correct class)
- Model said LONG with 5% probability → loss = -log(0.05) = 3.0 (very high, bad)
- Model said LONG with 90% probability → loss = -log(0.90) = 0.1 (low, good)

**Step 3: Calculate which direction to adjust**

This is called GRADIENT. It tells you:
- Which internal numbers are responsible for the error?
- Which direction should each internal number move to reduce the error?

```
Gradient says:
  "Internal number #4,521 was too high → decrease it slightly"
  "Internal number #892 was too low → increase it slightly"
  "Internal number #11,003 was fine → barely change it"
```

**Step 4: Adjust the internal numbers (tiny nudge)**

```
Old value of internal number #4,521: 0.73
Nudge amount (learning rate): 0.001
New value: 0.73 - (0.001 × gradient) = 0.729 (tiny change)
```

The nudge is TINY on purpose. If you adjust too much, you overfit to one example and forget the others.

**Step 5: Move to next bar. Repeat.**

```
Bar 2: [t_ATR=0.08, t_range=4, ...] → actual: SKIP
  Model guesses: LONG (still bad)
  Error: large
  Gradient: adjust differently
  Nudge internal numbers again

Bar 3: [t_ATR=0.91, t_range=58, ...] → actual: SHORT
  Model guesses: LONG
  Error: large
  Nudge again

...after 1,000 bars: model getting slightly better
...after 10,000 bars: model noticeably better
...after 140,000 bars: model has seen every pattern once
```

**Step 6: Repeat all 140,000 bars again (called an EPOCH)**

```
Epoch 1 (all 140,000 bars): 38% correct → many adjustments made
Epoch 2 (all 140,000 bars): 54% correct → fewer adjustments
Epoch 3 (all 140,000 bars): 67% correct → adjustments getting smaller
Epoch 4 (all 140,000 bars): 74% correct → small adjustments
Epoch 5 (all 140,000 bars): 79% correct → tiny adjustments
...
Epoch N (all 140,000 bars): 81% correct → almost no change → STOP
```

When the model stops improving = training complete.

---

## Part 4: What Does "Adjust" Mean Physically?

The model has millions of internal numbers (parameters).

**Neural Network example:**
- Each connection between neurons has a weight (a number)
- Training adjusts these weights
- After training: weights encode the patterns

```
Before training:
  weight_4521 = 0.73  (random)
  weight_892  = -0.21 (random)
  weight_11003= 0.55  (random)

After training:
  weight_4521 = 0.12  (learned: high ATR growth → reduce SKIP signal)
  weight_892  = 0.89  (learned: low range → increase SKIP signal)
  weight_11003= -0.44 (learned: RSI <30 → increase LONG signal)
```

The model never knows "ATR" or "RSI" — it just knows "column 4521 correlates with LONG."

**Decision Tree (LightGBM) example:**
- Each node in the tree = a threshold split
- Training finds the best threshold for each split
- After training: tree structure encodes the patterns

```
Before training: splits are random/arbitrary

After training:
  Node 1: "Is column 4521 (t_ATR%) > 0.65?"
            YES → go right (probably not SKIP)
            NO  → go left (probably SKIP)
  Node 2: "Is column 892 (t96_ATR%) < 0.30?"
            YES → go right (ATR was low in past, grew → LONG)
            NO  → go left (ATR was high throughout → SHORT)
```

---

## Part 5: Why LONG/SHORT Label Alone Is NOT Enough

If we only use LONG/SHORT/SKIP as label:

```
Bar A: price went up 13bps at H=96 → LONG
Bar B: price went up 80bps at H=5  → LONG

Same label. But completely different stories.
Model treats them the same → learns a blurry pattern
```

Bar A is a slow, weak move. Bar B is a fast, explosive move.
If the model learns them together as "LONG" — it learns a confused average.

**Real learning requires RICH labels:**

```
Bar A:
  mfe_up = 13bps    ← how far
  mfe_down = 2bps
  label_h1  = SKIP  ← barely moved at 1 bar
  label_h5  = SKIP  ← still nothing at 5 bars
  label_h20 = SKIP  ← nothing at 20 bars
  label_h96 = LONG  ← finally at 96 bars (slow move)
  first_touch_h = 96

Bar B:
  mfe_up = 80bps    ← how far (much bigger)
  mfe_down = 3bps
  label_h1  = LONG  ← already moved at 1 bar (explosive!)
  label_h5  = LONG  ← still going at 5 bars
  label_h20 = LONG  ← persisting at 20 bars
  label_h96 = LONG  ← still LONG at 96 bars
  first_touch_h = 1
```

Now the model learns:
- **Pattern A features** → slow, weak move (13bps at H=96 only)
- **Pattern B features** → fast, explosive move (80bps starting at H=1)

Two completely different behaviors. Two completely different what-to-do-after.

---

## Part 6: Multi-Target Learning

Instead of one label → we give the model MULTIPLE labels per bar.

```
Training row for Bar B:
INPUT:  [11,178 numbers]
TARGETS:
  direction_h96    = LONG    ← primary decision
  mfe_up_h96       = 80      ← how far (regression)
  mfe_down_h96     = 3
  label_h1         = LONG    ← timing
  label_h5         = LONG
  label_h10        = LONG
  label_h20        = LONG
  label_h32        = LONG
  label_h96        = LONG
  first_touch_h    = 1       ← when did it first hit 12bps?
  persistence_score = 6      ← how many horizons stayed LONG (out of 6)
```

The model now trains to predict ALL of these simultaneously.

By learning to predict timing (when does it move?) + magnitude (how far?) + persistence (how long?) → the model's internal representation becomes MUCH richer.

When you then ask it to predict direction → it uses that rich understanding → better decision.

---

## Part 7: Lookahead as Teacher — Structured Outcome Metrics

The model does NOT receive raw lookahead feature values as input.

Instead, we COMPUTE structured outcome metrics FROM the lookahead data — and these metrics become the LABELS that teach the model **what each market state MEANS.**

**We are NOT teaching the model to predict the future.**
We are showing it 140,000 complete stories: **"when the market was in THIS state → THIS is what actually happened."**

**The chain:**

```
Lookahead feature snapshots (historical truth — already happened)
        ↓
  Compute outcome metrics (what DID happen)
        ↓
  Metrics = LABELS = the meaning of each state
        ↓
  Model learns: "state X historically resolved like this"
```

**Structured outcome metrics computed FROM lookahead:**

| Metric | What it captures | Example (Bar B) |
|--------|-----------------|-----------------|
| mfe_up_bps | Max upward move that actually happened | 80 bps |
| mfe_down_bps | Max downward move that actually happened | 3 bps |
| direction | How it actually resolved (LONG / SHORT / SKIP) | LONG |
| time_to_peak | How many bars until the move peaked | 10 bars |
| persistence | How many of 8 horizons stayed in same direction | 6/8 |
| vol_expansion | Did ATR actually expand during the move? | True |
| volume_expansion | Did volume actually spike during the move? | True |

These are **historical facts** — not future guesses. They describe what actually happened after bar B, computed from real data. They teach the model what each state MEANS and are NEVER used as model input in live trading.

**Why not raw feature values?**

Raw values (ATR=0.91, RSI=25 at t+5...) don't directly tell the model "this state was a LONG" or "this state was a SHORT." There are thousands of them and the signal is buried.

Structured metrics (mfe_up=80bps, direction=LONG, time_to_peak=10 bars) cleanly describe WHAT HAPPENED — so the model can learn what each state MEANS.

```
AFTER Bar B — historical truth we computed:
  mfe_up_bps       = 80     (price DID move up 80bps — historical fact)
  mfe_down_bps     = 3      (price barely moved down — historical fact)
  direction        = LONG   (up > down AND up >= 12bps)
  time_to_peak     = 10     (peak was at t+10 — historical fact)
  persistence      = 6/8    (6 of 8 horizons stayed LONG — historical fact)
  vol_expansion    = True   (ATR DID expand — historical fact)
  volume_expansion = True   (volume DID spike — historical fact)

These 7 facts = the TEACHER for Bar B.
The model learns: "when the state looked like Bar B → this is what it meant."

In live trading: "this new bar's state resembles Bar B
                  → historically this resolved LONG with 80bps"
```

---

## Part 8: The Full Picture

**During training (2020-2023):**

```
For each bar t:

INPUT (what model sees):
  [t-96 snapshot: 1,242 values]
  [t-32 snapshot: 1,242 values]
  [t-20 snapshot: 1,242 values]
  [t-10 snapshot: 1,242 values]
  [t-5  snapshot: 1,242 values]
  [t-3  snapshot: 1,242 values]
  [t-2  snapshot: 1,242 values]
  [t-1  snapshot: 1,242 values]
  [t    snapshot: 1,242 values]  ← current bar
  = 11,178 numbers

TEACHER (structured outcome metrics computed from lookahead):
  direction:        LONG / SHORT / SKIP
  mfe_up_bps:       actual bps (max upward move over 96 bars)
  mfe_down_bps:     actual bps (max downward move over 96 bars)
  time_to_peak:     how many bars until peak move reached
  persistence:      how many of 8 horizons stayed in same direction
  vol_expansion:    did ATR expand during the move? (bool)
  volume_expansion: did volume spike during the move? (bool)
```

Model adjusts itself to learn the link between INPUT (state) and TEACHER (what actually happened).
Repeats for 140,000 bars × multiple epochs.
Stops when it can't improve anymore — it has learned what each state MEANS.

**In live trading (2026):**

```
New bar arrives (2026 bar)
↓
Compute 9 snapshots → 11,178 numbers
↓
Feed to brain (lookahead NOT available — future unknown)
↓
Brain applies learned patterns
↓
Output: LONG 78% / SHORT 14% / SKIP 8%
↓
Decision: LONG
```

The model never sees the future in live trading. It uses only what it learned from 140,000 historical complete stories (past+present+future) to predict from past+present alone.

---

## Part 9: What This Model Really Does — Conditional Outcome Modeling

**Not prediction. Resolution learning.**

A prediction model asks: "What will the price be?"

This model asks: **"Given the current market state, how does it typically resolve?"**

---

### The difference:

```
Prediction model:
  Input: market features
  Output: "Price will be $91,200 at t+10"

This model:
  Input: market features (11,178 columns = full state)
  Output: "direction=LONG 78%, mfe_up≈60bps, time_to_peak≈8 bars, persistence=6/8"
```

---

### What it learns:

The model learns the LINK between market states and how those states RESOLVE.

```
State:
  ATR growing rapidly + RSI deeply oversold + volume expanding
  → historically resolves: LONG 74% of time, ~65bps, peaks at ~8 bars

State:
  ATR flat + RSI neutral + volume low
  → historically resolves: SKIP 81% of time, <12bps move in either direction

State:
  ATR high + price extended above SMA200 + RSI overbought
  → historically resolves: SHORT 68% of time, ~45bps, peaks at ~12 bars
```

The model never explicitly has rules like the above. It discovers thousands of such patterns from 140,000 training examples.

---

### In live trading:

```
New bar arrives
↓
Compute 11,178 column state (lookback snapshots)
↓
Feed to model
↓
Model matches state to learned patterns
↓
Output: direction=LONG 78%, expected mfe_up=58bps, time_to_peak=9 bars
↓
Decision layer uses this to decide: ENTER LONG
```

---

### Why full context (11,178 columns) matters:

More context = more precise state matching.

```
Low context (100 features):
  "ATR high + RSI low" → model sees many states that look similar
  → blurry outcome distribution → uncertain decision

Full context (11,178 features):
  "ATR high AND growing for 10 bars AND volume spiked AND
   RSI was 45, then 35, then 22 AND price fell 3% from 20-day high..."
  → model sees very specific state
  → precise outcome distribution → confident decision
```

LightGBM handles 11,178 features via feature subsampling — each tree only sees a random subset, preventing overfitting while still using all available context.

---

## Part 10: How LightGBM Specifically Learns

### Step 1: The Building Block — One Decision Tree

A decision tree is a series of YES/NO questions on features. It splits the data into groups.

```
Tree asks: "Is keltner_width (at snapshot t) > 0.028?"

                    keltner_width > 0.028?
                   /                      \
                YES                        NO
         (wide channel)              (narrow channel)
         → market moving             → market quiet

Then in the YES branch:
         "Is t_rsi7 < 30?"
        /                \
      YES                 NO
  (oversold)         (not oversold)
  → lean LONG        → lean SHORT or SKIP
```

One tree is just a flowchart. Simple, fast, but WEAK — it misses most patterns.

---

### Step 2: Why One Tree Is Not Enough

If we train ONE tree on 140,000 bars → maybe 58% accuracy.

The problem:
- One tree can only ask a limited number of questions
- It misses complex interactions between features
- "ATR growing + RSI falling + volume spiking together → LONG" — one tree sees this roughly, not precisely

---

### Step 3: Boosting — Many Weak Trees = One Strong Model

This is the core idea of LightGBM. Build **many small trees** where **each tree fixes the mistakes of all previous trees.**

```
Round 1: Train Tree 1 on all 140,000 bars
  Bar 1 (actual LONG):  Tree 1 says LONG  ✓
  Bar 2 (actual LONG):  Tree 1 says SKIP  ✗  ← error
  Bar 3 (actual SHORT): Tree 1 says SHORT ✓
  Bar 4 (actual SHORT): Tree 1 says LONG  ✗  ← error
  Result: 58% correct, 42% wrong

Round 2: Train Tree 2 specifically on the ERRORS of Tree 1
  Tree 2 focuses on bars Tree 1 got wrong
  Tree 2 learns different splits that correct those specific errors
  Result: Trees 1+2 together → 67% correct

Round 3: Train Tree 3 on remaining errors of Trees 1+2
  Result: Trees 1+2+3 together → 73% correct

...
Round 1000: Trees 1 through 1000 together → 81% correct
```

Final answer = ALL 1000 trees vote together. Each tree adds a small correction.

---

### Step 4: The Gradient — How Each Tree Knows What to Fix

After each tree, the gradient tells us exactly:
- Which bars were learned wrong?
- In which direction should the next tree correct?
- By how much?

```
Bar A (actual LONG):
  After Tree 1: P(LONG)=0.35, P(SHORT)=0.35, P(SKIP)=0.30
  Error: LONG probability too low
  Gradient says: "next tree → PUSH LONG UP for states like Bar A"

Bar B (actual SKIP):
  After Tree 1: P(LONG)=0.60, P(SHORT)=0.20, P(SKIP)=0.20
  Error: LONG probability way too high
  Gradient says: "next tree → PUSH LONG DOWN for states like Bar B"

Tree 2 reads these gradients and learns:
  "When ATR=0.89 + RSI=28 + keltner wide → push LONG up"
  "When ATR=0.12 + RSI=52 + keltner narrow → push LONG down"
```

Each new tree is trained to produce those exact corrections. Not the full answer — just the adjustment.

---

### Step 5: How Each Tree Finds Its Best Split (LightGBM Specifically)

**Naive approach (slow):**
```
Check every feature × every possible value:
  "keltner_width > 0.010?" → does this reduce error?
  "keltner_width > 0.011?" → does this reduce error?
  ... × 11,178 features × thousands of values = billions of checks
```

**LightGBM approach — histograms (fast):**
```
For each feature, bin all values into 255 buckets:
  keltner_width: [0.000-0.005], [0.005-0.010], ..., [0.250-0.255]

Now check only 255 splits per feature instead of thousands.
11,178 features × 255 buckets = ~2.8 million checks (manageable)

Pick the split that reduces error most → use as the node split.
```

**Feature subsampling (prevents overfitting):**
```
Each tree only SEES a random subset of features — say 1,000 out of 11,178.

Tree 1 sees: [keltner_width, rsi7, t-10_ATR_slope, t-32_rsi7, ...]
Tree 2 sees: [std20, momentum10, t-5_vol_zscore, t-96_rsi7, ...]
Tree 3 sees: [body_bps, t-1_keltner_slope, t-20_rsi7_pct, ...]
...

Across 1000 trees → ALL 11,178 features get used many times.
But no single tree depends on too many → no overfitting.
```

---

### Step 6: Trace Through Our Direction Model (First 3 Trees)

**Tree 1 — first look at the data:**

```
Best split found: "Is t_keltner_width > 0.027?"

              keltner_width > 0.027?
             /                       \
          YES                         NO
    (68,000 bars)                 (72,000 bars)
    LONG: 38%, SHORT: 35%         LONG: 22%, SHORT: 20%
    SKIP: 27%                     SKIP: 58%

Second level — YES branch: "Is t_rsi7 < 32?"
    YES: LONG 54%, SHORT 25%, SKIP 21% → leaf: lean LONG
    NO:  LONG 24%, SHORT 44%, SKIP 32% → leaf: lean SHORT

Second level — NO branch: "Is t96_ATR_slope > 0.001?"
    YES: LONG 28%, SHORT 28%, SKIP 44% → leaf: slight SKIP
    NO:  LONG 16%, SHORT 14%, SKIP 70% → leaf: strong SKIP

Tree 1 result: 4 leaves, 61% accuracy
```

**Tree 2 — fixes Tree 1's errors:**

```
Tree 1 failed on: bars where RSI was 32-45 but price went up strongly
  → Tree 1 put them in SHORT leaf (wrong)
  → Gradient says: push LONG up for these bars

Tree 2 best split: "Is t-1_rsi7_percentile_rank < 15?"
  (was RSI in lowest 15% of last 96 bars — it's been falling)

    YES (RSI falling to extreme) → Tree 2 pushes LONG up
    NO  (RSI normal)             → small corrections elsewhere

Trees 1+2 together: 69% accuracy
```

**Tree 3 — fixes remaining errors:**

```
Tree 2 failed on: bars where RSI was low but ATR was also low
  → false signals (oversold but no volatility → price won't move 12bps)
  → Gradient says: push SKIP up for these

Tree 3 best split: "Is t_atr_percentile < 25?"
  (ATR in bottom 25% → market too quiet to reach 12bps)

This filters out weak RSI signals in low volatility environments.

Trees 1+2+3 together: 74% accuracy
```

**After 1000 trees:**

```
Tree 1:    caught keltner + RSI pattern              → +3% accuracy
Tree 2:    caught RSI falling-to-extreme pattern     → +8%
Tree 3:    filtered low-ATR false signals            → +5%
Tree 4:    caught volume spike + direction           → +3%
Tree 5:    caught t-10 → t snapshot journey          → +2%
...
Tree 247:  tiny correction for weekend Asia bars     → +0.1%
...
Tree 1000: micro-correction                          → +0.01%

Total: 81% accuracy on 140,000 training bars
```

---

### Step 7: What Does the Final Model Look Like?

```
1000 trees, each 6-8 levels deep.
Each tree = a flowchart of splits on features.

For any new bar:
  → run through all 1000 trees simultaneously
  → each tree gives its vote (LONG / SHORT / SKIP + confidence)
  → sum all 1000 votes
  → final output: LONG=78%, SHORT=14%, SKIP=8%

Time to run: ~10 milliseconds for 11,178 features through 1000 trees
```

---

### Step 8: Feature Importance — What Did LightGBM Find Useful?

After training, we can ask: which features were used most across all 1000 trees?

```
Rank 1: t_keltner_width            (used in 847 of 1000 trees) ← most important
Rank 2: t-1_rsi7_percentile_rank   (used in 791 trees)
Rank 3: t_atr_percentile           (used in 734 trees)
Rank 4: t-5_keltner_slope          (used in 698 trees) ← derivative matters!
Rank 5: t-10_rsi7                  (used in 654 trees) ← 10-bar-ago snapshot matters!
...
Rank 8,934: t-96_hh_count5         (used in 12 trees)  ← barely useful
```

This tells us:
- Which features ACTUALLY matter (data-driven, not our assumption)
- Which snapshot horizons matter most (was it t-1 or t-10 that mattered?)
- Which derivatives add value beyond base features

---

### Summary: How LightGBM Learns

```
1.  Start with zero knowledge
2.  Train Tree 1: finds the single most useful split across 11,178 features
3.  Measure errors: which bars were learned wrong? By how much?
4.  Train Tree 2: fixes those specific errors
5.  Measure remaining errors
6.  Train Tree 3: fixes those
7.  Repeat 1000 times
8.  Each tree is weak alone (one simple flowchart)
9.  Together they are strong (1000 flowcharts voting)
10. Each tree saw a random subset of features → no overfitting
11. Final: 81% accuracy on 140,000 training bars
```

The model never learned explicit rules. It built 1000 overlapping flowcharts that together capture every pattern in the data — including patterns no human could ever write explicitly.

---

## Summary

| Concept | What it means |
|---------|--------------|
| Model | A function: input numbers → output decision |
| Training | Showing 140,000 examples with correct answers |
| Learning | Adjusting internal numbers to reduce prediction error |
| Epoch | One complete pass through all training examples |
| Convergence | Model stops improving → training done |
| Rich labels | Multiple outcome metrics per bar → richer pattern learning |
| Lookahead | Historical future data → compute outcome metrics → become LABELS (teacher) |
| Outcome metrics | direction, mfe_up_bps, mfe_down_bps, time_to_peak, persistence, vol_expansion, volume_expansion |
| Conditional outcome modeling | Model learns: given THIS state → how does it typically RESOLVE? |
| State | 11,178 columns = full lookback journey (9 snapshots × 1,242 values) |
| Resolution | Apply learned state→outcome patterns to new bar → LONG / SHORT / SKIP |
| Decision tree | One flowchart of YES/NO splits on features → weak alone |
| Boosting | 1000 trees, each fixing previous tree's errors → strong together |
| Gradient | Direction + size of correction needed → tells next tree what to fix |
| Histogram | LightGBM bins features into 255 buckets → fast split search |
| Feature subsampling | Each tree sees random subset of features → prevents overfitting |
| Feature importance | Which features used most across 1000 trees → data-driven ranking |
