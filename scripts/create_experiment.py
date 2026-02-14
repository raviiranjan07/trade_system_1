"""
Create a new experiment with proper structure

Usage:
    python scripts/create_experiment.py --category ema --name "My Strategy Name"

This will:
1. Create experiment folder: experiments/{category}/EXP-XXX/
2. Generate template files (config.yaml, notes.md, etc.)
3. Return the experiment ID to use
"""

import argparse
from pathlib import Path
import yaml
from datetime import datetime
import pandas as pd

def get_next_exp_id(category):
    """Get next experiment ID by checking existing folders"""
    exp_dir = Path(f"experiments/{category}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Find existing experiments
    existing = [d.name for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("EXP-")]

    if not existing:
        return "EXP-001"

    # Get highest number
    nums = [int(e.split("-")[1]) for e in existing]
    next_num = max(nums) + 1
    return f"EXP-{next_num:03d}"

def create_experiment(category, name, description=""):
    """Create new experiment structure"""

    # Get exp ID
    exp_id = get_next_exp_id(category)
    exp_path = Path(f"experiments/{category}/{exp_id}")

    # Create directories
    exp_path.mkdir(parents=True, exist_ok=True)
    (exp_path / "plots").mkdir(exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Creating Experiment: {exp_id}")
    print(f"Category: {category}")
    print(f"Name: {name}")
    print(f"{'='*80}\n")

    # Create config.yaml
    config = {
        'experiment_id': exp_id,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'status': 'IN_PROGRESS',
        'category': category,
        'strategy': {
            'name': name,
            'description': description
        },
        'parameters': {
            'timeframe': 'FILL_IN',
            'symbol': 'BTCUSDT',
            'entry': {},
            'exit': {},
            'fees': {
                'bps': 8,
                'type': 'round_trip'
            }
        },
        'data': {
            'test_period_start': '2024-01-01',
            'test_period_end': '2025-12-30'
        }
    }

    with open(exp_path / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"[OK] Created: {exp_path / 'config.yaml'}")

    # Create notes.md template
    notes_template = f"""# {exp_id}: {name}

## What we're testing
[Describe in simple English what you're testing and why]

## Hypothesis
[What do you expect to happen?]

## Parameters
See config.yaml

## Results
[To be filled after backtest]

## What we learned
[Key insights from this experiment]

## Next steps
[What should we test next based on these results?]
"""

    with open(exp_path / 'notes.md', 'w') as f:
        f.write(notes_template)

    print(f"[OK] Created: {exp_path / 'notes.md'}")

    # Create empty metrics.json
    with open(exp_path / 'metrics.json', 'w') as f:
        f.write('{}')

    print(f"[OK] Created: {exp_path / 'metrics.json'}")

    # Create explanation.txt in simple English
    explanation_template = f"""EXPERIMENT {exp_id}: {name}
{'='*80}

WHAT ARE WE TESTING?
Write in simple English what you're testing. Explain like you're talking to someone
who doesn't know trading.

Example: "We're testing if buying when price crosses above a moving average line
is profitable. The moving average is like a smoothed version of the price that
shows the general trend."


WHY ARE WE TESTING THIS?
Explain why you think this might work. What's the logic behind it?

Example: "When price crosses above the average, it might mean the trend is
changing from down to up. Traders might start buying, pushing price higher."


HOW DOES IT WORK?
Explain the rules in simple steps:

1. Entry: [When do we buy/sell?]
   Example: "Buy when price crosses above EMA7"

2. Exit: [When do we close the trade?]
   Example: "Sell when profit reaches 12 bps OR price drops 10 bps from peak"

3. Filters: [Any conditions to skip trades?]
   Example: "Skip if price is too close to EMA (less than 5 bps away)"


WHAT HAPPENED? (Fill after running backtest)
- Total trades: [number]
- Win rate: [percentage]
- Total return: [profit or loss]
- Average winner: [how much winners make]
- Average loser: [how much losers lose]


WHAT DID WE LEARN? (Fill after analyzing results)
Write in simple English what worked and what didn't:

Example: "The strategy made money (+102%) but only won 35% of the time.
It works because winners are much bigger than losers (22 bps vs 10 bps).
The entry filter helped avoid choppy trades."


WHAT SHOULD WE DO NEXT?
Based on what we learned, what's the next thing to test?

Example: "The strategy is profitable but choppy. Next, test if a slower
moving average (EMA15 instead of EMA7) gives cleaner signals with fewer
false entries."


{'='*80}
"""

    with open(exp_path / 'explanation.txt', 'w') as f:
        f.write(explanation_template)

    print(f"[OK] Created: {exp_path / 'explanation.txt'}")

    print(f"\n{'='*80}")
    print(f"Experiment {exp_id} created successfully!")
    print(f"Location: {exp_path}")
    print(f"\nNext steps:")
    print(f"1. Edit explanation.txt - Explain in simple English what you're testing")
    print(f"2. Edit config.yaml - Fill in technical parameters")
    print(f"3. Edit notes.md - Document hypothesis and approach")
    print(f"4. Run your backtest and save results to {exp_id}/results.csv")
    print(f"5. Update explanation.txt with results and learnings")
    print(f"6. Save any charts to {exp_id}/plots/")
    print(f"7. Add summary to experiments/registry.csv")
    print(f"{'='*80}\n")

    return exp_id, exp_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create new experiment")
    parser.add_argument("--category", required=True, help="Experiment category (e.g., ema, rsi, ml)")
    parser.add_argument("--name", required=True, help="Strategy name")
    parser.add_argument("--description", default="", help="Brief description")

    args = parser.parse_args()

    exp_id, exp_path = create_experiment(args.category, args.name, args.description)
