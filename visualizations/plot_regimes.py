"""
Regime Visualization

Plots for visualizing market regime classifications over time.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


# Regime color scheme
REGIME_COLORS = {
    "TREND_HIGH_VOL": "#e74c3c",   # Red - strong moves
    "TREND_LOW_VOL": "#3498db",    # Blue - gradual trends
    "RANGE_LOW_VOL": "#2ecc71",    # Green - consolidation
    "HIGH_VOL": "#f39c12",         # Orange - choppy
}


def plot_regime_timeline(
    price_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    title: str = "Price with Market Regimes",
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot price chart with regime-colored background.

    Args:
        price_df: DataFrame with 'close' column indexed by time
        regime_df: DataFrame with 'regime' column indexed by time
        title: Chart title
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Align indices
    common_idx = price_df.index.intersection(regime_df.index)
    prices = price_df.loc[common_idx, "close"]
    regimes = regime_df.loc[common_idx, "regime"]

    # Plot price line
    ax.plot(prices.index, prices.values, color="black", linewidth=0.8, alpha=0.8)

    # Color background by regime
    current_regime = None
    start_idx = 0

    for i, (idx, regime) in enumerate(regimes.items()):
        if regime != current_regime:
            if current_regime is not None:
                # Fill previous regime block
                ax.axvspan(
                    regimes.index[start_idx],
                    regimes.index[i - 1],
                    alpha=0.3,
                    color=REGIME_COLORS.get(current_regime, "#cccccc"),
                )
            current_regime = regime
            start_idx = i

    # Fill last regime block
    if current_regime is not None:
        ax.axvspan(
            regimes.index[start_idx],
            regimes.index[-1],
            alpha=0.3,
            color=REGIME_COLORS.get(current_regime, "#cccccc"),
        )

    # Legend
    patches = [
        mpatches.Patch(color=color, alpha=0.5, label=regime)
        for regime, color in REGIME_COLORS.items()
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_regime_distribution(
    regime_df: pd.DataFrame,
    title: str = "Regime Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot regime distribution as pie and bar charts.

    Args:
        regime_df: DataFrame with 'regime' column
        title: Chart title
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Count regimes
    regime_counts = regime_df["regime"].value_counts()
    colors = [REGIME_COLORS.get(r, "#cccccc") for r in regime_counts.index]

    # Pie chart
    ax1.pie(
        regime_counts.values,
        labels=regime_counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax1.set_title("Regime Proportions")

    # Bar chart
    bars = ax1.bar(regime_counts.index, regime_counts.values, color=colors)
    ax2.bar(regime_counts.index, regime_counts.values, color=colors)
    ax2.set_title("Regime Counts")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis="x", rotation=45)

    # Add count labels on bars
    for bar, count in zip(ax2.patches, regime_counts.values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 100,
            f"{count:,}",
            ha="center",
            fontsize=9,
        )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_regime_transitions(
    regime_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot regime transition matrix as heatmap.

    Args:
        regime_df: DataFrame with 'regime' column
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Compute transition matrix
    regimes = regime_df["regime"]
    transitions = pd.crosstab(
        regimes.shift(1).dropna(),
        regimes.iloc[1:],
        normalize="index",
    )

    # Plot heatmap
    im = ax.imshow(transitions.values, cmap="YlOrRd", aspect="auto")

    # Labels
    ax.set_xticks(range(len(transitions.columns)))
    ax.set_yticks(range(len(transitions.index)))
    ax.set_xticklabels(transitions.columns, rotation=45, ha="right")
    ax.set_yticklabels(transitions.index)

    ax.set_xlabel("To Regime")
    ax.set_ylabel("From Regime")
    ax.set_title("Regime Transition Probabilities", fontsize=14, fontweight="bold")

    # Add values
    for i in range(len(transitions.index)):
        for j in range(len(transitions.columns)):
            val = transitions.iloc[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10)

    plt.colorbar(im, ax=ax, label="Probability")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


if __name__ == "__main__":
    # Demo with sample data
    print("Loading data...")

    outcomes_path = Path("data/outcomes/BTCUSDT_1m_outcomes.parquet")
    regimes_path = Path("data/regimes/BTCUSDT_1m_regimes.parquet")

    if outcomes_path.exists() and regimes_path.exists():
        outcomes_df = pd.read_parquet(outcomes_path)
        regime_df = pd.read_parquet(regimes_path)

        # Create price-like column from outcomes (using close proxy)
        # For demo, we'll use the index as time
        price_df = pd.DataFrame({"close": np.cumsum(np.random.randn(len(regime_df))) + 100}, index=regime_df.index)

        print("Plotting regime distribution...")
        plot_regime_distribution(regime_df, save_path="output/regime_distribution.png")

        print("Plotting regime transitions...")
        plot_regime_transitions(regime_df, save_path="output/regime_transitions.png")

        print("Done!")
    else:
        print("Data files not found. Run the pipeline first.")
