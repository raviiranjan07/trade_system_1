"""
Outcome Visualization

Plots for visualizing MFE/MAE distributions and expectancy analysis.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

# Regime color scheme (consistent with plot_regimes.py)
REGIME_COLORS = {
    "TREND_HIGH_VOL": "#e74c3c",
    "TREND_LOW_VOL": "#3498db",
    "RANGE_LOW_VOL": "#2ecc71",
    "HIGH_VOL": "#f39c12",
}


def plot_mfe_mae_distribution(
    outcome_df: pd.DataFrame,
    horizon: int = 30,
    direction: str = "long",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot MFE and MAE distributions for a specific horizon.

    Args:
        outcome_df: DataFrame with MFE/MAE columns
        horizon: Time horizon in minutes (10, 30, or 120)
        direction: 'long' or 'short'
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    mfe_col = f"mfe_{direction}_{horizon}m"
    mae_col = f"mae_{direction}_{horizon}m"

    mfe = outcome_df[mfe_col] * 100  # Convert to percentage
    mae = outcome_df[mae_col] * 100

    # MFE histogram
    axes[0].hist(mfe, bins=100, color="#2ecc71", alpha=0.7, edgecolor="white")
    axes[0].axvline(mfe.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {mfe.mean():.2f}%")
    axes[0].axvline(mfe.median(), color="orange", linestyle="--", linewidth=2, label=f"Median: {mfe.median():.2f}%")
    axes[0].set_title(f"MFE Distribution ({direction.upper()} {horizon}m)")
    axes[0].set_xlabel("MFE (%)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # MAE histogram
    axes[1].hist(mae, bins=100, color="#e74c3c", alpha=0.7, edgecolor="white")
    axes[1].axvline(mae.mean(), color="blue", linestyle="--", linewidth=2, label=f"Mean: {mae.mean():.2f}%")
    axes[1].axvline(mae.median(), color="cyan", linestyle="--", linewidth=2, label=f"Median: {mae.median():.2f}%")
    axes[1].set_title(f"MAE Distribution ({direction.upper()} {horizon}m)")
    axes[1].set_xlabel("MAE (%)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # MFE vs MAE scatter
    sample_size = min(5000, len(outcome_df))
    sample_idx = np.random.choice(len(outcome_df), sample_size, replace=False)
    axes[2].scatter(
        mae.iloc[sample_idx],
        mfe.iloc[sample_idx],
        alpha=0.3,
        s=1,
        c="#3498db",
    )
    axes[2].axhline(0, color="gray", linestyle="-", alpha=0.5)
    axes[2].axvline(0, color="gray", linestyle="-", alpha=0.5)
    axes[2].set_title(f"MFE vs MAE ({direction.upper()} {horizon}m)")
    axes[2].set_xlabel("MAE (%)")
    axes[2].set_ylabel("MFE (%)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_expectancy_by_regime(
    outcome_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    horizon: int = 30,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot expected MFE/MAE by market regime.

    Args:
        outcome_df: DataFrame with MFE/MAE columns
        regime_df: DataFrame with 'regime' column
        horizon: Time horizon in minutes
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Align indices
    common_idx = outcome_df.index.intersection(regime_df.index)
    outcomes = outcome_df.loc[common_idx]
    regimes = regime_df.loc[common_idx, "regime"]

    # Compute mean MFE/MAE by regime
    mfe_long = outcomes[f"mfe_long_{horizon}m"] * 100
    mae_long = outcomes[f"mae_long_{horizon}m"] * 100

    stats_by_regime = pd.DataFrame({
        "regime": regimes,
        "mfe_long": mfe_long,
        "mae_long": mae_long,
    }).groupby("regime").agg({
        "mfe_long": ["mean", "std", "count"],
        "mae_long": ["mean", "std"],
    })

    # Flatten column names
    stats_by_regime.columns = ["mfe_mean", "mfe_std", "count", "mae_mean", "mae_std"]
    stats_by_regime = stats_by_regime.sort_values("count", ascending=False)

    regimes_list = stats_by_regime.index.tolist()
    colors = [REGIME_COLORS.get(r, "#cccccc") for r in regimes_list]
    x = np.arange(len(regimes_list))
    width = 0.35

    # MFE by regime
    axes[0].bar(
        x - width / 2,
        stats_by_regime["mfe_mean"],
        width,
        label="Avg MFE",
        color=colors,
        alpha=0.8,
        yerr=stats_by_regime["mfe_std"] / 10,  # Scaled error bars
        capsize=3,
    )
    axes[0].bar(
        x + width / 2,
        stats_by_regime["mae_mean"],
        width,
        label="Avg MAE",
        color=colors,
        alpha=0.4,
        hatch="//",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(regimes_list, rotation=45, ha="right")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title(f"Mean MFE/MAE by Regime ({horizon}m horizon)")
    axes[0].legend()
    axes[0].axhline(0, color="gray", linestyle="-", alpha=0.5)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Expectancy (MFE + MAE) by regime
    expectancy = stats_by_regime["mfe_mean"] + stats_by_regime["mae_mean"]  # MAE is negative
    bar_colors = ["#2ecc71" if e > 0 else "#e74c3c" for e in expectancy]

    axes[1].bar(regimes_list, expectancy, color=bar_colors, alpha=0.8)
    axes[1].set_xticklabels(regimes_list, rotation=45, ha="right")
    axes[1].set_ylabel("Expectancy (%)")
    axes[1].set_title(f"Net Expectancy by Regime ({horizon}m horizon)")
    axes[1].axhline(0, color="gray", linestyle="-", linewidth=2)
    axes[1].grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, (regime, exp) in enumerate(zip(regimes_list, expectancy)):
        axes[1].text(
            i, exp + (0.02 if exp > 0 else -0.04),
            f"{exp:.2f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_outcome_over_time(
    outcome_df: pd.DataFrame,
    horizon: int = 30,
    window: int = 1000,
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot rolling average of MFE/MAE over time.

    Args:
        outcome_df: DataFrame with MFE/MAE columns
        horizon: Time horizon in minutes
        window: Rolling window size
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    mfe = outcome_df[f"mfe_long_{horizon}m"] * 100
    mae = outcome_df[f"mae_long_{horizon}m"] * 100

    mfe_rolling = mfe.rolling(window).mean()
    mae_rolling = mae.rolling(window).mean()
    expectancy_rolling = mfe_rolling + mae_rolling

    ax.plot(mfe_rolling.index, mfe_rolling.values, label="MFE (rolling)", color="#2ecc71", linewidth=1)
    ax.plot(mae_rolling.index, mae_rolling.values, label="MAE (rolling)", color="#e74c3c", linewidth=1)
    ax.plot(expectancy_rolling.index, expectancy_rolling.values, label="Expectancy (rolling)", color="#3498db", linewidth=2)

    ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax.fill_between(
        expectancy_rolling.index,
        0,
        expectancy_rolling.values,
        where=expectancy_rolling > 0,
        alpha=0.3,
        color="#2ecc71",
    )
    ax.fill_between(
        expectancy_rolling.index,
        0,
        expectancy_rolling.values,
        where=expectancy_rolling <= 0,
        alpha=0.3,
        color="#e74c3c",
    )

    ax.set_title(f"Rolling MFE/MAE/Expectancy ({horizon}m horizon, {window} bar window)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Percentage (%)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_horizon_comparison(
    outcome_df: pd.DataFrame,
    direction: str = "long",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare MFE/MAE across different horizons.

    Args:
        outcome_df: DataFrame with MFE/MAE columns
        direction: 'long' or 'short'
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    config = Config()
    horizons = config.get("outcomes.horizons", [10, 15, 30, 120])
    mfe_means = []
    mae_means = []
    mfe_stds = []
    mae_stds = []

    for h in horizons:
        mfe = outcome_df[f"mfe_{direction}_{h}m"] * 100
        mae = outcome_df[f"mae_{direction}_{h}m"] * 100
        mfe_means.append(mfe.mean())
        mae_means.append(mae.mean())
        mfe_stds.append(mfe.std())
        mae_stds.append(mae.std())

    x = np.arange(len(horizons))
    width = 0.35

    # Mean comparison
    axes[0].bar(x - width/2, mfe_means, width, label="MFE", color="#2ecc71", yerr=np.array(mfe_stds)/5, capsize=3)
    axes[0].bar(x + width/2, mae_means, width, label="MAE", color="#e74c3c", yerr=np.array(mae_stds)/5, capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{h}m" for h in horizons])
    axes[0].set_ylabel("Mean (%)")
    axes[0].set_title(f"Mean MFE/MAE by Horizon ({direction.upper()})")
    axes[0].legend()
    axes[0].axhline(0, color="gray", linestyle="-", alpha=0.5)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Expectancy comparison
    expectancy = [mfe + mae for mfe, mae in zip(mfe_means, mae_means)]
    colors = ["#2ecc71" if e > 0 else "#e74c3c" for e in expectancy]
    axes[1].bar([f"{h}m" for h in horizons], expectancy, color=colors, alpha=0.8)
    axes[1].set_ylabel("Expectancy (%)")
    axes[1].set_title(f"Net Expectancy by Horizon ({direction.upper()})")
    axes[1].axhline(0, color="gray", linestyle="-", linewidth=2)
    axes[1].grid(True, alpha=0.3, axis="y")

    for i, exp in enumerate(expectancy):
        axes[1].text(i, exp + (0.01 if exp > 0 else -0.02), f"{exp:.3f}%", ha="center", fontsize=10, fontweight="bold")

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
        outcome_df = pd.read_parquet(outcomes_path)
        regime_df = pd.read_parquet(regimes_path)

        Path("output").mkdir(exist_ok=True)

        print("Plotting MFE/MAE distribution...")
        plot_mfe_mae_distribution(outcome_df, horizon=30, save_path="output/mfe_mae_dist_30m.png")

        print("Plotting expectancy by regime...")
        plot_expectancy_by_regime(outcome_df, regime_df, horizon=30, save_path="output/expectancy_by_regime.png")

        print("Plotting horizon comparison...")
        plot_horizon_comparison(outcome_df, save_path="output/horizon_comparison.png")

        print("Done!")
    else:
        print("Data files not found. Run the pipeline first.")
