"""
State Vector Visualization

Plots for visualizing market state vectors and their patterns.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, List

# State vector columns
STATE_COLUMNS = [
    "ema50_slope_z",
    "ema200_slope_z",
    "trend_alignment",
    "return_5m_z",
    "return_15m_z",
    "rsi_z",
    "atr_percentile",
    "volume_z",
    "vwap_distance_z",
    "range_position",
]

# Short labels for display
STATE_LABELS = [
    "EMA50",
    "EMA200",
    "Trend",
    "Ret5m",
    "Ret15m",
    "RSI",
    "ATR%",
    "Vol",
    "VWAP",
    "Range",
]


def plot_state_heatmap(
    state_df: pd.DataFrame,
    n_samples: int = 200,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot heatmap of state vectors over time.

    Args:
        state_df: DataFrame with state vector columns
        n_samples: Number of time samples to show
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get state columns that exist
    cols = [c for c in STATE_COLUMNS if c in state_df.columns]
    labels = [STATE_LABELS[STATE_COLUMNS.index(c)] for c in cols]

    # Sample evenly across time
    step = max(1, len(state_df) // n_samples)
    sampled = state_df[cols].iloc[::step].T

    # Plot heatmap
    im = ax.imshow(
        sampled.values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
    )

    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(labels)

    # Show fewer x-ticks
    n_xticks = 10
    xtick_positions = np.linspace(0, sampled.shape[1] - 1, n_xticks, dtype=int)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([sampled.columns[i].strftime("%Y-%m-%d") for i in xtick_positions], rotation=45, ha="right")

    ax.set_title("State Vector Heatmap Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("State Dimension")

    plt.colorbar(im, ax=ax, label="Z-Score", shrink=0.8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_state_correlation(
    state_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot correlation matrix of state dimensions.

    Args:
        state_df: DataFrame with state vector columns
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get state columns that exist
    cols = [c for c in STATE_COLUMNS if c in state_df.columns]
    labels = [STATE_LABELS[STATE_COLUMNS.index(c)] for c in cols]

    # Compute correlation
    corr = state_df[cols].corr()

    # Plot heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_title("State Dimension Correlations", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_state_distributions(
    state_df: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot distribution of each state dimension.

    Args:
        state_df: DataFrame with state vector columns
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    cols = [c for c in STATE_COLUMNS if c in state_df.columns]
    n_cols = len(cols)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(cols):
        label = STATE_LABELS[STATE_COLUMNS.index(col)]
        data = state_df[col].dropna()

        axes[i].hist(data, bins=50, color="#3498db", alpha=0.7, edgecolor="white")
        axes[i].axvline(data.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {data.mean():.2f}")
        axes[i].axvline(0, color="gray", linestyle="-", alpha=0.5)
        axes[i].set_title(f"{label} ({col})")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("State Dimension Distributions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_state_by_regime(
    state_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot average state vector by regime.

    Args:
        state_df: DataFrame with state vector columns
        regime_df: DataFrame with 'regime' column
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    # Regime colors
    regime_colors = {
        "TREND_HIGH_VOL": "#e74c3c",
        "TREND_LOW_VOL": "#3498db",
        "RANGE_LOW_VOL": "#2ecc71",
        "HIGH_VOL": "#f39c12",
    }

    # Align indices
    common_idx = state_df.index.intersection(regime_df.index)
    states = state_df.loc[common_idx]
    regimes = regime_df.loc[common_idx, "regime"]

    # Get state columns
    cols = [c for c in STATE_COLUMNS if c in state_df.columns]
    labels = [STATE_LABELS[STATE_COLUMNS.index(c)] for c in cols]

    # Compute mean by regime
    combined = states[cols].copy()
    combined["regime"] = regimes
    mean_by_regime = combined.groupby("regime")[cols].mean()

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(cols))
    width = 0.2
    n_regimes = len(mean_by_regime)

    for i, (regime, row) in enumerate(mean_by_regime.iterrows()):
        offset = (i - n_regimes / 2 + 0.5) * width
        ax.bar(
            x + offset,
            row.values,
            width,
            label=regime,
            color=regime_colors.get(regime, "#cccccc"),
            alpha=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean Value")
    ax.set_title("Average State Vector by Regime", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_pca_states(
    state_df: pd.DataFrame,
    regime_df: Optional[pd.DataFrame] = None,
    n_samples: int = 5000,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot PCA projection of state vectors.

    Args:
        state_df: DataFrame with state vector columns
        regime_df: Optional DataFrame with 'regime' column for coloring
        n_samples: Number of samples to plot
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    from sklearn.decomposition import PCA

    # Get state columns
    cols = [c for c in STATE_COLUMNS if c in state_df.columns]
    data = state_df[cols].dropna()

    # Sample if needed
    if len(data) > n_samples:
        sample_idx = np.random.choice(len(data), n_samples, replace=False)
        data = data.iloc[sample_idx]

    # PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(data.values)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Scatter plot
    if regime_df is not None:
        # Color by regime
        regime_colors = {
            "TREND_HIGH_VOL": "#e74c3c",
            "TREND_LOW_VOL": "#3498db",
            "RANGE_LOW_VOL": "#2ecc71",
            "HIGH_VOL": "#f39c12",
        }
        common_idx = data.index.intersection(regime_df.index)
        regimes = regime_df.loc[common_idx, "regime"]
        colors = [regime_colors.get(r, "#cccccc") for r in regimes]

        scatter = axes[0].scatter(projected[:, 0], projected[:, 1], c=colors, alpha=0.5, s=5)

        # Legend
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=c, label=r) for r, c in regime_colors.items()]
        axes[0].legend(handles=patches, loc="upper right", fontsize=8)
    else:
        axes[0].scatter(projected[:, 0], projected[:, 1], alpha=0.5, s=5, c="#3498db")

    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0].set_title("PCA Projection of State Vectors")
    axes[0].grid(True, alpha=0.3)

    # Component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2"],
        index=[STATE_LABELS[STATE_COLUMNS.index(c)] for c in cols],
    )

    loadings.plot(kind="barh", ax=axes[1])
    axes[1].set_title("PCA Component Loadings")
    axes[1].set_xlabel("Loading")
    axes[1].axvline(0, color="gray", linestyle="-", alpha=0.5)
    axes[1].grid(True, alpha=0.3, axis="x")

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
        state_df = pd.read_parquet(outcomes_path)
        regime_df = pd.read_parquet(regimes_path)

        Path("output").mkdir(exist_ok=True)

        print("Plotting state heatmap...")
        plot_state_heatmap(state_df, save_path="output/state_heatmap.png")

        print("Plotting state correlation...")
        plot_state_correlation(state_df, save_path="output/state_correlation.png")

        print("Plotting state distributions...")
        plot_state_distributions(state_df, save_path="output/state_distributions.png")

        print("Plotting state by regime...")
        plot_state_by_regime(state_df, regime_df, save_path="output/state_by_regime.png")

        print("Plotting PCA projection...")
        plot_pca_states(state_df, regime_df, save_path="output/state_pca.png")

        print("Done!")
    else:
        print("Data files not found. Run the pipeline first.")
