"""
Expansion Module - Event-based labeling for trading signals.

This module provides tools for:
1. Analyzing price data to determine expansion thresholds
2. Labeling expansion events (binary: did price expand before reversing?)
3. Querying expansion rates from similar historical states

Key Concept:
    Instead of asking "how much did price move?" (MFE/MAE),
    we ask "did price expand fast in one direction before reversing?"

    EXPANSION = 1 if price hits +X% before hitting -Y%
    EXPANSION = 0 otherwise

Usage:
    from trade_system.expansion import ThresholdAnalyzer, ExpansionLabeler

    # Step 1: Analyze thresholds
    analyzer = ThresholdAnalyzer(ohlcv_df)
    thresholds = analyzer.analyze(horizon=5, percentile=0.75)

    # Step 2: Label expansions
    labeler = ExpansionLabeler(ohlcv_df)
    labels = labeler.label(config)

    # Step 3: Query expansion rate in similarity
    expansion_rate = query_expansion_rate(neighbors, labels)
"""

from .threshold_analyzer import ThresholdAnalyzer, ThresholdResult, save_thresholds, load_thresholds
from .expansion_labeler import ExpansionLabeler, ExpansionConfig, save_expansion_labels, load_expansion_labels
from .expansion_query import ExpansionQueryEngine

__all__ = [
    # Threshold analysis
    "ThresholdAnalyzer",
    "ThresholdResult",
    "save_thresholds",
    "load_thresholds",

    # Expansion labeling
    "ExpansionLabeler",
    "ExpansionConfig",
    "save_expansion_labels",
    "load_expansion_labels",

    # Expansion querying
    "ExpansionQueryEngine",
]
