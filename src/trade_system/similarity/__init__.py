"""
Similarity Engine Module.

KNN-based state vector matching for finding similar historical patterns.
Includes multi-horizon adaptive selection for optimal trade horizon.
Also includes Risk Profiler for case-based risk assessment.
"""

from .similarity_engine import SimilarityEngine
from .multi_horizon_engine import MultiHorizonEngine
from .risk_profiler import RiskProfiler

__all__ = ["SimilarityEngine", "MultiHorizonEngine", "RiskProfiler"]
