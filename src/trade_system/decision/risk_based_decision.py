"""
Risk-Based Decision Engine: Entry filter based on risk profile.

Instead of predicting direction, we filter entries based on:
- P(Case 1) < threshold (low wrong-direction probability)
- Median MAE < threshold (acceptable expected drawdown)
- Median recovery < threshold (reasonable hold time)

This is NOT direction prediction - it's risk filtering.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from ..config import Config


@dataclass
class RiskThresholds:
    """Configurable thresholds for entry filtering."""
    max_p_case1: float = 0.10        # Max 10% wrong direction probability
    max_mae_median: float = 30.0      # Max 30bp expected median MAE
    max_recovery_median: float = 50.0  # Max 50 bars expected recovery
    min_p_recovery: float = 0.80      # Min 80% recovery probability
    max_distance: float = 3.0         # Max neighbor distance
    min_edge_ratio: float = 1.5       # Min directional edge ratio (CRITICAL: filters noise)


class RiskBasedDecisionEngine:
    """
    Entry filter based on risk profile, not direction prediction.

    The key insight: We don't need to predict direction.
    We just need to filter out high-risk entries.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize with configurable thresholds.

        Args:
            config: Dictionary with threshold overrides, or None for defaults
        """
        if config is None:
            try:
                cfg = Config()
                config = cfg.get("risk_model", {})
            except Exception:
                config = {}

        self.thresholds = RiskThresholds(
            max_p_case1=config.get("max_p_case1", 0.10),
            max_mae_median=config.get("max_mae_median", 30.0),
            max_recovery_median=config.get("max_recovery_median", 50.0),
            min_p_recovery=config.get("min_p_recovery", 0.80),
            max_distance=config.get("max_distance", 3.0),
            min_edge_ratio=config.get("min_edge_ratio", 1.5),  # CRITICAL: filter noise
        )

    def should_enter(self, risk_profile: Dict) -> Tuple[bool, str]:
        """
        Determine if entry is allowed based on risk profile.

        This is NOT direction prediction - it's risk filtering.

        Args:
            risk_profile: Dictionary from RiskProfiler.get_risk_profile()

        Returns:
            Tuple of (should_enter: bool, reason: str)
        """
        # Check if profile is valid
        if risk_profile.get("status") != "OK":
            return False, f"Invalid profile: {risk_profile.get('status', 'UNKNOWN')}"

        # CRITICAL: Check directional edge - this filters out NOISE (50/50 states)
        edge_ratio = risk_profile.get("edge_ratio", 1.0)
        if edge_ratio < self.thresholds.min_edge_ratio:
            return False, f"No edge: ratio={edge_ratio:.2f} < {self.thresholds.min_edge_ratio:.1f}"

        # Check P(Case 1) - wrong direction probability
        p_case1 = risk_profile.get("p_case1", 1.0)
        if p_case1 > self.thresholds.max_p_case1:
            return False, f"P(Case1)={p_case1:.1%} > {self.thresholds.max_p_case1:.0%}"

        # Check P(Recovery) - probability of eventual win
        p_recovery = risk_profile.get("p_recovery", 0.0)
        if p_recovery < self.thresholds.min_p_recovery:
            return False, f"P(Recovery)={p_recovery:.1%} < {self.thresholds.min_p_recovery:.0%}"

        # Check median MAE
        mae_median = risk_profile.get("mae_median", float('inf'))
        if not np.isnan(mae_median) and mae_median > self.thresholds.max_mae_median:
            return False, f"MAE_median={mae_median:.1f}bp > {self.thresholds.max_mae_median:.0f}bp"

        # Check median recovery time
        recovery_median = risk_profile.get("recovery_median", float('inf'))
        if not np.isnan(recovery_median) and recovery_median > self.thresholds.max_recovery_median:
            return False, f"Recovery_median={recovery_median:.0f} > {self.thresholds.max_recovery_median:.0f} bars"

        # Check neighbor distance (optional - ensures similar states exist)
        distance_mean = risk_profile.get("distance_mean", float('inf'))
        if distance_mean > self.thresholds.max_distance:
            return False, f"Distance={distance_mean:.2f} > {self.thresholds.max_distance:.1f}"

        return True, "Risk acceptable"

    def get_entry_decision(self, risk_profile: Dict) -> Dict:
        """
        Get detailed entry decision with all checks.

        Args:
            risk_profile: Dictionary from RiskProfiler.get_risk_profile()

        Returns:
            Dictionary with decision details:
            {
                "action": "ENTER" or "SKIP",
                "reason": str,
                "checks": {
                    "p_case1": {"value": float, "threshold": float, "passed": bool},
                    "p_recovery": {"value": float, "threshold": float, "passed": bool},
                    "mae_median": {"value": float, "threshold": float, "passed": bool},
                    "recovery_median": {"value": float, "threshold": float, "passed": bool},
                    "distance": {"value": float, "threshold": float, "passed": bool},
                },
                "risk_score": float,
            }
        """
        should_enter, reason = self.should_enter(risk_profile)

        # Detailed checks
        edge_ratio = risk_profile.get("edge_ratio", 1.0)
        p_case1 = risk_profile.get("p_case1", 1.0)
        p_recovery = risk_profile.get("p_recovery", 0.0)
        mae_median = risk_profile.get("mae_median", float('nan'))
        recovery_median = risk_profile.get("recovery_median", float('nan'))
        distance_mean = risk_profile.get("distance_mean", float('nan'))

        checks = {
            "edge_ratio": {
                "value": edge_ratio,
                "threshold": self.thresholds.min_edge_ratio,
                "passed": edge_ratio >= self.thresholds.min_edge_ratio
            },
            "p_case1": {
                "value": p_case1,
                "threshold": self.thresholds.max_p_case1,
                "passed": p_case1 <= self.thresholds.max_p_case1
            },
            "p_recovery": {
                "value": p_recovery,
                "threshold": self.thresholds.min_p_recovery,
                "passed": p_recovery >= self.thresholds.min_p_recovery
            },
            "mae_median": {
                "value": mae_median,
                "threshold": self.thresholds.max_mae_median,
                "passed": np.isnan(mae_median) or mae_median <= self.thresholds.max_mae_median
            },
            "recovery_median": {
                "value": recovery_median,
                "threshold": self.thresholds.max_recovery_median,
                "passed": np.isnan(recovery_median) or recovery_median <= self.thresholds.max_recovery_median
            },
            "distance": {
                "value": distance_mean,
                "threshold": self.thresholds.max_distance,
                "passed": np.isnan(distance_mean) or distance_mean <= self.thresholds.max_distance
            },
        }

        return {
            "action": "ENTER" if should_enter else "SKIP",
            "reason": reason,
            "checks": checks,
            "risk_score": risk_profile.get("risk_score", 1.0),
            "target_bps": risk_profile.get("target_bps"),
            "horizon": risk_profile.get("horizon"),
        }

    def update_thresholds(self, **kwargs):
        """
        Update thresholds dynamically.

        Args:
            **kwargs: Threshold names and new values
        """
        for key, value in kwargs.items():
            if hasattr(self.thresholds, key):
                setattr(self.thresholds, key, value)

    def get_thresholds(self) -> Dict:
        """Get current threshold values."""
        return {
            "min_edge_ratio": self.thresholds.min_edge_ratio,  # CRITICAL
            "max_p_case1": self.thresholds.max_p_case1,
            "max_mae_median": self.thresholds.max_mae_median,
            "max_recovery_median": self.thresholds.max_recovery_median,
            "min_p_recovery": self.thresholds.min_p_recovery,
            "max_distance": self.thresholds.max_distance,
        }


if __name__ == "__main__":
    # Test the decision engine
    print("="*60)
    print("RISK-BASED DECISION ENGINE TEST")
    print("="*60)

    engine = RiskBasedDecisionEngine()
    print(f"\nDefault thresholds: {engine.get_thresholds()}")

    # Test with sample risk profiles
    test_profiles = [
        # Good profile - should enter
        {
            "status": "OK",
            "p_case1": 0.05,
            "p_recovery": 0.90,
            "mae_median": 20.0,
            "recovery_median": 30.0,
            "distance_mean": 1.5,
            "risk_score": 0.15,
        },
        # Bad profile - high P(Case1)
        {
            "status": "OK",
            "p_case1": 0.20,
            "p_recovery": 0.75,
            "mae_median": 40.0,
            "recovery_median": 60.0,
            "distance_mean": 2.0,
            "risk_score": 0.45,
        },
        # Bad profile - high MAE
        {
            "status": "OK",
            "p_case1": 0.08,
            "p_recovery": 0.85,
            "mae_median": 50.0,
            "recovery_median": 40.0,
            "distance_mean": 1.8,
            "risk_score": 0.35,
        },
    ]

    for i, profile in enumerate(test_profiles):
        print(f"\n--- Test Profile {i+1} ---")
        decision = engine.get_entry_decision(profile)
        print(f"Action: {decision['action']}")
        print(f"Reason: {decision['reason']}")
        print("Checks:")
        for name, check in decision['checks'].items():
            status = "PASS" if check['passed'] else "FAIL"
            print(f"  {name}: {check['value']:.2f} vs {check['threshold']:.2f} [{status}]")
