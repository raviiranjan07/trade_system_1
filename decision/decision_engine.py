from typing import Dict, List


class DecisionEngine:
    def __init__(
        self,
        capital: float,
        risk_per_trade: float = 0.005,  # 0.5%
        min_expectancy: float = 0.0,
        max_distance: float = 1.5,
        blocked_regimes: List[str] = None
    ):
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.min_expectancy = min_expectancy
        self.max_distance = max_distance
        self.blocked_regimes = blocked_regimes or ["HIGH_VOL"]

    def decide(
        self,
        similarity_result: Dict,
        regime: str
    ) -> Dict:
        """
        Balanced decision policy with configurable thresholds.
        """

        # -------------------------
        # 1️⃣ Hard filters (NO TRADE)
        # -------------------------
        if similarity_result.get("status") != "OK":
            return {"action": "NO_TRADE", "reason": "NO_DATA"}

        if similarity_result["expectancy"] <= self.min_expectancy:
            return {"action": "NO_TRADE", "reason": "NEGATIVE_EXPECTANCY"}

        if regime in self.blocked_regimes:
            return {"action": "NO_TRADE", "reason": "BLOCKED_REGIME"}

        if similarity_result["distance_mean"] > self.max_distance:
            return {"action": "NO_TRADE", "reason": "LOW_SIMILARITY"}

        # -------------------------
        # 2️⃣ Direction
        # -------------------------
        if similarity_result["mean_mfe"] > abs(similarity_result["mean_mae"]):
            direction = "LONG"
        else:
            direction = "SHORT"

        # -------------------------
        # 3️⃣ Risk parameters
        # -------------------------
        stop_pct = abs(similarity_result["mae_5pct"])
        take_profit_pct = similarity_result["mean_mfe"]

        if stop_pct <= 0:
            return {"action": "NO_TRADE", "reason": "INVALID_RISK"}

        # -------------------------
        # 4️⃣ Position sizing
        # -------------------------
        risk_amount = self.capital * self.risk_per_trade
        position_size = risk_amount / stop_pct

        # -------------------------
        # 5️⃣ Position cap (safety)
        # -------------------------
        max_position = self.capital * 1.0  # max 1x exposure

        position_size = min(position_size, max_position)
        
        return {
            "action": "TRADE",
            "direction": direction,
            "position_size": position_size,
            "stop_loss_pct": stop_pct,
            "take_profit_pct": take_profit_pct,
            "expectancy": similarity_result["expectancy"],
            "regime": regime,
        }
        
        

