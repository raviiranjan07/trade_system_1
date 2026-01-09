from typing import Dict, List


class DecisionEngine:
    def __init__(
        self,
        capital: float,
        risk_per_trade: float,
        min_expectancy: float,
        max_distance: float,
        blocked_regimes: List[str],
        min_mfe: float,
        max_leverage: float,
        stop_floor: float,
        use_stop_loss: bool = True,
    ):
        if any(
            v is None
            for v in (
                capital,
                risk_per_trade,
                min_expectancy,
                max_distance,
                blocked_regimes,
                min_mfe,
                max_leverage,
                stop_floor,
                use_stop_loss,
            )
        ):
            raise ValueError(
                "DecisionEngine requires explicit configuration for "
                "capital, risk_per_trade, min_expectancy, max_distance, "
                "blocked_regimes, min_mfe, max_leverage, stop_floor, and use_stop_loss."
            )

        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.min_expectancy = min_expectancy
        self.max_distance = max_distance
        self.blocked_regimes = blocked_regimes
        self.min_mfe = min_mfe
        self.max_leverage = max_leverage
        self.stop_floor = stop_floor
        self.use_stop_loss = use_stop_loss

    def decide(
        self,
        similarity_result: Dict,
        regime: str
    ) -> Dict:
        """
        Balanced decision policy with configurable thresholds.
        """

        # 1. Hard filters (NO TRADE)
        if similarity_result.get("status") != "OK":
            return {"action": "NO_TRADE", "reason": "NO_DATA"}

        if regime in self.blocked_regimes:
            return {"action": "NO_TRADE", "reason": "BLOCKED_REGIME"}

        if similarity_result["distance_mean"] > self.max_distance:
            return {"action": "NO_TRADE", "reason": "LOW_SIMILARITY"}

        # Directional stats (prefer nested structure; fall back to legacy keys)
        long_stats = similarity_result.get("long")
        short_stats = similarity_result.get("short")

        if long_stats is None:
            long_stats = {
                "expectancy": similarity_result.get("expectancy"),
                "mean_mfe": similarity_result.get("mean_mfe"),
                "mean_mae": similarity_result.get("mean_mae"),
                "mae_5pct": similarity_result.get("mae_5pct"),
            }

        # Evaluate each direction INDEPENDENTLY as separate trade options
        # A direction qualifies if its MFE >= min_mfe (MFE is always positive for both)
        candidates = []

        # Check LONG independently
        if long_stats and long_stats.get("mean_mfe") is not None:
            if long_stats["mean_mfe"] >= self.min_mfe:
                candidates.append(("LONG", long_stats))

        # Check SHORT independently
        if short_stats and short_stats.get("mean_mfe") is not None:
            if short_stats["mean_mfe"] >= self.min_mfe:
                candidates.append(("SHORT", short_stats))

        if not candidates:
            return {"action": "NO_TRADE", "reason": "LOW_MFE"}

        # Pick best from QUALIFIED candidates using MFE (higher = more profit potential)
        side, stats = max(candidates, key=lambda x: x[1].get("mean_mfe", 0))

        # 2. Direction
        direction = side

        # 3. Risk parameters
        stop_raw = stats.get("mae_5pct")
        if self.use_stop_loss:
            if stop_raw is None:
                return {"action": "NO_TRADE", "reason": "INVALID_STOP"}

            stop_pct = max(abs(stop_raw), self.stop_floor)
            take_profit_pct = stats.get("mean_mfe", 0.0)

            # 4. Position sizing (risk-based)
            if stop_pct <= 0:
                return {"action": "NO_TRADE", "reason": "INVALID_STOP"}

            position_size = (self.capital * self.risk_per_trade) / stop_pct
        else:
            # No stop loss: fixed notional based on risk_per_trade
            stop_pct = 0.0
            take_profit_pct = stats.get("mean_mfe", 0.0)
            position_size = self.capital * self.risk_per_trade

        # 5. Position cap (safety)
        max_position = self.capital * self.max_leverage  # cap exposure
        position_size = min(position_size, max_position)

        return {
            "action": "TRADE",
            "direction": direction,
            "position_size": position_size,
            "stop_loss_pct": stop_pct,
            "take_profit_pct": take_profit_pct,
            "expectancy": stats["expectancy"],
            "regime": regime,
            "side": side.lower(),
        }
