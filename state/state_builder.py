from state.state_schema import MarketState

def build_state(row) -> MarketState:
    return MarketState(
        ema50_slope_z=row["ema50_slope_z"],
        ema200_slope_z=row["ema200_slope_z"],
        trend_alignment=row["trend_alignment"],
        return_5m_z=row["return_5m_z"],
        return_15m_z=row["return_15m_z"],
        rsi_z=row["rsi_z"],
        atr_percentile=row["atr_percentile"],
        volume_z=row["volume_z"],
        vwap_distance_z=row["vwap_distance_z"],
        range_position=row["range_position"],
    )
