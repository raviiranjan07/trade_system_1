"""Zone Registry — tracks S/R zones across time with memory.

Each zone has:
- Identity: center, width, original position
- Role: support or resistance (can flip)
- Memory: touches, bounces, breaks, history
- State: active or dead

No future information used. Memory updates only after outcome is known.
"""

from collections import deque
import numpy as np


class Zone:
    """A single S/R zone with memory."""

    def __init__(self, zone_id: int, center: float, width: float,
                 low: float, high: float, role: str, bar: int):
        self.id = zone_id
        self.center = center
        self.width = width
        self.low = low
        self.high = high
        self.original_center = center
        self.original_width = width
        self.role = role  # "support" or "resistance"
        self.created_bar = bar
        self.last_touch_bar = bar
        self.active = True

        # Memory
        self.touches = 0
        self.bounces = 0
        self.breaks = 0
        self.support_bounces = 0
        self.support_breaks = 0
        self.resistance_bounces = 0
        self.resistance_breaks = 0
        self.history = deque(maxlen=10)

        # MFE tracking per outcome
        self.bounce_mfes = []    # MFE values for each bounce
        self.break_mfes = []     # MFE values for each break

    def get_bounce_ratio(self, prior_num=1, prior_den=2):
        """Bayesian smoothed bounce ratio."""
        return (self.bounces + prior_num) / (self.touches + prior_den)

    def get_break_ratio(self, prior_num=1, prior_den=2):
        """Bayesian smoothed break ratio."""
        return (self.breaks + prior_num) / (self.touches + prior_den)

    def get_recent_bounce_ratio(self, n=5):
        """Bounce ratio of last n touches."""
        recent = list(self.history)[-n:]
        if len(recent) == 0:
            return 0.5
        return sum(1 for x in recent if x == "bounce") / len(recent)

    def get_recent_break_ratio(self, n=5):
        """Break ratio of last n touches."""
        recent = list(self.history)[-n:]
        if len(recent) == 0:
            return 0.5
        return sum(1 for x in recent if x == "break") / len(recent)

    def get_pressure(self, min_touches=3):
        """Consecutive breaks from end of history."""
        if self.touches < min_touches:
            return 0
        pressure = 0
        for outcome in reversed(self.history):
            if outcome == "break":
                pressure += 1
            else:
                break
        return pressure

    def update_memory(self, outcome: str, role_at_touch: str, mfe_value: float = None):
        """Update memory after outcome is known.

        outcome: "bounce", "break", or "chop"
        role_at_touch: what role the zone had when touched
        mfe_value: favorable MFE in bps (how far price moved in the outcome direction)
        """
        self.touches += 1

        if outcome == "bounce":
            self.bounces += 1
            if mfe_value is not None:
                self.bounce_mfes.append(mfe_value)
            if role_at_touch == "support":
                self.support_bounces += 1
            else:
                self.resistance_bounces += 1
        elif outcome == "break":
            self.breaks += 1
            if mfe_value is not None:
                self.break_mfes.append(mfe_value)
            if role_at_touch == "support":
                self.support_breaks += 1
            else:
                self.resistance_breaks += 1
        # chop: touches++ but bounces/breaks unchanged

        if outcome != "chop":
            self.history.append(outcome)

    def get_avg_bounce_mfe(self):
        """Average MFE when zone bounced."""
        return np.mean(self.bounce_mfes) if len(self.bounce_mfes) > 0 else 0.0

    def get_avg_break_mfe(self):
        """Average MFE when zone broke."""
        return np.mean(self.break_mfes) if len(self.break_mfes) > 0 else 0.0

    def get_max_bounce_mfe(self):
        """Max MFE when zone bounced."""
        return max(self.bounce_mfes) if len(self.bounce_mfes) > 0 else 0.0

    def get_max_break_mfe(self):
        """Max MFE when zone broke."""
        return max(self.break_mfes) if len(self.break_mfes) > 0 else 0.0

    def get_last_outcome(self):
        """Most recent touch result. 1=bounce, 0=break, 0.5=no history."""
        if len(self.history) == 0:
            return 0.5
        last = self.history[-1]
        return 1.0 if last == "bounce" else 0.0

    def get_bounce_streak(self):
        """Consecutive bounces from end of history (opposite of pressure)."""
        streak = 0
        for outcome in reversed(self.history):
            if outcome == "bounce":
                streak += 1
            else:
                break
        return streak

    def get_bounce_mfe_trend(self):
        """Are recent bounces stronger or weaker? Positive = getting stronger."""
        if len(self.bounce_mfes) < 2:
            return 0.0
        recent = self.bounce_mfes[-3:] if len(self.bounce_mfes) >= 3 else self.bounce_mfes
        if len(recent) < 2:
            return 0.0
        return recent[-1] - recent[0]

    def get_chop_ratio(self):
        """Ratio of chop outcomes to total touches."""
        if self.touches == 0:
            return 0.0
        chop_count = self.touches - self.bounces - self.breaks
        return chop_count / self.touches

    def smooth_update_center(self, new_center: float, smoothing=0.9, drift_cap=0.5):
        """Update center with bounded smoothing."""
        updated = smoothing * self.center + (1 - smoothing) * new_center
        max_drift = drift_cap * self.original_width
        if abs(updated - self.original_center) > max_drift:
            return  # don't update, drift too far
        self.center = updated


class ZoneRegistry:
    """Maintains a registry of S/R zones across time."""

    def __init__(self, config: dict):
        cfg = config.get("zone_registry", {})
        self.tolerance_alpha = cfg.get("tolerance_alpha", 0.001)
        self.drift_smoothing = cfg.get("drift_smoothing", 0.9)
        self.drift_cap = cfg.get("drift_cap", 0.5)
        self.death_break_distance = cfg.get("death_break_distance", 50)
        self.death_consecutive_breaks = cfg.get("death_consecutive_breaks", 3)
        self.death_stale_bars = cfg.get("death_stale_bars", 100)
        self.role_flip_alpha = cfg.get("role_flip_alpha", 0.3)
        self.min_touch_gap = cfg.get("min_touch_gap", 5)

        filt_cfg = config.get("zone_filter", {})
        self.touch_threshold = filt_cfg.get("touch_threshold", 0.2)
        self.zone_width_min = filt_cfg.get("zone_width_min", 30)

        mem_cfg = config.get("memory", {})
        self.bayesian_prior_num = mem_cfg.get("bayesian_prior_num", 1)
        self.bayesian_prior_den = mem_cfg.get("bayesian_prior_den", 2)
        self.recent_history_size = mem_cfg.get("recent_history_size", 5)
        self.pressure_min_touches = mem_cfg.get("pressure_min_touches", 3)

        self.zones = []  # list of Zone objects
        self.next_id = 0

    def find_matching_zone(self, low: float, high: float, price: float):
        """Find a registered zone that matches the given range.

        Searches ALL zones (active and inactive).
        If an inactive zone matches, it gets reactivated.
        Returns matching Zone or None.
        Priority: 1. Range overlap, 2. Tolerance floor, 3. None
        """
        center = (low + high) / 2.0
        tolerance = self.tolerance_alpha * price  # e.g., 0.001 * 60000 = 60 (~10bps)

        best_zone = None
        best_dist = float("inf")

        for zone in self.zones:
            # Check range overlap
            overlap = zone.high >= low and zone.low <= high
            if overlap:
                dist = abs(center - zone.center)
                if dist < best_dist:
                    best_zone = zone
                    best_dist = dist
                continue

            # Check tolerance floor
            edge_dist = min(abs(low - zone.high), abs(high - zone.low))
            if edge_dist <= tolerance:
                dist = abs(center - zone.center)
                if dist < best_dist:
                    best_zone = zone
                    best_dist = dist

        # Reactivate if inactive
        if best_zone is not None and not best_zone.active:
            best_zone.active = True

        return best_zone

    def register_zone(self, low: float, high: float, role: str, bar: int) -> Zone:
        """Register a new zone."""
        center = (low + high) / 2.0
        width = high - low
        zone = Zone(self.next_id, center, width, low, high, role, bar)
        self.next_id += 1
        self.zones.append(zone)
        return zone

    def process_zone(self, low: float, high: float, role: str, bar: int, price: float) -> Zone:
        """Find or create a zone. Update center if matched."""
        match = self.find_matching_zone(low, high, price)
        if match:
            new_center = (low + high) / 2.0
            match.smooth_update_center(new_center, self.drift_smoothing, self.drift_cap)
            match.low = low
            match.high = high
            return match
        else:
            return self.register_zone(low, high, role, bar)

    def is_touch(self, dist_to_zone: float, zone_width: float) -> bool:
        """Check if price is touching a zone (normalized threshold)."""
        if zone_width < self.zone_width_min:
            return False
        if zone_width == 0:
            return False
        return (dist_to_zone / zone_width) <= self.touch_threshold

    def check_role_flip(self, zone: Zone, close: float, zone_width_price: float, price: float):
        """Check if zone should flip role based on current price."""
        threshold = max(self.role_flip_alpha * zone_width_price, self.tolerance_alpha * price)

        if zone.role == "support" and close < zone.low - threshold:
            zone.role = "resistance"
        elif zone.role == "resistance" and close > zone.high + threshold:
            zone.role = "support"

    def check_zone_death(self, bar: int, close: float):
        """Mark zones as dead if conditions met."""
        for zone in self.zones:
            if not zone.active:
                continue

            # Stale: no touch for N bars
            if bar - zone.last_touch_bar > self.death_stale_bars:
                zone.active = False
                continue

            # Consecutive breaks
            if zone.get_pressure(min_touches=1) >= self.death_consecutive_breaks:
                zone.active = False
                continue

            # Price broke through by large amount
            if zone.role == "support":
                dist = (zone.low - close) / zone.low * 10000
                if dist > self.death_break_distance:
                    zone.active = False
            elif zone.role == "resistance":
                dist = (close - zone.high) / zone.high * 10000
                if dist > self.death_break_distance:
                    zone.active = False

    def get_memory_features(self, zone: Zone, bar: int) -> dict:
        """Extract static memory features for a zone at given bar (14 features)."""
        return {
            "bounce_ratio": zone.get_bounce_ratio(self.bayesian_prior_num, self.bayesian_prior_den),
            "recent_bounce_ratio": zone.get_recent_bounce_ratio(self.recent_history_size),
            "pressure": zone.get_pressure(self.pressure_min_touches),
            "bars_since_touch": bar - zone.last_touch_bar,
            "touch_count_scaled": np.log1p(zone.touches),
            "level_type_binary": 1.0 if zone.role == "support" else 0.0,
            "avg_bounce_mfe": zone.get_avg_bounce_mfe(),
            "avg_break_mfe": zone.get_avg_break_mfe(),
            "max_bounce_mfe": zone.get_max_bounce_mfe(),
            "max_break_mfe": zone.get_max_break_mfe(),
            "last_outcome": zone.get_last_outcome(),
            "bounce_streak": zone.get_bounce_streak(),
            "bounce_mfe_trend": zone.get_bounce_mfe_trend(),
            "chop_ratio": zone.get_chop_ratio(),
        }
