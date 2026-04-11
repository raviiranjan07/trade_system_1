"""Brain Pipeline -- Step 8: Live Paper Trading Integration

Generates brain signals for the live bot.
Maintains S/R state across bars, logs decisions.

Usage:
  from brain.live import BrainLive
  brain_live = BrainLive(config)
  brain_live.startup()

  # Every 15min bar:
  signal = brain_live.on_new_bar(ohlcv_df)
"""

import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from brain.processing import compute_all_features
from brain.output import BrainInference


class BrainLive:
    """Live brain signal generator for paper trading."""

    def __init__(self, config: dict):
        self.config = config
        self.live_cfg = config.get("live", {})

        # Paths
        self.state_file = Path(self.live_cfg.get("state", {}).get(
            "state_file", "data/v12_trades/brain_state.json"))
        self.log_file = Path(self.live_cfg.get("monitoring", {}).get(
            "log_file", "data/risk_logs/brain/decisions.csv"))
        self.error_log = Path(self.live_cfg.get("error_handling", {}).get(
            "error_log", "data/risk_logs/brain/errors.csv"))

        # Thresholds
        self.long_thresh = self.live_cfg.get("thresholds", {}).get("long_confidence", 0.60)
        self.short_thresh = self.live_cfg.get("thresholds", {}).get("short_confidence", 0.65)

        # State
        self.bar_buffer = []  # last N bars for feature computation
        self.snapshot_count = config["dataset"]["snapshot_count"]
        self.sr_lookback = config["features"]["market_structure"].get("sr_lookback", 8)
        self.buffer_size = self.snapshot_count + self.sr_lookback + 10  # enough history

        # Inference engine
        model_dir = self.live_cfg.get("model_path", "experiments/brain/EXP-B001")
        config_path = self.live_cfg.get("config_path", "configs/base.yaml")
        self.brain = None  # loaded in startup()
        self.model_dir = model_dir
        self.config_path = config_path

        # Stats
        self.total_predictions = 0
        self.total_signals = 0

    def startup(self):
        """Initialize brain for live trading."""
        print("Brain Live: Starting up...")

        # Load model
        self.brain = BrainInference(self.model_dir, self.config_path)

        # Load state
        self._load_state()

        # Ensure log directories exist
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.error_log.parent.mkdir(parents=True, exist_ok=True)

        print(f"  Long threshold: {self.long_thresh}")
        print(f"  Short threshold: {self.short_thresh}")
        print(f"  Buffer size: {self.buffer_size}")
        print("Brain Live: Ready")

    def on_new_bar(self, bar: dict) -> dict:
        """Process a new bar and return brain signal.

        Args:
            bar: dict with keys: timestamp, open, high, low, close, volume

        Returns:
            dict with direction, confidence, mfe_up, mfe_down, context
        """
        try:
            # Add to buffer
            self.bar_buffer.append(bar)
            if len(self.bar_buffer) > self.buffer_size:
                self.bar_buffer = self.bar_buffer[-self.buffer_size:]

            # Need enough bars for features
            if len(self.bar_buffer) < self.snapshot_count + self.sr_lookback:
                return {"direction": "SKIP", "confidence": 0.0, "reason": "warming_up"}

            # Build OHLCV dataframe from buffer
            df = pd.DataFrame(self.bar_buffer)
            df.index = pd.to_datetime(df["timestamp"])
            df = df[["open", "high", "low", "close", "volume"]]

            # Compute features
            features = compute_all_features(df, self.config)

            # Build snapshot (last snapshot_count bars)
            feat_arr = features.values.astype(np.float32)
            if len(feat_arr) < self.snapshot_count:
                return {"direction": "SKIP", "confidence": 0.0, "reason": "not_enough_bars"}

            snapshot = feat_arr[-self.snapshot_count:]  # (8, 23)

            # Convert raw prices to relative bps
            snapshot = self._convert_raw_prices(snapshot, features.columns.tolist())

            # Predict
            result = self.brain.predict(snapshot)

            # Log
            self.total_predictions += 1
            if result["direction"] != "SKIP":
                self.total_signals += 1

            self._log_decision(bar, result)

            # Save state
            self._save_state()

            return result

        except Exception as e:
            error_action = self.live_cfg.get("error_handling", {}).get("on_model_error", "skip")
            self._log_error(bar, str(e))

            if error_action == "halt":
                raise
            return {"direction": "SKIP", "confidence": 0.0, "reason": f"error: {str(e)}"}

    def _convert_raw_prices(self, snapshot: np.ndarray, col_names: list) -> np.ndarray:
        """Convert raw price columns to relative bps."""
        raw_cols = ["support_range_low", "support_range_high",
                    "resistance_range_low", "resistance_range_high",
                    "window_low", "window_high"]

        if "window_low" in col_names and "window_high" in col_names:
            wl_idx = col_names.index("window_low")
            wh_idx = col_names.index("window_high")

            for row in range(len(snapshot)):
                midpoint = (snapshot[row, wh_idx] + snapshot[row, wl_idx]) / 2
                if midpoint > 0:
                    for col_name in raw_cols:
                        if col_name in col_names:
                            idx = col_names.index(col_name)
                            snapshot[row, idx] = (snapshot[row, idx] - midpoint) / midpoint * 10000

        return snapshot

    def _log_decision(self, bar: dict, result: dict):
        """Log every decision to CSV."""
        if not self.live_cfg.get("monitoring", {}).get("log_every_bar", True):
            if result["direction"] == "SKIP":
                return

        row = {
            "timestamp": bar.get("timestamp", ""),
            "close": bar.get("close", 0),
            "direction": result["direction"],
            "confidence": result["confidence"],
            "long_prob": result.get("long_prob", 0),
            "short_prob": result.get("short_prob", 0),
            "mfe_up": result.get("mfe_up", 0),
            "mfe_down": result.get("mfe_down", 0),
            "position": result.get("context", {}).get("position", ""),
            "zone_width": result.get("context", {}).get("zone_width", 0),
        }

        file_exists = self.log_file.exists()
        with open(self.log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _log_error(self, bar: dict, error_msg: str):
        """Log errors."""
        row = {
            "timestamp": bar.get("timestamp", ""),
            "error": error_msg,
        }
        file_exists = self.error_log.exists()
        with open(self.error_log, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _load_state(self):
        """Load persisted state."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
            self.total_predictions = state.get("total_predictions", 0)
            self.total_signals = state.get("total_signals", 0)
            print(f"  Loaded state: {self.total_predictions} predictions, {self.total_signals} signals")

    def _save_state(self):
        """Save state for persistence across restarts."""
        state = {
            "total_predictions": self.total_predictions,
            "total_signals": self.total_signals,
            "last_update": datetime.utcnow().isoformat(),
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
