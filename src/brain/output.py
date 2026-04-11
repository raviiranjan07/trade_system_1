"""Brain Pipeline -- Step 7: Output (Inference)

Load trained model and produce predictions for new bars.
Used by both evaluation and live trading.

Usage:
  from brain.output import BrainInference
  brain = BrainInference("experiments/brain/EXP-B001", "configs/base.yaml")
  result = brain.predict(snapshot)  # snapshot shape: (8, 23)
"""

import json
import numpy as np
import torch
from pathlib import Path


class BrainInference:
    """Inference engine for the trained brain model."""

    def __init__(self, model_dir: str, config_path: str):
        model_dir = Path(model_dir)

        # Load config
        try:
            from brain.config import load_config
            self.config = load_config(config_path)
        except ImportError:
            import yaml
            with open(config_path) as f:
                self.config = yaml.safe_load(f)

        # Load metadata
        data_dir = Path(self.config.get("tracking", {}).get("experiment_dir", "experiments/brain/")) / "datasets"
        with open(data_dir / "metadata.json") as f:
            self.meta = json.load(f)

        self.feature_names = self.meta["feature_names"]
        self.num_features = self.meta["snapshot_shape"][1]
        self.snapshot_count = self.meta["snapshot_shape"][0]

        # Load scaler
        scaler_data = np.load(data_dir / "scaler.npz")
        self.scaler_mean = scaler_data["mean"]
        self.scaler_std = scaler_data["std"]

        # Load model
        self.device = torch.device("cpu")  # inference always on CPU

        try:
            from brain.models.lstm import build_model
            self.model = build_model(self.config, self.num_features, self.snapshot_count)
        except ImportError:
            from brain.training import build_model_from_config
            self.model = build_model_from_config(self.config, self.num_features, self.snapshot_count)

        self.model.load_state_dict(
            torch.load(model_dir / "model.pt", weights_only=True, map_location=self.device)
        )
        self.model.eval()

        print(f"Brain loaded: {self.config['training']['architecture']}, {self.num_features} features")

    def predict(self, snapshot: np.ndarray) -> dict:
        """Predict direction and MFE from a snapshot.

        Args:
            snapshot: numpy array shape (snapshot_count, num_features)
                      Raw feature values (not normalized)

        Returns:
            dict with direction, confidence, mfe_up, mfe_down, context
        """
        # Normalize
        snapshot_norm = (snapshot.astype(np.float32) - self.scaler_mean) / self.scaler_std
        snapshot_norm = np.nan_to_num(snapshot_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Add batch dimension
        x = torch.tensor(snapshot_norm).unsqueeze(0)  # (1, 8, 23)

        with torch.no_grad():
            dir_logits, mfe_pred = self.model(x)
            probs = torch.softmax(dir_logits, dim=1).numpy()[0]
            mfe = mfe_pred.numpy()[0]

        # Direction
        long_prob = float(probs[0])
        short_prob = float(probs[1])

        if long_prob > short_prob:
            direction = "LONG"
            confidence = long_prob
        else:
            direction = "SHORT"
            confidence = short_prob

        # Check thresholds
        live_cfg = self.config.get("live", {}).get("thresholds", {})
        long_thresh = live_cfg.get("long_confidence", 0.60)
        short_thresh = live_cfg.get("short_confidence", 0.65)

        if direction == "LONG" and confidence < long_thresh:
            direction = "SKIP"
        elif direction == "SHORT" and confidence < short_thresh:
            direction = "SKIP"

        # MFE predictions (last horizon)
        num_horizons = len(self.config.get("labels", {}).get("mfe", {}).get("horizons", [1,2,3,4,5,6,7,8]))
        mfe_up = float(mfe[(num_horizons - 1) * 2])      # last horizon up
        mfe_down = float(mfe[(num_horizons - 1) * 2 + 1]) # last horizon down

        # S/R context from last snapshot
        context = self._extract_context(snapshot[-1])

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "long_prob": round(long_prob, 4),
            "short_prob": round(short_prob, 4),
            "mfe_up": round(mfe_up, 1),
            "mfe_down": round(mfe_down, 1),
            "context": context,
        }

    def _extract_context(self, features: np.ndarray) -> dict:
        """Extract S/R context from feature vector."""
        ctx = {}
        fn = self.feature_names

        if "zone_width" in fn:
            ctx["zone_width"] = round(float(features[fn.index("zone_width")]), 1)
        if "support_retest" in fn:
            ctx["support_touches"] = int(features[fn.index("support_retest")])
        if "resistance_retest" in fn:
            ctx["resistance_touches"] = int(features[fn.index("resistance_retest")])
        if "distance_to_support" in fn:
            ctx["dist_to_support"] = round(float(features[fn.index("distance_to_support")]), 1)
        if "distance_to_resistance" in fn:
            ctx["dist_to_resistance"] = round(float(features[fn.index("distance_to_resistance")]), 1)

        # Position
        dist_sup = ctx.get("dist_to_support", 999)
        dist_res = ctx.get("dist_to_resistance", 999)
        if dist_sup <= 5:
            ctx["position"] = "at_support"
        elif dist_res <= 5:
            ctx["position"] = "at_resistance"
        elif dist_sup < 0:
            ctx["position"] = "below_support"
        elif dist_res < 0:
            ctx["position"] = "above_resistance"
        else:
            ctx["position"] = "in_middle"

        return ctx
