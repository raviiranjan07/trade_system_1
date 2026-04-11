"""Brain Models -- LSTM Architecture

Input: (batch, snapshot_count, num_features)
Output: direction logits + MFE predictions
"""

import torch
import torch.nn as nn


class BrainLSTM(nn.Module):
    """LSTM brain with multi-task output (direction + MFE).

    Processes 8 snapshots sequentially. Each snapshot has all features.
    Uses last hidden state for predictions.
    """

    def __init__(self, num_features, hidden_size=128, num_layers=1, dropout=0.2,
                 num_mfe_horizons=8):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        # Direction head: binary classification (LONG=0, SHORT=1)
        self.direction_head = nn.Linear(hidden_size, 2)

        # MFE head: predict mfe_up and mfe_down at each horizon
        self.mfe_head = nn.Linear(hidden_size, num_mfe_horizons * 2)

        self.num_mfe_horizons = num_mfe_horizons

    def forward(self, x):
        """
        x: (batch, seq_len, features)
        Returns: direction_logits (batch, 2), mfe_pred (batch, num_horizons*2)
        """
        _, (h, _) = self.lstm(x)
        h = h[-1]  # last layer hidden state: (batch, hidden)
        h = self.dropout(h)

        direction = self.direction_head(h)
        mfe = self.mfe_head(h)

        return direction, mfe


class BrainMLP(nn.Module):
    """MLP brain -- flattens all snapshots into one vector.

    Input: (batch, snapshot_count, num_features) -> flatten -> (batch, snapshot_count * num_features)
    """

    def __init__(self, num_features, snapshot_count=8, hidden_size=128, dropout=0.2,
                 num_mfe_horizons=8):
        super().__init__()
        input_size = num_features * snapshot_count

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.direction_head = nn.Linear(hidden_size, 2)
        self.mfe_head = nn.Linear(hidden_size, num_mfe_horizons * 2)
        self.num_mfe_horizons = num_mfe_horizons

    def forward(self, x):
        """
        x: (batch, seq_len, features)
        """
        x = x.reshape(x.size(0), -1)  # flatten
        h = self.net(x)

        direction = self.direction_head(h)
        mfe = self.mfe_head(h)

        return direction, mfe


class BrainHybrid(nn.Module):
    """Hybrid brain -- dense per snapshot, then LSTM across snapshots.

    Each snapshot processed by a shared dense layer first,
    then LSTM processes the sequence of dense outputs.
    """

    def __init__(self, num_features, hidden_size=128, snapshot_dense=64,
                 num_layers=1, dropout=0.2, num_mfe_horizons=8):
        super().__init__()

        # Per-snapshot dense layer (shared across all snapshots)
        self.snapshot_net = nn.Sequential(
            nn.Linear(num_features, snapshot_dense),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # LSTM processes the sequence of dense outputs
        self.lstm = nn.LSTM(
            input_size=snapshot_dense,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        self.direction_head = nn.Linear(hidden_size, 2)
        self.mfe_head = nn.Linear(hidden_size, num_mfe_horizons * 2)
        self.num_mfe_horizons = num_mfe_horizons

    def forward(self, x):
        """
        x: (batch, seq_len, features)
        """
        batch, seq_len, feat = x.shape

        # Process each snapshot through dense layer
        x_flat = x.reshape(batch * seq_len, feat)
        x_dense = self.snapshot_net(x_flat)
        x_dense = x_dense.reshape(batch, seq_len, -1)

        # LSTM across snapshots
        _, (h, _) = self.lstm(x_dense)
        h = h[-1]
        h = self.dropout(h)

        direction = self.direction_head(h)
        mfe = self.mfe_head(h)

        return direction, mfe


class BrainGRU(nn.Module):
    """GRU brain -- same as LSTM but uses GRU cells."""

    def __init__(self, num_features, hidden_size=128, num_layers=1, dropout=0.2,
                 num_mfe_horizons=8):
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.direction_head = nn.Linear(hidden_size, 2)
        self.mfe_head = nn.Linear(hidden_size, num_mfe_horizons * 2)
        self.num_mfe_horizons = num_mfe_horizons

    def forward(self, x):
        _, h = self.gru(x)
        h = h[-1]
        h = self.dropout(h)
        direction = self.direction_head(h)
        mfe = self.mfe_head(h)
        return direction, mfe


def build_model(config, num_features, snapshot_count=8):
    """Build model from config."""
    arch = config["training"]["architecture"]
    model_cfg = config["training"]["model"]
    hidden = model_cfg.get("hidden_size", 128)
    layers = model_cfg.get("num_layers", 1)
    dropout = model_cfg.get("dropout", 0.2)
    snapshot_dense = model_cfg.get("snapshot_dense", 64)

    mfe_cfg = config["labels"].get("mfe", {})
    num_mfe_horizons = len(mfe_cfg.get("horizons", [1,2,3,4,5,6,7,8]))

    if arch == "lstm":
        return BrainLSTM(num_features, hidden, layers, dropout, num_mfe_horizons)
    elif arch == "mlp":
        return BrainMLP(num_features, snapshot_count, hidden, dropout, num_mfe_horizons)
    elif arch == "hybrid":
        return BrainHybrid(num_features, hidden, snapshot_dense, layers, dropout, num_mfe_horizons)
    elif arch == "gru":
        return BrainGRU(num_features, hidden, layers, dropout, num_mfe_horizons)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
