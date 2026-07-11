"""Model zoo — every trainable architecture, and nothing else.

Pure nn.Module definitions + the ARCHITECTURES registry. No training
logic, no I/O, no config loading. Adding an architecture is one class
+ one registry entry; the training pipeline (train.py) picks it up by
name from params.yaml (`architecture:` key).

Naming convention: contract-named classes (what it does), never
model-named (which product uses it).

The retired 2026 H1 zoo (MLPBinaryDir, LSTMAttention, LSTMAttentionV3
+ ONNX wrappers) lives in git history and archive/model_cards/.

Template for the first new-era architecture:

    class MyEncoder(nn.Module):
        \"\"\"One-line contract: inputs -> outputs.\"\"\"

        def __init__(self, input_size: int, hidden: int = 128):
            super().__init__()
            ...

        def forward(self, x):
            ...

    ARCHITECTURES = {"my_encoder": MyEncoder}

If ONNX export needs a different output shape than training (e.g. only
the direction head), add a thin wrapper module here too — export the
wrapper, train the full model.
"""

from __future__ import annotations

import torch  # noqa: F401  (architectures use torch ops)
import torch.nn as nn

ARCHITECTURES: dict[str, type[nn.Module]] = {
    # "my_encoder": MyEncoder,
}


def build(name: str, **kwargs) -> nn.Module:
    """Instantiate an architecture by registry name."""
    if name not in ARCHITECTURES:
        raise KeyError(
            f"Unknown architecture {name!r}. Available: {sorted(ARCHITECTURES)}. "
            "Define the class in training/architectures.py and register it.")
    return ARCHITECTURES[name](**kwargs)
