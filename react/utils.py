"""Small shared utilities: seeding, logging and checkpoints."""

from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Logger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def __call__(self, message: str):
        print(message)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    def flush(self):
        sys.stdout.flush()


def _to_python(value):
    """Convert NumPy scalars/containers to plain Python objects."""
    if isinstance(value, dict):
        return {key: _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_checkpoint(path, model, epoch, metrics, args=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": _to_python(metrics),
    }
    if args is not None:
        payload["args"] = vars(args)
    torch.save(payload, path)


def load_checkpoint(path, model, strict=False):
    # Checkpoints produced by this package contain only tensors and plain
    # Python metadata.  Older checkpoints may contain NumPy scalars, which
    # PyTorch >= 2.6 rejects under the default weights_only=True; fall back
    # for those trusted local files.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return checkpoint, missing, unexpected


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())
