"""Shared helpers for the command-line scripts."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/foo.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from react.engine import evaluate_stage1_video, evaluate_stage2_video
from react.metrics import VideoMetrics


def make_loader(dataset, batch_size, shuffle=False, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def evaluate_stage1_videos(model, loaders, activation, device, bank, return_metrics=False):
    metrics = VideoMetrics()
    metrics.reset_global()
    for loader in loaders:
        evaluate_stage1_video(loader, model, activation, metrics, device, bank)
    return metrics if return_metrics else metrics.summary()


def evaluate_stage2_videos(model, loaders, activation, device, return_metrics=False):
    metrics = VideoMetrics()
    metrics.reset_global()
    for loader in loaders:
        evaluate_stage2_video(loader, model, activation, metrics, device)
    return metrics if return_metrics else metrics.summary()
