"""CPU shape tests for the two REACT models.

Run with ``python tests/test_shapes.py`` (or pytest).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from react.models import VideoNas, VisionMoE


def _stage1_args():
    return argparse.Namespace(
        backbone="swin_T_224_1k",
        img_size=224,
        feature_dim=768,
        top_k=2,
        num_experts=4,
        drop_rate=0.0,
        pretrained_path=None,
        fix_backbone=False,
        position_embedding="sine",
    )


def test_stage1_shapes():
    torch.manual_seed(0)
    model = VisionMoE(_stage1_args()).eval()
    images = torch.randn(1, 4, 224, 224)
    with torch.no_grad():
        predictions, moe_loss = model(images)

    assert predictions[0][0].shape == (1, 768)
    assert predictions[0][1].shape == (1, 100)
    assert predictions[1][1].shape == (1, 6)
    assert predictions[2][1].shape == (1, 10)
    assert predictions[3][1].shape == (1, 15)
    assert moe_loss.ndim == 0


def test_stage2_shapes():
    torch.manual_seed(0)
    args = argparse.Namespace(top_k=2, num_experts=4, moe_hidden_dim=64, mask=True)
    model = VideoNas(
        args,
        num_layers_PG=2,
        num_layers_R=2,
        num_R=3,
        num_f_maps=32,
        dim=64,
        num_classes=100,
    ).eval()

    features = torch.randn(1, 16, 64)
    with torch.no_grad():
        out_list, out_i, out_v, out_t, moe_loss = model(features, apply_mask=True)

    assert len(out_list) == 4
    assert out_list[0].shape == (1, 100, 16)
    assert out_i[0].shape == (1, 6, 16)
    assert out_v[0].shape == (1, 10, 16)
    assert out_t[0].shape == (1, 15, 16)
    assert moe_loss.ndim == 0


if __name__ == "__main__":
    test_stage1_shapes()
    test_stage2_shapes()
    print("shape tests passed")
