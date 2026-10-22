"""Swin backbone adapter used by the REACT frame-level model."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from react.models.position_encoding import build_position_encoding
from react.models.swin_transformer import build_swin_transformer


SUPPORTED_BACKBONES = {
    "swin_T_224_1k",
    "swin_T_224_22k",
    "swin_B_224_22k",
    "swin_B_384_22k",
    "swin_L_224_22k",
    "swin_L_384_22k",
}


class SwinBackbone(nn.Module):
    """Expose the last Swin feature map together with its position encoding."""

    def __init__(self, backbone, position_embedding):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.num_channels = backbone.embed_dim * 8

    def forward(self, images):
        _, feature_map = self.backbone(images)
        position = self.position_embedding(feature_map).to(feature_map.dtype)
        return [feature_map], [feature_map], [position]


def _load_pretrained(backbone, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
        if not key.removeprefix("module.").startswith("head.")
    }
    incompatible = backbone.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained backbone from {checkpoint_path}: {incompatible}")


def _expand_patch_embed_to_four_channels(backbone):
    """Add a fourth input channel for the normalized frame timestamp."""
    projection = backbone.patch_embed.proj
    if projection.in_channels == 4:
        return
    if projection.in_channels != 3:
        raise ValueError(
            f"Expected a 3-channel patch embedding, got {projection.in_channels}"
        )

    expanded = nn.Conv2d(
        in_channels=4,
        out_channels=projection.out_channels,
        kernel_size=projection.kernel_size,
        stride=projection.stride,
        padding=projection.padding,
        bias=projection.bias is not None,
    )
    with torch.no_grad():
        expanded.weight[:, :3] = projection.weight
        expanded.weight[:, 3:] = projection.weight.mean(dim=1, keepdim=True)
        if projection.bias is not None:
            expanded.bias.copy_(projection.bias)
    backbone.patch_embed.proj = expanded


def build_backbone(args):
    if args.backbone not in SUPPORTED_BACKBONES:
        choices = ", ".join(sorted(SUPPORTED_BACKBONES))
        raise ValueError(f"Unsupported backbone {args.backbone!r}. Choose one of: {choices}")

    image_size = int(args.backbone.split("_")[-2])
    backbone = build_swin_transformer(
        args, args.backbone, image_size, num_classes=args.feature_dim
    )

    if args.pretrained_path:
        _load_pretrained(backbone, args.pretrained_path)

    _expand_patch_embed_to_four_channels(backbone)

    if args.fix_backbone:
        for name, parameter in backbone.named_parameters():
            if name.startswith(("layers.0", "layers.1")):
                parameter.requires_grad_(False)

    return SwinBackbone(backbone, build_position_encoding(args))
