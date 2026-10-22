"""Frame-level REACT model: CAMoE-guided Vision MoE."""

from __future__ import annotations

from torch import nn

from react.models.backbone import build_backbone
from react.models.camoe import FrameCAMoE


class VisionMoE(nn.Module):
    """Predict I/V/T and IVT labels from a single frame.

    The forward path is: the backbone feature map feeds three component CAM
    heads and the CAMoE decoder; the decoded feature map is added to the
    backbone feature map and projected to the 100 triplet logits.
    """

    def __init__(self, args):
        super().__init__()
        self.encoder = build_backbone(args)
        channels = self.encoder.num_channels

        self.cam_i = nn.Conv2d(channels, 6, kernel_size=1)
        self.cam_v = nn.Conv2d(channels, 10, kernel_size=1)
        self.cam_t = nn.Conv2d(channels, 15, kernel_size=1)
        self.cam_ivt = nn.Conv2d(channels, 100, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        cam_size = args.img_size // 32
        self.moe_decoder = FrameCAMoE(
            num_i=6,
            num_v=10,
            num_t=15,
            feat_dim=channels,
            cam_size=cam_size,
            patch_size=1,
            hidden_dim=128,
            layers=4,
            top_k=args.top_k,
            num_experts=args.num_experts,
        )

    def forward(self, images):
        _, features, _ = self.encoder(images)
        feature_map = features[0]

        cam_i = self.cam_i(feature_map)
        cam_v = self.cam_v(feature_map)
        cam_t = self.cam_t(feature_map)

        _, decoded_cams, moe_loss = self.moe_decoder(cam_i, cam_v, cam_t, feature_map)
        fused_feature = decoded_cams[-1] + feature_map
        cam_ivt = self.cam_ivt(fused_feature)

        pooled = self.pool(feature_map).flatten(1)
        predictions = [
            (pooled, self.pool(cam_ivt).flatten(1)),
            (cam_i, self.pool(cam_i).flatten(1)),
            (cam_v, self.pool(cam_v).flatten(1)),
            (cam_t, self.pool(cam_t).flatten(1)),
            (cam_ivt, self.pool(cam_ivt).flatten(1)),
        ]
        return predictions, moe_loss
