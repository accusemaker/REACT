"""Class-activation-guided mixture-of-experts (CAMoE) decoders.

Two variants are provided:

* :class:`FrameCAMoE` fuses the frame-level I/V/T CAMs with the backbone
  feature map and reconstructs a feature map that is used for triplet
  prediction.
* :class:`TemporalCAMoE` fuses the video-level I/V/T CAMs with the MS-TCN
  feature sequence and reconstructs a temporal feature sequence.
"""

from __future__ import annotations

import torch
from torch import nn

from react.models.moe import MoE, PatchEmbed


class FrameCAMoE(nn.Module):
    """Frame-level CAMoE decoder.

    Args:
        num_i / num_v / num_t: number of component classes.
        feat_dim: number of backbone feature channels (also the reconstructed
            feature dimension).
        cam_size: spatial size of the backbone feature map.
        patch_size: patch size used by the tokeniser (1 keeps every location).
        hidden_dim: token dimension inside the decoder.
        layers: number of transformer layers.
    """

    def __init__(
        self,
        num_i: int,
        num_v: int,
        num_t: int,
        feat_dim: int,
        cam_size: int,
        patch_size: int = 1,
        hidden_dim: int = 128,
        layers: int = 4,
        top_k: int = 2,
        num_experts: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cam_size = cam_size

        def branch(num_channels):
            projection = nn.Conv2d(num_channels, hidden_dim, kernel_size=1)
            patch = PatchEmbed(
                img_size=cam_size,
                patch_size=patch_size,
                in_chans=hidden_dim,
                embed_dim=hidden_dim,
            )
            num_patches = (cam_size // patch_size) ** 2
            position = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
            return projection, patch, position

        self.proj_i, self.patch_projection_i, self.pos_embed_i = branch(num_i)
        self.proj_v, self.patch_projection_v, self.pos_embed_v = branch(num_v)
        self.proj_t, self.patch_projection_t, self.pos_embed_t = branch(num_t)
        self.proj_ivt, self.patch_projection_ivt, self.pos_embed_ivt = branch(feat_dim)

        self.encoder_layer_fusion = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4
        )
        self.transformer_fusion = nn.TransformerEncoder(
            self.encoder_layer_fusion, num_layers=layers
        )

        self.moe = MoE(
            input_size=hidden_dim,
            output_size=hidden_dim,
            num_experts=num_experts,
            hidden_size=hidden_dim,
            noisy_gating=True,
            k=top_k,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.conv_i = nn.Conv2d(hidden_dim, num_i, kernel_size=1)
        self.conv_v = nn.Conv2d(hidden_dim, num_v, kernel_size=1)
        self.conv_t = nn.Conv2d(hidden_dim, num_t, kernel_size=1)
        self.conv_ivt = nn.Conv2d(hidden_dim, feat_dim, kernel_size=1)

    @staticmethod
    def _embed(projection, patch_projection, position, x):
        x = projection(x)
        x = patch_projection(x)
        return x + position[:, : x.size(1), :]

    def forward(self, cam_i, cam_v, cam_t, cam_ivt):
        batch_size = cam_i.shape[0]

        cam_i_embed = self._embed(
            self.proj_i, self.patch_projection_i, self.pos_embed_i, cam_i
        )
        cam_v_embed = self._embed(
            self.proj_v, self.patch_projection_v, self.pos_embed_v, cam_v
        )
        cam_t_embed = self._embed(
            self.proj_t, self.patch_projection_t, self.pos_embed_t, cam_t
        )
        cam_ivt_embed = self._embed(
            self.proj_ivt, self.patch_projection_ivt, self.pos_embed_ivt, cam_ivt
        )

        combined = torch.cat(
            (cam_i_embed, cam_v_embed, cam_t_embed, cam_ivt_embed), dim=1
        )
        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        combined = torch.cat((cls_token, combined), dim=1)

        fused = self.transformer_fusion(combined.transpose(0, 1)).transpose(0, 1)
        flattened = fused.reshape(-1, fused.size(-1))
        moe_output, moe_loss = self.moe(flattened)
        moe_output = moe_output.reshape(batch_size, -1, fused.size(-1))

        cls_token_out = moe_output[:, 0, :]
        cam_tokens = moe_output[:, 1:, :]
        tokens_per_cam = self.cam_size * self.cam_size

        cams = []
        for index in range(4):
            start = index * tokens_per_cam
            end = (index + 1) * tokens_per_cam
            cam = cam_tokens[:, start:end, :].view(
                batch_size, self.cam_size, self.cam_size, self.hidden_dim
            )
            cam = cam.permute(0, 3, 1, 2)
            if index == 0:
                cam = self.conv_i(cam)
            elif index == 1:
                cam = self.conv_v(cam)
            elif index == 2:
                cam = self.conv_t(cam)
            else:
                cam = self.conv_ivt(cam)
            cams.append(cam)

        return cls_token_out, cams, moe_loss


class TemporalCAMoE(nn.Module):
    """Video-level CAMoE decoder.

    The temporal CAMs and the MS-TCN feature sequence are concatenated along
    the channel axis, projected to ``hidden_dim``, refined by a small
    transformer, routed through the MoE, and projected back to the feature
    dimension.  The returned sequence keeps the temporal length ``T``.
    """

    def __init__(
        self,
        num_i: int = 6,
        num_v: int = 10,
        num_t: int = 15,
        feat_dim: int = 512,
        hidden_dim: int = 1024,
        layers: int = 2,
        top_k: int = 2,
        num_experts: int = 4,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        self.encoder_layer_fusion = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4
        )
        self.transformer_fusion = nn.TransformerEncoder(
            self.encoder_layer_fusion, num_layers=layers
        )
        self.moe = MoE(
            input_size=hidden_dim,
            output_size=hidden_dim,
            num_experts=num_experts,
            hidden_size=hidden_dim,
            noisy_gating=True,
            k=top_k,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.linear_ivt = nn.Linear(feat_dim + num_i + num_v + num_t, hidden_dim)
        self.out_ivt = nn.Linear(hidden_dim, feat_dim)

    def forward(self, seq_i, seq_v, seq_t, seq_feat):
        batch_size = seq_i.shape[0]

        combined = torch.cat((seq_i, seq_v, seq_t, seq_feat), dim=1)
        combined = combined.permute(0, 2, 1)
        combined = self.linear_ivt(combined)

        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        combined = torch.cat((cls_token, combined), dim=1)

        fused = self.transformer_fusion(combined.transpose(0, 1)).transpose(0, 1)
        flattened = fused.reshape(-1, fused.size(-1))
        moe_output, moe_loss = self.moe(flattened)
        moe_output = moe_output.reshape(batch_size, -1, fused.size(-1))

        sequence = moe_output[:, 1:, :]
        sequence = self.out_ivt(sequence).permute(0, 2, 1)
        return moe_loss, sequence
