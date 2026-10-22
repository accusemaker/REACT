"""Video-level REACT model: MS-TCN + temporal CAMoE."""

from __future__ import annotations

import copy

import torch
from torch import nn

from react.models.camoe import TemporalCAMoE
from react.models.tcn import BaseCausalTCN, FPN, Refinement


class VideoNas(nn.Module):
    """Temporal model that predicts per-frame I/V/T and IVT labels.

    Args:
        args: configuration namespace.  ``top_k``, ``num_experts``,
            ``moe_hidden_dim`` and ``mask`` are read from it.
        num_layers_PG: layers in the prediction-generation TCN.
        num_layers_R: layers in each refinement stage.
        num_R: number of refinement stages.
        num_f_maps: temporal feature dimension (512 in the paper).
        dim: input feature dimension (1024 for the released model).
        num_classes: number of triplet classes (100).
    """

    def __init__(
        self,
        args,
        num_layers_PG,
        num_layers_R,
        num_R,
        num_f_maps,
        dim,
        num_classes,
        num_i=6,
        num_v=10,
        num_t=15,
    ):
        super().__init__()
        self.args = args
        self.num_f_maps = num_f_maps

        self.PG = BaseCausalTCN(num_layers_PG, num_f_maps, dim, num_classes)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, kernel_size=1)
        self.conv_out_i = nn.Conv1d(num_f_maps, num_i, kernel_size=1)
        self.conv_out_v = nn.Conv1d(num_f_maps, num_v, kernel_size=1)
        self.conv_out_t = nn.Conv1d(num_f_maps, num_t, kernel_size=1)

        self.Rs = nn.ModuleList(
            [
                copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes))
                for _ in range(num_R)
            ]
        )

        self.fpn = FPN(num_f_maps)
        self.moe = TemporalCAMoE(
            num_i=num_i,
            num_v=num_v,
            num_t=num_t,
            feat_dim=num_f_maps,
            hidden_dim=args.moe_hidden_dim,
            layers=2,
            top_k=args.top_k,
            num_experts=args.num_experts,
        )

    def _make_mask(self, x):
        num_patches = x.numel()
        num_mask = int(num_patches * 0.75)
        mask = torch.cat(
            [
                torch.zeros(num_patches - num_mask, device=x.device),
                torch.ones(num_mask, device=x.device),
            ]
        )
        mask = mask[torch.randperm(mask.numel(), device=x.device)]
        return mask.view(x.shape)

    def forward(self, x, apply_mask=False):
        # x: [B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1)

        if self.args.mask and apply_mask:
            f, _ = self.PG(x, mask=self._make_mask(x))
        else:
            f, _ = self.PG(x)

        f_list = [f]
        for refinement in self.Rs:
            f, _ = refinement(f)
            f_list.append(f)

        f_list = self.fpn(f_list)

        out_list = []
        out_list_i = []
        out_list_v = []
        out_list_t = []
        moe_loss = None

        for f in f_list:
            cam_i = self.conv_out_i(f)
            cam_v = self.conv_out_v(f)
            cam_t = self.conv_out_t(f)
            out_list_i.append(cam_i)
            out_list_v.append(cam_v)
            out_list_t.append(cam_t)

            moe_loss, f_ivt = self.moe(cam_i, cam_v, cam_t, f)
            f_ivt = torch.nan_to_num(f_ivt, nan=0.0, posinf=1e4, neginf=-1e4)
            f_ivt = torch.clamp(f_ivt, min=-1e4, max=1e4)
            out_list.append(self.conv_out(f_ivt))

        return out_list, out_list_i, out_list_v, out_list_t, moe_loss
