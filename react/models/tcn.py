"""Temporal convolutional backbone used by the video-level REACT model.

Adapted from the Rendezvous implementation
(https://github.com/CAMMA-public/rendezvous, Apache-2.0).
"""

from __future__ import annotations

import copy

import torch.nn.functional as F
from torch import nn


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class BaseCausalTCN(nn.Module):
    """Dilated residual TCN with an optional input mask."""

    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, kernel_size=1)
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps))
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, kernel_size=1)
        self.channel_dropout = nn.Dropout2d()

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        x = self.channel_dropout(x.unsqueeze(3)).squeeze(3)
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        return out, self.conv_out(out)


class Refinement(nn.Module):
    """One refinement stage of the MS-TCN."""

    def __init__(self, num_layers, num_f_maps, num_classes):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps))
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, kernel_size=1)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out, self.conv_out(out)


class FPN(nn.Module):
    """Fuse the four temporal scales produced by the MS-TCN."""

    def __init__(self, num_f_maps):
        super().__init__()
        self.latlayer1 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1)
        self.latlayer2 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1)
        self.latlayer3 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1)

    @staticmethod
    def _upsample_add(x, y):
        return F.interpolate(x, size=y.size(2), mode="linear") + y

    def forward(self, out_list):
        p4 = out_list[3]
        c3 = out_list[2]
        c2 = out_list[1]
        c1 = out_list[0]
        # latlayer1 is reused for all three levels; latlayer2/latlayer3 are
        # kept for checkpoint compatibility.
        p3 = self._upsample_add(p4, self.latlayer1(c3))
        p2 = self._upsample_add(p3, self.latlayer1(c2))
        p1 = self._upsample_add(p2, self.latlayer1(c1))
        return [p1, p2, p3, p4]
