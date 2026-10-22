"""Loss functions for the two REACT stages."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from react.losses_asl import AsymmetricLossOptimized


TOOL_WEIGHT = [0.93487068, 0.94234964, 0.93487068, 1.18448115, 1.02368339, 0.97974447]
VERB_WEIGHT = [
    0.60002400,
    0.60002400,
    0.60002400,
    0.61682467,
    0.67082683,
    0.80163207,
    0.70562823,
    2.11208448,
    2.69230769,
    0.60062402,
]
TARGET_WEIGHT = [
    0.49752894,
    0.52041527,
    0.49752894,
    0.51394739,
    2.71899565,
    1.75577963,
    0.58509403,
    1.25228034,
    0.49752894,
    2.42993134,
    0.49802647,
    0.87266576,
    1.36074165,
    0.50150917,
    0.49802647,
]


def build_asl():
    return AsymmetricLossOptimized(
        gamma_neg=2,
        gamma_pos=0,
        clip=0,
        disable_torch_grad_focal_loss=True,
        eps=1e-5,
    )


def build_stage2_losses(device):
    return {
        "i": nn.BCEWithLogitsLoss(pos_weight=torch.tensor(TOOL_WEIGHT, device=device)),
        "v": nn.BCEWithLogitsLoss(pos_weight=torch.tensor(VERB_WEIGHT, device=device)),
        "t": nn.BCEWithLogitsLoss(pos_weight=torch.tensor(TARGET_WEIGHT, device=device)),
        "ivt": nn.BCEWithLogitsLoss(),
    }


def stage1_loss(predictions, labels, criterion, bank):
    """Frame-level ASL objective.

    ``predictions`` is the list returned by :class:`VisionMoE`:
    ``[(pooled, ivt), (cam_i, i), (cam_v, v), (cam_t, t), (cam_ivt, ivt)]``.
    """
    y_i, y_v, y_t, y_ivt = labels

    logit_ivt = predictions[0][1]
    logit_i = predictions[1][1]
    logit_v = predictions[2][1]
    logit_t = predictions[3][1]

    loss_i_direct = criterion(logit_i, y_i).mean()
    loss_v_direct = criterion(logit_v, y_v).mean()
    loss_t_direct = criterion(logit_t, y_t).mean()
    loss_direct = loss_i_direct + loss_v_direct + loss_t_direct

    component_logits = []
    for component in range(3):
        per_class = []
        for class_id in range((6, 10, 15)[component]):
            indices = np.where(bank[:, component + 1] == class_id)[0]
            per_class.append(torch.max(logit_ivt[:, indices], dim=-1).values)
        component_logits.append(torch.stack(per_class).permute(1, 0))

    loss_i = criterion(component_logits[0], y_i).mean()
    loss_v = criterion(component_logits[1], y_v).mean()
    loss_t = criterion(component_logits[2], y_t).mean()
    loss_ivt = criterion(logit_ivt, y_ivt).mean()
    loss_components = loss_i + loss_v + loss_t + loss_ivt

    total = loss_direct + loss_components
    parts = {
        "loss_i": loss_i.item(),
        "loss_v": loss_v.item(),
        "loss_t": loss_t.item(),
        "loss_ivt": loss_ivt.item(),
        "loss_i_direct": loss_i_direct.item(),
        "loss_v_direct": loss_v_direct.item(),
        "loss_t_direct": loss_t_direct.item(),
        "loss": total.item(),
    }
    return total, parts


def _resize_predictions(predictions, labels):
    """Match each FPN scale's temporal length to the label sequence."""
    resized = []
    for prediction in predictions:
        out = F.interpolate(prediction, size=labels.size(0), mode="nearest")
        resized.append((out, labels))
    return resized


def stage2_loss(predictions, component_predictions, labels, losses):
    """Video-level objective: 0.1 * (I + V + T) + IVT."""
    y_i, y_v, y_t, y_ivt = labels

    loss_i = 0
    loss_v = 0
    loss_t = 0
    loss_ivt = 0

    for out, target in _resize_predictions(predictions, y_ivt):
        loss_ivt = loss_ivt + losses["ivt"](out[0].transpose(0, 1), target.float())

    for out, target in _resize_predictions(component_predictions[0], y_i):
        loss_i = loss_i + losses["i"](out[0].transpose(0, 1), target.float())

    for out, target in _resize_predictions(component_predictions[1], y_v):
        loss_v = loss_v + losses["v"](out[0].transpose(0, 1), target.float())

    for out, target in _resize_predictions(component_predictions[2], y_t):
        loss_t = loss_t + losses["t"](out[0].transpose(0, 1), target.float())

    total = 0.1 * (loss_i + loss_v + loss_t) + loss_ivt
    parts = {
        "loss_i": loss_i.item(),
        "loss_v": loss_v.item(),
        "loss_t": loss_t.item(),
        "loss_ivt": loss_ivt.item(),
        "loss": total.item(),
    }
    return total, parts
