"""Evaluation utilities based on ivtmetrics."""

from __future__ import annotations

import numpy as np

# ivtmetrics still uses the removed NumPy aliases internally.
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

import torch
from ivtmetrics.recognition import Recognition


def _resolve_nan_fixed(self, classwise):
    """Treat undefined per-class AP values as NaN, as in the paper protocol."""
    equiv_nan = ["-0", "-0.", "-0.0", "-.0"]
    classwise = list(map(str, classwise))
    classwise = [np.nan if x in equiv_nan else x for x in classwise]
    classwise = [np.nan if x == "0.0" else x for x in classwise]
    return np.array(list(map(float, classwise)))


Recognition.resolve_nan = _resolve_nan_fixed


class VideoMetrics:
    def __init__(self):
        self.ivt = Recognition(100)
        self.i = Recognition(6)
        self.v = Recognition(10)
        self.t = Recognition(15)
        self.reset_global()

    def reset_global(self):
        self.ivt.reset_global()
        self.i.reset_global()
        self.v.reset_global()
        self.t.reset_global()

    def start_video(self):
        self.ivt.reset()
        self.i.reset()
        self.v.reset()
        self.t.reset()

    def end_video(self):
        self.ivt.video_end()
        self.i.video_end()
        self.v.video_end()
        self.t.video_end()

    def update_stage1(self, logits_ivt, labels, activation, bank):
        """Update metrics using the frame-level IVT logits."""
        y_i, y_v, y_t, y_ivt = labels
        component_logits = []
        for component in range(3):
            per_class = []
            for class_id in range((6, 10, 15)[component]):
                indices = np.where(bank[:, component + 1] == class_id)[0]
                per_class.append(torch.max(logits_ivt[:, indices], dim=-1).values)
            component_logits.append(torch.stack(per_class).permute(1, 0))

        probabilities = activation(logits_ivt).detach().cpu()
        self.i.update(y_i.float().detach().cpu(), activation(component_logits[0]).detach().cpu())
        self.v.update(y_v.float().detach().cpu(), activation(component_logits[1]).detach().cpu())
        self.t.update(y_t.float().detach().cpu(), activation(component_logits[2]).detach().cpu())
        self.ivt.update(y_ivt.float().detach().cpu(), probabilities)

    def update_stage2(self, predictions, component_predictions, labels, activation):
        """Update metrics using the video-level outputs."""
        y_i, y_v, y_t, y_ivt = labels

        activated = [
            activation(predictions[index][0].transpose(0, 1))
            for index in range(min(3, len(predictions)))
        ]
        probs_ivt = sum(activated) / len(activated)

        self.ivt.update(y_ivt.float().detach().cpu(), probs_ivt.detach().cpu())
        self.i.update(
            y_i.float().detach().cpu(),
            activation(component_predictions[0][0][0].transpose(0, 1)).detach().cpu(),
        )
        self.v.update(
            y_v.float().detach().cpu(),
            activation(component_predictions[1][0][0].transpose(0, 1)).detach().cpu(),
        )
        self.t.update(
            y_t.float().detach().cpu(),
            activation(component_predictions[2][0][0].transpose(0, 1)).detach().cpu(),
        )

    def compute(self, ignore_null=True):
        # ``ignore_null`` in ivtmetrics drops the final six entries of the
        # 100-class triplet vector.  It must not be applied to the separate
        # 6/10/15-class component recognizers.
        return {
            "i": self.i.compute_video_AP(ignore_null=False),
            "v": self.v.compute_video_AP(ignore_null=False),
            "t": self.t.compute_video_AP(ignore_null=False),
            "iv": self.ivt.compute_video_AP("iv", ignore_null=ignore_null),
            "it": self.ivt.compute_video_AP("it", ignore_null=ignore_null),
            "ivt": self.ivt.compute_video_AP("ivt", ignore_null=ignore_null),
        }

    def summary(self):
        metrics = self.compute()
        return {
            "mAP_i": metrics["i"]["mAP"],
            "mAP_v": metrics["v"]["mAP"],
            "mAP_t": metrics["t"]["mAP"],
            "mAP_iv": metrics["iv"]["mAP"],
            "mAP_it": metrics["it"]["mAP"],
            "mAP_ivt": metrics["ivt"]["mAP"],
        }


def log_test_report(metrics: VideoMetrics, logger, tag: str = "best"):
    """Print the per-category AP / mean AP / top-k report.

    The layout follows the project's standard test log.
    """
    mAP_i = metrics.i.compute_video_AP(ignore_null=False)
    mAP_v = metrics.v.compute_video_AP(ignore_null=False)
    mAP_t = metrics.t.compute_video_AP(ignore_null=False)
    mAP_iv = metrics.ivt.compute_video_AP("iv", ignore_null=True)
    mAP_it = metrics.ivt.compute_video_AP("it", ignore_null=True)
    mAP_ivt = metrics.ivt.compute_video_AP("ivt", ignore_null=True)

    top5 = [
        metrics.ivt.topK(5, "i"),
        metrics.ivt.topK(5, "v"),
        metrics.ivt.topK(5, "t"),
        metrics.ivt.topK(5, "iv"),
        metrics.ivt.topK(5, "it"),
        metrics.ivt.topK(5, "ivt"),
    ]
    top10 = [
        metrics.ivt.topK(10, "i"),
        metrics.ivt.topK(10, "v"),
        metrics.ivt.topK(10, "t"),
        metrics.ivt.topK(10, "iv"),
        metrics.ivt.topK(10, "it"),
        metrics.ivt.topK(10, "ivt"),
    ]
    top20 = [
        metrics.ivt.topK(20, "i"),
        metrics.ivt.topK(20, "v"),
        metrics.ivt.topK(20, "t"),
        metrics.ivt.topK(20, "iv"),
        metrics.ivt.topK(20, "it"),
        metrics.ivt.topK(20, "ivt"),
    ]

    logger(f"======== {tag} ==============")
    logger("-" * 50)
    logger("Test Results")
    logger("Per-category AP: ")
    logger(f"I   : {mAP_i['AP']}")
    logger(f"V   : {mAP_v['AP']}")
    logger(f"T   : {mAP_t['AP']}")
    logger(f"IV  : {mAP_iv['AP']}")
    logger(f"IT  : {mAP_it['AP']}")
    logger(f"IVT : {mAP_ivt['AP']}")
    logger("-" * 50)

    logger("Mean AP:  I  |  V  |  T  |  IV  |  IT  |  IVT ")
    logger(
        ":::::: : "
        f"{mAP_i['mAP']:.4f} | {mAP_v['mAP']:.4f} | {mAP_t['mAP']:.4f} | "
        f"{mAP_iv['mAP']:.4f} | {mAP_it['mAP']:.4f} | {mAP_ivt['mAP']:.4f} "
    )

    for name, values in (("top 5", top5), ("top 10", top10), ("top 20", top20)):
        logger(f"{name}:  I  |  V  |  T  |  IV  |  IT  |  IVT ")
        logger(
            ":::::: : "
            + " | ".join(f"{value:.4f}" for value in values)
            + " "
        )
    logger("=" * 50)
