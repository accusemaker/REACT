#!/usr/bin/env python3
"""Run Stage 1 inference and export per-video frame features for Stage 2."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from _common import make_loader

from react.data.cholec import build_eval_transform, build_frame_datasets, check_dataset_layout
from react.data.splits import get_records
from react.engine import evaluate_stage1_video, export_video_features
from react.metrics import VideoMetrics, log_test_report
from react.models import VisionMoE
from react.utils import Logger, load_checkpoint, resolve_device, set_seed, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--dataset_variant", default="cholect45-crossval")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="destination .pkl file")
    parser.add_argument("--evaluate", action="store_true", help="also compute Stage 1 mAP")

    parser.add_argument("--backbone", default="swin_B_384_22k")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--feature_dim", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--drop_rate", type=float, default=0.0)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20000912)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--videos",
        nargs="+",
        default=None,
        help="optional subset of video names (smoke tests); default: all videos",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.position_embedding = "sine"
    args.pretrained_path = None
    args.fix_backbone = False

    set_seed(args.seed)
    device = resolve_device(args.device)
    check_dataset_layout(args.data_dir)

    model = VisionMoE(args).to(device)
    checkpoint, missing, unexpected = load_checkpoint(args.checkpoint, model)
    print(
        f"Loaded {args.checkpoint} (epoch={checkpoint.get('epoch')}, "
        f"missing={len(missing)}, unexpected={len(unexpected)})"
    )
    model.eval()

    records = get_records(args.dataset_variant, args.fold)
    transform = build_eval_transform(args.img_size)
    videos = args.videos or records["all"]
    datasets = build_frame_datasets(args.data_dir, videos, transform)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features_by_video = {}
    metrics = VideoMetrics()
    metrics.reset_global()
    activation = torch.nn.Sigmoid()
    bank = metrics.ivt.bank

    with torch.no_grad():
        for video, dataset in zip(videos, datasets):
            loader = make_loader(
                dataset, args.batch_size, num_workers=args.num_workers
            )
            if args.evaluate:
                features = evaluate_stage1_video(
                    loader,
                    model,
                    activation,
                    metrics,
                    device,
                    bank,
                    return_features=True,
                )
            else:
                features = export_video_features(loader, model, device)
            features_by_video[video] = features.numpy()
            print(f"{video}: {tuple(features_by_video[video].shape)}")

    with output_path.open("wb") as handle:
        pickle.dump(features_by_video, handle)
    print(f"Wrote {len(features_by_video)} videos to {output_path}")

    if args.evaluate:
        summary = metrics.summary()
        write_json(output_path.with_suffix(".metrics.json"), summary)
        report_logger = Logger(output_path.with_suffix(".log"))
        log_test_report(metrics, report_logger, tag="best")


if __name__ == "__main__":
    main()
