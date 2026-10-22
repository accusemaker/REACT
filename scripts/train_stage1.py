#!/usr/bin/env python3
"""Train the frame-level REACT model (Stage 1)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

from _common import evaluate_stage1_videos, make_loader

from react.data.cholec import (
    build_eval_transform,
    build_frame_datasets,
    build_train_transform,
    check_dataset_layout,
)
from react.data.splits import get_records
from react.engine import train_stage1_epoch
from react.losses import build_asl
from react.metrics import VideoMetrics, log_test_report
from react.models import VisionMoE
from react.utils import (
    Logger,
    count_parameters,
    resolve_device,
    save_checkpoint,
    set_seed,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="CholecT45/50 root directory")
    parser.add_argument("--dataset_variant", default="cholect45-crossval")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--output_dir", default="outputs")

    parser.add_argument("--backbone", default="swin_B_384_22k")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--feature_dim", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--drop_rate", type=float, default=0.0)
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--fix_backbone", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=58)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--power", type=float, default=0.1)
    parser.add_argument("--decay_rate", type=float, default=0.99)
    parser.add_argument("--train_fraction", type=float, default=0.25)
    parser.add_argument("--moe_loss_coef", type=float, default=0.0)

    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--select_split", choices=["val", "test"], default="val")
    parser.add_argument("--test_at_end", action="store_true")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20000912)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--augmentation_list",
        nargs="+",
        default=["original", "vflip", "hflip", "contrast", "rot90"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.position_embedding = "sine"

    set_seed(args.seed)
    device = resolve_device(args.device)
    check_dataset_layout(args.data_dir)

    records = get_records(args.dataset_variant, args.fold)
    train_transform = build_train_transform(args.img_size, args.augmentation_list)
    eval_transform = build_eval_transform(args.img_size)

    train_dataset = ConcatDataset(
        build_frame_datasets(args.data_dir, records["train"], train_transform)
    )
    val_datasets = build_frame_datasets(args.data_dir, records["val"], eval_transform)
    test_datasets = build_frame_datasets(args.data_dir, records["test"], eval_transform)

    train_loader = make_loader(
        train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loaders = [
        make_loader(dataset, args.batch_size, num_workers=args.num_workers)
        for dataset in val_datasets
    ]
    test_loaders = [
        make_loader(dataset, args.batch_size, num_workers=args.num_workers)
        for dataset in test_datasets
    ]

    model = VisionMoE(args).to(device)
    criterion = build_asl()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=args.power, total_iters=args.warmup_epochs
    )
    decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, decay], milestones=[args.warmup_epochs + 1]
    )

    run_dir = Path(args.output_dir) / f"fold{args.fold}" / "stage1"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(run_dir / "train.log")
    write_json(run_dir / "args.json", vars(args))

    bank = VideoMetrics().ivt.bank
    activation = torch.nn.Sigmoid()

    logger(
        f"Stage 1 | fold {args.fold} | train={len(records['train'])} "
        f"val={len(records['val'])} test={len(records['test'])} | "
        f"params={count_parameters(model) / 1e6:.1f}M"
    )

    best_score = -1.0
    best_path = run_dir / "best.pth"
    last_path = run_dir / "last.pth"

    for epoch in range(args.epochs):
        stats = train_stage1_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            device,
            bank,
            epoch,
            logger,
            limit_batches=args.train_fraction,
            moe_loss_coef=args.moe_loss_coef,
        )
        logger(
            f"epoch {epoch:03d} train loss={stats.get('loss', 0):.4f} "
            f"ivt={stats.get('loss_ivt', 0):.4f}"
        )

        is_last = epoch == args.epochs - 1
        if (epoch + 1) % args.val_interval == 0 or is_last:
            val_summary = evaluate_stage1_videos(
                model, val_loaders, activation, device, bank
            )
            logger(
                f"epoch {epoch:03d} val mAP_ivt={val_summary['mAP_ivt']:.4f} "
                f"(I={val_summary['mAP_i']:.4f} V={val_summary['mAP_v']:.4f} "
                f"T={val_summary['mAP_t']:.4f})"
            )

            if args.select_split == "test":
                select_summary = evaluate_stage1_videos(
                    model, test_loaders, activation, device, bank
                )
                logger(
                    f"epoch {epoch:03d} test mAP_ivt={select_summary['mAP_ivt']:.4f}"
                )
            else:
                select_summary = val_summary

            save_checkpoint(last_path, model, epoch, val_summary, args)
            if select_summary["mAP_ivt"] > best_score:
                best_score = select_summary["mAP_ivt"]
                save_checkpoint(best_path, model, epoch, select_summary, args)
                logger(
                    f"epoch {epoch:03d} saved best checkpoint "
                    f"(mAP_ivt={best_score:.4f})"
                )

    if args.test_at_end:
        from react.utils import load_checkpoint

        checkpoint, missing, unexpected = load_checkpoint(best_path, model)
        logger(
            f"loaded best checkpoint from epoch {checkpoint.get('epoch')} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        test_metrics = evaluate_stage1_videos(
            model, test_loaders, activation, device, bank, return_metrics=True
        )
        test_summary = test_metrics.summary()
        log_test_report(test_metrics, logger, tag="best")
        write_json(run_dir / "test_metrics.json", test_summary)


if __name__ == "__main__":
    main()
