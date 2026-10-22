#!/usr/bin/env python3
"""Train and evaluate the video-level REACT model (Stage 2)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

from _common import evaluate_stage2_videos, make_loader

from react.data.features import build_feature_datasets, load_feature_file
from react.data.splits import get_records
from react.engine import train_stage2_epoch
from react.losses import build_stage2_losses
from react.metrics import log_test_report
from react.models import VideoNas
from react.utils import (
    Logger,
    count_parameters,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    set_seed,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--dataset_variant", default="cholect45-crossval")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--feature_file", required=True)
    parser.add_argument("--output_dir", default="outputs")

    parser.add_argument("--input_dim", type=int, default=1024)
    parser.add_argument("--num_f_maps", type=int, default=512)
    parser.add_argument("--num_layers_PG", type=int, default=11)
    parser.add_argument("--num_layers_R", type=int, default=10)
    parser.add_argument("--num_R", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--moe_hidden_dim", type=int, default=1024)
    parser.add_argument("--mask", action="store_true", default=True)
    parser.add_argument("--no_mask", dest="mask", action="store_false")

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=200)
    parser.add_argument("--power", type=float, default=0.1)
    parser.add_argument("--decay_rate", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--train_fraction", type=float, default=1.0)

    parser.add_argument("--val_interval", type=int, default=50)
    parser.add_argument("--select_split", choices=["val", "test"], default="val")
    parser.add_argument("--test_at_end", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_split", choices=["val", "test"], default="test")
    parser.add_argument("--checkpoint", default=None)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20000912)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    records = get_records(args.dataset_variant, args.fold)
    feature_dict = load_feature_file(args.feature_file)

    train_datasets = build_feature_datasets(
        feature_dict, args.data_dir, records["train"]
    )
    val_datasets = build_feature_datasets(feature_dict, args.data_dir, records["val"])
    test_datasets = build_feature_datasets(feature_dict, args.data_dir, records["test"])

    train_loader = make_loader(
        ConcatDataset(train_datasets),
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loaders = [
        make_loader(dataset, batch_size=1, num_workers=args.num_workers)
        for dataset in val_datasets
    ]
    test_loaders = [
        make_loader(dataset, batch_size=1, num_workers=args.num_workers)
        for dataset in test_datasets
    ]

    model = VideoNas(
        args,
        args.num_layers_PG,
        args.num_layers_R,
        args.num_R,
        args.num_f_maps,
        args.input_dim,
        100,
    ).to(device)

    run_dir = Path(args.output_dir) / f"fold{args.fold}" / "stage2"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(run_dir / "train.log")
    write_json(run_dir / "args.json", vars(args))
    logger(
        f"Stage 2 | fold {args.fold} | train={len(records['train'])} "
        f"val={len(records['val'])} test={len(records['test'])} | "
        f"params={count_parameters(model) / 1e6:.1f}M"
    )

    activation = torch.nn.Sigmoid()

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval_only requires --checkpoint")
        checkpoint, missing, unexpected = load_checkpoint(args.checkpoint, model)
        logger(
            f"Loaded {args.checkpoint} (epoch={checkpoint.get('epoch')}, "
            f"missing={len(missing)}, unexpected={len(unexpected)})"
        )
        loaders = test_loaders if args.eval_split == "test" else val_loaders
        eval_metrics = evaluate_stage2_videos(
            model, loaders, activation, device, return_metrics=True
        )
        summary = eval_metrics.summary()
        tag = "latest" if "latest" in str(args.checkpoint) else "best"
        log_test_report(eval_metrics, logger, tag=tag)
        write_json(run_dir / f"{args.eval_split}_metrics.json", summary)
        return

    losses = build_stage2_losses(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=args.power / 2, total_iters=args.warmup_epochs
    )
    decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, decay], milestones=[args.warmup_epochs + 1]
    )

    best_score = -1.0
    best_path = run_dir / "best.pth"
    last_path = run_dir / "last.pth"

    for epoch in range(args.epochs):
        stats = train_stage2_epoch(
            train_loader,
            model,
            losses,
            optimizer,
            scheduler,
            device,
            epoch,
            logger,
            limit_batches=args.train_fraction,
        )
        logger(
            f"epoch {epoch:03d} train loss={stats.get('loss', 0):.4f} "
            f"ivt={stats.get('loss_ivt', 0):.4f}"
        )

        is_last = epoch == args.epochs - 1
        if (epoch + 1) % args.val_interval == 0 or is_last:
            val_summary = evaluate_stage2_videos(
                model, val_loaders, activation, device
            )
            logger(
                f"epoch {epoch:03d} val mAP_ivt={val_summary['mAP_ivt']:.4f} "
                f"(I={val_summary['mAP_i']:.4f} V={val_summary['mAP_v']:.4f} "
                f"T={val_summary['mAP_t']:.4f})"
            )

            if args.select_split == "test":
                select_summary = evaluate_stage2_videos(
                    model, test_loaders, activation, device
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
        checkpoint, missing, unexpected = load_checkpoint(best_path, model)
        logger(
            f"Loaded best checkpoint from epoch {checkpoint.get('epoch')} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        test_metrics = evaluate_stage2_videos(
            model, test_loaders, activation, device, return_metrics=True
        )
        test_summary = test_metrics.summary()
        log_test_report(test_metrics, logger, tag="best")
        write_json(run_dir / "test_metrics.json", test_summary)


if __name__ == "__main__":
    main()
