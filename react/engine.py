"""Training and evaluation loops for both REACT stages."""

from __future__ import annotations

import torch

from react.losses import stage1_loss, stage2_loss


def train_stage1_epoch(
    loader,
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    bank,
    epoch,
    logger,
    limit_batches=1.0,
    moe_loss_coef=0.0,
    log_interval=50,
):
    model.train()
    running = {}
    count = 0
    for batch_index, (images, labels, _) in enumerate(loader):
        if limit_batches < 1.0 and batch_index > len(loader) * limit_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = [target.to(device, non_blocking=True) for target in labels]

        predictions, moe_loss = model(images)
        loss, parts = stage1_loss(predictions, labels, criterion, bank)
        if moe_loss_coef:
            loss = loss + moe_loss_coef * moe_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        count += 1
        for key, value in parts.items():
            running[key] = running.get(key, 0.0) + value

        if log_interval and batch_index % log_interval == 0:
            logger(
                f"epoch {epoch} batch {batch_index}/{len(loader)} "
                f"loss={parts['loss']:.4f} ivt={parts['loss_ivt']:.4f}"
            )

    if scheduler is not None:
        scheduler.step()

    return {key: value / max(count, 1) for key, value in running.items()}


@torch.no_grad()
def evaluate_stage1_video(loader, model, activation, metrics, device, bank, return_features=False):
    model.eval()
    metrics.start_video()
    features = []

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = [target.to(device, non_blocking=True) for target in labels]
        predictions, _ = model(images)
        metrics.update_stage1(predictions[0][1], labels, activation, bank)
        if return_features:
            features.append(predictions[0][0].cpu())

    metrics.end_video()
    if return_features:
        return torch.cat(features, dim=0)
    return None


@torch.no_grad()
def export_video_features(loader, model, device):
    model.eval()
    features = []
    for images, _, _ in loader:
        images = images.to(device, non_blocking=True)
        predictions, _ = model(images)
        features.append(predictions[0][0].cpu())
    return torch.cat(features, dim=0)


def train_stage2_epoch(
    loader,
    model,
    losses,
    optimizer,
    scheduler,
    device,
    epoch,
    logger,
    limit_batches=1.0,
    log_interval=1,
):
    model.train()
    running = {}
    count = 0

    for batch_index, (features, labels, _) in enumerate(loader):
        if limit_batches < 1.0 and batch_index > len(loader) * limit_batches:
            break

        features = features.to(device, non_blocking=True)
        labels = [target[0].to(device, non_blocking=True) for target in labels]

        out_list, out_i, out_v, out_t, _ = model(features, apply_mask=True)
        loss, parts = stage2_loss(
            out_list, (out_i, out_v, out_t), labels, losses
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        count += 1
        for key, value in parts.items():
            running[key] = running.get(key, 0.0) + value

        if log_interval and batch_index % log_interval == 0:
            logger(
                f"epoch {epoch} batch {batch_index}/{len(loader)} "
                f"loss={parts['loss']:.4f} ivt={parts['loss_ivt']:.4f}"
            )

    if scheduler is not None:
        scheduler.step()

    return {key: value / max(count, 1) for key, value in running.items()}


@torch.no_grad()
def evaluate_stage2_video(loader, model, activation, metrics, device):
    model.eval()
    metrics.start_video()

    for features, labels, _ in loader:
        features = features.to(device, non_blocking=True)
        labels = [target[0].to(device, non_blocking=True) for target in labels]
        out_list, out_i, out_v, out_t, _ = model(features, apply_mask=False)
        metrics.update_stage2(out_list, (out_i, out_v, out_t), labels, activation)

    metrics.end_video()
