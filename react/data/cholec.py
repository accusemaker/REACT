"""Frame-level CholecT45/50 dataset used by the first stage.

The dataset layout and label handling are adapted from Rendezvous
(https://github.com/CAMMA-public/rendezvous, Apache-2.0).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(img_size: int, augmentation_list):
    ops = [transforms.Resize((img_size, img_size))]
    for name in augmentation_list:
        if name == "original":
            continue
        if name == "vflip":
            ops.append(transforms.RandomVerticalFlip(0.4))
        elif name == "hflip":
            ops.append(transforms.RandomHorizontalFlip(0.4))
        elif name == "contrast":
            ops.append(transforms.RandomAutocontrast(p=0.5))
        elif name == "rot90":
            ops.append(transforms.RandomRotation(90, expand=True))
        elif name == "colorjitter":
            ops.append(
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.2, saturation=0.0, hue=0.0
                )
            )
        elif name == "sharpness":
            ops.append(transforms.RandomAdjustSharpness(sharpness_factor=1.6, p=0.5))
        else:
            raise ValueError(f"Unknown augmentation {name!r}")
    ops.extend(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


def build_eval_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class FrameDataset(Dataset):
    """One video; returns a 4-channel image and its multi-label targets.

    The fourth channel is a normalized timestamp in ``[0, 1]``:
    ``(frame_id + 1) / (num_frames + 10)``.
    """

    def __init__(self, dataset_dir, video, transform):
        self.video = video
        self.transform = transform
        self.img_dir = os.path.join(dataset_dir, "data", video)

        self.triplet_labels = np.loadtxt(
            os.path.join(dataset_dir, "triplet", f"{video}.txt"),
            dtype=int,
            delimiter=",",
        )
        self.tool_labels = np.loadtxt(
            os.path.join(dataset_dir, "instrument", f"{video}.txt"),
            dtype=int,
            delimiter=",",
        )
        self.verb_labels = np.loadtxt(
            os.path.join(dataset_dir, "verb", f"{video}.txt"),
            dtype=int,
            delimiter=",",
        )
        self.target_labels = np.loadtxt(
            os.path.join(dataset_dir, "target", f"{video}.txt"),
            dtype=int,
            delimiter=",",
        )

        lengths = {
            len(self.triplet_labels),
            len(self.tool_labels),
            len(self.verb_labels),
            len(self.target_labels),
        }
        if len(lengths) != 1:
            raise ValueError(f"Label files for {video} have different lengths: {lengths}")

    def __len__(self):
        return len(self.triplet_labels)

    def __getitem__(self, index):
        frame_id = int(self.triplet_labels[index, 0])
        image_path = os.path.join(self.img_dir, f"{frame_id:06d}.png")
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        time_stamp = (frame_id + 1) / (len(self.triplet_labels) + 10)
        time_channel = torch.full(
            (1, image.shape[1], image.shape[2]), time_stamp, dtype=image.dtype
        )
        image = torch.cat((image, time_channel), dim=0)

        labels = (
            self.tool_labels[index, 1:],
            self.verb_labels[index, 1:],
            self.target_labels[index, 1:],
            self.triplet_labels[index, 1:],
        )
        return image, labels, image_path


def build_frame_datasets(dataset_dir, records, transform):
    return [FrameDataset(dataset_dir, video, transform) for video in records]


def check_dataset_layout(dataset_dir):
    dataset_dir = Path(dataset_dir)
    required = ["data", "triplet", "instrument", "verb", "target"]
    missing = [name for name in required if not (dataset_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{dataset_dir} is missing the following directories: {missing}"
        )
