"""Video-level feature dataset used by the second stage."""

from __future__ import annotations

import os
import pickle

import numpy as np
from torch.utils.data import Dataset


def load_feature_file(path):
    with open(path, "rb") as handle:
        features = pickle.load(handle)
    if not isinstance(features, dict):
        raise TypeError("The feature file must contain a dict {video_id: array}")

    # Legacy feature files use two-digit keys ("79"); accept both forms.
    normalized = {}
    for key, value in features.items():
        video = key if str(key).startswith("VID") else f"VID{str(key).zfill(2)}"
        normalized[video] = value
    return normalized


class FeatureDataset(Dataset):
    """A whole video as one sample.

    Consecutive feature vectors that are exactly identical are removed,
    together with the corresponding labels, following the standard data
    pipeline.
    """

    def __init__(self, features, dataset_dir, video, dedup=True):
        self.video = video
        self.features = np.asarray(features, dtype=np.float32)

        triplet_file = os.path.join(dataset_dir, "triplet", f"{video}.txt")
        self.triplet_labels = np.loadtxt(triplet_file, dtype=int, delimiter=",")

        if len(self.features) != len(self.triplet_labels):
            raise ValueError(
                f"{video}: {len(self.features)} feature rows but "
                f"{len(self.triplet_labels)} label rows"
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

        keep = np.arange(len(self.features))
        if dedup and len(self.features) > 1:
            diff = self.features[1:] - self.features[:-1]
            identical = np.where(np.sum(diff, axis=-1) == 0)[0]
            removed = np.unique(np.concatenate((identical, identical + 1)))
            keep = np.array([i for i in range(len(self.features)) if i not in removed])
            if len(keep) == 0:  # degenerate two-frame video with identical features
                keep = np.array([0])

        self.features = self.features[keep]
        self.triplet_labels = self.triplet_labels[keep]
        self.tool_labels = self.tool_labels[keep]
        self.verb_labels = self.verb_labels[keep]
        self.target_labels = self.target_labels[keep]
        self.frame_ids = self.triplet_labels[:, 0].astype(int)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        labels = (
            self.tool_labels[:, 1:],
            self.verb_labels[:, 1:],
            self.target_labels[:, 1:],
            self.triplet_labels[:, 1:],
        )
        return self.features, labels, self.frame_ids


def build_feature_datasets(feature_dict, dataset_dir, records, dedup=True):
    datasets = []
    for video in records:
        if video not in feature_dict:
            raise KeyError(f"{video} is missing from the feature file")
        datasets.append(FeatureDataset(feature_dict[video], dataset_dir, video, dedup))
    return datasets
