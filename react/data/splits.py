"""Official CholecT45 / CholecT50 cross-validation splits.

Only the five-fold cross-validation protocols used in the paper are kept.
Each fold lists the nine (CholecT45) or ten (CholecT50) held-out videos.
"""

from __future__ import annotations

CHOLECT45_CROSSVAL = {
    1: [79, 2, 51, 6, 25, 14, 66, 23, 50],
    2: [80, 32, 5, 15, 40, 47, 26, 48, 70],
    3: [31, 57, 36, 18, 52, 68, 10, 8, 73],
    4: [42, 29, 60, 27, 65, 75, 22, 49, 12],
    5: [78, 43, 62, 35, 74, 1, 56, 4, 13],
}

CHOLECT50_CROSSVAL = {
    1: [79, 2, 51, 6, 25, 14, 66, 23, 50, 111],
    2: [80, 32, 5, 15, 40, 47, 26, 48, 70, 96],
    3: [31, 57, 36, 18, 52, 68, 10, 8, 73, 103],
    4: [42, 29, 60, 27, 65, 75, 22, 49, 12, 110],
    5: [78, 43, 62, 35, 74, 1, 56, 4, 13, 92],
}

SPLITS = {
    "cholect45-crossval": CHOLECT45_CROSSVAL,
    "cholect50-crossval": CHOLECT50_CROSSVAL,
}


def _vid(video_id: int) -> str:
    return f"VID{str(video_id).zfill(2)}"


def get_records(variant: str, fold: int) -> dict[str, list[str]]:
    """Return train/val/test/all video names for one fold.

    The validation split uses the last five videos in the canonical fold
    order; the remaining videos form the training set.
    """
    if variant not in SPLITS:
        raise ValueError(f"Unknown dataset variant {variant!r}. Choose from {list(SPLITS)}")
    folds = SPLITS[variant]
    if fold not in folds:
        raise ValueError(f"fold must be one of {sorted(folds)}, got {fold}")

    test = [_vid(v) for v in folds[fold]]
    train_val = [
        _vid(v) for k in sorted(folds) if k != fold for v in folds[k]
    ]

    if "crossval" in variant:
        val = train_val[-5:]
        train = train_val[:-5]
    else:  # pragma: no cover - kept for future non-crossval variants
        val = []
        train = train_val

    return {
        "train": train,
        "val": val,
        "test": test,
        "all": test + train + val,
    }
