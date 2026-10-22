#!/usr/bin/env bash
# Train and evaluate REACT on all five CholecT45/50 cross-validation folds.
set -euo pipefail

DATA_DIR="${DATA_DIR:?Set DATA_DIR to the CholecT45/50 root directory}"
DATASET_VARIANT="${DATASET_VARIANT:-cholect45-crossval}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
export OUTPUT_DIR
GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"
PRETRAINED_PATH="${PRETRAINED_PATH:-}"

for K in 1 2 3 4 5; do
    echo "============================================================"
    echo "Fold ${K}"
    echo "============================================================"

    PRETRAINED_ARGS=()
    if [ -n "${PRETRAINED_PATH}" ]; then
        PRETRAINED_ARGS=(--pretrained_path "${PRETRAINED_PATH}")
    fi

    "${PYTHON}" scripts/train_stage1.py \
        --data_dir "${DATA_DIR}" \
        --dataset_variant "${DATASET_VARIANT}" \
        --fold "${K}" \
        --output_dir "${OUTPUT_DIR}" \
        --device "cuda:${GPU}" \
        --test_at_end \
        "${PRETRAINED_ARGS[@]}"

    "${PYTHON}" scripts/export_features.py \
        --data_dir "${DATA_DIR}" \
        --dataset_variant "${DATASET_VARIANT}" \
        --fold "${K}" \
        --checkpoint "${OUTPUT_DIR}/fold${K}/stage1/best.pth" \
        --output "${OUTPUT_DIR}/fold${K}/features.pkl" \
        --device "cuda:${GPU}" \
        --evaluate

    "${PYTHON}" scripts/train_stage2.py \
        --data_dir "${DATA_DIR}" \
        --dataset_variant "${DATASET_VARIANT}" \
        --fold "${K}" \
        --feature_file "${OUTPUT_DIR}/fold${K}/features.pkl" \
        --output_dir "${OUTPUT_DIR}" \
        --device "cuda:${GPU}" \
        --test_at_end
done

"${PYTHON}" - <<'PY'
import json
import statistics
from pathlib import Path

output_dir = Path(__import__("os").environ.get("OUTPUT_DIR", "outputs"))
keys = ["mAP_i", "mAP_v", "mAP_t", "mAP_iv", "mAP_it", "mAP_ivt"]
values = {key: [] for key in keys}

for fold in range(1, 6):
    path = output_dir / f"fold{fold}" / "stage2" / "test_metrics.json"
    if not path.exists():
        print(f"missing {path}")
        continue
    metrics = json.loads(path.read_text())
    for key in keys:
        values[key].append(metrics[key])

print("\nStage 2 test results (mean ± std over folds)")
for key in keys:
    scores = values[key]
    if not scores:
        continue
    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    print(f"{key:8s}: {mean:.4f} ± {std:.4f}")
PY
