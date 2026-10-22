#!/usr/bin/env bash
# Quick sanity check: 1 epoch on ~1% of the batches + feature export of two videos.
# This is NOT a full training run; it only verifies the environment and scripts.
set -euo pipefail

DATA_DIR="${DATA_DIR:?Set DATA_DIR to the CholecT45/50 root directory}"
DATASET_VARIANT="${DATASET_VARIANT:-cholect45-crossval}"
OUTPUT_DIR="${OUTPUT_DIR:-work/smoke}"
FOLD="${FOLD:-1}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"

"${PYTHON}" scripts/train_stage1.py \
    --data_dir "${DATA_DIR}" \
    --dataset_variant "${DATASET_VARIANT}" \
    --fold "${FOLD}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "cuda:${GPU}" \
    --epochs 1 \
    --train_fraction 0.01 \
    --val_interval 1 \
    --num_workers 4 \
    --test_at_end

"${PYTHON}" scripts/export_features.py \
    --data_dir "${DATA_DIR}" \
    --dataset_variant "${DATASET_VARIANT}" \
    --fold "${FOLD}" \
    --checkpoint "${OUTPUT_DIR}/fold${FOLD}/stage1/best.pth" \
    --output "${OUTPUT_DIR}/features_2vids.pkl" \
    --videos VID31 VID79 \
    --evaluate \
    --batch_size 16 \
    --num_workers 4 \
    --device "cuda:${GPU}"

echo
echo "Smoke test done."
echo "Stage 1 log : ${OUTPUT_DIR}/fold${FOLD}/stage1/train.log"
echo "Features    : ${OUTPUT_DIR}/features_2vids.pkl"
