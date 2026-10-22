#!/usr/bin/env bash
# Full single-fold pipeline:
#   Stage 1 training -> feature export -> Stage 2 training -> test evaluation.
#
# Example (run from the repository root):
#   DATA_DIR=/path/to/CholecT45 \
#   OUTPUT_DIR=/path/to/output/run_fold1 \
#   PRETRAINED_PATH=/path/to/swin_base_patch4_window12_384_22k.pth \
#   FOLD=1 GPU=0 bash scripts/run_one_fold.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:?Set DATA_DIR to the CholecT45/50 root directory}"
DATASET_VARIANT="${DATASET_VARIANT:-cholect45-crossval}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
FOLD="${FOLD:-1}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"
PRETRAINED_PATH="${PRETRAINED_PATH:-}"

PRETRAINED_ARGS=()
if [ -n "${PRETRAINED_PATH}" ]; then
    PRETRAINED_ARGS=(--pretrained_path "${PRETRAINED_PATH}")
fi

echo "============================================================"
echo "REACT single fold"
echo "  data   : ${DATA_DIR}"
echo "  output : ${OUTPUT_DIR}"
echo "  fold   : ${FOLD}"
echo "  gpu    : cuda:${GPU}"
echo "============================================================"

echo "[1/3] Stage 1 training"
"${PYTHON}" scripts/train_stage1.py \
    --data_dir "${DATA_DIR}" \
    --dataset_variant "${DATASET_VARIANT}" \
    --fold "${FOLD}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "cuda:${GPU}" \
    --test_at_end \
    "${PRETRAINED_ARGS[@]}"

echo "[2/3] Stage 1 inference / feature export"
"${PYTHON}" scripts/export_features.py \
    --data_dir "${DATA_DIR}" \
    --dataset_variant "${DATASET_VARIANT}" \
    --fold "${FOLD}" \
    --checkpoint "${OUTPUT_DIR}/fold${FOLD}/stage1/best.pth" \
    --output "${OUTPUT_DIR}/fold${FOLD}/features.pkl" \
    --device "cuda:${GPU}" \
    --evaluate

echo "[3/3] Stage 2 training and test evaluation"
"${PYTHON}" scripts/train_stage2.py \
    --data_dir "${DATA_DIR}" \
    --dataset_variant "${DATASET_VARIANT}" \
    --fold "${FOLD}" \
    --feature_file "${OUTPUT_DIR}/fold${FOLD}/features.pkl" \
    --output_dir "${OUTPUT_DIR}" \
    --device "cuda:${GPU}" \
    --test_at_end

echo
echo "Done."
echo "Stage 1 metrics : ${OUTPUT_DIR}/fold${FOLD}/stage1/test_metrics.json"
echo "Stage 2 metrics : ${OUTPUT_DIR}/fold${FOLD}/stage2/test_metrics.json"
