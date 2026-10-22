# REACT

**REACT: Relation-Aware Class Activation-Guided Mixture-of-Experts for Spatio-Temporal Triplet Recognition**

[Paper (IEEE TMI)](https://doi.org/10.1109/TMI.2026.3716009)

This repository is the **core code release** accompanying the paper. It
contains the complete two-stage training pipeline, frame-level feature
extraction, and inference/evaluation code for spatio-temporal surgical action
triplet recognition.

## Method

REACT is a relation-aware, class activation-guided Mixture-of-Experts
framework that jointly models spatial details and temporal dynamics while
capturing inter-component relations.
Class Activation Maps (CAMs) serve as weak semantic priors to guide expert
routing, and Feature Modulation Experts (FMEs) align heterogeneous instrument,
verb, and target features into a shared semantic space.

- **Class Activation-guided Mixture-of-Experts (CAMoE).** CAMs guide dynamic
  expert routing through noisy top-k gating, enabling component-specialized
  processing for structurally heterogeneous triplet elements.
- **Feature Modulation Experts (FMEs).** Each expert applies a lightweight
  FiLM-style affine transformation `K(Z) ⊙ Z + B(Z)` with per-expert
  modulation parameters.
- **Hierarchical frame–video modeling.** The frame-level stage extracts CAMoE
  representations from temporally encoded frames; the video-level stage
  refines them with an MS-TCN and video-level CAMoE for temporal relation
  modeling.
- **Two-stage training.** The frame-level stage uses asymmetric loss (ASL);
  the video-level stage uses binary cross-entropy with the spatial backbone
  frozen.

The default configuration is CholecT45 cross-validation, `top_k=2`, and four
experts in each stage.

## Repository

```text
react/
├── react/          # data, models, losses, metrics, engine
├── scripts/        # train_stage1.py, export_features.py, train_stage2.py
├── tests/          # CPU shape tests
└── tools/          # legacy checkpoint converter
```

## Installation

```bash
pip install -r requirements.txt
# optional
pip install -e .
```

Tested with Python 3.10, PyTorch 2.0.1 and CUDA 11.8. Newer versions should
also work.

## Data

Download CholecT45/CholecT50 from the official release and arrange it as:

```text
CholecT45/
├── data/VID01/000000.png ...
├── triplet/VID01.txt
├── instrument/VID01.txt
├── verb/VID01.txt
└── target/VID01.txt
```

The default protocol is CholecT45 (`--dataset_variant cholect45-crossval`).
If the CholecT45 files live inside a directory named `CholecT50`, pass that
directory as `DATA_DIR`; the split still selects the 45 CholecT45 videos.

## Training

Quick check (one epoch on 1% of the batches, two-video feature export):

```bash
DATA_DIR=/path/to/CholecT45 OUTPUT_DIR=work/smoke bash scripts/run_smoke.sh
```

Full single fold:

```bash
DATA_DIR=/path/to/CholecT45 \
OUTPUT_DIR=work/run_fold1 \
PRETRAINED_PATH=/path/to/swin_base_patch4_window12_384_22k.pth \
FOLD=1 GPU=0 \
bash scripts/run_one_fold.sh
```

All five folds:

```bash
DATA_DIR=/path/to/CholecT45 \
OUTPUT_DIR=work/run_5fold \
PRETRAINED_PATH=/path/to/swin_base_patch4_window12_384_22k.pth \
GPU=0 \
bash scripts/run_5fold.sh
```

## Inference / evaluation

Frame-level inference and feature export:

```bash
python scripts/export_features.py \
  --data_dir /path/to/CholecT45 \
  --fold 1 \
  --checkpoint outputs/fold1/stage1/best.pth \
  --output outputs/fold1/features.pkl \
  --evaluate
```

Video-level evaluation (no training):

```bash
python scripts/train_stage2.py \
  --data_dir /path/to/CholecT45 \
  --fold 1 \
  --feature_file outputs/fold1/features.pkl \
  --checkpoint outputs/fold1/stage2/best.pth \
  --eval_only --eval_split test \
  --output_dir outputs
```

The evaluation writes a per-category AP report (mean AP and top-5/10/20) to the log.

## Checkpoints

Training checkpoints contain `{"model": state_dict, "epoch": ..., "metrics": ...}`.
Existing checkpoints from earlier versions can be converted without modifying the input checkpoints:

```bash
python tools/convert_legacy_checkpoint.py \
  --stage 1 --input /path/to/legacy_stage1.pth \
  --output outputs/legacy_stage1.pth

python tools/convert_legacy_checkpoint.py \
  --stage 2 --input /path/to/legacy_stage2.pth \
  --output outputs/legacy_stage2.pth
```

## Tests

```bash
python tests/test_shapes.py
```

## Citation

```bibtex
@ARTICLE{11618546,
  author={Qu, Bohan and Xiao, Yuhao and Liu, Wenxuan and Zheng, Wenbo and Wang, Zheng and Zhong, Xian and Sheng, Bin},
  journal={IEEE Transactions on Medical Imaging},
  title={REACT: Relation-Aware Class Activation-Guided Mixture-of-Experts for Spatio-Temporal Triplet Recognition},
  year={2026},
  volume={45},
  number={10},
  pages={5138-5152},
  keywords={Modeling;Videos;Cams;Computer aided manufacturing;Instruments;Labeling;Routing;Training;Visualization;Modulation;Relation-aware modeling;mixture-of-experts;spatio-temporal triplet recognition;surgical video understanding;medical image analysis},
  doi={10.1109/TMI.2026.3716009}}
```

## Acknowledgements

This code builds on [Rendezvous](https://github.com/CAMMA-public/rendezvous),
[Swin Transformer](https://github.com/microsoft/Swin-Transformer), and
[ASL](https://github.com/Alibaba-MIIL/ASL). See `THIRD_PARTY_NOTICES.md`.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
