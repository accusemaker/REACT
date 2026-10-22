# Third-party notices

This repository contains code adapted from the following projects.  The
original copyright headers are retained in the relevant source files where
required.

## Rendezvous

- Repository: https://github.com/CAMMA-public/rendezvous
- Authors: C. I. Nwoye et al., CAMMA, University of Strasbourg
- License: Apache-2.0

The CholecT45/50 dataset handling, the MS-TCN/refinement/FPN temporal model,
and parts of the evaluation protocol are adapted from Rendezvous.  The
original headers are retained in the corresponding files.

## Swin Transformer

- Repository: https://github.com/microsoft/Swin-Transformer
- Authors: Ze Liu et al., Microsoft Research
- License: MIT

`react/models/swin_transformer.py` is adapted from the official Swin
Transformer implementation.  The original header is retained.

## DETR position encoding

- Repository: https://github.com/facebookresearch/detr
- License: Apache-2.0

`react/models/position_encoding.py` follows the sine position encoding used in
DETR.  The original header is retained.

## Asymmetric loss

- Repository: https://github.com/Alibaba-MIIL/ASL
- License: Apache-2.0

`react/losses_asl.py` is adapted from the asymmetric loss implementation.
The original header is retained.

## ivtmetrics

`ivtmetrics` is used as an external dependency for mAP computation and is not
vendored in this repository.

## Dataset

CholecT45 / CholecT50 are not distributed with this repository.  Please follow
the license and citation requirements of the dataset release.
