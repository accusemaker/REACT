#!/usr/bin/env python3
"""Convert earlier research checkpoints to the cleaned REACT layout.

Stage 1 renames the MoCo wrapper (``encoder_q.0`` -> ``encoder.backbone``) and
the decoder (``moe`` -> ``moe_decoder``), and drops the unused momentum /
contrastive / prototype / head-tail parameters.  Stage 2 only drops the small
set of unused decoder heads.  Nothing in the input checkpoint is modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def convert_stage1(state):
    converted = {}
    for key, value in state.items():
        if key == "bank":
            continue
        if key.startswith(("conv1x1.", "cam_disen.", "encoder_k.", "queue")):
            continue
        if any(token in key for token in ("prototpye", "prototype", "variance")):
            continue
        if key.startswith("moe.moe.experts.") and ".norm." in key:
            continue
        if key.startswith("moe.mm_layernorm."):
            continue

        if key.startswith("encoder_q.0."):
            key = "encoder.backbone." + key[len("encoder_q.0.") :]
        elif key.startswith("encoder_q.1."):
            continue
        elif key.startswith("moe."):
            key = "moe_decoder." + key[len("moe.") :]

        converted[key] = value
    return converted


def convert_stage2(state):
    converted = {}
    for key, value in state.items():
        if key.startswith(("moe.out_i.", "moe.out_v.", "moe.out_t.", "moe.mm_layernorm.")):
            continue
        # The refinement stages define an unused top-level projection;
        # the conv_1x1 layers inside ``layers.*`` are used and must be kept.
        parts = key.split(".")
        if len(parts) > 2 and parts[0] == "Rs" and parts[2] == "conv_1x1":
            continue
        if key.startswith("moe.moe.experts.") and ".norm." in key:
            continue
        converted[key] = value
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Earlier checkpoints may contain NumPy scalars in their metadata.
    # They are trusted local inputs, so disable weights_only for this converter.
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    converter = convert_stage1 if args.stage == 1 else convert_stage2
    converted = converter(state)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": converted}, output)
    print(
        f"stage {args.stage}: {len(state)} -> {len(converted)} tensors, "
        f"wrote {output}"
    )


if __name__ == "__main__":
    main()
