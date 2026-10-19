#!/usr/bin/env python3
"""Write the tensor group report of one EfficientSAM3 checkpoint.

For each tensor group (image encoder, neck, text encoder, detector, geometry encoder,
segmentation head, scoring, point and box decoder) the report gives the parameter count and the
origin of the architecture. The script then builds the EfficientSAM3 image model with the
checkpoint alone (no SAM3 checkpoint) and records the missing and unexpected keys, which shows
whether the checkpoint loads without `facebook/sam3`.

With --compare-to, the script also reports which tensor groups have identical values in a second
checkpoint. Identical groups between a stage 1 and a stage 3 checkpoint were not changed by the
stage 3 fine-tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import esam3_graphs as graphs  # noqa: E402

# (prefix after "detector." is removed, group name, architecture origin)
GROUPS = [
    ("backbone.vision_backbone.trunk.", "image_encoder",
     "EfficientSAM3 student (EfficientViT / RepViT / TinyViT)"),
    ("backbone.vision_backbone.sam2_convs.", "neck_point_box", "SAM3 neck architecture"),
    ("backbone.vision_backbone.convs.", "neck_detector", "SAM3 neck architecture"),
    ("backbone.language_backbone.", "text_encoder",
     "SAM3 text encoder, or the MobileCLIP student when the group is near 42M parameters"),
    ("transformer.", "detector_transformer", "SAM3"),
    ("geometry_encoder.", "geometry_encoder", "SAM3"),
    ("segmentation_head.", "segmentation_head", "SAM3"),
    ("dot_prod_scoring.", "scoring", "SAM3"),
    ("tracker.sam_prompt_encoder.", "point_box_prompt_encoder", "SAM3 tracker (SAM2 design)"),
    ("tracker.sam_mask_decoder.", "point_box_mask_decoder", "SAM3 tracker (SAM2 design)"),
    ("tracker.", "tracker_memory", "SAM3 tracker (video memory; not used for images)"),
]


def load_state_dict(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        checkpoint = checkpoint["model"]
    return checkpoint


def normalized_key(key: str) -> str:
    key = key.replace("student_trunk.", "")
    return key[len("detector."):] if key.startswith("detector.") else key


def group_of(key: str) -> tuple[str, str]:
    for prefix, name, origin in GROUPS:
        if key.startswith(prefix):
            return name, origin
    return "other", "unknown"


def group_digests(state: dict) -> dict:
    digests: dict = defaultdict(hashlib.sha256)
    for key in sorted(state):
        name, _ = group_of(normalized_key(key))
        tensor = state[key]
        digests[name].update(normalized_key(key).encode())
        digests[name].update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return {name: digest.hexdigest() for name, digest in digests.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone-type", default="efficientvit")
    parser.add_argument("--model-name", default="b1")
    parser.add_argument("--text-encoder", default=None,
                        help="student text encoder type, for example MobileCLIP-S0")
    parser.add_argument("--compare-to", help="second checkpoint for the identical-group check")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    graphs.add_repository_to_path(args.repository)
    state = load_state_dict(args.checkpoint)
    groups: dict = defaultdict(lambda: {"parameters": 0, "tensors": 0})
    for key, tensor in state.items():
        name, origin = group_of(normalized_key(key))
        groups[name]["parameters"] += tensor.numel()
        groups[name]["tensors"] += 1
        groups[name]["origin"] = origin
    report: dict = {
        "checkpoint": Path(args.checkpoint).name,
        "bytes": Path(args.checkpoint).stat().st_size,
        "total_parameters": sum(t.numel() for t in state.values()),
        "groups": {name: dict(values) for name, values in sorted(groups.items())},
        "has_point_box_decoder": "point_box_mask_decoder" in groups,
    }

    from sam3.model_builder import build_efficientsam3_image_model

    kwargs = {"text_encoder_type": args.text_encoder,
              "text_encoder_context_length": graphs.TEXT_CONTEXT_LENGTH} if args.text_encoder else {}
    model = build_efficientsam3_image_model(
        checkpoint_path=None, backbone_type=args.backbone_type, model_name=args.model_name,
        enable_inst_interactivity=report["has_point_box_decoder"], load_from_HF=False,
        device="cpu", **kwargs)
    model_keys = set(model.state_dict())
    loadable = {}
    for key, tensor in state.items():
        key = normalized_key(key)
        if key.startswith("tracker."):
            key = "inst_interactive_predictor.model." + key[len("tracker."):]
        loadable[key] = tensor
    missing = sorted(model_keys - set(loadable))
    unexpected = sorted(set(loadable) - model_keys)
    report["load_without_sam3_checkpoint"] = {
        "missing_keys": len(missing),
        "missing_key_examples": missing[:10],
        "unexpected_keys": len(unexpected),
        "unexpected_key_examples": unexpected[:10],
        "loads": len(missing) == 0,
    }

    if args.compare_to:
        own = group_digests(state)
        other = group_digests(load_state_dict(args.compare_to))
        report["identical_groups_in"] = {
            "checkpoint": Path(args.compare_to).name,
            "identical": sorted(n for n in own if other.get(n) == own[n]),
            "different": sorted(n for n in own if n in other and other[n] != own[n]),
            "only_here": sorted(n for n in own if n not in other),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("checkpoint", "total_parameters",
                                             "has_point_box_decoder")}))
    for name, values in report["groups"].items():
        print(f"  {name:28s} {values['parameters'] / 1e6:9.2f}M  {values['origin']}")
    print(f"  loads without SAM3 checkpoint: {report['load_without_sam3_checkpoint']['loads']} "
          f"(missing {len(missing)}, unexpected {len(unexpected)})")
    if "identical_groups_in" in report:
        print(f"  identical groups in {Path(args.compare_to).name}: "
              f"{report['identical_groups_in']['identical']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
