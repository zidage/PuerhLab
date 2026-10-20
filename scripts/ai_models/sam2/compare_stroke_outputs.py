#!/usr/bin/env python3
"""Compare the stroke prompts of #458 step E3 between ONNX Runtime and PyTorch.

Check name: StrokePromptLogitsMatchPytorchWithinTolerance.

`segmentation_runtime_report strokes` writes the corrective sequences of each prompt (the points
and labels of every interaction). This script replays every sequence on both sides with the same
preprocessed pixels: `SAM2ImagePredictor.predict` (the upstream code path) and the exported
`point_decoder.onnx` on ONNX Runtime CPU. Each side feeds its own previous selection back as
`mask_input` where the sequence does, so the check covers several positive points, negative
points, and the mask input path that step E1 did not test.

Selection as in the Rust program: one point uses the multimask output with the best predicted IoU;
several points use the single-mask output (`multimask_output=False`, with the stability fallback
of `MaskDecoder`). The union sequence decodes each stroke alone with positive points; the host
combination (maximum and minimum of the logits) is the same on both sides.

The requirement is IoU >= 0.99 of the thresholded masks (logit > 0) at the photo size for every
decoder call. With --pad-points N, the script also runs every call padded to N points on both
sides (--pad-mode label: padding points with label -1; repeat: copies of the last point) and
reports the IoU of the padded masks against the unpadded PyTorch masks and of the padded ONNX
masks against the padded PyTorch masks (`PaddedPromptMatchesUnpaddedPytorch`, reported, not
required).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import sam2_graphs as graphs  # noqa: E402
from compare_onnx_outputs import (  # noqa: E402
    LOGIT_LIMIT,
    REQUIRED_IOU,
    iou,
    select_single_mask,
    upsample_logits,
)

PADDING_LABEL = -1.0


def pad(points, labels, total, repeat):
    """The padding of `segmentation_runtime_report --pad-points --pad-mode`."""
    if len(points) > total:
        raise ValueError(f"{len(points)} points do not fit into --pad-points {total}")
    point, label = (points[-1], labels[-1]) if repeat else ([0.0, 0.0], PADDING_LABEL)
    return (list(points) + [list(point)] * (total - len(points)),
            list(labels) + [label] * (total - len(labels)))


def interactions(sequence: dict):
    """Per step: the decoder points, labels, and whether the previous selection is the mask input."""
    combination = sequence["combination"]
    points, labels = [], []
    for step in sequence["steps"]:
        new_points = [list(p) for p in step["points"]]
        label = step["label"]
        if combination == "union_of_strokes":
            yield step, new_points, [1] * len(new_points), False
            continue
        points += new_points
        labels += [label] * len(new_points)
        yield step, list(points), list(labels), combination == "points_and_previous_mask" and step["step"] > 0


def combine(combination: str, label: int, current, decoded):
    if combination != "union_of_strokes" or current is None:
        return decoded
    return np.maximum(current, decoded) if label == 1 else np.minimum(current, -decoded)


class OnnxDecoder:
    def __init__(self, session, features):
        self.session = session
        self.features = features

    def __call__(self, points, labels, mask_input, multimask):
        coords = [[x * graphs.IMAGE_SIZE, y * graphs.IMAGE_SIZE] for x, y in points]
        labels = [float(v) for v in labels]
        size = graphs.LOW_RES_MASK_SIZE
        image_embed, feats_0, feats_1 = self.features
        logits, scores, _ = self.session.run(None, {
            "image_embed": image_embed, "high_res_feats_0": feats_0, "high_res_feats_1": feats_1,
            "point_coords": np.array([coords], np.float32),
            "point_labels": np.array([labels], np.float32),
            "mask_input": (mask_input[None, None].astype(np.float32) if mask_input is not None
                           else np.zeros((1, 1, size, size), np.float32)),
            "has_mask_input": np.array([1.0 if mask_input is not None else 0.0], np.float32),
        })
        logits, scores = logits[0], scores[0]
        if multimask:
            token = 1 + int(np.argmax(scores[1:]))
            selected = logits[token]
        else:
            selected, _ = select_single_mask(logits, scores)
        return np.clip(selected, -LOGIT_LIMIT, LOGIT_LIMIT)


def torch_decode(predictor, points, labels, mask_input, width, height, multimask):
    kwargs = {
        "point_coords": np.array([[x * width, y * height] for x, y in points], np.float32),
        "point_labels": np.array(labels, np.int32),
        "return_logits": True,
    }
    if mask_input is not None:
        kwargs["mask_input"] = mask_input[None].astype(np.float32)
    _, scores, low = predictor.predict(multimask_output=multimask, **kwargs)
    return low[int(np.argmax(scores))] if multimask else low[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", required=True, choices=sorted(graphs.VARIANTS))
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--report", required=True, help="strokes report of the Rust program")
    parser.add_argument("--pad-points", type=int, default=0)
    parser.add_argument("--pad-mode", choices=("label", "repeat"), default="label")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import onnxruntime as ort

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    graphs.add_repository_to_path(args.repository)
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    _, checkpoint_name = graphs.VARIANTS[args.variant]
    predictor = SAM2ImagePredictor(
        graphs.build_model(args.variant, str(Path(args.weights) / checkpoint_name)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    sessions = {n: ort.InferenceSession(str(Path(args.onnx_dir) / f"{n}.onnx"), options,
                                        providers=["CPUExecutionProvider"])
                for n in ("point_encoder", "point_decoder")}
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))

    results = []
    for image_entry in report["images"]:
        image = Image.open(Path(args.photos) / image_entry["file"]).convert("RGB")
        width, height = image.size
        predictor.set_image(image)
        pixels = graphs.preprocess_image(image).numpy()
        features = sessions["point_encoder"].run(None, {"pixel_values": pixels})
        onnx_decode = OnnxDecoder(sessions["point_decoder"], features)
        for prompt in image_entry["prompts"]:
            for name, sequence in prompt.get("sequences", {}).items():
                if "steps" not in sequence:
                    continue
                combination = sequence["combination"]
                current = {"torch": None, "onnx": None, "padded": None, "padded_torch": None}
                for step, points, labels, use_mask in interactions(sequence):
                    multimask = len(points) == 1
                    previous = {k: (v if use_mask else None) for k, v in current.items()}
                    decoded = {
                        "torch": torch_decode(predictor, points, labels, previous["torch"],
                                              width, height, multimask),
                        "onnx": onnx_decode(points, labels, previous["onnx"], multimask),
                    }
                    if args.pad_points:
                        padded_points, padded_labels = pad(points, labels, args.pad_points,
                                                           args.pad_mode == "repeat")
                        decoded["padded"] = onnx_decode(padded_points, padded_labels,
                                                        previous["padded"], multimask)
                        decoded["padded_torch"] = torch_decode(
                            predictor, padded_points, padded_labels, previous["padded_torch"],
                            width, height, multimask)
                    for side, logits in decoded.items():
                        current[side] = combine(combination, step["label"], current[side], logits)
                    masks = {side: upsample_logits(v, height, width) > 0
                             for side, v in current.items() if v is not None}
                    result = {
                        "image": image_entry["file"], "prompt": prompt["name"], "sequence": name,
                        "step": step["step"], "decoder_points": len(points),
                        "negative_points": sum(1 for v in labels if v == 0),
                        "mask_input": use_mask,
                        "max_abs_logit_diff": float(np.abs(decoded["onnx"] - decoded["torch"]).max()),
                        "iou": iou(masks["torch"], masks["onnx"]),
                    }
                    result["passed"] = result["iou"] >= REQUIRED_IOU
                    if args.pad_points:
                        result["padded_iou"] = iou(masks["torch"], masks["padded"])
                        result["padded_torch_iou"] = iou(masks["torch"], masks["padded_torch"])
                        result["padded_onnx_vs_padded_torch_iou"] = iou(masks["padded_torch"],
                                                                        masks["padded"])
                    results.append(result)
                    print(f"[compare] {Path(image_entry['file']).stem} {prompt['name']} {name} "
                          f"step {step['step']}: IoU {result['iou']:.5f}"
                          + (f", padded {result['padded_iou']:.5f}" if args.pad_points else ""),
                          flush=True)

    passed = all(r["passed"] for r in results)
    summary = {
        "check": "StrokePromptLogitsMatchPytorchWithinTolerance", "variant": args.variant,
        "required_iou": REQUIRED_IOU, "onnxruntime": ort.__version__, "passed": passed,
        "comparisons": len(results), "failed": sum(not r["passed"] for r in results),
        "minimum_iou": min(r["iou"] for r in results),
        "maximum_abs_logit_diff": max(r["max_abs_logit_diff"] for r in results),
        "with_mask_input": sum(r["mask_input"] for r in results),
        "with_negative_points": sum(r["negative_points"] > 0 for r in results),
        "maximum_decoder_points": max(r["decoder_points"] for r in results),
    }
    if args.pad_points:
        def spread(key):
            values = sorted(r[key] for r in results)
            return {"minimum_iou": values[0], "median_iou": values[len(values) // 2],
                    "below_0_99": sum(v < REQUIRED_IOU for v in values)}

        summary["PaddedPromptMatchesUnpaddedPytorch"] = {
            "pad_points": args.pad_points, "pad_mode": args.pad_mode,
            "onnx_padded_vs_pytorch_unpadded": spread("padded_iou"),
            "pytorch_padded_vs_pytorch_unpadded": spread("padded_torch_iou"),
            "onnx_padded_vs_pytorch_padded": spread("padded_onnx_vs_padded_torch_iou"),
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**summary, "results": results}, indent=2), encoding="utf-8")
    print(f"[compare] {'PASS' if passed else 'FAIL'}: {len(results) - summary['failed']}"
          f"/{len(results)} decoder calls, minimum IoU {summary['minimum_iou']:.5f}; report {out}")
    if args.pad_points:
        print(f"[compare] padded: {summary['PaddedPromptMatchesUnpaddedPytorch']}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
