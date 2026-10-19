#!/usr/bin/env python3
"""Compare the exported EfficientSAM3 ONNX graphs with the upstream PyTorch predictor.

Check name: OnnxMaskLogitsMatchPytorchWithinTolerance.

For every prompt in --prompts, the script runs the upstream PyTorch code path
(`Sam3Processor.set_image` + `predict_inst` for points and boxes, `set_text_prompt` for text)
and the ONNX graphs on ONNX Runtime CPU with the same preprocessed pixels. It reports the
maximum absolute logit difference and the IoU of the thresholded masks (logit > 0) at the
original image size. The requirement is IoU >= 0.99 for every compared mask.

The script also writes the PyTorch masks as 8-bit PNG files (255 = inside) under
<out>/reference_masks/. The Rust measurement program compares its own masks with them.
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
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import esam3_graphs as graphs  # noqa: E402

REQUIRED_IOU = 0.99
TEXT_SCORE_THRESHOLD = 0.5
STABILITY_DELTA = 0.05
STABILITY_THRESHOLD = 0.98


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def upsample_logits(logits: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(logits, dtype=np.float32))
    while tensor.dim() < 4:
        tensor = tensor.unsqueeze(0)
    return F.interpolate(tensor, (height, width), mode="bilinear",
                         align_corners=False)[0].numpy()


def select_single_mask(logits: np.ndarray, scores: np.ndarray):
    """`MaskDecoder._dynamic_multimask_via_stability` for one prompt set."""
    single = logits[0]
    area_inner = (single > STABILITY_DELTA).sum()
    area_outer = (single > -STABILITY_DELTA).sum()
    stability = area_inner / area_outer if area_outer > 0 else 1.0
    if stability >= STABILITY_THRESHOLD:
        return single, float(scores[0])
    best = 1 + int(np.argmax(scores[1:]))
    return logits[best], float(scores[best])


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def to_model_points(points, width: int, height: int):
    coords = np.array([[p[0] * graphs.IMAGE_SIZE, p[1] * graphs.IMAGE_SIZE] for p in points],
                      dtype=np.float32)
    labels = np.array([p[2] for p in points], dtype=np.float32)
    pixel_coords = np.array([[p[0] * width, p[1] * height] for p in points], dtype=np.float32)
    return coords, labels, pixel_coords


def compare_point_prompts(model, sessions, image_entry, image, pixels, out_dir, results):
    from sam3.model.sam3_image_processor import Sam3Processor

    width, height = image.size
    processor = Sam3Processor(model, device="cpu")
    state = processor.set_image(image)
    encoder_out = sessions["point_encoder"].run(None, {"pixel_values": pixels.numpy()})
    image_embed, feats_0, feats_1 = encoder_out
    stem = Path(image_entry["file"]).stem

    prompt_sets = [(p["name"], p["points"], None) for p in image_entry.get("points", [])]
    prompt_sets += [(b["name"], None, b["box"]) for b in image_entry.get("boxes", [])]
    for name, points, box in prompt_sets:
        if points is not None:
            coords, labels, pixel_coords = to_model_points(points, width, height)
            torch_kwargs = {"point_coords": pixel_coords, "point_labels": labels.astype(np.int32)}
        else:
            x0, y0, x1, y1 = box
            coords = np.array([[x0, y0], [x1, y1]], dtype=np.float32) * graphs.IMAGE_SIZE
            labels = np.array([2, 3], dtype=np.float32)
            torch_kwargs = {"box": np.array([x0 * width, y0 * height, x1 * width, y1 * height],
                                            dtype=np.float32)}
        onnx_logits, onnx_scores, _ = sessions["point_decoder"].run(None, {
            "image_embed": image_embed, "high_res_feats_0": feats_0, "high_res_feats_1": feats_1,
            "point_coords": coords[None], "point_labels": labels[None],
            "mask_input": np.zeros((1, 1, graphs.LOW_RES_MASK_SIZE, graphs.LOW_RES_MASK_SIZE),
                                   np.float32),
            "has_mask_input": np.zeros(1, np.float32),
        })
        onnx_logits, onnx_scores = onnx_logits[0], onnx_scores[0]

        for multimask in (True, False):
            _, torch_scores, torch_low = model.predict_inst(
                state, multimask_output=multimask, return_logits=True, **torch_kwargs)
            if multimask:
                onnx_candidates = [(onnx_logits[i], float(onnx_scores[i])) for i in (1, 2, 3)]
            else:
                onnx_candidates = [select_single_mask(onnx_logits, onnx_scores)]
            for index, (onnx_low, onnx_score) in enumerate(onnx_candidates):
                reference_low = torch_low[index]
                onnx_low_clamped = np.clip(onnx_low, -32.0, 32.0)
                reference_mask = upsample_logits(reference_low, height, width)[0] > 0
                onnx_mask = upsample_logits(onnx_low_clamped, height, width)[0] > 0
                label = f"{stem}/{name}/{'multi' if multimask else 'single'}{index}"
                result = {
                    "image": image_entry["file"], "prompt": name, "kind": "point_box",
                    "multimask": multimask, "mask_index": index,
                    "max_abs_logit_diff": float(np.abs(onnx_low_clamped - reference_low).max()),
                    "score_torch": float(torch_scores[index]), "score_onnx": onnx_score,
                    "iou": iou(reference_mask, onnx_mask),
                    "mask_fraction": float(reference_mask.mean()),
                }
                result["passed"] = result["iou"] >= REQUIRED_IOU
                results.append(result)
                save_mask(reference_mask, out_dir / "reference_masks" / f"{label.replace('/', '__')}.png")
                print(f"[compare] {label}: IoU {result['iou']:.5f} "
                      f"max|dlogit| {result['max_abs_logit_diff']:.4f}")


def compare_text_prompts(model, sessions, image_entry, image, pixels, out_dir, results):
    from sam3.model.sam3_image_processor import Sam3Processor

    width, height = image.size
    stem = Path(image_entry["file"]).stem
    processor = Sam3Processor(model, device="cpu", confidence_threshold=TEXT_SCORE_THRESHOLD)
    for text in image_entry.get("text", []):
        state = processor.set_image(image)
        state = processor.set_text_prompt(text, state)
        torch_scores = state["scores"].numpy()
        torch_masks = state["masks"].squeeze(1).numpy()

        tokens = graphs.tokenize(model, text).numpy()
        language_mask, language_features, language_embeds = sessions["text_encoder"].run(
            None, {"tokenized_ids": tokens})
        # The exporter removes graph inputs that the detector does not read (language_embeds).
        grounding = sessions["detector_grounding"]
        available = {"pixel_values": pixels.numpy(), "language_features": language_features,
                     "language_mask": language_mask, "language_embeds": language_embeds}
        _, scores, mask_logits = grounding.run(
            None, {i.name: available[i.name] for i in grounding.get_inputs()})
        keep = scores[0] > TEXT_SCORE_THRESHOLD
        onnx_scores = scores[0][keep]
        onnx_masks = upsample_logits(mask_logits[0][keep][None], height, width) > 0 \
            if keep.any() else np.zeros((0, height, width), bool)
        label = f"{stem}/text:{text}"
        result = {"image": image_entry["file"], "prompt": text, "kind": "text",
                  "detections_torch": int(len(torch_scores)),
                  "detections_onnx": int(keep.sum()), "matches": []}
        torch_order = np.argsort(-torch_scores)
        onnx_order = np.argsort(-onnx_scores)
        for rank, (ti, oi) in enumerate(zip(torch_order, onnx_order)):
            match_iou = iou(torch_masks[ti], onnx_masks[oi])
            result["matches"].append({"rank": rank, "score_torch": float(torch_scores[ti]),
                                      "score_onnx": float(onnx_scores[oi]), "iou": match_iou})
            save_mask(torch_masks[ti], out_dir / "reference_masks"
                      / f"{stem}__text_{text.replace(' ', '_')}__{rank}.png")
        result["passed"] = (result["detections_torch"] == result["detections_onnx"]
                            and all(m["iou"] >= REQUIRED_IOU for m in result["matches"]))
        results.append(result)
        print(f"[compare] {label}: detections torch {result['detections_torch']} onnx "
              f"{result['detections_onnx']}, IoU {[round(m['iou'], 5) for m in result['matches']]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--point-checkpoint")
    parser.add_argument("--text-checkpoint")
    parser.add_argument("--backbone-type", default="efficientvit")
    parser.add_argument("--model-name", default="b1")
    parser.add_argument("--text-encoder", default="MobileCLIP-S0")
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True, help="output JSON report path")
    args = parser.parse_args()

    import onnxruntime as ort

    graphs.add_repository_to_path(args.repository)
    torch.set_grad_enabled(False)
    onnx_dir = Path(args.onnx_dir)
    out_path = Path(args.out)
    out_dir = out_path.parent
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    def open_session(name: str):
        return ort.InferenceSession(str(onnx_dir / f"{name}.onnx"), options,
                                    providers=["CPUExecutionProvider"])

    results: list[dict] = []
    if args.point_checkpoint:
        model = graphs.build_point_model(args.point_checkpoint, args.backbone_type,
                                         args.model_name)
        sessions = {n: open_session(n) for n in ("point_encoder", "point_decoder")}
        for entry in prompts["images"]:
            image = Image.open(Path(args.photos) / entry["file"]).convert("RGB")
            pixels = graphs.preprocess_image(image)
            compare_point_prompts(model, sessions, entry, image, pixels, out_dir, results)
        del model
    if args.text_checkpoint:
        model = graphs.build_text_model(args.text_checkpoint, args.backbone_type,
                                        args.model_name, args.text_encoder)
        sessions = {n: open_session(n) for n in ("text_encoder", "detector_grounding")}
        for entry in prompts["images"]:
            image = Image.open(Path(args.photos) / entry["file"]).convert("RGB")
            pixels = graphs.preprocess_image(image)
            compare_text_prompts(model, sessions, entry, image, pixels, out_dir, results)

    passed = all(r["passed"] for r in results)
    report = {"check": "OnnxMaskLogitsMatchPytorchWithinTolerance", "required_iou": REQUIRED_IOU,
              "onnxruntime": ort.__version__, "passed": passed, "results": results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[compare] {'PASS' if passed else 'FAIL'}: {sum(r['passed'] for r in results)}"
          f"/{len(results)} comparisons; report {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
