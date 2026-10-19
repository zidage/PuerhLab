#!/usr/bin/env python3
"""Compare the exported SAM 2.1 ONNX graphs with the upstream PyTorch predictor (#458, E1).

Check name: OnnxMaskLogitsMatchPytorchWithinTolerance.

For every point and box prompt in --prompts, the script runs `SAM2ImagePredictor.set_image` and
`predict` (the upstream code path) and the ONNX graphs on ONNX Runtime CPU with the same
preprocessed pixels. It reports the maximum absolute low-resolution logit difference and the IoU
of the thresholded masks (logit > 0) at the original image size. The requirement is IoU >= 0.99
for every compared mask.

The PyTorch masks are written as 8-bit PNG files (255 = inside) under <out>/reference_masks/ with
the names that `segmentation_runtime_report --reference-masks` reads.
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

import sam2_graphs as graphs  # noqa: E402

REQUIRED_IOU = 0.99
STABILITY_DELTA = 0.05
STABILITY_THRESHOLD = 0.98
LOGIT_LIMIT = 32.0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def upsample_logits(logits: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(logits, dtype=np.float32))[None, None]
    return F.interpolate(tensor, (height, width), mode="bilinear", align_corners=False)[0, 0].numpy()


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


def compare_image(predictor, sessions, entry, image, out_dir, results) -> None:
    width, height = image.size
    stem = Path(entry["file"]).stem
    predictor.set_image(image)
    pixels = graphs.preprocess_image(image).numpy()
    image_embed, feats_0, feats_1 = sessions["point_encoder"].run(None, {"pixel_values": pixels})

    for prompt in graphs.prompt_sets(entry):
        coords, labels = graphs.model_prompt(prompt)
        onnx_logits, onnx_scores, _ = sessions["point_decoder"].run(None, {
            "image_embed": image_embed, "high_res_feats_0": feats_0, "high_res_feats_1": feats_1,
            "point_coords": np.array([coords], np.float32),
            "point_labels": np.array([labels], np.float32),
            "mask_input": np.zeros((1, 1, graphs.LOW_RES_MASK_SIZE, graphs.LOW_RES_MASK_SIZE),
                                   np.float32),
            "has_mask_input": np.zeros(1, np.float32),
        })
        onnx_logits, onnx_scores = onnx_logits[0], onnx_scores[0]
        torch_kwargs = {}
        if prompt["points"]:
            torch_kwargs["point_coords"] = np.array(
                [[x * width, y * height] for x, y in prompt["points"]], np.float32)
            torch_kwargs["point_labels"] = np.array(prompt["labels"], np.int32)
        if prompt["box"] is not None:
            x0, y0, x1, y1 = prompt["box"]
            torch_kwargs["box"] = np.array([x0 * width, y0 * height, x1 * width, y1 * height],
                                           np.float32)

        for multimask in (True, False):
            _, torch_scores, torch_low = predictor.predict(
                multimask_output=multimask, return_logits=True, **torch_kwargs)
            if multimask:
                candidates = [(onnx_logits[i], float(onnx_scores[i])) for i in (1, 2, 3)]
            else:
                candidates = [select_single_mask(onnx_logits, onnx_scores)]
            for index, (onnx_low, onnx_score) in enumerate(candidates):
                reference_low = torch_low[index]
                onnx_low = np.clip(onnx_low, -LOGIT_LIMIT, LOGIT_LIMIT)
                reference_mask = upsample_logits(reference_low, height, width) > 0
                onnx_mask = upsample_logits(onnx_low, height, width) > 0
                label = f"{stem}__{prompt['name']}__{'multi' if multimask else 'single'}{index}"
                result = {
                    "image": entry["file"], "prompt": prompt["name"], "multimask": multimask,
                    "mask_index": index,
                    "max_abs_logit_diff": float(np.abs(onnx_low - reference_low).max()),
                    "score_torch": float(torch_scores[index]), "score_onnx": onnx_score,
                    "iou": iou(reference_mask, onnx_mask),
                    "mask_fraction": float(reference_mask.mean()),
                }
                result["passed"] = result["iou"] >= REQUIRED_IOU
                results.append(result)
                save_mask(reference_mask, out_dir / "reference_masks" / f"{label}.png")
                print(f"[compare] {label}: IoU {result['iou']:.5f} "
                      f"max|dlogit| {result['max_abs_logit_diff']:.4f}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", required=True, choices=sorted(graphs.VARIANTS))
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True, help="output JSON report path")
    args = parser.parse_args()

    import onnxruntime as ort

    graphs.add_repository_to_path(args.repository)
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    torch.set_grad_enabled(False)
    _, checkpoint_name = graphs.VARIANTS[args.variant]
    model = graphs.build_model(args.variant, str(Path(args.weights) / checkpoint_name))
    predictor = SAM2ImagePredictor(model)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sessions = {n: ort.InferenceSession(str(Path(args.onnx_dir) / f"{n}.onnx"), options,
                                        providers=["CPUExecutionProvider"])
                for n in ("point_encoder", "point_decoder")}
    out_path = Path(args.out)
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    results: list[dict] = []
    for entry in prompts["images"]:
        image = Image.open(Path(args.photos) / entry["file"]).convert("RGB")
        compare_image(predictor, sessions, entry, image, out_path.parent, results)

    passed = all(r["passed"] for r in results)
    report = {"check": "OnnxMaskLogitsMatchPytorchWithinTolerance", "variant": args.variant,
              "required_iou": REQUIRED_IOU, "onnxruntime": ort.__version__, "passed": passed,
              "minimum_iou": min(r["iou"] for r in results),
              "maximum_abs_logit_diff": max(r["max_abs_logit_diff"] for r in results),
              "results": results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[compare] {'PASS' if passed else 'FAIL'}: {sum(r['passed'] for r in results)}"
          f"/{len(results)} comparisons, minimum IoU {report['minimum_iou']:.5f}; "
          f"report {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
