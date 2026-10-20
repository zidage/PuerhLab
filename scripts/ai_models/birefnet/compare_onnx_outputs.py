#!/usr/bin/env python3
"""Compare the exported BiRefNet ONNX graph with the PyTorch model (#458, step E2).

Check name: OnnxMatteMatchesPytorchWithinTolerance.

For every photo in --prompts, the script runs the Hugging Face BiRefNet model in PyTorch and
`subject_matte.onnx` on ONNX Runtime CPU with the same preprocessed pixels. It reports the maximum
and mean absolute alpha difference at the model size and the IoU of the thresholded mattes
(alpha > 0.5) at the original photo size. The requirement is IoU >= 0.99 for every photo.

Outputs under the directory of --out:

- reference_alpha/<stem>.png: the PyTorch alpha at the photo size, 8 bit, for
  `segmentation_runtime_report matte --reference-alpha`;
- alpha/<stem>.npy: the ONNX Runtime CPU alpha at the model size (float16), for probe_mattes.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import birefnet_graphs as graphs  # noqa: E402

REQUIRED_IOU = 0.99


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", required=True, choices=sorted(graphs.VARIANTS))
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--prompts", required=True, help="the quality set; only the photo list "
                                                         "is used")
    parser.add_argument("--out", required=True, help="output JSON report path")
    args = parser.parse_args()

    import onnxruntime as ort

    torch.set_grad_enabled(False)
    size = graphs.image_size(args.variant)
    model = graphs.SubjectMatte(graphs.build_model(args.weights, args.variant)).eval()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(Path(args.onnx_dir) / "subject_matte.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    out_path = Path(args.out)
    (out_path.parent / "reference_alpha").mkdir(parents=True, exist_ok=True)
    (out_path.parent / "alpha").mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    results: list[dict] = []
    for entry in prompts["images"]:
        image = graphs.load_rgb(str(Path(args.photos) / entry["file"]))
        width, height = image.size
        stem = Path(entry["file"]).stem
        pixels = graphs.preprocess(image, size)
        started = time.perf_counter()
        reference = model(pixels)[0, 0].numpy()
        torch_seconds = time.perf_counter() - started
        started = time.perf_counter()
        onnx_alpha = session.run(None, {"pixel_values": pixels.numpy()})[0][0, 0]
        onnx_seconds = time.perf_counter() - started
        reference_photo = graphs.alpha_to_photo(reference, width, height)
        onnx_photo = graphs.alpha_to_photo(onnx_alpha, width, height)
        difference = np.abs(onnx_alpha - reference)
        result = {
            "image": entry["file"],
            "max_abs_alpha_diff": float(difference.max()),
            "mean_abs_alpha_diff": float(difference.mean()),
            "iou": iou(reference_photo > 0.5, onnx_photo > 0.5),
            "foreground_fraction": float((reference_photo > 0.5).mean()),
            "torch_cpu_seconds": round(torch_seconds, 2),
            "onnxruntime_cpu_seconds": round(onnx_seconds, 2),
        }
        result["passed"] = result["iou"] >= REQUIRED_IOU
        results.append(result)
        Image.fromarray(np.round(reference_photo * 255).astype(np.uint8), mode="L").save(
            out_path.parent / "reference_alpha" / f"{stem}.png")
        np.save(out_path.parent / "alpha" / f"{stem}.npy", onnx_alpha.astype(np.float16))
        print(f"[compare] {args.variant} {stem}: IoU {result['iou']:.5f} "
              f"max|dalpha| {result['max_abs_alpha_diff']:.4f} "
              f"torch {torch_seconds:.1f} s, ort {onnx_seconds:.1f} s", flush=True)

    passed = all(r["passed"] for r in results)
    report = {"check": "OnnxMatteMatchesPytorchWithinTolerance", "variant": args.variant,
              "required_iou": REQUIRED_IOU, "onnxruntime": ort.__version__,
              "torch": torch.__version__, "passed": passed,
              "minimum_iou": min(r["iou"] for r in results),
              "maximum_abs_alpha_diff": max(r["max_abs_alpha_diff"] for r in results),
              "results": results}
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[compare] {'PASS' if passed else 'FAIL'}: {sum(r['passed'] for r in results)}"
          f"/{len(results)} photos, minimum IoU {report['minimum_iou']:.5f}; report {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
