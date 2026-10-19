#!/usr/bin/env python3
"""Make the edge refinement comparison images for the EfficientSAM3 point and box masks.

For every point and box prompt in --prompts, the script runs point_encoder.onnx and
point_decoder.onnx on ONNX Runtime CPU, selects one mask (the best multimask token for one point,
the single-mask selection otherwise), and makes the mask at the photo size with the long edge
capped at 4096 in two ways:

  (a) bilinear: bilinear upsampling of the 288 x 288 logits, threshold logit > 0;
  (b) guided: the guided filter of He et al. (2013) applied to sigmoid(bilinear logits) with the
      photo luminance as the guide, threshold 0.5.

It writes one comparison image per prompt with up to four crops along the mask edge. Each crop
row shows the photo, (a), and (b). The rows come from the edge pixels with the largest photo
gradient, so they show hair, fur, foliage, or hard edges when the photo has them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import esam3_graphs as graphs  # noqa: E402

MAXIMUM_AXIS = 4096
CROP_SIZE = 256
CROP_COUNT = 4
OVERLAY_COLOR = np.array([30, 144, 255], np.float32)


def preprocess(image: Image.Image) -> np.ndarray:
    resized = image.resize((graphs.IMAGE_SIZE, graphs.IMAGE_SIZE), Image.Resampling.BILINEAR)
    pixels = (np.asarray(resized, np.float32) / 255.0 - 0.5) / 0.5
    return pixels.transpose(2, 0, 1)[None]


def guided_filter(guide: np.ndarray, source: np.ndarray, radius: int, eps: float) -> np.ndarray:
    size = (2 * radius + 1, 2 * radius + 1)

    def box(x):
        return cv2.boxFilter(x, -1, size, borderType=cv2.BORDER_REFLECT)

    mean_guide, mean_source = box(guide), box(source)
    covariance = box(guide * source) - mean_guide * mean_source
    variance = box(guide * guide) - mean_guide * mean_guide
    a = covariance / (variance + eps)
    b = mean_source - a * mean_guide
    return box(a) * guide + box(b)


def select_logits(logits: np.ndarray, scores: np.ndarray, multimask: bool) -> np.ndarray:
    if multimask:
        return logits[1 + int(np.argmax(scores[1:]))]
    single = logits[0]
    inner, outer = (single > 0.05).sum(), (single > -0.05).sum()
    if outer == 0 or inner / outer >= 0.98:
        return single
    return logits[1 + int(np.argmax(scores[1:]))]


def overlay(photo: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = photo.astype(np.float32)
    result[mask] = result[mask] * 0.45 + OVERLAY_COLOR * 0.55
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    result[edge > 0] = (255, 210, 0)
    return result.astype(np.uint8)


def edge_crops(photo: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    ys, xs = np.nonzero(edge)
    if len(ys) == 0:
        return []
    order = np.argsort(-gradient[ys, xs])
    centers: list[tuple[int, int]] = []
    height, width = mask.shape
    for index in order:
        y, x = int(ys[index]), int(xs[index])
        if all(abs(y - cy) > CROP_SIZE or abs(x - cx) > CROP_SIZE for cy, cx in centers):
            centers.append((min(max(y, CROP_SIZE // 2), height - CROP_SIZE // 2),
                            min(max(x, CROP_SIZE // 2), width - CROP_SIZE // 2)))
        if len(centers) == CROP_COUNT:
            break
    return centers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--radius", type=int, default=8,
                        help="guided filter radius in pixels of the refined mask")
    parser.add_argument("--eps", type=float, default=1e-3)
    args = parser.parse_args()

    import onnxruntime as ort

    onnx_dir = Path(args.onnx_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = ort.InferenceSession(str(onnx_dir / "point_encoder.onnx"),
                                   providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(onnx_dir / "point_decoder.onnx"),
                                   providers=["CPUExecutionProvider"])
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    timings = []
    for entry in prompts["images"]:
        image = Image.open(Path(args.photos) / entry["file"]).convert("RGB")
        scale = min(1.0, MAXIMUM_AXIS / max(image.size))
        width, height = round(image.width * scale), round(image.height * scale)
        photo = np.asarray(image.resize((width, height), Image.Resampling.LANCZOS)
                           if scale < 1.0 else image)
        image_embed, feats_0, feats_1 = encoder.run(None, {"pixel_values": preprocess(image)})
        guide = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        prompt_sets = [(p["name"], p["points"], None) for p in entry.get("points", [])]
        prompt_sets += [(b["name"], None, b["box"]) for b in entry.get("boxes", [])]
        for name, points, box in prompt_sets:
            if points is not None:
                coords = [[p[0] * graphs.IMAGE_SIZE, p[1] * graphs.IMAGE_SIZE] for p in points]
                labels = [float(p[2]) for p in points]
            else:
                coords = [[box[0] * graphs.IMAGE_SIZE, box[1] * graphs.IMAGE_SIZE],
                          [box[2] * graphs.IMAGE_SIZE, box[3] * graphs.IMAGE_SIZE]]
                labels = [2.0, 3.0]
            logits, scores, _ = decoder.run(None, {
                "image_embed": image_embed, "high_res_feats_0": feats_0,
                "high_res_feats_1": feats_1,
                "point_coords": np.array([coords], np.float32),
                "point_labels": np.array([labels], np.float32),
                "mask_input": np.zeros((1, 1, 288, 288), np.float32),
                "has_mask_input": np.zeros(1, np.float32),
            })
            low = select_logits(logits[0], scores[0], points is not None and len(points) == 1)
            started = time.perf_counter()
            upsampled = cv2.resize(np.clip(low, -32, 32), (width, height),
                                   interpolation=cv2.INTER_LINEAR)
            bilinear = upsampled > 0
            bilinear_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            probability = 1.0 / (1.0 + np.exp(-upsampled))
            guided = guided_filter(guide, probability.astype(np.float32), args.radius,
                                   args.eps) > 0.5
            guided_ms = (time.perf_counter() - started) * 1000
            changed = float(np.logical_xor(bilinear, guided).mean())
            timings.append({"image": entry["file"], "prompt": name, "width": width,
                            "height": height, "bilinear_ms": bilinear_ms,
                            "guided_ms": guided_ms, "changed_pixel_fraction": changed})

            rows = []
            for cy, cx in edge_crops(photo, bilinear):
                window = (slice(cy - CROP_SIZE // 2, cy + CROP_SIZE // 2),
                          slice(cx - CROP_SIZE // 2, cx + CROP_SIZE // 2))
                rows.append(np.concatenate([photo[window], overlay(photo[window], bilinear[window]),
                                            overlay(photo[window], guided[window])], axis=1))
            if rows:
                stem = Path(entry["file"]).stem
                Image.fromarray(np.concatenate(rows, axis=0)).save(
                    out_dir / f"{stem}__{name}__photo_bilinear_guided.png")
            print(f"[refine] {entry['file']} {name}: bilinear {bilinear_ms:.1f} ms, guided "
                  f"{guided_ms:.1f} ms, changed pixels {changed * 100:.2f}%")
    (out_dir / "refine_timings.json").write_text(json.dumps(
        {"radius": args.radius, "eps": args.eps, "results": timings}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
