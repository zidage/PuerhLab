#!/usr/bin/env python3
"""Make the edge refinement comparison images for the EfficientSAM3 point and box masks.

For every point and box prompt in --prompts, the script runs point_encoder.onnx and
point_decoder.onnx on ONNX Runtime CPU, selects one mask (the best multimask token for one point,
the single-mask selection otherwise), and makes the mask at the photo size with the long edge
capped at 4096 in four ways:

  (a) bilinear: bilinear upsampling of the 288 x 288 logits of the whole photo, threshold
      logit > 0;
  (b) guided: the guided filter of He et al. (2013) applied to sigmoid(bilinear logits) with the
      photo luminance as the guide, threshold 0.5;
  (c) rectangle: when the rectangle below covers less than --max-rectangle-fraction of the photo,
      a second encoder pass on a rectangle of the source photo around the subject. The
      rectangle is the bounding box of (a), the box prompt, and the positive points, with a margin,
      grown toward a square. The prompt is mapped into the rectangle; the bounding box of (a) is
      added as a box prompt. The rectangle is taken from the full-resolution photo, so a small
      subject gets more encoder pixels. Outside the rectangle the mask is empty. For a larger
      rectangle (c) is (a);
  (d) matte: a soft alpha in [0, 1] from (c), not thresholded. Holes and islands smaller than
      --clean-cells^2 logit cells are removed from (c) first. The unknown band is --band-cells
      logit cells on each side of the (c) edge (a trimap: alpha is 1 inside the band and 0 outside
      it). --matte projection (default) projects each band pixel color onto the line between the
      local mean color of the sure foreground and of the sure background (both within
      2 x band), then smooths the alpha with a small color guided filter; where the two mean
      colors are closer than --min-contrast, alpha is the cleaned (c) mask. --matte guided applies the
      color guided filter of He et al. to sigmoid(logits) with radius --matte-radius-cells.

The IoU between (a) and (c) is the quality cost of encoding the whole photo for this prompt (plan
section 8.6, cases 5 and 6).

Per prompt it writes:

  <stem>__<prompt>__photo_bilinear_guided.png: up to four crops along the mask edge. Each row shows
      the photo, (a), (b), (c), (d) as an overlay, and (d) as a gray matte;
  <stem>__<prompt>__subject.jpg: the subject rectangle with (a) and (d) side by side.
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
SUBJECT_VIEW_AXIS = 1400
RECTANGLE_MARGIN = 0.15
OVERLAY_COLOR = np.array([30, 144, 255], np.float32)


def preprocess(image: Image.Image) -> np.ndarray:
    resized = image.resize((graphs.IMAGE_SIZE, graphs.IMAGE_SIZE), Image.Resampling.BILINEAR)
    pixels = (np.asarray(resized, np.float32) / 255.0 - 0.5) / 0.5
    return pixels.transpose(2, 0, 1)[None]


def box_filter(x: np.ndarray, radius: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * radius + 1, 2 * radius + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, source: np.ndarray, radius: int, eps: float) -> np.ndarray:
    mean_guide, mean_source = box_filter(guide, radius), box_filter(source, radius)
    covariance = box_filter(guide * source, radius) - mean_guide * mean_source
    variance = box_filter(guide * guide, radius) - mean_guide * mean_guide
    a = covariance / (variance + eps)
    b = mean_source - a * mean_guide
    return box_filter(a, radius) * guide + box_filter(b, radius)


def color_guided_filter(guide: np.ndarray, source: np.ndarray, radius: int, eps: float,
                        subsample: int) -> np.ndarray:
    """The color guided filter of He et al. (2013), computed at 1/subsample resolution (the fast
    guided filter of He and Sun, 2015). guide is H x W x 3 in [0, 1]; source is H x W."""
    height, width = source.shape
    small = (max(1, width // subsample), max(1, height // subsample))
    g = cv2.resize(guide, small, interpolation=cv2.INTER_AREA)
    p = cv2.resize(source, small, interpolation=cv2.INTER_AREA)
    r = max(1, radius // subsample)
    mean_g = box_filter(g, r)
    mean_p = box_filter(p, r)
    mean_gp = box_filter(g * p[..., None], r)
    cov_gp = mean_gp - mean_g * mean_p[..., None]
    sigma = np.empty(p.shape + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            value = box_filter(g[..., i] * g[..., j], r) - mean_g[..., i] * mean_g[..., j]
            sigma[..., i, j] = sigma[..., j, i] = value
    sigma += eps * np.eye(3, dtype=np.float32)
    a = np.linalg.solve(sigma, cov_gp[..., None])[..., 0]
    b = mean_p - (a * mean_g).sum(axis=-1)
    mean_a = cv2.resize(box_filter(a, r), (width, height), interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(box_filter(b, r), (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip((mean_a * guide).sum(axis=-1) + mean_b, 0.0, 1.0)


def clean_mask(mask: np.ndarray, min_area: float) -> np.ndarray:
    """Fill holes and remove islands smaller than min_area pixels (as SAM
    postprocess_small_regions)."""
    out = mask.copy()
    for fill in (True, False):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (~out if fill else out).astype(np.uint8), connectivity=8)
        small = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] < min_area]
        if small:
            out[np.isin(labels, small)] = fill
    return out


def band_matte(guide: np.ndarray, logits: np.ndarray, cell: float, args) -> np.ndarray:
    """Soft alpha for one rectangle. guide is H x W x 3 in [0, 1]; logits are upsampled to H x W."""
    hard = clean_mask(logits > 0, (args.clean_cells * cell) ** 2)
    band = max(2, round(args.band_cells * cell))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    sure_foreground = cv2.erode(hard.astype(np.uint8), kernel) > 0
    sure_background = cv2.dilate(hard.astype(np.uint8), kernel) == 0
    if args.matte == "guided":
        probability = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        alpha = color_guided_filter(guide, probability,
                                    max(2, round(args.matte_radius_cells * cell)),
                                    args.matte_eps, args.matte_subsample)
    else:
        radius = 2 * band
        weights = {}
        for name, region in (("foreground", sure_foreground), ("background", sure_background)):
            w = region.astype(np.float32)
            total = box_filter(w, radius)
            mean = box_filter(guide * w[..., None], radius) / np.maximum(total, 1e-6)[..., None]
            weights[name] = (mean, total > 1e-3)
        (fg, has_fg), (bg, has_bg) = weights["foreground"], weights["background"]
        direction = fg - bg
        length2 = (direction * direction).sum(axis=-1)
        alpha = np.clip(((guide - bg) * direction).sum(axis=-1) / np.maximum(length2, 1e-6),
                        0.0, 1.0)
        usable = has_fg & has_bg & (length2 > args.min_contrast ** 2)
        alpha = np.where(usable, alpha, hard).astype(np.float32)
        alpha = color_guided_filter(guide, alpha, args.smooth_radius, args.matte_eps, 1)
    alpha[sure_foreground] = 1.0
    alpha[sure_background] = 0.0
    return alpha


def select_logits(logits: np.ndarray, scores: np.ndarray, multimask: bool) -> np.ndarray:
    if multimask:
        return logits[1 + int(np.argmax(scores[1:]))]
    single = logits[0]
    inner, outer = (single > 0.05).sum(), (single > -0.05).sum()
    if outer == 0 or inner / outer >= 0.98:
        return single
    return logits[1 + int(np.argmax(scores[1:]))]


def overlay(photo: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Binary mask: blue fill and a yellow edge. Soft mask (float): blue blend by alpha."""
    result = photo.astype(np.float32)
    if mask.dtype == bool:
        result[mask] = result[mask] * 0.45 + OVERLAY_COLOR * 0.55
        edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8))
        result[edge > 0] = (255, 210, 0)
    else:
        alpha = (mask * 0.55)[..., None]
        result = result * (1.0 - alpha) + OVERLAY_COLOR * alpha
    return result.astype(np.uint8)


def matte_view(mask: np.ndarray) -> np.ndarray:
    gray = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


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


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def subject_rectangle(mask: np.ndarray, points, box) -> tuple[float, float, float, float] | None:
    """Normalized rectangle around the mask, the box prompt, and the positive points, with a
    margin, grown toward a square in photo pixels. None when the mask is empty."""
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    x0, x1 = xs.min() / width, (xs.max() + 1) / width
    y0, y1 = ys.min() / height, (ys.max() + 1) / height
    for x, y, label in points or []:
        if label == 1:
            x0, x1, y0, y1 = min(x0, x), max(x1, x), min(y0, y), max(y1, y)
    if box is not None:
        x0, y0, x1, y1 = min(x0, box[0]), min(y0, box[1]), max(x1, box[2]), max(y1, box[3])
    # Photo pixels from here.
    x0, x1, y0, y1 = x0 * width, x1 * width, y0 * height, y1 * height
    margin = RECTANGLE_MARGIN * max(x1 - x0, y1 - y0)
    x0, x1, y0, y1 = x0 - margin, x1 + margin, y0 - margin, y1 + margin
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = min(side, width), min(side, height)
    x0 = min(max(cx - w / 2, 0.0), width - w)
    y0 = min(max(cy - h / 2, 0.0), height - h)
    return x0 / width, y0 / height, (x0 + w) / width, (y0 + h) / height


def run_decoder(decoder, features, coords, labels) -> tuple[np.ndarray, np.ndarray]:
    image_embed, feats_0, feats_1 = features
    logits, scores, _ = decoder.run(None, {
        "image_embed": image_embed, "high_res_feats_0": feats_0,
        "high_res_feats_1": feats_1,
        "point_coords": np.array([coords], np.float32),
        "point_labels": np.array([labels], np.float32),
        "mask_input": np.zeros((1, 1, 288, 288), np.float32),
        "has_mask_input": np.zeros(1, np.float32),
    })
    return logits[0], scores[0]


def prompt_inputs(points, box) -> tuple[list, list]:
    size = graphs.IMAGE_SIZE
    coords, labels = [], []
    if box is not None:
        coords += [[box[0] * size, box[1] * size], [box[2] * size, box[3] * size]]
        labels += [2.0, 3.0]
    for x, y, label in points or []:
        coords.append([x * size, y * size])
        labels.append(float(label))
    return coords, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--radius", type=int, default=8,
                        help="gray guided filter radius in pixels of the refined mask")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--matte", choices=["projection", "guided"], default="projection")
    parser.add_argument("--matte-radius-cells", type=float, default=0.5,
                        help="--matte guided: filter radius in logit cells of the rectangle pass")
    parser.add_argument("--matte-eps", type=float, default=1e-4)
    parser.add_argument("--smooth-radius", type=int, default=2,
                        help="--matte projection: guided smoothing radius in pixels")
    parser.add_argument("--min-contrast", type=float, default=0.08,
                        help="--matte projection: minimum RGB distance of the two mean colors")
    parser.add_argument("--band-cells", type=float, default=1.0,
                        help="half width of the unknown band in logit cells")
    parser.add_argument("--clean-cells", type=float, default=3.0,
                        help="holes and islands smaller than this many cells squared are removed")
    parser.add_argument("--matte-subsample", type=int, default=2)
    parser.add_argument("--max-rectangle-fraction", type=float, default=0.5,
                        help="run the rectangle pass only below this fraction of the photo area")
    parser.add_argument("--images", default="",
                        help="comma-separated file names; empty runs every image")
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
    selected = {name for name in args.images.split(",") if name}
    timings = []
    for entry in prompts["images"]:
        if selected and entry["file"] not in selected:
            continue
        image = Image.open(Path(args.photos) / entry["file"]).convert("RGB")
        scale = min(1.0, MAXIMUM_AXIS / max(image.size))
        width, height = round(image.width * scale), round(image.height * scale)
        photo = np.asarray(image.resize((width, height), Image.Resampling.LANCZOS)
                           if scale < 1.0 else image)
        features = encoder.run(None, {"pixel_values": preprocess(image)})
        guide = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        prompt_sets = [(p["name"], p["points"], None) for p in entry.get("points", [])]
        prompt_sets += [(b["name"], None, b["box"]) for b in entry.get("boxes", [])]
        for name, points, box in prompt_sets:
            multimask = points is not None and len(points) == 1
            logits, scores = run_decoder(decoder, features, *prompt_inputs(points, box))
            low = select_logits(logits, scores, multimask)
            started = time.perf_counter()
            upsampled = cv2.resize(np.clip(low, -32, 32), (width, height),
                                   interpolation=cv2.INTER_LINEAR)
            bilinear = upsampled > 0
            bilinear_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            probability = (1.0 / (1.0 + np.exp(-upsampled))).astype(np.float32)
            guided = guided_filter(guide, probability, args.radius, args.eps) > 0.5
            guided_ms = (time.perf_counter() - started) * 1000
            changed = float(np.logical_xor(bilinear, guided).mean())

            # (c) rectangle pass and (d) matte.
            result = {"image": entry["file"], "prompt": name, "width": width, "height": height,
                      "mask_fraction": float(bilinear.mean()), "bilinear_ms": bilinear_ms,
                      "guided_ms": guided_ms, "changed_pixel_fraction": changed}
            rectangle = subject_rectangle(bilinear, points, box)
            matte = np.zeros(bilinear.shape, np.float32)
            rectangle_area = (0.0 if rectangle is None else
                              (rectangle[2] - rectangle[0]) * (rectangle[3] - rectangle[1]))
            result["rectangle_area_fraction"] = rectangle_area
            if rectangle is not None and rectangle_area < args.max_rectangle_fraction:
                rx0, ry0, rx1, ry1 = rectangle
                started = time.perf_counter()
                source_box = (round(rx0 * image.width), round(ry0 * image.height),
                              round(rx1 * image.width), round(ry1 * image.height))
                rect_features = encoder.run(None, {"pixel_values": preprocess(image.crop(source_box))})
                rect_encode_ms = (time.perf_counter() - started) * 1000

                def to_rect(x, y):
                    return (x - rx0) / (rx1 - rx0), (y - ry0) / (ry1 - ry0)

                ys, xs = np.nonzero(bilinear)
                mask_box = [*to_rect(xs.min() / width, ys.min() / height),
                            *to_rect((xs.max() + 1) / width, (ys.max() + 1) / height)]
                if box is not None:
                    rect_box = [*to_rect(box[0], box[1]), *to_rect(box[2], box[3])]
                else:
                    rect_box = mask_box
                rect_points = []
                for x, y, label in points or []:
                    px, py = to_rect(x, y)
                    if 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0:
                        rect_points.append([px, py, label])
                rect_logits, rect_scores = run_decoder(
                    decoder, rect_features, *prompt_inputs(rect_points, rect_box))
                rect_low = select_logits(rect_logits, rect_scores, False)
                px0, py0 = round(rx0 * width), round(ry0 * height)
                px1, py1 = round(rx1 * width), round(ry1 * height)
                rect_up = cv2.resize(np.clip(rect_low, -32, 32), (px1 - px0, py1 - py0),
                                     interpolation=cv2.INTER_LINEAR)
                rect_mask = np.zeros_like(bilinear)
                rect_mask[py0:py1, px0:px1] = rect_up > 0
                result.update({
                    "rectangle": [round(v, 4) for v in rectangle],
                    "rectangle_source_pixels": [source_box[2] - source_box[0],
                                                source_box[3] - source_box[1]],
                    "rectangle_encode_ms_cpu": rect_encode_ms,
                    "rectangle_iou": iou(bilinear, rect_mask),
                    "rectangle_mask_fraction": float(rect_mask.mean()),
                })
            else:
                # The rectangle covers most of the photo: a second pass adds no encoder pixels.
                px0, py0, px1, py1 = 0, 0, width, height
                rect_low, rect_up, rect_mask = low, upsampled, bilinear
            if rectangle is not None:
                started = time.perf_counter()
                cell = max(px1 - px0, py1 - py0) / rect_low.shape[0]
                rect_guide = photo[py0:py1, px0:px1].astype(np.float32) / 255.0
                matte[py0:py1, px0:px1] = band_matte(rect_guide, rect_up, cell, args)
                result.update({
                    "logit_cell_pixels": cell,
                    "matte_ms": (time.perf_counter() - started) * 1000,
                    "matte_soft_pixel_fraction": float(((matte > 0.05) & (matte < 0.95)).mean()),
                })
            else:
                rect_mask = bilinear
            timings.append(result)

            stem = Path(entry["file"]).stem
            rows = []
            for cy, cx in edge_crops(photo, bilinear):
                window = (slice(cy - CROP_SIZE // 2, cy + CROP_SIZE // 2),
                          slice(cx - CROP_SIZE // 2, cx + CROP_SIZE // 2))
                rows.append(np.concatenate([
                    photo[window], overlay(photo[window], bilinear[window]),
                    overlay(photo[window], guided[window]),
                    overlay(photo[window], rect_mask[window]),
                    overlay(photo[window], matte[window]), matte_view(matte[window])], axis=1))
            if rows:
                Image.fromarray(np.concatenate(rows, axis=0)).save(
                    out_dir / f"{stem}__{name}__photo_bilinear_guided.png")
            if rectangle is not None:
                view = (slice(py0, py1), slice(px0, px1))
                pair = np.concatenate([overlay(photo[view], bilinear[view]),
                                       overlay(photo[view], matte[view])], axis=1)
                factor = min(1.0, SUBJECT_VIEW_AXIS / max(pair.shape[:2]))
                pair_image = Image.fromarray(pair)
                if factor < 1.0:
                    pair_image = pair_image.resize(
                        (round(pair.shape[1] * factor), round(pair.shape[0] * factor)),
                        Image.Resampling.LANCZOS)
                pair_image.save(out_dir / f"{stem}__{name}__subject.jpg", quality=90)
            print(f"[refine] {entry['file']} {name}: bilinear {bilinear_ms:.1f} ms, guided "
                  f"{guided_ms:.1f} ms, changed pixels {changed * 100:.2f}%, rectangle IoU "
                  f"{result.get('rectangle_iou', float('nan')):.3f}, matte "
                  f"{result.get('matte_ms', float('nan')):.0f} ms")
    (out_dir / "refine_timings.json").write_text(json.dumps(
        {"radius": args.radius, "eps": args.eps, "matte": args.matte,
         "matte_radius_cells": args.matte_radius_cells, "matte_eps": args.matte_eps,
         "matte_subsample": args.matte_subsample, "smooth_radius": args.smooth_radius,
         "min_contrast": args.min_contrast, "band_cells": args.band_cells,
         "clean_cells": args.clean_cells,
         "results": timings}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
