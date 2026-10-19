#!/usr/bin/env python3
"""Compare encoder preprocessing for the EfficientSAM3 point and box masks.

For every point and box prompt in --prompts, the script runs point_encoder.onnx and
point_decoder.onnx on ONNX Runtime CPU with three pipelines and selects one mask as
refine_edges_prototype.py does (the best multimask token for one point, the single-mask selection
otherwise):

  stretch:   the photo is stretched to 1008 x 1008 without padding (Sam3Processor.set_image, the
             SM1 pipeline);
  letterbox: the long edge is scaled to 1008, the aspect ratio is kept, and the rest of the square
             is gray (normalized value 0) on the right or the bottom;
  dual:      both encoder passes; the mask with the higher decoder IoU score, then holes and
             islands smaller than --clean-cells^2 logit cells are removed.

The decoder IoU score of the selected token is the score of each pipeline. Per photo it writes
<stem>.jpg with one row per prompt (stretch, letterbox, dual), and compare_preprocessing.json with
the scores, the mask fractions, and the dual choice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import esam3_graphs as graphs  # noqa: E402

LOW_RES = 288
VIEW_AXIS = 640
OVERLAY_COLOR = np.array([30, 144, 255], np.float32)


def encoder_input(image: Image.Image, letterbox: bool) -> tuple[np.ndarray, float, float]:
    """Encoder pixels and the scale (sx, sy) from normalized photo to normalized model
    coordinates."""
    size = graphs.IMAGE_SIZE
    if letterbox:
        scale = size / max(image.size)
        width, height = round(image.width * scale), round(image.height * scale)
        canvas = Image.new("RGB", (size, size), (128, 128, 128))
        canvas.paste(image.resize((width, height), Image.Resampling.BILINEAR), (0, 0))
        sx, sy = width / size, height / size
    else:
        canvas = image.resize((size, size), Image.Resampling.BILINEAR)
        sx = sy = 1.0
    pixels = (np.asarray(canvas, np.float32) / 255.0 - 0.5) / 0.5
    return pixels.transpose(2, 0, 1)[None], sx, sy


def decode(decoder, features, points, box, sx, sy) -> tuple[np.ndarray, float]:
    size = graphs.IMAGE_SIZE
    coords, labels = [], []
    if box is not None:
        coords += [[box[0] * sx * size, box[1] * sy * size], [box[2] * sx * size, box[3] * sy * size]]
        labels += [2.0, 3.0]
    for x, y, label in points or []:
        coords.append([x * sx * size, y * sy * size])
        labels.append(float(label))
    image_embed, feats_0, feats_1 = features
    logits, scores, _ = decoder.run(None, {
        "image_embed": image_embed, "high_res_feats_0": feats_0, "high_res_feats_1": feats_1,
        "point_coords": np.array([coords], np.float32),
        "point_labels": np.array([labels], np.float32),
        "mask_input": np.zeros((1, 1, LOW_RES, LOW_RES), np.float32),
        "has_mask_input": np.zeros(1, np.float32),
    })
    logits, scores = logits[0], scores[0]
    best = 1 + int(np.argmax(scores[1:]))
    if points is not None and len(points) == 1 and box is None:
        return logits[best], float(scores[best])
    single = logits[0]
    inner, outer = (single > 0.05).sum(), (single > -0.05).sum()
    if outer == 0 or inner / outer >= 0.98:
        return single, float(scores[0])
    return logits[best], float(scores[best])


def to_mask(logits: np.ndarray, sx: float, sy: float, width: int, height: int) -> np.ndarray:
    region = logits[:max(1, round(sy * LOW_RES)), :max(1, round(sx * LOW_RES))]
    return cv2.resize(np.clip(region, -32, 32), (width, height),
                      interpolation=cv2.INTER_LINEAR) > 0


def clean_mask(mask: np.ndarray, min_area: float) -> np.ndarray:
    out = mask.copy()
    for fill in (True, False):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (~out if fill else out).astype(np.uint8), connectivity=8)
        small = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] < min_area]
        if small:
            out[np.isin(labels, small)] = fill
    return out


def view(photo: np.ndarray, mask: np.ndarray, title: str) -> Image.Image:
    result = photo.astype(np.float32)
    result[mask] = result[mask] * 0.45 + OVERLAY_COLOR * 0.55
    image = Image.fromarray(result.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 330, 16], fill=(0, 0, 0))
    draw.text((4, 3), title, fill=(255, 255, 255))
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clean-cells", type=float, default=3.0)
    args = parser.parse_args()

    import onnxruntime as ort

    onnx_dir, out_dir = Path(args.onnx_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = ort.InferenceSession(str(onnx_dir / "point_encoder.onnx"),
                                   providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(onnx_dir / "point_decoder.onnx"),
                                   providers=["CPUExecutionProvider"])
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    results = []
    for entry in prompts["images"]:
        image = Image.open(Path(args.photos) / entry["file"]).convert("RGB")
        scale = VIEW_AXIS / max(image.size)
        width, height = round(image.width * scale), round(image.height * scale)
        photo = np.asarray(image.resize((width, height), Image.Resampling.LANCZOS))
        encoded = {}
        for name, letterbox in (("stretch", False), ("letterbox", True)):
            pixels, sx, sy = encoder_input(image, letterbox)
            encoded[name] = (encoder.run(None, {"pixel_values": pixels}), sx, sy)
        prompt_sets = [(p["name"], p["points"], None) for p in entry.get("points", [])]
        prompt_sets += [(b["name"], None, b["box"]) for b in entry.get("boxes", [])]
        rows = []
        for name, points, box in prompt_sets:
            record = {"image": entry["file"], "prompt": name}
            masks, tiles = {}, []
            for pipeline, (features, sx, sy) in encoded.items():
                logits, score = decode(decoder, features, points, box, sx, sy)
                masks[pipeline] = (to_mask(logits, sx, sy, width, height), score, logits, sx, sy)
                record[pipeline] = {"score": score,
                                    "mask_fraction": float(masks[pipeline][0].mean())}
                tiles.append(view(photo, masks[pipeline][0], f"{pipeline} score {score:.2f}"))
            choice = max(masks, key=lambda k: masks[k][1])
            _, score, logits, sx, sy = masks[choice]
            cell = max(width, height) / LOW_RES
            dual = clean_mask(masks[choice][0], (args.clean_cells * cell) ** 2)
            record["dual"] = {"choice": choice, "score": score, "mask_fraction": float(dual.mean())}
            tiles.append(view(photo, dual, f"dual ({choice}) + clean"))
            results.append(record)
            row = Image.new("RGB", (sum(t.width for t in tiles), height))
            x = 0
            for tile in tiles:
                row.paste(tile, (x, 0))
                x += tile.width
            ImageDraw.Draw(row).text((4, 20), name, fill=(255, 255, 0))
            rows.append(row)
            print(f"[preprocess] {entry['file']} {name}: stretch {record['stretch']['score']:.2f}, "
                  f"letterbox {record['letterbox']['score']:.2f}, dual -> {choice}")
        sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)))
        y = 0
        for row in rows:
            sheet.paste(row, (0, y))
            y += row.height
        sheet.save(out_dir / f"{Path(entry['file']).stem}.jpg", quality=88)
    (out_dir / "compare_preprocessing.json").write_text(
        json.dumps({"clean_cells": args.clean_cells, "results": results}, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
