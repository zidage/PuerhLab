#!/usr/bin/env python3
"""Mask uncertainty and grid-hole metrics of a point and box graph pair (#458, step E1).

The script runs point_encoder.onnx and point_decoder.onnx on ONNX Runtime CPU for every point and
box prompt in --prompts and measures the mask that the overlay shows (the best multimask token for
one point, else the single-mask selection with the stability check). It works for the SAM 2.1
graphs and for the EfficientSAM3 graphs of SM1: `model_input` in export_report.json gives the
input size and the normalization, and a report without it is an EfficientSAM3 export.

Metrics per prompt (plan section 8.16.1):

  uncertain     share of low-resolution logits with |logit| < 2
  sure_fg       share of low-resolution logits > 4
  grid_energy   share of the high-pass logit spectrum on the lines of the embedding cell period
                (harmonics of the image_embed size), or on --grid-lines
  hole_frac     enclosed background holes / mask area, at a 900 px long edge
  island_frac   foreground outside the largest component / mask area, at a 900 px long edge

With --logit-dir the script also writes a color image of the selected logits for every prompt.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

STABILITY_DELTA = 0.05
STABILITY_THRESHOLD = 0.98
LOGIT_LIMIT = 32.0
VIEW_EDGE = 900
ESAM3_INPUT = {"image_size": 1008, "low_res_mask_size": 288,
               "pixel_mean": [0.5, 0.5, 0.5], "pixel_std": [0.5, 0.5, 0.5]}


def preprocess(image: Image.Image, spec: dict) -> np.ndarray:
    """Stretch to the square input (bilinear with antialias), then normalize. NCHW float32."""
    size = spec["image_size"]
    tensor = TF.to_tensor(image)
    tensor = TF.resize(tensor, [size, size], antialias=True)
    tensor = TF.normalize(tensor, spec["pixel_mean"], spec["pixel_std"])
    return tensor.unsqueeze(0).numpy()


def prompt_sets(entry: dict) -> list[dict]:
    sets = [{"name": p["name"], "points": p["points"], "box": None} for p in entry.get("points", [])]
    sets += [{"name": b["name"], "points": [], "box": b["box"]} for b in entry.get("boxes", [])]
    return sets


def model_prompt(prompt: dict, size: int):
    coords, labels = [], []
    if prompt["box"] is not None:
        x0, y0, x1, y1 = prompt["box"]
        coords += [[x0 * size, y0 * size], [x1 * size, y1 * size]]
        labels += [2.0, 3.0]
    coords += [[p[0] * size, p[1] * size] for p in prompt["points"]]
    labels += [float(p[2]) for p in prompt["points"]]
    return np.array([coords], np.float32), np.array([labels], np.float32)


def select_overlay_token(logits: np.ndarray, scores: np.ndarray, multimask: bool) -> int:
    if multimask:
        return 1 + int(np.argmax(scores[1:]))
    single = logits[0]
    outer = (single > -STABILITY_DELTA).sum()
    stability = (single > STABILITY_DELTA).sum() / outer if outer > 0 else 1.0
    return 0 if stability >= STABILITY_THRESHOLD else 1 + int(np.argmax(scores[1:]))


def grid_energy(low: np.ndarray, frequencies: list[int]) -> float:
    """Share of the high-pass spectrum energy on the given row and column frequencies."""
    side = low.shape[0]
    x = np.tanh(low / 4)
    x = x - cv2.GaussianBlur(x, (0, 0), 6)
    power = np.abs(np.fft.fft2(x)) ** 2
    total = power.sum() + 1e-9
    f = np.fft.fftfreq(side) * side
    lines = np.zeros_like(power, bool)
    for k in frequencies:
        selected = np.isclose(np.abs(f), k, atol=1)
        lines[selected, :] = True
        lines[:, selected] = True
    return float(power[lines].sum() / total)


def holes_and_islands(mask: np.ndarray):
    area = max(int(mask.sum()), 1)
    height, width = mask.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), connectivity=8)
    hole = sum(stats[i, cv2.CC_STAT_AREA] for i in range(1, count)
               if stats[i, 0] > 0 and stats[i, 1] > 0
               and stats[i, 0] + stats[i, 2] < width and stats[i, 1] + stats[i, 3] < height)
    count2, _, stats2, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    largest = stats2[1:, cv2.CC_STAT_AREA].max() if count2 > 1 else 0
    return hole / area, (area - largest) / area, count - 1, count2 - 1


def logit_image(low: np.ndarray) -> np.ndarray:
    """Logits as color: blue negative, white near 0, red positive (saturated at |logit| 8)."""
    v = np.clip(low / 8.0, -1.0, 1.0)
    rgb = np.ones(low.shape + (3,), np.float32)
    pos, neg = np.clip(v, 0, 1), np.clip(-v, 0, 1)
    rgb[..., 1] -= pos + neg
    rgb[..., 2] -= pos
    rgb[..., 0] -= neg
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--label", required=True, help="model name for the report")
    parser.add_argument("--photos", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--grid-lines", help="comma-separated spectrum frequencies; default: "
                                             "harmonics of the image_embed size")
    parser.add_argument("--logit-dir", help="write the selected logits as images here")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import onnxruntime as ort

    onnx_dir = Path(args.onnx_dir)
    export_report = json.loads((onnx_dir / "export_report.json").read_text(encoding="utf-8"))
    spec = export_report.get("model_input") or ESAM3_INPUT
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    encoder = ort.InferenceSession(str(onnx_dir / "point_encoder.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(onnx_dir / "point_decoder.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    low_side = spec["low_res_mask_size"]
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    logit_dir = Path(args.logit_dir) if args.logit_dir else None
    if logit_dir:
        logit_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    frequencies = None
    for entry in prompts["images"]:
        image = Image.open(Path(args.photos) / entry["file"]).convert("RGB")
        width, height = image.size
        scale = VIEW_EDGE / max(width, height)
        view_size = (round(width * scale), round(height * scale))
        started = time.perf_counter()
        image_embed, feats_0, feats_1 = encoder.run(None, {"pixel_values": preprocess(image, spec)})
        encode_ms = (time.perf_counter() - started) * 1000
        if frequencies is None:
            if args.grid_lines:
                frequencies = [int(v) for v in args.grid_lines.split(",")]
            else:
                cell = image_embed.shape[-1]
                frequencies = list(range(cell, low_side // 2 + 1, cell))
        for prompt in prompt_sets(entry):
            coords, labels = model_prompt(prompt, spec["image_size"])
            logits, scores, object_score = decoder.run(None, {
                "image_embed": image_embed, "high_res_feats_0": feats_0,
                "high_res_feats_1": feats_1, "point_coords": coords, "point_labels": labels,
                "mask_input": np.zeros((1, 1, low_side, low_side), np.float32),
                "has_mask_input": np.zeros(1, np.float32),
            })
            multimask = prompt["box"] is None and len(prompt["points"]) == 1
            token = select_overlay_token(logits[0], scores[0], multimask)
            low = np.clip(logits[0][token], -LOGIT_LIMIT, LOGIT_LIMIT)
            view = cv2.resize(low, view_size, interpolation=cv2.INTER_LINEAR) > 0
            hole, island, hole_count, components = holes_and_islands(view)
            row = {
                "image": entry["file"], "prompt": prompt["name"], "token": token,
                "score": round(float(scores[0][token]), 4),
                "object_score_logit": round(float(object_score[0][0]), 3),
                "uncertain": round(float((np.abs(low) < 2).mean()), 4),
                "sure_fg": round(float((low > 4).mean()), 4),
                "grid_energy": round(grid_energy(low, frequencies), 4),
                "hole_frac": round(float(hole), 4), "island_frac": round(float(island), 4),
                "holes": int(hole_count), "components": int(components),
                "mask_fraction": round(float(view.mean()), 4),
                "encoder_cpu_ms": round(encode_ms),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            if logit_dir:
                stem = Path(entry["file"]).stem
                Image.fromarray(logit_image(low)).save(logit_dir / f"{stem}__{prompt['name']}.png")

    report = {"label": args.label, "onnx_dir": str(onnx_dir), "model_input": spec,
              "grid_frequencies": frequencies, "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[probe] {len(rows)} prompts; report {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
