#!/usr/bin/env python3
"""Reference masks for the port check of the Alcedo Studio segmentation service (SM3.A.2).

The Rust test `rust/puerh_mind/tests/segmentation_port_check.rs` runs the sidecar segmentation
engine on the same pixels and prompts and compares its masks with the masks of this script.

For each photo of --photos (8-bit RGB, long edge at most 4096 px; the lab input set
`build/tmp/explore/e4/input_4096/photos`), the script:

1. writes the exact RGB8 pixels that it uses (`<photo>.rgb`, row-major), so that both sides read
   the same values and no JPEG decoder difference enters the comparison;
2. preprocesses them as the lab MVP does (torchvision `ToTensor`, `Resize((1024, 1024),
   antialias=True)`, ImageNet `Normalize`) and runs `point_encoder.onnx` once on ONNX Runtime CPU;
3. runs `point_decoder.onnx` for each prompt and applies the MVP selection rules (master plan
   section 8.17.3): a click returns tokens 1 to 3 sorted by area with the best-scored token
   selected; another prompt returns token 0 when its stability is at least 0.98, else the
   best-scored token 1 to 3; logits clipped to +-32.

Prompts per photo (normalized coordinates of the lab quality prompts,
`build/tmp/sm1/quality/prompts.json`): every point set (one point is a click), every box, a
4-point stroke across the center of the first box, and a subtract click inside the first box with
the selected box logits as `mask_input`.

Output (--out): `manifest.json` and little-endian float32 files (`*.f32`) of the mask input and
the logits of every returned mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageOps
from torchvision.transforms import Normalize, Resize, ToTensor

IMAGE_SIZE = 1024
LOGIT_SIZE = 256
LOGIT_LIMIT = 32.0
STABILITY_DELTA = 0.05
STABILITY_THRESHOLD = 0.98
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rgb8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(rgb, dtype=np.uint8).copy()


def preprocess(rgb8: np.ndarray) -> np.ndarray:
    tensor = ToTensor()(rgb8)
    tensor = Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)(tensor)
    tensor = Normalize(list(MEAN), list(STD))(tensor)
    return tensor.unsqueeze(0).numpy().astype(np.float32)


def model_prompt(points: list[list[float]], box: list[float] | None):
    coords, labels = [], []
    if box is not None:
        x0, y0, x1, y1 = box
        coords += [[min(x0, x1) * IMAGE_SIZE, min(y0, y1) * IMAGE_SIZE],
                   [max(x0, x1) * IMAGE_SIZE, max(y0, y1) * IMAGE_SIZE]]
        labels += [2.0, 3.0]
    coords += [[x * IMAGE_SIZE, y * IMAGE_SIZE] for x, y, _ in points]
    labels += [float(label) for _, _, label in points]
    return np.array([coords], np.float32), np.array([labels], np.float32)


def level(logits: np.ndarray, scores: np.ndarray, token: int) -> dict:
    clipped = np.clip(logits[token], -LOGIT_LIMIT, LOGIT_LIMIT).astype(np.float32)
    return {"token": token, "score": float(scores[token]), "area": int((clipped > 0).sum()),
            "logits": clipped}


def select_masks(logits: np.ndarray, scores: np.ndarray, is_click: bool):
    if is_click:
        best = 1 + int(np.argmax(scores[1:]))
        levels = sorted((level(logits, scores, token) for token in (1, 2, 3)),
                        key=lambda item: (item["area"], item["token"]))
        return levels, next(i for i, item in enumerate(levels) if item["token"] == best)
    first = logits[0]
    inner = int((first > STABILITY_DELTA).sum())
    outer = int((first > -STABILITY_DELTA).sum())
    stability = inner / outer if outer > 0 else 1.0
    token = 0 if stability >= STABILITY_THRESHOLD else 1 + int(np.argmax(scores[1:]))
    return [level(logits, scores, token)], 0


def prompts_for(entry: dict) -> list[dict]:
    prompts = [{"name": f"points_{item['name']}", "points": item["points"], "box": None}
               for item in entry.get("points", [])]
    prompts += [{"name": f"box_{item['name']}", "points": [], "box": item["box"]}
                for item in entry.get("boxes", [])]
    if entry.get("boxes"):
        x0, y0, x1, y1 = entry["boxes"][0]["box"]
        cy = (y0 + y1) / 2
        stroke = [[x0 + (x1 - x0) * (0.4 + 0.2 * k / 3), cy, 1] for k in range(4)]
        prompts.append({"name": "stroke_box_center", "points": stroke, "box": None})
        prompts.append({"name": "box_then_subtract", "points": [[(x0 + x1) / 2, y0 + 0.1 * (y1 - y0), 0]],
                        "box": entry["boxes"][0]["box"], "mask_input_from": f"box_{entry['boxes'][0]['name']}"})
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--onnx-dir", type=Path, required=True,
                        help="directory with point_encoder.onnx and point_decoder.onnx")
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    encoder_path = args.onnx_dir / "point_encoder.onnx"
    decoder_path = args.onnx_dir / "point_decoder.onnx"
    encoder = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])
    entries = {entry["file"]: entry for entry in json.loads(args.prompts.read_text())["images"]}
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {"onnxruntime": ort.__version__, "torch": torch.__version__,
                "encoder_sha256": file_sha256(encoder_path),
                "decoder_sha256": file_sha256(decoder_path), "photos": []}
    for photo_path in sorted(args.photos.iterdir()):
        entry = entries.get(photo_path.name)
        if entry is None:
            continue
        stem = photo_path.stem
        rgb8 = read_rgb8(photo_path)
        height, width = rgb8.shape[:2]
        (args.out / f"{stem}.rgb").write_bytes(rgb8.tobytes())
        image_embed, feats_0, feats_1 = encoder.run(None, {"pixel_values": preprocess(rgb8)})
        selected_by_name: dict[str, np.ndarray] = {}
        photo = {"name": stem, "width": int(width), "height": int(height),
                 "pixels": f"{stem}.rgb", "prompts": []}
        for index, prompt in enumerate(prompts_for(entry)):
            coords, labels = model_prompt(prompt["points"], prompt["box"])
            mask_input = selected_by_name.get(prompt.get("mask_input_from", ""))
            outputs = decoder.run(None, {
                "image_embed": image_embed, "high_res_feats_0": feats_0, "high_res_feats_1": feats_1,
                "point_coords": coords, "point_labels": labels,
                "mask_input": (mask_input[None, None] if mask_input is not None
                               else np.zeros((1, 1, LOGIT_SIZE, LOGIT_SIZE), np.float32)),
                "has_mask_input": np.array([1.0 if mask_input is not None else 0.0], np.float32)})
            logits, scores = outputs[0][0], outputs[1][0]
            is_click = prompt["box"] is None and len(prompt["points"]) == 1
            levels, selected = select_masks(logits, scores, is_click)
            selected_by_name[prompt["name"]] = levels[selected]["logits"]
            record = {"name": prompt["name"], "points": prompt["points"], "box": prompt["box"],
                      "mask_input": None, "selected_index": selected, "masks": []}
            if mask_input is not None:
                record["mask_input"] = f"{stem}_{index}_mask_input.f32"
                (args.out / record["mask_input"]).write_bytes(mask_input.astype("<f4").tobytes())
            for level_index, item in enumerate(levels):
                logits_file = f"{stem}_{index}_level{level_index}.f32"
                (args.out / logits_file).write_bytes(item["logits"].astype("<f4").tobytes())
                record["masks"].append({"token": item["token"], "score": item["score"],
                                        "area": item["area"], "logits": logits_file})
            photo["prompts"].append(record)
        manifest["photos"].append(photo)
        print(f"{stem}: {width}x{height}, {len(photo['prompts'])} prompts")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
