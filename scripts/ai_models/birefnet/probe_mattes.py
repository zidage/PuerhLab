#!/usr/bin/env python3
"""Quality metrics of the BiRefNet subject mattes on the quality photo set (#458, step E2).

The script reads the alpha files of compare_onnx_outputs.py (`alpha/<stem>.npy`, ONNX Runtime
CPU output at the model size) and reports per photo:

- foreground: share of pixels with alpha > 0.5;
- soft: share of pixels with 0.05 < alpha < 0.95;
- edge width: soft pixels divided by the boundary pixels of the alpha > 0.5 mask, in model pixels.
  A hard segmentation gives about 1 to 2; a soft matte on hair gives more;
- components: connected parts of the alpha > 0.5 mask with at least 1 % of the foreground area;
- island and hole fraction (plan section 8.16.1), relative to the foreground area;
- subject recall: share of the SAM 2.1 box mask of the photo (the E1 reference mask of the box
  prompt, the subject that the quality set names) that is inside the foreground;
- outside box: share of the foreground outside the box of that prompt (other subjects or
  background that the model includes);
- for each --region of the photo: the soft share and the mean alpha in that rectangle (hair and
  fur areas).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SOFT_LOW = 0.05
SOFT_HIGH = 0.95
COMPONENT_SHARE = 0.01


def parse_region(text: str) -> tuple[str, str, tuple[float, float, float, float]]:
    """`DSC00674.jpg:hair:0.3,0.0,0.8,0.4` -> (file, name, box)."""
    file, name, box = text.split(":", 2)
    values = tuple(float(v) for v in box.split(","))
    if len(values) != 4:
        raise ValueError(f"region {text!r} needs four box values")
    return file, name, values


def holes_and_islands(mask: np.ndarray) -> tuple[float, float, int]:
    area = int(mask.sum())
    if area == 0:
        return 0.0, 0.0, 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    islands = float((area - areas.max()) / area)
    components = int((areas >= COMPONENT_SHARE * area).sum())
    background = (~mask).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background, 4)
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    holes = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count) if i not in border)
    return float(holes / area), islands, components


def boundary_pixels(mask: np.ndarray) -> int:
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    return int((mask & (eroded == 0)).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpha-dir", required=True, help="alpha/ of compare_onnx_outputs.py")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--subject-masks", required=True,
                        help="E1 reference masks (<stem>__<box name>__single0.png)")
    parser.add_argument("--label", required=True)
    parser.add_argument("--region", action="append", default=[], type=parse_region)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    results = []
    for entry in prompts["images"]:
        stem = Path(entry["file"]).stem
        alpha = np.load(Path(args.alpha_dir) / f"{stem}.npy").astype(np.float32)
        size = alpha.shape[0]
        mask = alpha > 0.5
        soft = (alpha > SOFT_LOW) & (alpha < SOFT_HIGH)
        holes, islands, components = holes_and_islands(mask)
        boundary = boundary_pixels(mask)
        box_prompt = entry["boxes"][0]
        subject = np.asarray(Image.open(
            Path(args.subject_masks) / f"{stem}__{box_prompt['name']}__single0.png"
        ).convert("L").resize((size, size), Image.Resampling.NEAREST)) > 127
        x0, y0, x1, y1 = (round(v * size) for v in box_prompt["box"])
        inside_box = np.zeros_like(mask)
        inside_box[y0:y1, x0:x1] = True
        result = {
            "image": entry["file"],
            "foreground": float(mask.mean()),
            "soft": float(soft.mean()),
            "edge_width_px": float(soft.sum() / boundary) if boundary else 0.0,
            "components": components,
            "hole_fraction": holes,
            "island_fraction": islands,
            "subject": box_prompt["name"],
            "subject_recall": float((mask & subject).sum() / max(subject.sum(), 1)),
            "outside_box": float((mask & ~inside_box).sum() / max(mask.sum(), 1)),
            "regions": {},
        }
        for file, name, (rx0, ry0, rx1, ry1) in args.region:
            if file != entry["file"]:
                continue
            crop = (slice(round(ry0 * size), round(ry1 * size)),
                    slice(round(rx0 * size), round(rx1 * size)))
            result["regions"][name] = {"soft": float(soft[crop].mean()),
                                       "mean_alpha": float(alpha[crop].mean())}
        results.append(result)
        print(f"[probe] {args.label} {stem}: fg {result['foreground']:.3f} "
              f"soft {result['soft']:.4f} edge {result['edge_width_px']:.1f} px "
              f"parts {components} recall {result['subject_recall']:.3f} "
              f"outside {result['outside_box']:.3f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": args.label, "soft_range": [SOFT_LOW, SOFT_HIGH],
                               "results": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
