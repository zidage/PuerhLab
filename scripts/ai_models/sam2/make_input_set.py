#!/usr/bin/env python3
"""Scale the quality photos and the E1 reference masks to the application input size (#458, E4).

The application gives the segmentation model at most a 4096 px long edge (plan section 9.3, SM4,
and `kMaximumRasterMaskAxis`, P3). The E4 measurements therefore read the photos and the reference
masks at that size, not at the camera size. The script processes one file at a time:

  <out>/photos/<file>.jpg            photo with the long edge scaled to at most --max-edge (Lanczos)
  <out>/reference_masks/<file>.png   E1 reference mask scaled to the size of its photo
                                     (area average, then threshold 127)

Photos that already fit are copied without a change.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def scaled_size(width: int, height: int, max_edge: int) -> tuple[int, int]:
    scale = min(1.0, max_edge / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--reference-masks", required=True)
    parser.add_argument("--max-edge", type=int, default=4096)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    (out / "photos").mkdir(parents=True, exist_ok=True)
    (out / "reference_masks").mkdir(parents=True, exist_ok=True)
    sizes = {}
    for path in sorted(Path(args.photos).glob("*.jpg")):
        with Image.open(path) as image:
            original = image.size
            size = scaled_size(*original, args.max_edge)
            target = out / "photos" / path.name
            if size == original:
                shutil.copyfile(path, target)
            else:
                image.convert("RGB").resize(size, Image.Resampling.LANCZOS).save(
                    target, quality=95)
        sizes[path.stem] = {"original": list(original), "input": list(size)}
        print(f"[photo] {path.name}: {original[0]}x{original[1]} -> {size[0]}x{size[1]}")

    masks = 0
    for path in sorted(Path(args.reference_masks).glob("*.png")):
        stem = path.name.split("__", 1)[0]
        if stem not in sizes:
            continue
        with Image.open(path) as mask:
            resized = mask.convert("L").resize(tuple(sizes[stem]["input"]),
                                               Image.Resampling.BOX)
        resized.point(lambda v: 255 if v > 127 else 0).save(out / "reference_masks" / path.name)
        masks += 1
    (out / "input_set.json").write_text(json.dumps(
        {"max_edge": args.max_edge, "photos": sizes, "reference_masks": masks}, indent=2),
        encoding="utf-8")
    print(f"[masks] {masks} reference masks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
