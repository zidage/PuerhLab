#!/usr/bin/env python3
"""Build model comparison sheets from overlay and logit images (#458, step E1).

Each --model is `label=overlay_dir` or `label=overlay_dir|logit_dir`. The overlay directories hold
the `<stem>__<prompt>.png` files of `segmentation_runtime_report --overlay-dir`; the logit
directories hold the files of `probe_masks.py --logit-dir`.

Output:
  <out-dir>/<stem>.jpg          one row per prompt, one column per model (overlays)
  <out-dir>/logits_<name>.jpg   one row per --logit-prompt, one column per model (logits)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HEADER = 26
LABEL_WIDTH = 150
BACKGROUND = (24, 24, 24)
TEXT = (235, 235, 235)


def font(size: int):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit(image: Image.Image, edge: int) -> Image.Image:
    scale = edge / max(image.size)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                        Image.Resampling.LANCZOS)


def grid(rows: list[tuple[str, list[Image.Image | None]]], columns: list[str], tile: tuple[int, int],
         title: str) -> Image.Image:
    tw, th = tile
    sheet = Image.new("RGB", (LABEL_WIDTH + tw * len(columns), HEADER * 2 + th * len(rows)),
                      BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 4), title, fill=TEXT, font=font(16))
    for c, name in enumerate(columns):
        draw.text((LABEL_WIDTH + c * tw + 6, HEADER + 4), name, fill=TEXT, font=font(14))
    for r, (label, images) in enumerate(rows):
        y = HEADER * 2 + r * th
        draw.text((8, y + 6), label, fill=TEXT, font=font(14))
        for c, image in enumerate(images):
            x = LABEL_WIDTH + c * tw
            if image is None:
                draw.text((x + 8, y + 8), "not available", fill=(160, 160, 160), font=font(13))
                continue
            sheet.paste(image, (x + (tw - image.width) // 2, y + (th - image.height) // 2))
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--images", help="comma-separated photo files; default: all")
    parser.add_argument("--logit-prompt", action="append", default=[],
                        help="<file>:<prompt> for the logit sheet")
    parser.add_argument("--logit-sheet-name", default="key_prompts")
    parser.add_argument("--crop", action="append", default=[],
                        help="<file>:<prompt>:x0,y0,x1,y1 (normalized) for the detail sheet")
    parser.add_argument("--crop-sheet-name", default="edge_detail")
    parser.add_argument("--tile-edge", type=int, default=320)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    models = []
    for item in args.model:
        label, paths = item.split("=", 1)
        overlay, _, logits = paths.partition("|")
        models.append((label, Path(overlay), Path(logits) if logits else None))
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    wanted = set(args.images.split(",")) if args.images else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [m[0] for m in models]

    for entry in prompts["images"]:
        if wanted is not None and entry["file"] not in wanted:
            continue
        stem = Path(entry["file"]).stem
        names = [p["name"] for p in entry.get("points", [])] + [b["name"] for b in entry.get("boxes", [])]
        rows, tile = [], (0, 0)
        for name in names:
            images = []
            for _, overlay_dir, _ in models:
                path = overlay_dir / f"{stem}__{name}.png"
                image = fit(Image.open(path).convert("RGB"), args.tile_edge) if path.exists() else None
                if image is not None:
                    tile = (max(tile[0], image.width), max(tile[1], image.height))
                images.append(image)
            rows.append((name, images))
        sheet = grid(rows, columns, tile, f"{entry['file']}  ({prompts.get('cases', {}).get(entry['file'], '')})")
        sheet.save(out_dir / f"{stem}.jpg", quality=args.quality, optimize=True)
        print(f"[sheet] {stem}.jpg {sheet.size}")

    if args.logit_prompt:
        rows, tile = [], (0, 0)
        for item in args.logit_prompt:
            file, prompt = item.split(":", 1)
            stem = Path(file).stem
            images = []
            for _, _, logit_dir in models:
                path = logit_dir / f"{stem}__{prompt}.png" if logit_dir else None
                image = fit(Image.open(path).convert("RGB"), args.tile_edge) \
                    if path is not None and path.exists() else None
                if image is not None:
                    tile = (max(tile[0], image.width), max(tile[1], image.height))
                images.append(image)
            rows.append((f"{stem}\n{prompt}", images))
        sheet = grid(rows, columns, tile, "Selected low-resolution logits (blue < 0 < red, white = |logit| near 0)")
        sheet.save(out_dir / f"logits_{args.logit_sheet_name}.jpg", quality=args.quality, optimize=True)
        print(f"[sheet] logits_{args.logit_sheet_name}.jpg {sheet.size}")

    if args.crop:
        rows, tile = [], (0, 0)
        for item in args.crop:
            file, prompt, box = item.split(":", 2)
            x0, y0, x1, y1 = (float(v) for v in box.split(","))
            stem = Path(file).stem
            images = []
            for _, overlay_dir, _ in models:
                path = overlay_dir / f"{stem}__{prompt}.png"
                image = None
                if path.exists():
                    full = Image.open(path).convert("RGB")
                    w, h = full.size
                    image = fit(full.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h))),
                                args.tile_edge)
                    tile = (max(tile[0], image.width), max(tile[1], image.height))
                images.append(image)
            rows.append((f"{stem}\n{prompt}", images))
        sheet = grid(rows, columns, tile, "Detail crops of the overlays (2048 px long edge)")
        sheet.save(out_dir / f"{args.crop_sheet_name}.jpg", quality=args.quality, optimize=True)
        print(f"[sheet] {args.crop_sheet_name}.jpg {sheet.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
