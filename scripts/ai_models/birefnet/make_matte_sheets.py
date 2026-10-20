#!/usr/bin/env python3
"""Build the comparison sheets of the E2 report from the matte images (#458, step E2).

Each --model is `label=dir`. A matte directory holds the `<stem>__subject.png` (overlay) and
`<stem>__alpha.png` files of `segmentation_runtime_report matte --overlay-dir`. A reference
directory (`--reference label=dir:prompt-kind`) holds the E1 overlays `<stem>__<prompt>.png`; the
sheet shows the overlay of the first box prompt of the photo.

Output:
  <out-dir>/overview.jpg           one row per photo, one column per model (overlays)
  <out-dir>/<stem>.jpg             overlays and alpha mattes of one photo
  <out-dir>/<crop sheet>_alpha.jpg     detail crops of the alpha mattes
  <out-dir>/<crop sheet>_composite.jpg detail crops of the photo composited on a flat color
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HEADER = 26
LABEL_WIDTH = 150
BACKGROUND = (24, 24, 24)
TEXT = (235, 235, 235)
COMPOSITE_COLOR = (40, 170, 90)


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


def grid(rows: list[tuple[str, list[Image.Image | None]]], columns: list[str],
         tile: tuple[int, int], title: str) -> Image.Image:
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
                draw.text((x + 8, y + 8), "-", fill=(160, 160, 160), font=font(13))
                continue
            sheet.paste(image, (x + (tw - image.width) // 2, y + (th - image.height) // 2))
    return sheet


def open_rgb(path: Path) -> Image.Image | None:
    return Image.open(path).convert("RGB") if path.exists() else None


def crop(image: Image.Image, box: tuple[float, ...]) -> Image.Image:
    w, h = image.size
    x0, y0, x1, y1 = box
    return image.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))


def composite(photo: Image.Image, alpha: Image.Image) -> Image.Image:
    rgb = np.asarray(photo.resize(alpha.size, Image.Resampling.LANCZOS), np.float32)
    a = np.asarray(alpha.convert("L"), np.float32)[..., None] / 255.0
    flat = np.array(COMPOSITE_COLOR, np.float32)
    return Image.fromarray(np.round(rgb * a + flat * (1.0 - a)).astype(np.uint8))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="label=matte dir")
    parser.add_argument("--reference", help="label=E1 overlay dir (box prompt overlays)")
    parser.add_argument("--photos", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--crop", action="append", default=[],
                        help="<file>:<name>:x0,y0,x1,y1 (normalized) for the detail sheets")
    parser.add_argument("--crop-sheet-name", default="hair_detail")
    parser.add_argument("--tile-edge", type=int, default=320)
    parser.add_argument("--overview-edge", type=int, default=220)
    parser.add_argument("--crop-edge", type=int, default=360)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    models = [(label, Path(path)) for label, path in (m.split("=", 1) for m in args.model)]
    reference = tuple(args.reference.split("=", 1)) if args.reference else None
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [label for label, _ in models] + ([reference[0]] if reference else [])

    overview_rows, overview_tile = [], (0, 0)
    for entry in prompts["images"]:
        stem = Path(entry["file"]).stem
        overlays = [open_rgb(d / f"{stem}__subject.png") for _, d in models]
        alphas = [open_rgb(d / f"{stem}__alpha.png") for _, d in models]
        if reference:
            box = entry["boxes"][0]["name"]
            overlays.append(open_rgb(Path(reference[1]) / f"{stem}__{box}.png"))
            alphas.append(None)
        tiles = [[fit(i, args.tile_edge) if i else None for i in row] for row in (overlays, alphas)]
        tile = tuple(max((getattr(i, attr) for row in tiles for i in row if i), default=1)
                     for attr in ("width", "height"))
        sheet = grid([("overlay", tiles[0]), ("alpha", tiles[1])], columns, tile,
                     f"{entry['file']}  ({prompts.get('cases', {}).get(entry['file'], '')})")
        sheet.save(out_dir / f"{stem}.jpg", quality=args.quality, optimize=True)
        print(f"[sheet] {stem}.jpg {sheet.size}")
        small = [fit(i, args.overview_edge) if i else None for i in overlays]
        overview_tile = (max([overview_tile[0]] + [i.width for i in small if i]),
                         max([overview_tile[1]] + [i.height for i in small if i]))
        overview_rows.append((stem, small))
    sheet = grid(overview_rows, columns, overview_tile, "Subject mattes on the quality set")
    sheet.save(out_dir / "overview.jpg", quality=args.quality, optimize=True)
    print(f"[sheet] overview.jpg {sheet.size}")

    if args.crop:
        alpha_rows, composite_rows, tile = [], [], (0, 0)
        for item in args.crop:
            file, name, box_text = item.split(":", 2)
            box = tuple(float(v) for v in box_text.split(","))
            stem = Path(file).stem
            photo = Image.open(Path(args.photos) / file).convert("RGB")
            alpha_tiles, composite_tiles = [], []
            for _, directory in models:
                alpha = open_rgb(directory / f"{stem}__alpha.png")
                if alpha is None:
                    alpha_tiles.append(None)
                    composite_tiles.append(None)
                    continue
                alpha_tiles.append(fit(crop(alpha, box), args.crop_edge))
                composite_tiles.append(fit(crop(composite(photo, alpha), box), args.crop_edge))
            if reference:
                alpha_tiles.append(fit(crop(photo, box), args.crop_edge))
                composite_tiles.append(alpha_tiles[-1])
            for image in alpha_tiles:
                if image is not None:
                    tile = (max(tile[0], image.width), max(tile[1], image.height))
            alpha_rows.append((f"{stem}\n{name}", alpha_tiles))
            composite_rows.append((f"{stem}\n{name}", composite_tiles))
        crop_columns = [label for label, _ in models] + (["photo"] if reference else [])
        for rows, kind, title in (
                (alpha_rows, "alpha", "Detail crops of the alpha mattes (2048 px long edge)"),
                (composite_rows, "composite",
                 "Detail crops: photo x alpha on a flat green background (2048 px long edge)")):
            sheet = grid(rows, crop_columns, tile, title)
            sheet.save(out_dir / f"{args.crop_sheet_name}_{kind}.jpg", quality=args.quality,
                       optimize=True)
            print(f"[sheet] {args.crop_sheet_name}_{kind}.jpg {sheet.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
