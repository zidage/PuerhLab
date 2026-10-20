#!/usr/bin/env python3
"""Build the click candidate sheets of the report from the candidate overlays (#458, step E4).

Each --model is `label=report_json|overlay_dir`: the report and the --overlay-dir of
`segmentation_runtime_report candidates`. The overlay directory holds `<stem>__candidates.jpg`
(all candidates of the overlay grid), `<stem>__<prompt>__live_level<k>.jpg` (the three masks of
the live point decoder, from small to large), and `<stem>__<prompt>__level<k>.jpg` (the candidate
levels under the pointer, from small to large).

Output:
  <out-dir>/candidate_maps.jpg  one row per photo, one column per model (all candidates)
  <out-dir>/<stem>.jpg          one row per single-point prompt and model: the three live point
                                decoder levels and the candidate levels, with the area share and
                                the reference IoU under each tile. Yellow marks the largest level,
                                cyan the level with the best predicted IoU (the default).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HEADER = 26
LABEL_WIDTH = 170
CAPTION = 20
BACKGROUND = (24, 24, 24)
TEXT = (235, 235, 235)
MUTED = (160, 160, 160)
LARGEST_COLOR = (255, 210, 0)
BEST_SCORE_COLOR = (0, 220, 255)


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


def caption(level: dict) -> str:
    iou = level.get("reference_iou")
    iou_text = "—" if iou is None else f"{iou:.2f}"
    return f"area {level['area_fraction']:.3f}  IoU {iou_text}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--grid", help="grid of the levels; default: the overlay grid (largest)")
    parser.add_argument("--max-levels", type=int, default=4)
    parser.add_argument("--tile-edge", type=int, default=300)
    parser.add_argument("--quality", type=int, default=72)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    models = []
    for item in args.model:
        label, paths = item.split("=", 1)
        report_path, overlay_dir = paths.split("|", 1)
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        models.append((label, report, Path(overlay_dir)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edge = args.tile_edge

    # All candidates per photo.
    photos = [image["file"] for image in models[0][1]["images"]]
    tile_h = edge + CAPTION
    sheet = Image.new("RGB", (LABEL_WIDTH + edge * len(models),
                              HEADER * 2 + tile_h * len(photos)), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 4), "Click candidates (white outline: candidate border, grey: no candidate)",
              fill=TEXT, font=font(16))
    for c, (label, _, _) in enumerate(models):
        draw.text((LABEL_WIDTH + c * edge + 6, HEADER + 4), label, fill=TEXT, font=font(14))
    for r, file in enumerate(photos):
        y = HEADER * 2 + r * tile_h
        draw.text((8, y + 6), Path(file).stem, fill=TEXT, font=font(14))
        for c, (label, report, overlay_dir) in enumerate(models):
            grid = args.grid or str(max(report["generator"]["points_per_side"]))
            image_report = next(i for i in report["images"] if i["file"] == file)
            path = overlay_dir / f"{Path(file).stem}__candidates.jpg"
            x = LABEL_WIDTH + c * edge
            if path.exists():
                tile = fit(Image.open(path).convert("RGB"), edge)
                sheet.paste(tile, (x + (edge - tile.width) // 2, y))
            g = image_report["grids"][grid]
            draw.text((x + 6, y + edge + 2),
                      f"{g['candidates']} candidates, coverage {g['coverage']:.2f}",
                      fill=MUTED, font=font(12))
    sheet.save(out_dir / "candidate_maps.jpg", quality=args.quality)

    # Levels per prompt.
    for file in photos:
        stem = Path(file).stem
        rows = []
        for label, report, overlay_dir in models:
            grid = args.grid or str(max(report["generator"]["points_per_side"]))
            image_report = next(i for i in report["images"] if i["file"] == file)
            for prompt in image_report["prompts"]:
                rows.append((label, grid, prompt, overlay_dir))
        if not rows:
            continue
        columns = ([f"Live level {k}" for k in range(3)]
                   + [f"Candidate level {k}" for k in range(args.max_levels)])
        sheet = Image.new("RGB", (LABEL_WIDTH + edge * len(columns),
                                  HEADER * 2 + tile_h * len(rows)), BACKGROUND)
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 4), f"{stem}: highlight under the pointer (levels: small to large; "
                  "yellow: largest, cyan: best score)", fill=TEXT, font=font(16))
        for c, name in enumerate(columns):
            draw.text((LABEL_WIDTH + c * edge + 6, HEADER + 4), name, fill=TEXT, font=font(14))
        for r, (label, grid, prompt, overlay_dir) in enumerate(rows):
            y = HEADER * 2 + r * tile_h
            draw.text((8, y + 6), f"{prompt['name']}", fill=TEXT, font=font(14))
            draw.text((8, y + 26), f"{label}, {prompt['intent']}", fill=MUTED, font=font(12))
            live = prompt["live_point_decoder"]
            live_default = live["default_best_score"]["level"]
            tiles = []
            for k, level in enumerate(live["levels"]):
                marks = []
                if k == len(live["levels"]) - 1:
                    marks.append(LARGEST_COLOR)
                if k == live_default:
                    marks.append(BEST_SCORE_COLOR)
                tiles.append((overlay_dir / f"{stem}__{prompt['name']}__live_level{k}.jpg",
                              caption(level), marks))
            candidates = prompt["candidates"][grid]
            defaults = candidates["defaults"]
            for level in candidates["levels"][:args.max_levels]:
                marks = []
                if defaults["largest"]["level"] == level["level"]:
                    marks.append(LARGEST_COLOR)
                if defaults["best_score"]["level"] == level["level"]:
                    marks.append(BEST_SCORE_COLOR)
                tiles.append((overlay_dir / f"{stem}__{prompt['name']}__level{level['level']}.jpg",
                              caption(level), marks))
            if not candidates["levels"]:
                tiles.append((None, "no candidate under the pointer", None))
            elif len(candidates["levels"]) > args.max_levels:
                draw.text((8, y + 46),
                          f"{len(candidates['levels']) - args.max_levels} more candidate levels",
                          fill=MUTED, font=font(11))
            for c, (path, text, marks) in enumerate(tiles):
                x = LABEL_WIDTH + c * edge
                if path is not None and path.exists():
                    tile = fit(Image.open(path).convert("RGB"), edge - 6)
                    tx, ty = x + (edge - tile.width) // 2, y + 3
                    sheet.paste(tile, (tx, ty))
                    for i, color in enumerate(marks or []):
                        inset = i * 3
                        draw.rectangle((tx - 3 + inset, ty - 3 + inset,
                                        tx + tile.width + 2 - inset, ty + tile.height + 2 - inset),
                                       outline=color, width=3)
                draw.text((x + 6, y + edge + 2), text, fill=MUTED, font=font(12))
        sheet.save(out_dir / f"{stem}.jpg", quality=args.quality)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
