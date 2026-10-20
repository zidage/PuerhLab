#!/usr/bin/env python3
"""Build the stroke selection sheets of the report (#458, step E3).

Each --model is `label=report_json|mask_dir`: the report and the --mask-dir of
`segmentation_runtime_report strokes`. The mask directory holds
`<stem>__<prompt>__<sequence>__step<k>.png`, the selection after interaction k of each corrective
sequence, at the evaluation size of the report.

Output: <out-dir>/<stem>.jpg, one row per single-point prompt and model. Columns: one click (the
E4 default), the first stroke (half of the inner chord), a stroke along the whole inner chord, the
half-chord stroke after the corrective strokes, the click after the corrective clicks, and the
largest mask of the click after the corrective (mostly subtract) strokes. Green marks positive points and add strokes, red negative points and subtract
strokes; the yellow line is the edge of the reference mask. Under each tile: the reference IoU and
the number of interactions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

HEADER = 26
LABEL_WIDTH = 170
CAPTION = 20
BACKGROUND = (24, 24, 24)
TEXT = (235, 235, 235)
MUTED = (160, 160, 160)
MASK_COLOR = (30, 144, 255)
REFERENCE_COLOR = (255, 210, 0)
POSITIVE = (40, 220, 60)
NEGATIVE = (235, 40, 40)
COLUMNS = (
    ("One click (E4 default)", "click_then_clicks", "first"),
    ("Stroke (half chord)", "stroke_then_strokes", "first"),
    ("Stroke (full chord)", "full_stroke_then_strokes", "first"),
    ("Stroke + corrections", "stroke_then_strokes", "last"),
    ("Click + corrections", "click_then_clicks", "last"),
    ("Largest mask + strokes", "largest_mask_click_then_strokes", "last"),
)


def font(size: int):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def overlay(photo: Image.Image, mask: Image.Image, reference: Image.Image) -> Image.Image:
    tint = Image.new("RGB", photo.size, MASK_COLOR)
    image = Image.composite(Image.blend(photo, tint, 0.55), photo, mask.point(lambda v: 255 * (v > 127)))
    edge = reference.filter(ImageFilter.FIND_EDGES).point(lambda v: 255 * (v > 0))
    edge = edge.filter(ImageFilter.MaxFilter(3))
    return Image.composite(Image.new("RGB", photo.size, REFERENCE_COLOR), image, edge)


def draw_steps(image: Image.Image, steps: list[dict], scale: float) -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    radius = max(3, round(max(width, height) / 160))
    for step in steps:
        color = POSITIVE if step["label"] == 1 else NEGATIVE
        points = [(x * width, y * height) for x, y in step["points"]]
        if step.get("stroke"):
            stroke = step["stroke"]
            draw.line([(stroke["start"][0] * width, stroke["start"][1] * height),
                       (stroke["end"][0] * width, stroke["end"][1] * height)],
                      fill=color, width=max(2, radius // 2))
        for x, y in points:
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color,
                         outline=(0, 0, 0))


def tile(photo, reference, mask_dir, stem, prompt, sequence_name, which, edge):
    sequence = prompt["sequences"][sequence_name]
    if "steps" not in sequence:
        return None, "unavailable"
    steps = sequence["steps"]
    index = 0 if which == "first" else len(steps) - 1
    mask = Image.open(mask_dir / f"{stem}__{prompt['name']}__{sequence_name}__step{index}.png")
    mask = mask.convert("L")
    image = overlay(photo, mask, reference)
    scale = edge / max(image.size)
    image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                         Image.Resampling.LANCZOS)
    draw_steps(image, steps[: index + 1], scale)
    iou = steps[index]["reference_iou"]
    return image, f"IoU {iou:.2f}  {index + 1} interaction{'s' if index else ''}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--photos", required=True, help="photos of the report (the 4096 px set)")
    parser.add_argument("--reference-masks", required=True)
    parser.add_argument("--tile-edge", type=int, default=300)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    models = []
    for item in args.model:
        label, paths = item.split("=", 1)
        report_path, mask_dir = paths.split("|")
        models.append((label, json.loads(Path(report_path).read_text(encoding="utf-8")),
                       Path(mask_dir)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    title_font, label_font, caption_font = font(16), font(14), font(13)

    first_report = models[0][1]
    for image_index, image_entry in enumerate(first_report["images"]):
        stem = Path(image_entry["file"]).stem
        size = (image_entry["evaluation_width"], image_entry["evaluation_height"])
        photo = Image.open(Path(args.photos) / image_entry["file"]).convert("RGB")
        photo = photo.resize(size, Image.Resampling.LANCZOS)
        rows = []
        for prompt_index, prompt in enumerate(image_entry["prompts"]):
            if prompt.get("status") == "unavailable":
                continue
            reference = Image.open(Path(args.reference_masks) / prompt["reference"]).convert("L")
            reference = reference.resize(size, Image.Resampling.BILINEAR).point(
                lambda v: 255 * (v > 127))
            for label, report, mask_dir in models:
                model_prompt = report["images"][image_index]["prompts"][prompt_index]
                assert model_prompt["name"] == prompt["name"]
                tiles = [tile(photo, reference, mask_dir, stem, model_prompt, name, which,
                              args.tile_edge) for _, name, which in COLUMNS]
                rows.append((f"{prompt['name']} ({prompt['intent']})", label, tiles))
        if not rows:
            continue
        tile_width = max(t[0].width for _, _, tiles in rows for t in tiles if t[0])
        tile_height = max(t[0].height for _, _, tiles in rows for t in tiles if t[0])
        sheet = Image.new("RGB", (LABEL_WIDTH + len(COLUMNS) * tile_width,
                                  HEADER + len(rows) * (tile_height + CAPTION)), BACKGROUND)
        draw = ImageDraw.Draw(sheet)
        for column, (title, _, _) in enumerate(COLUMNS):
            draw.text((LABEL_WIDTH + column * tile_width + 6, 4), title, fill=TEXT, font=title_font)
        for row, (prompt_label, model_label, tiles) in enumerate(rows):
            top = HEADER + row * (tile_height + CAPTION)
            draw.text((6, top + 6), prompt_label, fill=TEXT, font=label_font)
            draw.text((6, top + 26), model_label, fill=MUTED, font=label_font)
            for column, (image, text) in enumerate(tiles):
                left = LABEL_WIDTH + column * tile_width
                if image is not None:
                    sheet.paste(image, (left, top))
                draw.text((left + 4, top + tile_height + 2), text, fill=MUTED, font=caption_font)
        path = out_dir / f"{stem}.jpg"
        sheet.save(path, quality=args.quality)
        print(f"[sheet] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
