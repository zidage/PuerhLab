#!/usr/bin/env python3
"""Edge stage of the SAM 2.1 stroke selection (#458 step E5).

For every person of --targets (edge_targets.json) on the E4 input set (4096 px long edge), the
script makes the selection with the E3 practice and then the alpha matte at the photo size in
several ways, so that the edge stages can be compared on the same coarse selection.

Selection (E3, section 9): SAM 2.1 point graphs on ONNX Runtime CPU (the E3 check showed the same
masks as DirectML). The first interaction is one stroke along the whole inner chord of the
reference region through `start`, on the principal axis of the region, with --stroke-points
points. Corrective strokes follow the E3 simulation: through the deepest pixel of the larger error
region (missed: add stroke, extra: subtract stroke), along the principal axis of that error
component, over half of its inner chord, at a margin of half of its depth. Every decoder run gets
all points of the selection and the previous low-resolution logits as `mask_input`, single-mask
output. The sequence stops at reference IoU >= --stop-iou (0.95; E3 used 0.90) or after
--max-interactions. The
reference is the E1 box mask of base plus; it bounds the simulated strokes, it is not an edge
ground truth.

Coarse logits:

  photo     : the 256 x 256 logits of the photo encoder pass.
  rectangle : when the subject rectangle covers less than --max-rectangle-fraction of the photo, a
              second encoder pass on that rectangle of the 4096 px input (the SM1 method (c)). The
              prompt is every stroke point inside the rectangle and the bounding box of the photo
              selection as a box. For a larger rectangle, rectangle is photo.
  tiles     : a second encoder pass on square tiles of --tile-size pixels along the edge of the
              photo selection, blended into one logit map at the photo size (see tile_pass), and
              used only within --tile-band-cells photo logit cells of the photo selection edge
              (limit_to_edge_band). The tile logit cell is --tile-size / 256 pixels for any
              subject size.

Edge stages, applied to the three coarse logits:

  bilinear   : bilinear upsampling of the logits, threshold 0 (the E3 and E4 masks).
  guided     : guided-filter upsampling of a soft matte (the Apple Vision method). The soft matte
               is the cleaned mask smoothed over half a logit cell (1 inside, 0 outside; the SAM
               logits themselves are not a matte, see guided_matte). The color guided filter of
               He et al. with the photo as the guide runs at --guided-cell-pixels pixels per logit
               cell, and its coefficients are applied at the photo size (the fast guided filter of
               He and Sun). Radius --guided-radius-cells logit cells, eps --guided-eps. Not
               thresholded.
  projection : the SM1 projection matte (method (d) of refine_edges_prototype.py, its defaults): a
               trimap band of one logit cell on each side of the edge, each band pixel color
               projected onto the line between the local mean colors of the sure foreground and
               the sure background, then a small color guided filter.

Sensitivity variants on the tile pass (metrics for every region, sheets for the hair regions):
guided on sigmoid(logits), guided radius 0.5 and 2 cells, eps 1e-2 and 1e-4, projection band 2
cells, the whole tiles without the edge band (bilinear, projection), and the whole tiles without
`mask_input` (bilinear).

Outputs under --out-dir:

  edges.json                 : selection steps, coarse IoU, times, and the metrics per region;
  alpha/<stem>__<variant>.png: 8-bit alpha of the main variants at the photo size;
  logits/<stem>__<coarse>.npy: the coarse logits (256 x 256; the tile pass at the photo size,
                               float16).

--sheet-dir gets <stem>.jpg per target (the selection with its strokes, then one block per region
with the photo crop and, per main variant, the photo with the background darkened by 2 EV and
desaturated through the alpha, where a halo or a missed strand is visible, and the alpha itself)
and <stem>__hair_settings.jpg with the sensitivity variants on the hair regions.

Metrics (no ground truth alpha exists for these photos; they are proxies):

  soft width    : pixels with 0.05 < alpha < 0.95 divided by the length of the alpha = 0.5
                  contour, in pixels. Large on hair is expected; on a hard edge it is blur, noise,
                  or an alpha below 1 inside the subject.
  specks        : connected components of alpha > 0.5 (islands) and of alpha <= 0.5 (holes)
                  smaller than a quarter of a logit cell area that do not touch the crop border.
  edge gradient : mean photo gradient on the alpha = 0.5 contour divided by the mean gradient in
                  the band of one logit cell around the bilinear edge. Above 1: the contour lies
                  on image edges.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# After HERE: both directories have a compare_onnx_outputs module.
sys.path.append(os.path.join(os.path.dirname(HERE), "efficientsam3"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy import ndimage  # noqa: E402

import sam2_graphs as graphs  # noqa: E402
from compare_onnx_outputs import LOGIT_LIMIT, select_single_mask  # noqa: E402
from refine_edges_prototype import (  # noqa: E402
    band_matte,
    clean_mask,
    color_guided_filter,
    subject_rectangle,
)

EVALUATION_EDGE = 1024
MARGIN_OF_SHORT_SIDE = 0.01
STEP = 0.5
COARSE = ("photo", "rectangle", "tiles")
COARSE_TITLES = {"photo": "photo pass", "rectangle": "rectangle pass", "tiles": "tile pass",
                 "tiles_unlimited": "tile pass, whole tiles",
                 "tiles_without_mask_input": "tile pass, whole tiles, no mask_input"}
# SM1 projection matte defaults (refine_edges_prototype.py).
PROJECTION_ARGS = SimpleNamespace(matte="projection", band_cells=1.0, clean_cells=3.0,
                                  smooth_radius=2, min_contrast=0.08, matte_eps=1e-4,
                                  matte_radius_cells=0.5, matte_subsample=2)


# --- Stroke simulation (the E3 `strokes` mode of segmentation_runtime_report.rs) ---------------

class Region:
    """A binary region on the evaluation frame and the distance of its pixels to its edge
    (outside the frame counts as outside)."""

    def __init__(self, inside: np.ndarray) -> None:
        self.inside = inside
        padded = np.pad(inside, 1, constant_values=False)
        self.distance = ndimage.distance_transform_edt(padded)[1:-1, 1:-1].astype(np.float32)

    def distance_at(self, point) -> float:
        x, y = int(np.floor(point[0])), int(np.floor(point[1]))
        height, width = self.inside.shape
        if x < 0 or y < 0 or x >= width or y >= height:
            return 0.0
        return float(self.distance[y, x])

    def deepest(self):
        if not self.inside.any():
            return None
        index = int(np.argmax(self.distance))
        y, x = divmod(index, self.inside.shape[1])
        return (x + 0.5, y + 0.5), float(self.distance[y, x])

    def principal_direction(self, point) -> tuple[float, float]:
        x, y = int(np.floor(point[0])), int(np.floor(point[1]))
        if not self.inside[y, x]:
            return 1.0, 0.0
        labels, _ = ndimage.label(self.inside)  # 4-connected
        ys, xs = np.nonzero(labels == labels[y, x])
        xs, ys = xs.astype(np.float64), ys.astype(np.float64)
        cxx, cyy = xs.var(), ys.var()
        cxy = (xs * ys).mean() - xs.mean() * ys.mean()
        angle = 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)
        return float(np.cos(angle)), float(np.sin(angle))

    def inner_chord(self, point, direction, margin) -> tuple[float, float]:
        if self.distance_at(point) < margin:
            return 0.0, 0.0

        def reach(sign):
            t = 0.0
            while True:
                n = t + STEP
                if self.distance_at((point[0] + sign * n * direction[0],
                                     point[1] + sign * n * direction[1])) < margin:
                    return t
                t = n

        return reach(-1.0), reach(1.0)


def stroke_along_chord(point, direction, chord, fraction):
    back, forward = chord
    total = back + forward
    if total <= 0.0:
        return None
    length = min(max(fraction, 0.0), 1.0) * total
    offset = min(max(-length / 2.0, -back), forward - length)
    at = lambda t: (point[0] + t * direction[0], point[1] + t * direction[1])  # noqa: E731
    return at(offset), at(offset + length)


def sample_stroke(stroke, count):
    (x0, y0), (x1, y1) = stroke
    if count <= 1:
        return [((x0 + x1) / 2, (y0 + y1) / 2)]
    return [(x0 + k / (count - 1) * (x1 - x0), y0 + k / (count - 1) * (y1 - y0))
            for k in range(count)]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


class Decoder:
    def __init__(self, session, features):
        self.session = session
        self.features = features

    def __call__(self, coords, labels, mask_input=None, box=None):
        """coords in model-input pixels. box (model-input pixels) goes first with labels 2, 3."""
        coords, labels = list(coords), list(labels)
        if box is not None:
            coords = [[box[0], box[1]], [box[2], box[3]]] + coords
            labels = [2.0, 3.0] + labels
        size = graphs.LOW_RES_MASK_SIZE
        image_embed, feats_0, feats_1 = self.features
        logits, scores, _ = self.session.run(None, {
            "image_embed": image_embed, "high_res_feats_0": feats_0, "high_res_feats_1": feats_1,
            "point_coords": np.array([coords], np.float32),
            "point_labels": np.array([labels], np.float32),
            "mask_input": (mask_input[None, None].astype(np.float32) if mask_input is not None
                           else np.zeros((1, 1, size, size), np.float32)),
            "has_mask_input": np.array([0.0 if mask_input is None else 1.0], np.float32),
        })
        selected, _ = select_single_mask(logits[0], scores[0])
        return np.clip(selected, -LOGIT_LIMIT, LOGIT_LIMIT).astype(np.float32)


def upsample(logits: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(logits, (width, height), interpolation=cv2.INTER_LINEAR)


def select_person(decoder, target, reference, args):
    """The E3 selection: a whole-chord stroke, then corrective strokes. Returns the logits, the
    points (normalized) with labels, and the steps."""
    height, width = reference.shape
    region = Region(reference)
    margin = MARGIN_OF_SHORT_SIDE * min(width, height)
    start = (target["start"][0] * width, target["start"][1] * height)
    depth = region.distance_at(start)
    if depth < margin:
        margin = max(1.5, depth / 2)
    direction = region.principal_direction(start)
    stroke = stroke_along_chord(start, direction, region.inner_chord(start, direction, margin), 1.0)
    if stroke is None:
        raise RuntimeError(f"{target['file']}: start point outside the reference at the margin")
    points = sample_stroke(stroke, args.stroke_points)
    labels = [1.0] * len(points)

    def model(ps):
        return [[x / width * graphs.IMAGE_SIZE, y / height * graphs.IMAGE_SIZE] for x, y in ps]

    logits = decoder(model(points), labels)
    steps = [{"step": 0, "label": 1, "stroke": [[v / s for v, s in zip(p, (width, height))]
                                                 for p in stroke],
              "points": [[x / width, y / height] for x, y in points]}]
    for step in range(args.max_interactions):
        mask = upsample(logits, width, height) > 0
        steps[-1]["reference_iou"] = iou(mask, reference)
        if steps[-1]["reference_iou"] >= args.stop_iou or step == args.max_interactions - 1:
            break
        missed, extra = Region(reference & ~mask), Region(mask & ~reference)
        dm, de = missed.deepest(), extra.deepest()
        if dm is None and de is None:
            break
        if de is not None and (dm is None or de[1] > dm[1]):
            error, label, (deep, depth) = extra, 0.0, de
        else:
            error, label, (deep, depth) = missed, 1.0, dm
        if depth < 1.5:
            break
        direction = error.principal_direction(deep)
        chord = error.inner_chord(deep, direction, depth * 0.5)
        correction = stroke_along_chord(deep, direction, chord, 0.5)
        if correction is not None and np.hypot(correction[1][0] - correction[0][0],
                                               correction[1][1] - correction[0][1]) >= 1.0:
            new_points = sample_stroke(correction, args.stroke_points)
        else:
            correction, new_points = None, [deep]
        points += new_points
        labels += [label] * len(new_points)
        logits = decoder(model(points), labels, mask_input=logits)
        steps.append({
            "step": step + 1, "label": int(label),
            "error_region": "missed" if label else "extra", "error_region_depth_px": depth,
            "stroke": (None if correction is None else
                       [[v / s for v, s in zip(p, (width, height))] for p in correction]),
            "points": [[x / width, y / height] for x, y in new_points]})
    normalized = [[x / width, y / height, label] for (x, y), label in zip(points, labels)]
    return logits, normalized, steps


def tile_pass(encoder, decoder_session, image, logits_up, points, args):
    """Second encoder pass on square tiles of --tile-size pixels of the 4096 px input along the edge
    of the photo selection. Tiles are on a grid with --tile-overlap; a tile runs when it holds a
    pixel within one photo logit cell of the edge. The prompt of a tile: the stroke points inside
    it, one positive point at the deepest selected pixel and one negative point at the deepest
    unselected pixel of the tile (if deeper than one tile logit cell), and the photo logits of the
    tile as `mask_input` (resampled to 256 x 256). The tile logits are blended with weights that
    fall to 0 over the overlap at the tile border; outside the tiles the photo logits stay.
    Returns the blended logits with and without `mask_input` ("whole tiles"), the tile windows,
    the encoder time, and the tile logit cell in pixels. limit_to_edge_band() then restricts the
    result to the edge band of the photo selection."""
    width, height = image.size
    size = min(args.tile_size, width, height)
    mask = logits_up > 0
    cell = max(width, height) / graphs.LOW_RES_MASK_SIZE
    radius = max(1, round(cell))
    band = cv2.dilate(contour(mask).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)) > 0
    stride = max(1, int(size * (1.0 - args.tile_overlap)))

    def starts(extent):
        values = list(range(0, max(extent - size, 0) + 1, stride))
        if values[-1] < extent - size:
            values.append(extent - size)
        return values

    ramp = np.minimum(np.arange(size) + 1, size - np.arange(size)).astype(np.float32)
    ramp = np.minimum(ramp / max(1.0, size * args.tile_overlap / 2), 1.0)
    weight = np.outer(ramp, ramp)
    low = graphs.LOW_RES_MASK_SIZE
    sums = {"with": np.zeros((height, width), np.float32),
            "without": np.zeros((height, width), np.float32)}
    total = np.zeros((height, width), np.float32)
    windows, encoder_ms = [], 0.0
    for y0 in starts(height):
        for x0 in starts(width):
            window = (slice(y0, y0 + size), slice(x0, x0 + size))
            if not band[window].any():
                continue
            windows.append([x0, y0, x0 + size, y0 + size])
            started = time.perf_counter()
            features = encoder.run(None, {"pixel_values": graphs.preprocess_image(
                image.crop((x0, y0, x0 + size, y0 + size))).numpy()})
            encoder_ms += (time.perf_counter() - started) * 1000
            decoder = Decoder(decoder_session, features)
            coords, labels = [], []
            for x, y, label in points:
                u, v = (x * width - x0) / size, (y * height - y0) / size
                if 0.0 <= u < 1.0 and 0.0 <= v < 1.0:
                    coords.append([u * graphs.IMAGE_SIZE, v * graphs.IMAGE_SIZE])
                    labels.append(float(label))
            small = cv2.resize(mask[window].astype(np.uint8), (low, low),
                               interpolation=cv2.INTER_NEAREST) > 0
            for inside, label in ((small, 1.0), (~small, 0.0)):
                depth = ndimage.distance_transform_edt(np.pad(inside, 1))[1:-1, 1:-1]
                if depth.max() > 1.0:
                    y, x = np.unravel_index(int(np.argmax(depth)), depth.shape)
                    scale = graphs.IMAGE_SIZE / low
                    coords.append([(x + 0.5) * scale, (y + 0.5) * scale])
                    labels.append(label)
            mask_input = cv2.resize(logits_up[window], (low, low), interpolation=cv2.INTER_AREA)
            for key, previous in (("with", mask_input), ("without", None)):
                tile_logits = decoder(coords, labels, mask_input=previous)
                sums[key][window] += weight * cv2.resize(tile_logits, (size, size),
                                                         interpolation=cv2.INTER_LINEAR)
            total[window] += weight
    blended = {key: np.where(total > 0, value / np.maximum(total, 1e-6), logits_up)
               .astype(np.float32) for key, value in sums.items()}
    return blended, windows, encoder_ms, size / low


def limit_to_edge_band(photo_logits, tile_logits, cell, band_cells):
    """The tile logits within band_cells photo logit cells of the photo selection edge, the photo
    logits beyond one more cell, a linear blend between. A tile then moves the edge but does not
    decide a whole region (a low-contrast region can get opposite decisions in two tiles, with a
    straight seam along the tile border)."""
    edge = contour(photo_logits > 0)
    distance = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.clip((band_cells * cell + cell - distance) / cell, 0.0, 1.0)
    return (weight * tile_logits + (1.0 - weight) * photo_logits).astype(np.float32)


# --- Edge stages --------------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -LOGIT_LIMIT, LOGIT_LIMIT)))).astype(np.float32)


def guided_matte(guide, logits_up, cell, spec, cell_pixels):
    """Guided-filter upsampling. The soft matte is built on a grid of `cell_pixels` pixels per
    logit cell: 'matte' is the cleaned mask (logit > 0, holes and islands below 3 x 3 cells
    removed) smoothed with a Gaussian of half a cell, so that it is 1 inside and 0 outside;
    'sigmoid' is sigmoid(logits). The color guided filter runs on the same grid and its
    coefficients are applied at the photo size."""
    height, width = logits_up.shape
    subsample = max(1, round(cell / cell_pixels))
    small = (max(1, width // subsample), max(1, height // subsample))
    low = cv2.resize(logits_up, small, interpolation=cv2.INTER_AREA)
    small_cell = cell / subsample
    if spec["source"] == "sigmoid":
        matte = sigmoid(low)
    else:
        hard = clean_mask(low > 0, (PROJECTION_ARGS.clean_cells * small_cell) ** 2)
        matte = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 0.5 * small_cell)
    source = cv2.resize(matte, (width, height), interpolation=cv2.INTER_LINEAR)
    radius = max(subsample, round(spec["radius_cells"] * cell))
    return color_guided_filter(guide, source, radius, spec["eps"], subsample)


def projection_matte(guide, logits_up, cell, spec):
    return band_matte(guide, logits_up, cell,
                      SimpleNamespace(**{**vars(PROJECTION_ARGS),
                                         "band_cells": spec["band_cells"]}))


def edge_stage(spec, guide, logits_up, cell, args):
    if spec["edge"] == "bilinear":
        return (logits_up > 0).astype(np.float32)
    if spec["edge"] == "guided":
        return guided_matte(guide, logits_up, cell, spec, args.guided_cell_pixels)
    return projection_matte(guide, logits_up, cell, spec)


def variant_specs(args):
    """Main variants (every region) and sensitivity variants (on the tile pass; sheets for the hair
    regions)."""
    guided = {"edge": "guided", "source": "matte", "radius_cells": args.guided_radius_cells,
              "eps": args.guided_eps}
    projection = {"edge": "projection", "band_cells": PROJECTION_ARGS.band_cells}
    main = {}
    for coarse in COARSE:
        main[f"{coarse}_bilinear"] = {"coarse": coarse, "edge": "bilinear"}
        main[f"{coarse}_guided"] = {"coarse": coarse, **guided}
        main[f"{coarse}_projection"] = {"coarse": coarse, **projection}
    sensitivity = {
        "guided_sigmoid": {"coarse": "tiles", **guided, "source": "sigmoid"},
        "guided_r0.5": {"coarse": "tiles", **guided, "radius_cells": 0.5},
        "guided_r2": {"coarse": "tiles", **guided, "radius_cells": 2.0},
        "guided_eps1e-2": {"coarse": "tiles", **guided, "eps": 1e-2},
        "guided_eps1e-4": {"coarse": "tiles", **guided, "eps": 1e-4},
        "projection_band2": {"coarse": "tiles", **projection, "band_cells": 2.0},
        "tiles_unlimited_bilinear": {"coarse": "tiles_unlimited", "edge": "bilinear"},
        "tiles_unlimited_projection": {"coarse": "tiles_unlimited", **projection},
        "tiles_without_mask_input_bilinear": {"coarse": "tiles_without_mask_input",
                                              "edge": "bilinear"},
    }
    return main, sensitivity


def variant_title(name, spec, args):
    """Two lines: the coarse pass and the edge stage with its settings."""
    coarse = COARSE_TITLES[spec["coarse"]]
    if spec["edge"] == "bilinear":
        return f"{coarse}\nbilinear"
    if spec["edge"] == "projection":
        return f"{coarse}\nprojection, band {spec['band_cells']:g}"
    return (f"{coarse}\nguided ({spec['source']}), r {spec['radius_cells']:g}, "
            f"eps {spec['eps']:g}")


# --- Metrics ------------------------------------------------------------------------------------

def contour(mask: np.ndarray) -> np.ndarray:
    edge = np.zeros(mask.shape, bool)
    edge[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    edge[:-1, :] |= mask[:-1, :] != mask[1:, :]
    return edge


def specks(alpha: np.ndarray, max_area: float) -> tuple[int, int]:
    """Small islands (alpha > 0.5) and holes (alpha <= 0.5) that do not touch the crop border."""
    counts = []
    height, width = alpha.shape
    for region in (alpha > 0.5, alpha <= 0.5):
        count, _, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8),
                                                              connectivity=8)
        small = 0
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            touches = x == 0 or y == 0 or x + w == width or y + h == height
            if area < max_area and not touches:
                small += 1
        counts.append(small)
    return counts[0], counts[1]


def region_metrics(alpha, gradient, band, cell):
    edge = contour(alpha > 0.5)
    length = int(edge.sum())
    soft = int(((alpha > 0.05) & (alpha < 0.95)).sum())
    islands, holes = specks(alpha, 0.25 * cell * cell)
    band_gradient = float(gradient[band].mean()) if band.any() else 0.0
    return {
        "contour_px": length,
        "soft_px": soft,
        "soft_width_px": soft / length if length else 0.0,
        "islands": islands,
        "holes": holes,
        "edge_gradient": (float(gradient[edge].mean()) / band_gradient
                          if length and band_gradient > 0 else 0.0),
    }


# --- Sheets -------------------------------------------------------------------------------------

def font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def darken_background(photo: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """The photo where alpha is 1; 2 EV darker and desaturated where alpha is 0."""
    pixels = photo.astype(np.float32)
    gray = pixels @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    background = np.repeat(gray[..., None], 3, axis=2) * 0.25
    a = np.clip(alpha, 0.0, 1.0)[..., None]
    return np.clip(pixels * a + background * (1.0 - a), 0, 255).astype(np.uint8)


def crop_window(region, width, height):
    size = min(region["size"], width, height)
    cx, cy = region["center"][0] * width, region["center"][1] * height
    x0 = int(round(min(max(cx - size / 2, 0), width - size)))
    y0 = int(round(min(max(cy - size / 2, 0), height - size)))
    return x0, y0, size


def overview_image(photo, alpha, target, steps, rectangle, tiles, edge):
    height, width = photo.shape[:2]
    scale = min(1.0, edge / max(width, height))
    image = Image.fromarray(darken_background(photo, alpha)).resize(
        (round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    label_font = font(18)
    for tile in tiles:
        draw.rectangle([v * scale for v in tile], outline=(200, 90, 255), width=1)
    if rectangle is not None:
        draw.rectangle([v * scale for v in rectangle], outline=(0, 200, 255), width=3)
    for region in target["regions"]:
        x0, y0, size = crop_window(region, width, height)
        box = (x0 * scale, y0 * scale, (x0 + size) * scale, (y0 + size) * scale)
        draw.rectangle(box, outline=(255, 210, 0), width=3)
        draw.text((box[0] + 4, box[1] + 2), region["name"], fill=(255, 210, 0), font=label_font)
    for step in steps:
        color = (60, 220, 60) if step["label"] else (240, 60, 60)
        if step["stroke"]:
            (x0, y0), (x1, y1) = step["stroke"]
            draw.line([(x0 * image.width, y0 * image.height),
                       (x1 * image.width, y1 * image.height)], fill=color, width=4)
        for x, y in step["points"]:
            r = 6
            draw.ellipse([x * image.width - r, y * image.height - r,
                          x * image.width + r, y * image.height + r], fill=color,
                         outline=(0, 0, 0))
    return image


def region_block(photo, alphas, titles, region, tile, note=""):
    height, width = photo.shape[:2]
    x0, y0, size = crop_window(region, width, height)
    window = (slice(y0, y0 + size), slice(x0, x0 + size))
    header = 34
    canvas = Image.new("RGB", ((1 + len(alphas)) * tile, header + 2 * tile), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    label_font = font(16)
    draw.text((6, 4), f"{region['name']} ({region['kind']}, {size} px crop of the 4096 px input)"
              f"{note}", fill=(255, 255, 255), font=font(24))

    def put(array, column, row):
        image = Image.fromarray(array).resize(
            (tile, tile), Image.Resampling.LANCZOS if size > tile else Image.Resampling.NEAREST)
        canvas.paste(image, (column * tile, header + row * tile))

    put(photo[window], 0, 0)
    for k, (name, alpha) in enumerate(alphas.items(), start=1):
        put(darken_background(photo[window], alpha[window]), k, 0)
        put((np.clip(alpha[window], 0, 1) * 255).astype(np.uint8), k, 1)
        draw.rectangle([k * tile, header, (k + 1) * tile - 1, header + 40], fill=(0, 0, 0))
        draw.multiline_text((k * tile + 4, header + 2), titles[name], fill=(255, 230, 0),
                            font=label_font, spacing=2)
    draw.text((4, header + tile + 4), "row 1: background\n-2 EV, desaturated\nrow 2: alpha",
              fill=(200, 200, 200), font=label_font)
    return canvas


def save_sheet(path, blocks, quality):
    total_width = max(b.width for b in blocks)
    sheet = Image.new("RGB", (total_width, sum(b.height for b in blocks) + 8 * len(blocks)),
                      (12, 12, 12))
    y = 0
    for block in blocks:
        sheet.paste(block, (0, y))
        y += block.height + 8
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=quality)


# --- Main ---------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--photos", required=True, help="E4 input set photos (4096 px)")
    parser.add_argument("--reference-masks", required=True, help="E4 input set reference masks")
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sheet-dir", default="")
    parser.add_argument("--stroke-points", type=int, default=4)
    parser.add_argument("--stop-iou", type=float, default=0.95,
                        help="0.95, above the 0.90 of E3: a user corrects a visibly missing "
                             "region before the edge matters")
    parser.add_argument("--max-interactions", type=int, default=5)
    parser.add_argument("--max-rectangle-fraction", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=1024,
                        help="tile pass: tile side in pixels of the 4096 px input")
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--tile-band-cells", type=float, default=2.0,
                        help="tile pass: the tile logits apply within this many photo logit "
                             "cells of the photo selection edge")
    parser.add_argument("--guided-radius-cells", type=float, default=1.0)
    parser.add_argument("--guided-eps", type=float, default=1e-3)
    parser.add_argument("--guided-cell-pixels", type=float, default=4.0,
                        help="pixels per logit cell of the low-resolution guided filter grid")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tile", type=int, default=288)
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--only", default="", help="comma-separated photo stems")
    args = parser.parse_args()

    import onnxruntime as ort

    cv2.setNumThreads(args.threads)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    sessions = {n: ort.InferenceSession(str(Path(args.onnx_dir) / f"{n}.onnx"), options,
                                        providers=["CPUExecutionProvider"])
                for n in ("point_encoder", "point_decoder")}
    targets = json.loads(Path(args.targets).read_text(encoding="utf-8"))["targets"]
    only = {s for s in args.only.split(",") if s}
    out_dir = Path(args.out_dir)
    (out_dir / "alpha").mkdir(parents=True, exist_ok=True)
    (out_dir / "logits").mkdir(parents=True, exist_ok=True)
    main_specs, sensitivity_specs = variant_specs(args)
    all_specs = {**main_specs, **sensitivity_specs}
    titles = {name: variant_title(name, spec, args) for name, spec in all_specs.items()}

    results = []
    for target in targets:
        stem = Path(target["file"]).stem
        if only and stem not in only:
            continue
        image = Image.open(Path(args.photos) / target["file"]).convert("RGB")
        width, height = image.size
        photo = np.asarray(image)
        guide = photo.astype(np.float32) / 255.0
        scale = EVALUATION_EDGE / max(width, height)
        eval_size = (round(width * scale), round(height * scale))
        reference_full = np.asarray(Image.open(Path(args.reference_masks) / target["reference"])
                                    .convert("L").resize((width, height),
                                                         Image.Resampling.BILINEAR)) > 127
        reference = cv2.resize(reference_full.astype(np.uint8) * 255, eval_size,
                               interpolation=cv2.INTER_AREA) > 127

        started = time.perf_counter()
        features = sessions["point_encoder"].run(
            None, {"pixel_values": graphs.preprocess_image(image).numpy()})
        encoder_ms = (time.perf_counter() - started) * 1000
        decoder = Decoder(sessions["point_decoder"], features)
        logits, points, steps = select_person(decoder, target, reference, args)
        np.save(out_dir / "logits" / f"{stem}__photo.npy", logits)
        result = {"file": target["file"], "name": target["name"], "width": width,
                  "height": height, "reference": target["reference"],
                  "encoder_cpu_ms": encoder_ms, "interactions": len(steps),
                  "decoder_points": len(points),
                  "negative_points": sum(1 for p in points if p[2] == 0), "steps": steps}

        coarse = {"photo": {"window": (0, 0, width, height),
                            "up": upsample(logits, width, height),
                            "cell": max(width, height) / graphs.LOW_RES_MASK_SIZE}}
        photo_mask = coarse["photo"]["up"] > 0
        result["photo_logit_cell_px"] = coarse["photo"]["cell"]
        rectangle = subject_rectangle(photo_mask, points, None)
        area = (0.0 if rectangle is None else
                (rectangle[2] - rectangle[0]) * (rectangle[3] - rectangle[1]))
        result["rectangle_area_fraction"] = area
        result["rectangle"] = None
        if rectangle is not None and area < args.max_rectangle_fraction:
            rx0, ry0, rx1, ry1 = rectangle
            px0, py0 = round(rx0 * width), round(ry0 * height)
            px1, py1 = round(rx1 * width), round(ry1 * height)
            crop = image.crop((px0, py0, px1, py1))
            started = time.perf_counter()
            crop_features = sessions["point_encoder"].run(
                None, {"pixel_values": graphs.preprocess_image(crop).numpy()})
            result["rectangle_encoder_cpu_ms"] = (time.perf_counter() - started) * 1000
            cw, ch = px1 - px0, py1 - py0
            ys, xs = np.nonzero(photo_mask)
            size = graphs.IMAGE_SIZE
            box = [(xs.min() - px0) / cw * size, (ys.min() - py0) / ch * size,
                   (xs.max() + 1 - px0) / cw * size, (ys.max() + 1 - py0) / ch * size]
            coords, labels = [], []
            for x, y, label in points:
                u, v = (x * width - px0) / cw, (y * height - py0) / ch
                if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
                    coords.append([u * size, v * size])
                    labels.append(float(label))
            rect_logits = Decoder(sessions["point_decoder"], crop_features)(coords, labels, box=box)
            np.save(out_dir / "logits" / f"{stem}__rectangle.npy", rect_logits)
            coarse["rectangle"] = {"window": (px0, py0, px1, py1),
                                   "up": upsample(rect_logits, cw, ch),
                                   "cell": max(cw, ch) / graphs.LOW_RES_MASK_SIZE}
            result["rectangle"] = [px0, py0, px1, py1]
            result["rectangle_logit_cell_px"] = coarse["rectangle"]["cell"]
        else:
            coarse["rectangle"] = coarse["photo"]

        tiles, windows, tile_encoder_ms, tile_cell = tile_pass(
            sessions["point_encoder"], sessions["point_decoder"], image, coarse["photo"]["up"],
            points, args)
        limited = limit_to_edge_band(coarse["photo"]["up"], tiles["with"],
                                     coarse["photo"]["cell"], args.tile_band_cells)
        for name, value in (("tiles", limited), ("tiles_unlimited", tiles["with"]),
                            ("tiles_without_mask_input", tiles["without"])):
            coarse[name] = {"window": (0, 0, width, height), "up": value, "cell": tile_cell}
            np.save(out_dir / "logits" / f"{stem}__{name}.npy", value.astype(np.float16))
        result["tiles"] = windows
        result["tile_count"] = len(windows)
        result["tile_encoder_cpu_ms"] = tile_encoder_ms
        result["tile_logit_cell_px"] = tile_cell

        alphas, times = {}, {}
        for name, spec in all_specs.items():
            c = coarse[spec["coarse"]]
            x0, y0, x1, y1 = c["window"]
            started = time.perf_counter()
            matte = edge_stage(spec, guide[y0:y1, x0:x1], c["up"], c["cell"], args)
            times[name] = (time.perf_counter() - started) * 1000
            alpha = np.zeros((height, width), np.float32)
            alpha[y0:y1, x0:x1] = matte
            alphas[name] = alpha
            if name in main_specs:
                Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8)).save(
                    out_dir / "alpha" / f"{stem}__{name}.png")
        result["edge_cpu_ms"] = times
        eval_masks = {v: cv2.resize(a, eval_size, interpolation=cv2.INTER_AREA) > 0.5
                      for v, a in alphas.items()}
        result["reference_iou"] = {v: iou(m, reference) for v, m in eval_masks.items()}
        result["photo_vs_rectangle_iou"] = iou(eval_masks["photo_bilinear"],
                                               eval_masks["rectangle_bilinear"])
        result["photo_vs_tiles_iou"] = iou(eval_masks["photo_bilinear"],
                                           eval_masks["tiles_bilinear"])
        result["photo_vs_tiles_unlimited_iou"] = iou(eval_masks["photo_bilinear"],
                                                     eval_masks["tiles_unlimited_bilinear"])
        result["photo_vs_tiles_without_mask_input_iou"] = iou(
            eval_masks["photo_bilinear"], eval_masks["tiles_without_mask_input_bilinear"])

        # Metrics per region. The band and the speck size use the logit cell of the photo pass.
        gray = cv2.GaussianBlur(cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(np.float32),
                                (0, 0), 1.0)
        gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0),
                                 cv2.Sobel(gray, cv2.CV_32F, 0, 1))
        cell = coarse["photo"]["cell"]
        band_radius = max(1, round(cell))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_radius + 1,) * 2)
        band = cv2.dilate(contour(photo_mask).astype(np.uint8), kernel) > 0
        result["regions"] = [{"name": r["name"], "kind": r["kind"]} for r in target["regions"]]
        result["metrics"] = {}
        for name, alpha in alphas.items():
            result["metrics"][name] = {}
            for region in target["regions"]:
                x0, y0, size = crop_window(region, width, height)
                window = (slice(y0, y0 + size), slice(x0, x0 + size))
                result["metrics"][name][region["name"]] = region_metrics(
                    alpha[window], gradient[window], band[window], cell)
        results.append(result)
        print(f"[edges] {stem}: {len(steps)} interactions, {len(points)} points, reference IoU "
              f"{steps[-1].get('reference_iou', float('nan')):.3f}, rectangle "
              f"{'none' if result['rectangle'] is None else f'{area:.3f}'}, "
              f"{len(windows)} tiles, edge ms "
              + ", ".join(f"{k} {v:.0f}" for k, v in times.items() if k in main_specs),
              flush=True)

        if args.sheet_dir:
            sheet_dir = Path(args.sheet_dir)
            main_alphas = {n: alphas[n] for n in main_specs}
            blocks = [overview_image(photo, alphas["photo_bilinear"], target, steps,
                                     result["rectangle"], windows, args.tile * 3)]
            note = "" if result["rectangle"] is not None else "; rectangle pass = photo pass"
            blocks += [region_block(photo, main_alphas, titles, r, args.tile, note)
                       for r in target["regions"]]
            save_sheet(sheet_dir / f"{stem}.jpg", blocks, args.quality)
            hair = [r for r in target["regions"] if r["kind"] == "hair"]
            if hair:
                chosen = {n: alphas[n] for n in ("tiles_guided", "tiles_projection",
                                                 *sensitivity_specs)}
                save_sheet(sheet_dir / f"{stem}__hair_settings.jpg",
                           [region_block(photo, chosen, titles, r, args.tile) for r in hair],
                           args.quality)

    (out_dir / "edges.json").write_text(json.dumps({
        "onnx_dir": args.onnx_dir, "stroke_points": args.stroke_points,
        "stop_iou": args.stop_iou, "max_interactions": args.max_interactions,
        "max_rectangle_fraction": args.max_rectangle_fraction,
        "tile_size": args.tile_size, "tile_overlap": args.tile_overlap,
        "tile_band_cells": args.tile_band_cells,
        "guided_cell_pixels": args.guided_cell_pixels,
        "variants": all_specs, "titles": titles, "main_variants": list(main_specs),
        "projection": vars(PROJECTION_ARGS), "threads": args.threads,
        "opencv": cv2.__version__, "results": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
