#!/usr/bin/env python3
"""Tables of the E5 edge report (#458 step E5) from the edges.json of refine_stroke_edges.py.

With --e3-report (the E3 `strokes` report of the same variant), the script also checks that the
first stroke of refine_stroke_edges.py is the E3 stroke: for every target whose start point and
reference mask are those of an E3 prompt, the E3 sweep entry with the same point count, the
principal axis, and the whole inner chord must have the same end points
(StrokePlacementMatchesE3, maximum distance in evaluation-frame pixels).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else float("nan")


def summarize(results, names):
    rows = {}
    for name in names:
        row = {"hair": [], "hard": [], "hard_specks": 0, "hair_specks": 0,
               "hard_gradient": [], "hair_gradient": [], "cpu_ms": [], "reference_iou": []}
        for result in results:
            kinds = {r["name"]: r["kind"] for r in result["regions"]}
            for region, m in result["metrics"][name].items():
                kind = kinds[region]
                row[kind].append(m["soft_width_px"])
                row[f"{kind}_specks"] += m["islands"] + m["holes"]
                row[f"{kind}_gradient"].append(m["edge_gradient"])
            row["cpu_ms"].append(result["edge_cpu_ms"][name])
            row["reference_iou"].append(result["reference_iou"][name])
        rows[name] = {
            "hair_soft_width_px_median": median(row["hair"]),
            "hard_soft_width_px_median": median(row["hard"]),
            "hard_soft_width_px_max": max(row["hard"]),
            "hair_specks": row["hair_specks"], "hard_specks": row["hard_specks"],
            "hair_edge_gradient_median": median(row["hair_gradient"]),
            "hard_edge_gradient_median": median(row["hard_gradient"]),
            "cpu_ms_median": median(row["cpu_ms"]), "cpu_ms_max": max(row["cpu_ms"]),
            "reference_iou_min": min(row["reference_iou"]),
        }
    return rows


def check_strokes(edges, e3_report, targets):
    e3 = {image["file"]: image for image in e3_report["images"]}
    starts = {(t["file"], t["name"]): t["start"] for t in targets}
    count = edges["stroke_points"]
    checks = []
    for result in edges["results"]:
        image = e3.get(result["file"])
        if image is None:
            continue
        first = result["steps"][0]
        start_point = starts[(result["file"], result["name"])]
        for prompt in image["prompts"]:
            if (max(abs(a - b) for a, b in zip(prompt["point"], start_point)) > 1e-4
                    or prompt["reference"] != result["reference"]):
                continue
            entry = next((s for s in prompt.get("sweep", [])
                          if s["count"] == count and s["direction"] == "principal"
                          and s["length_of_inner_chord"] == 1.0), None)
            if entry is None:
                continue
            width, height = image["evaluation_width"], image["evaluation_height"]
            start = [first["stroke"][0][0] * width, first["stroke"][0][1] * height]
            end = [first["stroke"][1][0] * width, first["stroke"][1][1] * height]
            theirs = [[entry["stroke"][k][0] * width, entry["stroke"][k][1] * height]
                      for k in ("start", "end")]
            distance = max(max(abs(a - b) for a, b in zip(p, q)) for p, q in
                           ((start, theirs[0]), (end, theirs[1])))
            swapped = max(max(abs(a - b) for a, b in zip(p, q)) for p, q in
                          ((start, theirs[1]), (end, theirs[0])))
            checks.append({"file": result["file"], "e3_prompt": prompt["name"],
                           "max_endpoint_distance_px": min(distance, swapped),
                           "e3_reference_iou_4_points": entry["result"]["single_mask_iou"]})
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--e3-report", default="")
    parser.add_argument("--targets", default="", help="edge_targets.json (for --e3-report)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    edges = json.loads(Path(args.edges).read_text(encoding="utf-8"))
    results = edges["results"]
    main_names = edges["main_variants"]
    sensitivity = [n for n in edges["variants"] if n not in main_names]

    selection = []
    for r in results:
        selection.append({
            "file": r["file"], "name": r["name"], "interactions": r["interactions"],
            "decoder_points": r["decoder_points"], "negative_points": r["negative_points"],
            "selection_reference_iou": r["steps"][-1].get("reference_iou"),
            "rectangle_area_fraction": r["rectangle_area_fraction"],
            "rectangle_used": r["rectangle"] is not None,
            "photo_logit_cell_px": r["photo_logit_cell_px"],
            "rectangle_logit_cell_px": r.get("rectangle_logit_cell_px"),
            "photo_vs_rectangle_iou": r["photo_vs_rectangle_iou"],
            "reference_iou": {n: r["reference_iou"][n] for n in main_names},
            "encoder_cpu_ms": r["encoder_cpu_ms"],
            "rectangle_encoder_cpu_ms": r.get("rectangle_encoder_cpu_ms"),
            "tile_count": r["tile_count"], "tile_logit_cell_px": r["tile_logit_cell_px"],
            "tile_encoder_cpu_ms": r["tile_encoder_cpu_ms"],
            "photo_vs_tiles_iou": r["photo_vs_tiles_iou"],
            "photo_vs_tiles_unlimited_iou": r["photo_vs_tiles_unlimited_iou"],
            "photo_vs_tiles_without_mask_input_iou": r["photo_vs_tiles_without_mask_input_iou"],
            "tiles_without_mask_input_reference_iou":
                r["reference_iou"]["tiles_without_mask_input_bilinear"],
        })
    per_region = []
    for r in results:
        kinds = {x["name"]: x["kind"] for x in r["regions"]}
        for region in kinds:
            per_region.append({
                "file": r["file"], "region": region, "kind": kinds[region],
                **{n: {k: r["metrics"][n][region][k]
                       for k in ("soft_width_px", "islands", "holes", "edge_gradient")}
                   for n in main_names}})
    small = [r for r in results if r["rectangle"] is not None]
    report = {
        "selection": selection,
        "main": summarize(results, main_names),
        "main_small_subjects": summarize(small, main_names) if small else {},
        "sensitivity": summarize(results, ["tiles_guided", "tiles_projection"] + sensitivity),
        "per_region": per_region,
    }
    if args.e3_report:
        checks = check_strokes(
            edges, json.loads(Path(args.e3_report).read_text(encoding="utf-8")),
            json.loads(Path(args.targets).read_text(encoding="utf-8"))["targets"])
        report["StrokePlacementMatchesE3"] = checks

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Selection")
    for s in selection:
        print(f"  {s['file']:<16} {s['interactions']} interactions, {s['decoder_points']:>2} points "
              f"({s['negative_points']} negative), IoU {s['selection_reference_iou']:.3f}, rectangle "
              f"{s['rectangle_area_fraction']:.3f} {'used' if s['rectangle_used'] else 'not used'}, "
              f"cell {s['photo_logit_cell_px']:.1f}/"
              f"{(s['rectangle_logit_cell_px'] or s['photo_logit_cell_px']):.1f} px, photo vs "
              f"rectangle {s['photo_vs_rectangle_iou']:.3f}; {s['tile_count']} tiles, photo vs tiles "
              f"{s['photo_vs_tiles_iou']:.3f} (whole tiles {s['photo_vs_tiles_unlimited_iou']:.3f}, no mask_input "
              f"{s['photo_vs_tiles_without_mask_input_iou']:.3f}), reference IoU tiles "
              f"{s['reference_iou']['tiles_bilinear']:.3f} (no mask_input "
              f"{s['tiles_without_mask_input_reference_iou']:.3f})")
    for title, table in (("Main variants, all targets", report["main"]),
                         ("Main variants, targets with a rectangle pass", report["main_small_subjects"]),
                         ("Sensitivity (tile pass)", report["sensitivity"])):
        print(title)
        print("  variant                 hair width  hard width (max)  hair specks  hard specks"
              "  hair grad  hard grad  cpu ms med/max  min ref IoU")
        for name, row in table.items():
            print(f"  {name:<22} {row['hair_soft_width_px_median']:>10.1f}  "
                  f"{row['hard_soft_width_px_median']:>10.1f} ({row['hard_soft_width_px_max']:>5.1f})"
                  f"  {row['hair_specks']:>11}  {row['hard_specks']:>11}  "
                  f"{row['hair_edge_gradient_median']:>9.2f}  {row['hard_edge_gradient_median']:>9.2f}"
                  f"  {row['cpu_ms_median']:>6.0f}/{row['cpu_ms_max']:<6.0f}  "
                  f"{row['reference_iou_min']:.3f}")
    print("Per region (soft width px / specks / edge gradient)")
    for row in per_region:
        print(f"  {Path(row['file']).stem:<11} {row['region']:<28} {row['kind']:<4} " + "  ".join(
            f"{n.replace('rectangle', 'rect')}: {row[n]['soft_width_px']:.1f}/"
            f"{row[n]['islands'] + row[n]['holes']}/{row[n]['edge_gradient']:.2f}"
            for n in main_names))
    if args.e3_report:
        print("StrokePlacementMatchesE3")
        for c in report["StrokePlacementMatchesE3"]:
            print(f"  {c['file']} {c['e3_prompt']}: {c['max_endpoint_distance_px']:.2f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
