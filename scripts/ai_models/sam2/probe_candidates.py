#!/usr/bin/env python3
"""Summarize the click candidate reports of `segmentation_runtime_report candidates` (#458, E4).

Input: one or more reports (--report LABEL=PATH). Output: a JSON summary and Markdown tables on
stdout with

- per grid: the time to the candidates (encoder, candidate decoder, host filters), the candidate
  count, and the coverage (share of the photo inside at least one candidate);
- per single-point prompt: the reference IoU of the live point decoder (best predicted IoU token)
  and of the candidate levels (default level, best level under the pointer, level steps), and the
  hover result (share of nearby pointer positions with the same highlight);
- counts: prompts whose default selection is the intended mask (IoU >= --target-iou), prompts
  whose intended mask is one level step away, and prompts whose intended mask is not reachable.

With --upstream-dir and --mask-dir, it also compares the Rust candidate masks of one grid with the
masks of the upstream `SAM2AutomaticMaskGenerator` (`compare_candidate_outputs.py
--upstream-out`) on the 256 x 256 model grid: the share of upstream masks that have a Rust
candidate with IoU >= --match-iou, and the reverse.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def mean_of(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def load_rust_masks(directory: Path, stem: str) -> np.ndarray:
    info = json.loads((directory / f"{stem}.json").read_text(encoding="utf-8"))
    side = info["side"] or 256
    raw = np.fromfile(directory / f"{stem}.u8", dtype=np.uint8)
    return raw.reshape(-1, side, side).astype(bool)


def match_share(a: np.ndarray, b: np.ndarray, threshold: float) -> float | None:
    """Share of the masks in `a` that have a mask in `b` with IoU >= threshold."""
    if len(a) == 0:
        return None
    if len(b) == 0:
        return 0.0
    fa = a.reshape(len(a), -1).astype(np.float32)
    fb = b.reshape(len(b), -1).astype(np.float32)
    intersection = fa @ fb.T
    union = fa.sum(1)[:, None] + fb.sum(1)[None, :] - intersection
    iou = np.where(union > 0, intersection / np.maximum(union, 1), 1.0)
    return float((iou.max(1) >= threshold).mean())


def summarize(label: str, report: dict, args) -> dict:
    grids = [str(g) for g in report["generator"]["points_per_side"]]
    summary = {"label": label, "variant": report.get("variant"), "grids": {}, "prompts": [],
               "encoder_warm": report["graphs"].get("point_encoder", {}).get("warm")}
    for grid in grids:
        rows = [image["grids"][grid] for image in report["images"]]
        summary["grids"][grid] = {
            "time_to_candidates_ms_median": median(r["time_to_candidates_ms"] for r in rows),
            "time_to_candidates_ms_max": max(r["time_to_candidates_ms"] for r in rows),
            "decoder_ms_median": median(r["decoder_ms"] for r in rows),
            "host_ms_median": median(r["filter_ms"] + r["nms_ms"] for r in rows),
            "candidates_median": median(r["candidates"] for r in rows),
            "candidates_range": [min(r["candidates"] for r in rows),
                                 max(r["candidates"] for r in rows)],
            "coverage_median": median(r["coverage"] for r in rows),
            "coverage_min": min(r["coverage"] for r in rows),
            "warm_candidate_pass": rows[0].get("warm_candidate_pass"),
        }
    target = args.target_iou
    for image in report["images"]:
        for prompt in image["prompts"]:
            live = prompt["live_point_decoder"]
            live_ious = [level["reference_iou"] for level in live["levels"]]
            live_best = (max(range(len(live_ious)), key=lambda i: live_ious[i])
                         if all(v is not None for v in live_ious) and live_ious else None)
            entry = {
                "photo": image["file"], "prompt": prompt["name"], "intent": prompt["intent"],
                "has_reference": prompt["reference"] is not None,
                "live": {
                    "default_iou": live["default_best_score"]["reference_iou"],
                    "best_iou": live["best_reachable_iou"],
                    "levels": len(live["levels"]),
                    "level_ious": live_ious,
                    "default_level": live["default_best_score"]["level"],
                    "best_level": live_best,
                    "hover_same": live["default_best_score"]["hover"].get("same_highlight_share"),
                    "hover_same_kept_inside": live["default_best_score"].get(
                        "hover_kept_inside_highlight", {}).get("same_highlight_share"),
                    "decode_ms": live["decode_ms"],
                },
                "grids": {},
            }
            for grid in grids:
                candidates = prompt["candidates"][grid]
                best = candidates["best_reachable"]
                grid_entry = {"levels": len(candidates["levels"]),
                              "best_iou": best["reference_iou"] if best else None,
                              "lookup_us": candidates["lookup_us"]}
                for rule, default in candidates["defaults"].items():
                    grid_entry[rule] = {
                        "default_iou": default["reference_iou"],
                        "steps": default["steps_to_best_reachable"],
                        "hover_same": default["hover"].get("same_highlight_share"),
                    }
                entry["grids"][grid] = grid_entry
            summary["prompts"].append(entry)

    def counts(select):
        reference = [p for p in summary["prompts"] if p["has_reference"]]
        result = {"prompts_with_reference": len(reference), "default_is_target": 0,
                  "target_one_step_away": 0, "target_further_away": 0, "target_not_reachable": 0}
        for p in reference:
            default_iou, best_iou, steps = select(p)
            if default_iou is not None and default_iou >= target:
                result["default_is_target"] += 1
            elif best_iou is not None and best_iou >= target:
                key = "target_one_step_away" if steps == 1 else "target_further_away"
                result[key] += 1
            else:
                result["target_not_reachable"] += 1
        return result

    def live_select(p, rule):
        # The live decoder levels are the three tokens sorted by area (level 2 is the largest).
        live = p["live"]
        default = live["default_level"] if rule == "best_score" else len(live["level_ious"]) - 1
        steps = None if live["best_level"] is None else abs(live["best_level"] - default)
        return live["level_ious"][default], live["best_iou"], steps

    summary["counts"] = {
        "live_point_decoder_best_score": counts(lambda p: live_select(p, "best_score")),
        "live_point_decoder_largest": counts(lambda p: live_select(p, "largest")),
    }
    for grid in grids:
        for rule in ("largest", "best_score"):
            summary["counts"][f"candidates_{grid}_{rule}"] = counts(
                lambda p, g=grid, r=rule: (p["grids"][g][r]["default_iou"],
                                           p["grids"][g]["best_iou"], p["grids"][g][r]["steps"]))
    hover = {"live_point_decoder": median(p["live"]["hover_same"] for p in summary["prompts"]),
             "live_point_decoder_kept_inside": median(
                 p["live"]["hover_same_kept_inside"] for p in summary["prompts"])}
    hover_mean = {"live_point_decoder": mean_of(p["live"]["hover_same"] for p in summary["prompts"]),
                  "live_point_decoder_kept_inside": mean_of(
                      p["live"]["hover_same_kept_inside"] for p in summary["prompts"])}
    for grid in grids:
        for rule in ("largest", "best_score"):
            hover[f"candidates_{grid}_{rule}"] = median(
                p["grids"][grid][rule]["hover_same"] for p in summary["prompts"])
    for grid in grids:
        for rule in ("largest", "best_score"):
            hover_mean[f"candidates_{grid}_{rule}"] = mean_of(
                p["grids"][grid][rule]["hover_same"] for p in summary["prompts"])
    summary["hover_same_highlight_share_median"] = hover
    summary["hover_same_highlight_share_mean"] = hover_mean

    if args.upstream_dir and args.mask_dir:
        grid = args.upstream_grid
        matches = {}
        for image in report["images"]:
            stem = Path(image["file"]).stem
            upstream_path = Path(args.upstream_dir) / f"{stem}.npz"
            rust_dir = Path(args.mask_dir) / grid
            if not upstream_path.exists() or not (rust_dir / f"{stem}.json").exists():
                continue
            upstream = np.load(upstream_path)["masks"].astype(bool)
            rust = load_rust_masks(rust_dir, stem)
            matches[image["file"]] = {
                "upstream": len(upstream), "rust": len(rust),
                "upstream_matched_by_rust": match_share(upstream, rust, args.match_iou),
                "rust_matched_by_upstream": match_share(rust, upstream, args.match_iou),
            }
        summary["upstream_comparison"] = {
            "grid": grid, "match_iou": args.match_iou, "photos": matches,
            "upstream_matched_by_rust_median": median(
                m["upstream_matched_by_rust"] for m in matches.values()),
            "rust_matched_by_upstream_median": median(
                m["rust_matched_by_upstream"] for m in matches.values()),
        }
    return summary


def fmt(value, digits=2):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_tables(summaries: list[dict]) -> None:
    print("| Model | Grid | Time to candidates, median / max (ms) | Decoder, median (ms) | Host, median (ms) | Candidates, median (range) | Coverage, median / min |")
    print("| --- | ---: | --- | ---: | ---: | --- | --- |")
    for s in summaries:
        for grid, g in s["grids"].items():
            print(f"| {s['label']} | {grid} x {grid} | {g['time_to_candidates_ms_median']:.0f} / "
                  f"{g['time_to_candidates_ms_max']:.0f} | {g['decoder_ms_median']:.0f} | "
                  f"{g['host_ms_median']:.0f} | {g['candidates_median']:.0f} "
                  f"({g['candidates_range'][0]}-{g['candidates_range'][1]}) | "
                  f"{g['coverage_median']:.2f} / {g['coverage_min']:.2f} |")
    print()
    for s in summaries:
        print(f"Counts, {s['label']}:")
        print("| Selection | Default is the target | One level step | More steps | Not reachable |")
        print("| --- | ---: | ---: | ---: | ---: |")
        for name, c in s["counts"].items():
            print(f"| {name} | {c['default_is_target']} | {c['target_one_step_away']} | "
                  f"{c['target_further_away']} | {c['target_not_reachable']} |")
        print()
        print("Hover, median share of nearby pointer positions with the same highlight:",
              {k: fmt(v) for k, v in s["hover_same_highlight_share_median"].items()})
        print("Hover, mean share:",
              {k: fmt(v) for k, v in s["hover_same_highlight_share_mean"].items()})
        print()
    for s in summaries:
        grids = list(s["grids"])
        grid = grids[-1]
        print(f"Per prompt, {s['label']}, candidates {grid} x {grid} "
              "(reference IoU: live default / live best | candidate largest / best score / best level, levels):")
        print("| Photo | Prompt | Intent | Live default | Live best | Largest | Best score | Best level | Levels | Hover live / kept inside / best score |")
        print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for p in s["prompts"]:
            g = p["grids"][grid]
            print(f"| `{Path(p['photo']).stem}` | {p['prompt']} | {p['intent']} | "
                  f"{fmt(p['live']['default_iou'])} | {fmt(p['live']['best_iou'])} | "
                  f"{fmt(g['largest']['default_iou'])} | {fmt(g['best_score']['default_iou'])} | "
                  f"{fmt(g['best_iou'])} | {g['levels']} | "
                  f"{fmt(p['live']['hover_same'])} / {fmt(p['live']['hover_same_kept_inside'])} / "
                  f"{fmt(g['best_score']['hover_same'])} |")
        print()
        if "upstream_comparison" in s:
            u = s["upstream_comparison"]
            print(f"Upstream generator comparison ({u['grid']} x {u['grid']}, IoU >= {u['match_iou']}): "
                  f"upstream matched by Rust {fmt(u['upstream_matched_by_rust_median'])}, "
                  f"Rust matched by upstream {fmt(u['rust_matched_by_upstream_median'])} (medians)")
            print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="append", required=True, help="LABEL=PATH")
    parser.add_argument("--target-iou", type=float, default=0.85)
    parser.add_argument("--upstream-dir")
    parser.add_argument("--mask-dir", help="--mask-dir of the first report")
    parser.add_argument("--upstream-grid", default="32")
    parser.add_argument("--match-iou", type=float, default=0.8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    summaries = []
    for index, item in enumerate(args.report):
        label, path = item.split("=", 1)
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        if index > 0:
            args.upstream_dir = None
        summaries.append(summarize(label, report, args))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print_tables(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
