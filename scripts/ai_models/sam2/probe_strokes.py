#!/usr/bin/env python3
"""Tables of the stroke selection report (#458, step E3).

Each --report is `label=report_json[|comparison_report_json[|gpu_memory_json]]`: the report of
`segmentation_runtime_report strokes` for one model, optionally a report of the same model with
other decoder settings (--pad-points or --decoder-sessions per-point-count), and the output of
`run_with_gpu_memory.ps1` for the first report. The script prints the tables of the report and
writes them as JSON to --out.

A selection reaches the target if its reference IoU is >= 0.85 (the rule of step E4).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

TARGET_IOU = 0.85


def prompts(report: dict):
    for image in report["images"]:
        for prompt in image["prompts"]:
            if prompt.get("status") == "unavailable":
                continue
            yield image["file"], prompt


def stem(file: str) -> str:
    return Path(file).stem


def reached(value) -> bool:
    return value is not None and value >= TARGET_IOU


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def sweep_entries(prompt: dict, **match):
    for entry in prompt["sweep"]:
        if all(entry.get(k) == v for k, v in match.items()):
            yield entry


def primary_stroke(prompt: dict, report: dict) -> dict | None:
    primary = report["strokes"]["primary"]
    for entry in sweep_entries(prompt, direction="principal",
                               length_of_inner_chord=primary["length_of_inner_chord"],
                               count=primary["count"], spacing_of_short_side=None):
        return entry
    return None


def first_interaction(report: dict) -> dict:
    rows = {
        "click, best predicted IoU (E4 default)": [],
        "click, single-mask output": [],
        "stroke, single-mask output": [],
        "stroke, best predicted IoU of the multimask output": [],
    }
    intents = defaultdict(lambda: defaultdict(int))
    for _, prompt in prompts(report):
        click = prompt["click"]["result"]
        stroke = primary_stroke(prompt, report)
        values = [click["best_score_iou"], click["single_mask_iou"],
                  stroke and stroke["result"]["single_mask_iou"],
                  stroke and stroke["result"]["best_score_iou"]]
        for (name, items), value in zip(rows.items(), values):
            items.append(value)
            intents[name][prompt["intent"]] += reached(value)
    return {name: {"reached": sum(map(reached, values)), "prompts": len(values),
                   "object": intents[name]["object"], "part": intents[name]["part"],
                   "median_iou": median(values)}
            for name, values in rows.items()}


def per_prompt(reports: dict) -> list[dict]:
    rows = []
    first = next(iter(reports.values()))
    for file, prompt in prompts(first):
        row = {"photo": stem(file), "prompt": prompt["name"], "intent": prompt["intent"]}
        for label, report in reports.items():
            match = next(p for f, p in prompts(report) if f == file and p["name"] == prompt["name"])
            stroke = primary_stroke(match, report)
            sequence = match["sequences"]["stroke_then_strokes"]
            clicks = match["sequences"]["click_then_clicks"]
            row[label] = {
                "click": match["click"]["result"]["best_score_iou"],
                "stroke": stroke and stroke["result"]["single_mask_iou"],
                "strokes_to_target": sequence.get("interactions_to_iou_0_85"),
                "clicks_to_target": clicks.get("interactions_to_iou_0_85"),
            }
        rows.append(row)
    return rows


def sweep_table(report: dict, key: str, values, **fixed) -> list[dict]:
    """Targets reached and median IoU per value of `key`; the other sweep settings are `fixed`."""
    table = []
    for value in values:
        ious, prompts_reached, strokes = [], 0, 0
        for _, prompt in prompts(report):
            entries = list(sweep_entries(prompt, **{key: value}, **fixed))
            if not entries:
                continue
            prompt_ious = [e["result"]["single_mask_iou"] for e in entries]
            ious += prompt_ious
            strokes += len(entries)
            prompts_reached += reached(median(prompt_ious)) if len(entries) > 1 else reached(prompt_ious[0])
        table.append({key: value, "strokes": strokes,
                      "share_reached": sum(map(reached, ious)) / len(ious) if ious else None,
                      "prompts_reached": prompts_reached, "median_iou": median(ious)})
    return table


def length_count_grid(report: dict, counts=(2, 3, 4, 5, 8)) -> dict:
    """Prompts reached by the principal-axis stroke and the share of all strokes reached, per
    length (rows) and point count (columns)."""
    grid = {}
    for length in report["strokes"]["lengths_of_inner_chord"]:
        row = {}
        for count in counts:
            principal, every = [], []
            for _, prompt in prompts(report):
                for entry in sweep_entries(prompt, length_of_inner_chord=length, count=count,
                                           spacing_of_short_side=None):
                    every.append(entry["result"]["single_mask_iou"])
                    if entry["direction"] == "principal":
                        principal.append(entry["result"]["single_mask_iou"])
            row[count] = {"principal_prompts_reached": sum(map(reached, principal)),
                          "all_directions_share_reached": sum(map(reached, every)) / len(every)}
        grid[length] = row
    return grid


def spacing_table(report: dict) -> list[dict]:
    table = []
    for spacing in report["strokes"]["spacings_of_short_side"]:
        ious, counts = [], []
        for _, prompt in prompts(report):
            for entry in sweep_entries(prompt, spacing_of_short_side=spacing):
                ious.append(entry["result"]["single_mask_iou"])
                counts.append(entry["count"])
        table.append({"spacing_of_short_side": spacing, "strokes": len(ious),
                      "median_points": median(counts), "maximum_points": max(counts),
                      "share_reached": sum(map(reached, ious)) / len(ious),
                      "median_iou": median(ious)})
    return table


def sequence_table(report: dict, limit: int) -> list[dict]:
    table = []
    names = next(iter(prompts(report)))[1]["sequences"].keys()
    for name in names:
        counts = defaultdict(int)
        finals, adds, subtracts = [], [], []
        for _, prompt in prompts(report):
            sequence = prompt["sequences"][name]
            if "steps" not in sequence:
                counts["unavailable"] += 1
                continue
            n = sequence["interactions_to_iou_0_85"]
            counts[n if n is not None else "not reached"] += 1
            finals.append(sequence["final_reference_iou"])
            steps = sequence["steps"]
            for before, after in zip(steps, steps[1:]):
                gain = after["reference_iou"] - before["reference_iou"]
                (adds if after["label"] == 1 else subtracts).append(gain)
        reached_counts = {k: counts.get(k, 0) for k in range(1, limit + 1)}
        table.append({
            "sequence": name, "reached_after": reached_counts,
            "not_reached": counts.get("not reached", 0),
            "unavailable": counts.get("unavailable", 0),
            "mean_interactions_not_reached_as_limit_plus_one": statistics.mean(
                [k for k, c in reached_counts.items() for _ in range(c)]
                + [limit + 1] * counts.get("not reached", 0)),
            "median_final_iou": median(finals),
            "add_corrections": len(adds), "median_add_gain": median(adds),
            "subtract_corrections": len(subtracts), "median_subtract_gain": median(subtracts),
            "subtract_corrections_that_lowered_iou": sum(g < 0 for g in subtracts),
            "add_corrections_that_lowered_iou": sum(g < 0 for g in adds),
        })
    return table


def timing_table(report: dict, comparison: dict | None) -> list[dict]:
    rows = []
    timing = report["decoder_time_per_stroke"]["points"]
    comparison_timing = comparison["decoder_time_per_stroke"]["points"] if comparison else {}
    for count, entry in timing.items():
        row = {"points": int(count)}
        for session in ("new_session", "report_session"):
            for mask in ("without_mask_input", "with_mask_input"):
                row[f"{session}_{mask}_p50_ms"] = entry[session][mask]["p50_ms"]
        if count in comparison_timing:
            row["comparison_report_session_without_mask_input_p50_ms"] = (
                comparison_timing[count]["report_session"]["without_mask_input"]["p50_ms"])
            row["comparison_report_session_with_mask_input_p50_ms"] = (
                comparison_timing[count]["report_session"]["with_mask_input"]["p50_ms"])
        rows.append(row)
    return rows


def comparison_agreement(report: dict, comparison: dict) -> dict:
    """Reference IoU of the same sweep stroke and corrective sequence, the report against the comparison report."""
    differences, decisions, total = [], 0, 0
    sequence_differences = []
    for (file, prompt), (_, comparison_prompt) in zip(prompts(report), prompts(comparison)):
        assert prompt["name"] == comparison_prompt["name"]
        for a, b in zip(prompt["sweep"], comparison_prompt["sweep"]):
            ia, ib = a["result"]["single_mask_iou"], b["result"]["single_mask_iou"]
            differences.append(abs(ia - ib))
            decisions += reached(ia) != reached(ib)
            total += 1
        for name, sequence in prompt["sequences"].items():
            other = comparison_prompt["sequences"].get(name, {})
            if "steps" in sequence and "steps" in other:
                sequence_differences.append(
                    abs(sequence["final_reference_iou"] - other["final_reference_iou"]))
    return {"sweep_strokes": total, "median_abs_iou_difference": median(differences),
            "maximum_abs_iou_difference": max(differences),
            "target_decision_changed": decisions,
            "sequence_final_iou_maximum_abs_difference": max(sequence_differences),
            "first_interaction_comparison": first_interaction(comparison),
            "sequences_comparison": sequence_table(comparison, comparison["strokes"]["corrective_steps"] + 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    reports, comparison_reports, memory = {}, {}, {}
    for item in args.report:
        label, paths = item.split("=", 1)
        parts = paths.split("|")
        reports[label] = json.loads(Path(parts[0]).read_text(encoding="utf-8"))
        if len(parts) > 1 and parts[1]:
            comparison_reports[label] = json.loads(Path(parts[1]).read_text(encoding="utf-8"))
        if len(parts) > 2 and parts[2]:
            memory[label] = json.loads(Path(parts[2]).read_text(encoding="utf-8"))

    out = {"per_prompt": per_prompt(reports), "models": {}}
    for label, report in reports.items():
        settings = report["strokes"]
        primary = settings["primary"]
        limit = settings["corrective_steps"] + 1
        comparison = comparison_reports.get(label)
        model = {
            "first_interaction": first_interaction(report),
            "count_principal": sweep_table(report, "count", [2, 3, 4, 5, 8],
                                           direction="principal",
                                           length_of_inner_chord=primary["length_of_inner_chord"],
                                           spacing_of_short_side=None),
            "count_all_strokes": sweep_table(report, "count", [2, 3, 4, 5, 8],
                                             spacing_of_short_side=None),
            "length_principal": sweep_table(report, "length_of_inner_chord",
                                            settings["lengths_of_inner_chord"],
                                            direction="principal", count=primary["count"],
                                            spacing_of_short_side=None),
            "direction": sweep_table(report, "direction",
                                     ["principal", "0", "45", "90", "135"],
                                     length_of_inner_chord=primary["length_of_inner_chord"],
                                     count=primary["count"], spacing_of_short_side=None),
            "length_by_count": length_count_grid(report),
            "spacing": spacing_table(report),
            "sequences": sequence_table(report, limit),
            "timing": timing_table(report, comparison),
            "load_ms": {k: v.get("load_ms") for k, v in report["graphs"].items()},
        }
        if comparison:
            model["comparison_agreement"] = comparison_agreement(report, comparison)
        if label in memory:
            model["gpu_memory_mb"] = {
                "peak_dedicated": memory[label]["peak_dedicated_bytes"] / 2**20,
                "peak_shared": memory[label]["peak_shared_bytes"] / 2**20,
            }
        out["models"][label] = model

    for label, model in out["models"].items():
        print(f"\n## {label}\n")
        print("| First interaction | Targets reached | Object | Part | Median IoU |")
        print("| --- | ---: | ---: | ---: | ---: |")
        for name, row in model["first_interaction"].items():
            print(f"| {name} | {row['reached']} / {row['prompts']} | {row['object']} | "
                  f"{row['part']} | {fmt(row['median_iou'])} |")
        for key in ("count_principal", "count_all_strokes", "length_principal", "direction"):
            print(f"\n{key}:")
            for row in model[key]:
                first_key = next(iter(row))
                print(f"  {first_key}={row[first_key]}: strokes {row['strokes']}, "
                      f"share {fmt(row['share_reached'])}, prompts {row['prompts_reached']}, "
                      f"median {fmt(row['median_iou'], 3)}")
        print("\nlength by count (principal prompts reached / all-direction share):")
        for length, row in model["length_by_count"].items():
            print(f"  {length}: " + "  ".join(
                f"{c}: {v['principal_prompts_reached']} / {v['all_directions_share_reached']:.2f}"
                for c, v in row.items()))
        print("\nspacing:")
        for row in model["spacing"]:
            print(f"  {row}")
        print("\nsequences:")
        for row in model["sequences"]:
            print(f"  {row}")
        print("\ntiming:")
        for row in model["timing"]:
            print(f"  {row}")
        if "comparison_agreement" in model:
            agreement = {k: v for k, v in model["comparison_agreement"].items()
                         if not k.endswith("_comparison")}
            print(f"\ncomparison agreement: {agreement}")
            for name, row in model["comparison_agreement"]["first_interaction_comparison"].items():
                print(f"  comparison {name}: {row['reached']} / {row['prompts']}")
            for row in model["comparison_agreement"]["sequences_comparison"]:
                print(f"  comparison {row['sequence']}: {row['reached_after']} "
                      f"not reached {row['not_reached']}")
        if "gpu_memory_mb" in model:
            print(f"\nGPU memory: {model['gpu_memory_mb']}")
    print("\nper prompt:")
    for row in out["per_prompt"]:
        cells = [f"{row['photo']} {row['prompt']} ({row['intent']})"]
        for label in reports:
            r = row[label]
            cells.append(f"{label}: click {fmt(r['click'])} stroke {fmt(r['stroke'])} "
                         f"strokes {r['strokes_to_target']} clicks {r['clicks_to_target']}")
        print("  " + " | ".join(cells))

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[probe] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
