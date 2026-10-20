#!/usr/bin/env python3
"""Export the SAM 2.1 point and box graphs to ONNX (#458, steps E1 and E4).

For each variant in --variants, the script writes the graphs of --graphs to <out-root>/<variant>/
(point_encoder.onnx, point_decoder.onnx, and the batched point-grid graph
candidate_decoder.onnx of step E4) and export_report.json. The report records the opset, the input and output
shapes, the file sizes and SHA-256 values, the parameter counts, and the exporter error of each
graph that fails. Its `model_input` entry tells `segmentation_runtime_report` the input size,
the low-resolution mask size, and the pixel normalization (see sam2_graphs.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import sam2_graphs as graphs  # noqa: E402


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_onnx(path: Path) -> dict:
    import onnx

    model = onnx.load(str(path), load_external_data=False)

    def shape_of(value_info):
        return [d.dim_param or d.dim_value for d in value_info.type.tensor_type.shape.dim]

    files = [path] + sorted(path.parent.glob(path.name + ".data"))
    return {
        "inputs": {i.name: shape_of(i) for i in model.graph.input},
        "outputs": {o.name: shape_of(o) for o in model.graph.output},
        "opset": [{"domain": o.domain or "ai.onnx", "version": o.version}
                  for o in model.opset_import],
        "node_count": len(model.graph.node),
        "operator_types": sorted({n.op_type for n in model.graph.node}),
        "files": [{"name": f.name, "bytes": f.stat().st_size, "sha256": sha256_of(f)}
                  for f in files],
    }


def export_graph(name: str, module, inputs: tuple, input_names: list[str],
                 output_names: list[str], dynamic_axes: dict, out_dir: Path, opset: int,
                 report: dict) -> None:
    path = out_dir / f"{name}.onnx"
    entry: dict = {"path": path.name}
    report["graphs"][name] = entry
    started = time.perf_counter()
    try:
        with torch.no_grad():
            torch.onnx.export(module, inputs, str(path), input_names=input_names,
                              output_names=output_names, dynamic_axes=dynamic_axes,
                              opset_version=opset, dynamo=False)
        entry["status"] = "exported"
        entry["export_seconds"] = round(time.perf_counter() - started, 2)
        entry.update(describe_onnx(path))
        print(f"[export] {out_dir.name}/{name}: {entry['files'][0]['bytes'] / 1e6:.1f} MB")
    except Exception as error:  # Report the real exporter error; the run continues.
        entry["status"] = "failed"
        entry["error"] = f"{type(error).__name__}: {error}"
        entry["traceback"] = traceback.format_exc()
        print(f"[export] {out_dir.name}/{name} FAILED: {entry['error']}", file=sys.stderr)


def parameter_count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def export_variant(variant: str, args) -> dict:
    _, checkpoint_name = graphs.VARIANTS[variant]
    checkpoint = Path(args.weights) / checkpoint_name
    out_dir = Path(args.out_root) / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    model = graphs.build_model(variant, str(checkpoint))
    encoder = graphs.PointEncoder(model).eval()
    decoder = graphs.PointDecoder(model).eval()
    report: dict = {
        "model_family": "sam2.1",
        "variant": variant,
        "torch": torch.__version__,
        "opset": args.opset,
        "checkpoint": {"name": checkpoint_name, "bytes": checkpoint.stat().st_size,
                       "sha256": sha256_of(checkpoint)},
        "model_input": {
            "image_size": graphs.IMAGE_SIZE,
            "low_res_mask_size": graphs.LOW_RES_MASK_SIZE,
            "pixel_mean": list(graphs.NORMALIZE_MEAN),
            "pixel_std": list(graphs.NORMALIZE_STD),
            "resize": "stretch to image_size x image_size, bilinear with antialias, no padding",
        },
        "parameters": {
            "image_encoder": parameter_count(model.image_encoder),
            "prompt_encoder": parameter_count(model.sam_prompt_encoder),
            "mask_decoder": parameter_count(model.sam_mask_decoder),
            "whole_checkpoint": parameter_count(model),
        },
        "graphs": {},
    }
    pixels = torch.randn(1, 3, graphs.IMAGE_SIZE, graphs.IMAGE_SIZE)
    with torch.no_grad():
        image_embed, feats_0, feats_1 = encoder(pixels)
    report["point_feature_shapes"] = {
        "image_embed": list(image_embed.shape),
        "high_res_feats_0": list(feats_0.shape),
        "high_res_feats_1": list(feats_1.shape),
    }
    if "point_encoder" in args.graphs:
        export_graph("point_encoder", encoder, (pixels,), ["pixel_values"],
                     ["image_embed", "high_res_feats_0", "high_res_feats_1"], {}, out_dir,
                     args.opset, report)

    if "point_decoder" in args.graphs:
        coords = torch.tensor([[[400.0, 300.0], [600.0, 500.0]]])
        labels = torch.tensor([[1.0, 0.0]])
        mask_input = torch.zeros(1, 1, graphs.LOW_RES_MASK_SIZE, graphs.LOW_RES_MASK_SIZE)
        has_mask_input = torch.zeros(1)
        export_graph(
            "point_decoder", decoder,
            (image_embed, feats_0, feats_1, coords, labels, mask_input, has_mask_input),
            ["image_embed", "high_res_feats_0", "high_res_feats_1", "point_coords",
             "point_labels", "mask_input", "has_mask_input"],
            ["mask_logits", "iou_scores", "object_score_logits"],
            {"point_coords": {1: "point_count"}, "point_labels": {1: "point_count"}},
            out_dir, args.opset, report)

    if "candidate_decoder" in args.graphs:
        candidates = graphs.CandidateDecoder(model).eval()
        # --candidate-batch 0: five prompts at export time and a dynamic prompt axis. Otherwise
        # the prompt count is fixed to that value.
        count = args.candidate_batch or 5
        grid = graphs.grid_points(16)[:count] * graphs.IMAGE_SIZE
        dynamic = {} if args.candidate_batch else {
            "point_coords": {0: "prompt_count"}, "mask_logits": {0: "prompt_count"},
            "iou_scores": {0: "prompt_count"}, "stability_scores": {0: "prompt_count"}}
        export_graph(
            "candidate_decoder", candidates, (image_embed, feats_0, feats_1, grid[:, None, :]),
            ["image_embed", "high_res_feats_0", "high_res_feats_1", "point_coords"],
            ["mask_logits", "iou_scores", "stability_scores"],
            dynamic, out_dir, args.opset, report)
        report["candidate_decoder"] = {
            "stability_offset": candidates.stability_offset,
            "tokens": "multimask tokens 1-3 of the mask decoder",
            "prompt_count": args.candidate_batch or "dynamic",
            "float16": args.candidate_float16,
        }
        entry = report["graphs"]["candidate_decoder"]
        if args.candidate_float16 and entry["status"] == "exported":
            import onnx
            from onnxruntime.transformers.float16 import convert_float_to_float16

            # Float16 inside the graph, float32 inputs and outputs, so the host code stays the
            # same. The point graphs stay float32.
            path = out_dir / "candidate_decoder.onnx"
            onnx.save(convert_float_to_float16(onnx.load(str(path)), keep_io_types=True),
                      str(path))
            entry.update(describe_onnx(path))

    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repository", required=True, help="SAM 2 repository checkout")
    parser.add_argument("--weights", required=True, help="directory with the .pt checkpoints")
    parser.add_argument("--variants", default="tiny,small,base_plus,large")
    parser.add_argument("--graphs", default="point_encoder,point_decoder,candidate_decoder",
                        type=lambda text: {g.strip() for g in text.split(",") if g.strip()})
    parser.add_argument("--candidate-batch", type=int, default=0,
                        help="fixed prompt count of candidate_decoder.onnx; 0 is dynamic")
    parser.add_argument("--candidate-float16", action="store_true",
                        help="convert candidate_decoder.onnx to float16 inside the graph")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    graphs.add_repository_to_path(args.repository)
    failed = []
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        report = export_variant(variant, args)
        failed += [f"{variant}/{n}" for n, g in report["graphs"].items()
                   if g["status"] != "exported"]
    print(f"[export] failed graphs: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
