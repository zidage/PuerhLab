#!/usr/bin/env python3
"""Export the EfficientSAM3 graphs that Alcedo Studio needs to ONNX.

Graphs (see esam3_graphs.py for the tensor layout):

  point_encoder.onnx       from --point-checkpoint (stage 1 checkpoint with tracker weights)
  point_decoder.onnx       from --point-checkpoint
  text_encoder.onnx        from --text-checkpoint (stage 3 fine-tuned checkpoint)
  detector_grounding.onnx  from --text-checkpoint

The script writes export_report.json next to the graphs. The report records the opset, the
input and output shapes, the file sizes, and the exporter error of each graph that fails.
It also records the token ids of the text prompts in --prompts, so that the host tokenizer can
be compared with the Python BPE tokenizer.
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

import esam3_graphs as graphs  # noqa: E402


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
        dims = []
        for dim in value_info.type.tensor_type.shape.dim:
            dims.append(dim.dim_param if dim.dim_param else dim.dim_value)
        return dims

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
        print(f"[export] {name}: {entry['files'][0]['bytes'] / 1e6:.1f} MB")
    except Exception as error:  # Report the real exporter error; the run continues.
        entry["status"] = "failed"
        entry["error"] = f"{type(error).__name__}: {error}"
        entry["traceback"] = traceback.format_exc()
        print(f"[export] {name} FAILED: {entry['error']}", file=sys.stderr)


def export_point_graphs(args, out_dir: Path, report: dict) -> None:
    model = graphs.build_point_model(args.point_checkpoint, args.backbone_type, args.model_name)
    encoder = graphs.PointEncoder(model).eval()
    decoder = graphs.PointDecoder(model).eval()
    pixels = torch.randn(1, 3, graphs.IMAGE_SIZE, graphs.IMAGE_SIZE)
    with torch.no_grad():
        image_embed, feats_0, feats_1 = encoder(pixels)
    report["point_feature_shapes"] = {
        "image_embed": list(image_embed.shape),
        "high_res_feats_0": list(feats_0.shape),
        "high_res_feats_1": list(feats_1.shape),
    }
    export_graph("point_encoder", encoder, (pixels,), ["pixel_values"],
                 ["image_embed", "high_res_feats_0", "high_res_feats_1"], {}, out_dir,
                 args.opset, report)

    coords = torch.tensor([[[400.0, 300.0], [600.0, 500.0]]])
    labels = torch.tensor([[1.0, 0.0]])
    mask_input = torch.zeros(1, 1, graphs.LOW_RES_MASK_SIZE, graphs.LOW_RES_MASK_SIZE)
    has_mask_input = torch.zeros(1)
    export_graph(
        "point_decoder", decoder,
        (image_embed, feats_0, feats_1, coords, labels, mask_input, has_mask_input),
        ["image_embed", "high_res_feats_0", "high_res_feats_1", "point_coords", "point_labels",
         "mask_input", "has_mask_input"],
        ["mask_logits", "iou_scores", "object_score_logits"],
        {"point_coords": {1: "point_count"}, "point_labels": {1: "point_count"}},
        out_dir, args.opset, report)


def export_text_graphs(args, out_dir: Path, report: dict, prompts: dict) -> None:
    model = graphs.build_text_model(args.text_checkpoint, args.backbone_type, args.model_name,
                                    args.text_encoder)
    text_encoder = graphs.TextEncoder(model).eval()
    grounding = graphs.DetectorGrounding(model).eval()

    texts = sorted({t for image in prompts.get("images", []) for t in image.get("text", [])})
    report["text_tokens"] = {t: graphs.tokenize(model, t)[0].tolist() for t in texts}

    tokens = graphs.tokenize(model, "dog")
    export_graph("text_encoder", text_encoder, (tokens,), ["tokenized_ids"],
                 ["language_mask", "language_features", "language_embeds"], {}, out_dir,
                 args.opset, report)
    with torch.no_grad():
        language_mask, language_features, language_embeds = text_encoder(tokens)
    pixels = torch.randn(1, 3, graphs.IMAGE_SIZE, graphs.IMAGE_SIZE)
    export_graph(
        "detector_grounding", grounding,
        (pixels, language_features, language_mask, language_embeds),
        ["pixel_values", "language_features", "language_mask", "language_embeds"],
        ["boxes_xyxy", "scores", "mask_logits"], {}, out_dir, args.opset, report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repository", required=True,
                        help="EfficientSAM3 repository checkout at the recorded commit")
    parser.add_argument("--point-checkpoint")
    parser.add_argument("--text-checkpoint")
    parser.add_argument("--backbone-type", default="efficientvit")
    parser.add_argument("--model-name", default="b1")
    parser.add_argument("--text-encoder", default="MobileCLIP-S0")
    parser.add_argument("--prompts", help="prompts.json; its text prompts are tokenized")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if not args.point_checkpoint and not args.text_checkpoint:
        parser.error("give --point-checkpoint, --text-checkpoint, or both")

    graphs.add_repository_to_path(args.repository)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8")) if args.prompts else {}
    report: dict = {
        "torch": torch.__version__,
        "opset": args.opset,
        "image_size": graphs.IMAGE_SIZE,
        "low_res_mask_size": graphs.LOW_RES_MASK_SIZE,
        "normalize_mean": graphs.NORMALIZE_MEAN,
        "normalize_std": graphs.NORMALIZE_STD,
        "resize": "stretch to image_size x image_size, bilinear with antialias, no padding",
        "point_checkpoint": args.point_checkpoint,
        "text_checkpoint": args.text_checkpoint,
        "graphs": {},
    }
    if args.point_checkpoint:
        export_point_graphs(args, out_dir, report)
    if args.text_checkpoint:
        export_text_graphs(args, out_dir, report, prompts)

    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    failed = [n for n, g in report["graphs"].items() if g["status"] != "exported"]
    print(f"[export] report: {out_dir / 'export_report.json'}; failed graphs: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
