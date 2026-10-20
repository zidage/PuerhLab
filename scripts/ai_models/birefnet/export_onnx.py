#!/usr/bin/env python3
"""Export the BiRefNet subject matte graphs to ONNX (#458, step E2).

For each variant in --variants, the script writes <out-root>/<variant>/subject_matte.onnx and
export_report.json. The report records the Hugging Face revision, the opset, the input and output
shapes, the file sizes and SHA-256 values, the parameter count, and the exporter error if the
export fails. Its `model_input` entry tells `segmentation_runtime_report matte` the input size and
the pixel normalization (see birefnet_graphs.py).

BiRefNet uses `torchvision.ops.deform_conv2d` in its ASPP decoder blocks. The TorchScript
exporter has no symbolic for it. Two decompositions into standard operators are available:

- `--deform-conv grid_sample` (default): `birefnet_graphs.deform_conv2d_grid_sample`, one
  GridSample and one 1 x 1 Conv per kernel point. The script first checks it against torchvision
  on random inputs (`GridSampleDeformConvMatchesTorchvision`, required maximum difference 1e-5
  relative to the largest output).
- `--deform-conv gathernd`: the symbolic of `deform_conv2d_onnx_exporter` (the method of the
  BiRefNet ONNX tutorial). It gathers all kernel points at once; DirectML runs out of memory with
  it (see the E2 report).
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

import birefnet_graphs as graphs  # noqa: E402


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

    operator_counts: dict[str, int] = {}
    for node in model.graph.node:
        operator_counts[node.op_type] = operator_counts.get(node.op_type, 0) + 1
    files = [path] + sorted(path.parent.glob(path.name + ".data"))
    return {
        "inputs": {i.name: shape_of(i) for i in model.graph.input},
        "outputs": {o.name: shape_of(o) for o in model.graph.output},
        "opset": [{"domain": o.domain or "ai.onnx", "version": o.version}
                  for o in model.opset_import],
        "node_count": len(model.graph.node),
        "operator_counts": dict(sorted(operator_counts.items())),
        "files": [{"name": f.name, "bytes": f.stat().st_size, "sha256": sha256_of(f)}
                  for f in files],
    }


def register_deform_conv2d(exporter_source: str) -> None:
    sys.path.insert(0, os.path.abspath(exporter_source))
    import deform_conv2d_onnx_exporter

    if deform_conv2d_onnx_exporter.JitScalarType is None:
        # PyTorch 2.14 has the TorchScript exporter types in a new module. Without them, the
        # exporter uses the dtype tables of PyTorch 1.x, which do not accept JitScalarType keys.
        from torch.onnx._internal.torchscript_exporter._type_utils import JitScalarType

        deform_conv2d_onnx_exporter.JitScalarType = JitScalarType

    known_size = deform_conv2d_onnx_exporter.get_tensor_dim_size

    def tensor_dim_size(tensor, dim):
        # The TorchScript graph has no static size for some decoder tensors. Their strides are
        # known; for a contiguous NCHW tensor with batch 1 they give the sizes (the fix of
        # tutorials/BiRefNet_pth2onnx.ipynb in the BiRefNet repository).
        size = known_size(tensor, dim)
        if size is not None:
            return size
        strides = tensor.type().strides()
        if dim == 3:
            return strides[2]
        if dim == 2:
            return strides[1] // strides[2]
        if dim == 1:
            return strides[0] // strides[1]
        if dim == 0:
            return 1
        raise ValueError(f"no size for dimension {dim}")

    deform_conv2d_onnx_exporter.get_tensor_dim_size = tensor_dim_size
    deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()


REQUIRED_DEFORM_CONV_DIFFERENCE = 1e-5


def export_variant(variant: str, args) -> dict:
    repository, size, backbone, task = graphs.VARIANTS[variant]
    directory = graphs.variant_directory(args.weights, variant)
    out_dir = Path(args.out_root) / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    model = graphs.build_model(args.weights, variant)
    if args.deform_conv == "grid_sample":
        graphs.use_grid_sample_deform_conv(model)
    matte = graphs.SubjectMatte(model).eval()
    revisions = json.loads(Path(args.revisions).read_text(encoding="utf-8"))
    weights = directory / "model.safetensors"
    report: dict = {
        "model_family": "birefnet",
        "variant": variant,
        "repository": repository,
        "revision": revisions.get(repository),
        "backbone": backbone,
        "task": task,
        "torch": torch.__version__,
        "opset": args.opset,
        "deform_conv": args.deform_conv,
        "weights": {"name": weights.name, "bytes": weights.stat().st_size,
                    "sha256": sha256_of(weights)},
        "model_input": {
            "image_size": size,
            "low_res_mask_size": size,
            "pixel_mean": list(graphs.NORMALIZE_MEAN),
            "pixel_std": list(graphs.NORMALIZE_STD),
            "resize": "stretch to image_size x image_size, bilinear with antialias, no padding",
            "output": "alpha in [0, 1] (sigmoid in the graph)",
        },
        "parameters": sum(p.numel() for p in model.parameters()),
        "graphs": {},
    }
    path = out_dir / "subject_matte.onnx"
    entry: dict = {"path": path.name}
    report["graphs"]["subject_matte"] = entry
    pixels = torch.randn(1, 3, size, size)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            torch.onnx.export(matte, (pixels,), str(path), input_names=["pixel_values"],
                              output_names=["alpha"], opset_version=args.opset, dynamo=False)
        entry["status"] = "exported"
        entry["export_seconds"] = round(time.perf_counter() - started, 2)
        entry.update(describe_onnx(path))
        print(f"[export] {variant}/subject_matte: {entry['files'][0]['bytes'] / 1e6:.1f} MB")
    except Exception as error:  # Report the real exporter error; the run continues.
        entry["status"] = "failed"
        entry["error"] = f"{type(error).__name__}: {error}"
        entry["traceback"] = traceback.format_exc()
        print(f"[export] {variant}/subject_matte FAILED: {entry['error']}", file=sys.stderr)
    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True,
                        help="directory with one Hugging Face snapshot directory per variant")
    parser.add_argument("--revisions", required=True,
                        help="JSON file: repository -> Hugging Face commit of the snapshot")
    parser.add_argument("--deform-conv", choices=("grid_sample", "gathernd"),
                        default="grid_sample")
    parser.add_argument("--deform-exporter",
                        help="src directory of the deform_conv2d_onnx_exporter checkout "
                             "(--deform-conv gathernd)")
    parser.add_argument("--variants", default=",".join(graphs.VARIANTS))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    if args.deform_conv == "gathernd":
        if not args.deform_exporter:
            parser.error("--deform-conv gathernd needs --deform-exporter")
        register_deform_conv2d(args.deform_exporter)
    else:
        difference = graphs.check_grid_sample_deform_conv()
        passed = difference <= REQUIRED_DEFORM_CONV_DIFFERENCE
        print(f"[export] GridSampleDeformConvMatchesTorchvision: maximum relative difference "
              f"{difference:.2e} ({'passed' if passed else 'FAILED'})")
        if not passed:
            return 1
    failed = []
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        report = export_variant(variant, args)
        failed += [f"{variant}/{n}" for n, g in report["graphs"].items()
                   if g["status"] != "exported"]
    print(f"[export] failed graphs: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
