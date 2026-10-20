#!/usr/bin/env python3
"""Convert an exported BiRefNet graph to float16 weights and activations (#458, step E2).

The float32 Swin-L graphs need more than the 8 GB of dedicated memory of the test GPU on DirectML.
This script writes <out-dir>/subject_matte.onnx with float16 tensors inside the graph and float32
inputs and outputs (`keep_io_types`), so the Rust program runs it without a change. It copies
export_report.json and adds a `float16` entry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16


# GridSample stays in float32: its grid coordinates need the float32 precision for the sampling
# positions of the 256 x 256 and larger decoder maps. Range (the base grid of the sampling
# positions) has no float16 type in ONNX.
BLOCKED_OPERATORS = ["GridSample", "Range"]


def remove_duplicate_casts(model: onnx.ModelProto) -> int:
    """The converter adds one Cast node per blocked consumer of a tensor, with the same node name
    and the same output name when one tensor feeds several GridSample nodes. ONNX Runtime rejects
    the duplicate names. Keep the first of the identical nodes."""
    seen: set[tuple] = set()
    kept = []
    for node in model.graph.node:
        key = (node.op_type, node.name, tuple(node.input), tuple(node.output))
        if node.op_type == "Cast" and key in seen:
            continue
        seen.add(key)
        kept.append(node)
    removed = len(model.graph.node) - len(kept)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    source = Path(args.onnx_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(source / "subject_matte.onnx"))
    converted = convert_float_to_float16(model, keep_io_types=True,
                                         op_block_list=BLOCKED_OPERATORS)
    removed = remove_duplicate_casts(converted)
    path = out_dir / "subject_matte.onnx"
    onnx.save(converted, str(path))
    report = json.loads((source / "export_report.json").read_text(encoding="utf-8"))
    report["float16"] = {
        "source": str(source / "subject_matte.onnx"),
        "converter": "onnxruntime.transformers.float16.convert_float_to_float16",
        "keep_io_types": True,
        "op_block_list": BLOCKED_OPERATORS,
        "duplicate_casts_removed": removed,
        "bytes": path.stat().st_size,
    }
    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[fp16] {path}: {path.stat().st_size / 1e6:.1f} MB, {removed} duplicate casts removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
