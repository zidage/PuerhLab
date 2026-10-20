#!/usr/bin/env python3
"""Compare the SAM 2.1 candidate graph with the upstream PyTorch generator (#458, step E4).

Check `CandidateDecoderLogitsMatchPytorchWithinTolerance`: for each photo, the script runs
`SAM2ImagePredictor.set_image` and `_predict` for the point grid of `SAM2AutomaticMaskGenerator`
(single positive points, `multimask_output=True`, the batched path of the generator) and
`candidate_decoder.onnx` on ONNX Runtime CPU with the same encoder outputs. It compares the
256 x 256 logits (maximum absolute difference, IoU of logit > 0), the predicted IoU scores, and
the stability scores (`calculate_stability_score` on the 256 x 256 logits). The requirement
applies to the masks that pass the generator filters in PyTorch (predicted IoU > 0.8, stability
>= 0.95), because only these become candidates: IoU >= 0.99 for every such mask, and the same
filter result for at least 99 percent of all masks. The other masks are reported.

A graph with a dynamic prompt axis is also compared with a smaller batch, so the check covers
that axis. A graph with a fixed prompt count (`export_onnx.py --candidate-batch`) is compared
with full batches only.

With --upstream-out, the script also runs `SAM2AutomaticMaskGenerator.generate` (the upstream
generator with its default thresholds) on the photo resized to --upstream-long-edge and writes
the masks, resized to the 256 x 256 model grid, to <upstream-out>/<stem>.npz. The Rust program
`segmentation_runtime_report candidates` writes its own candidate masks; `probe_candidates.py`
compares the two sets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import sam2_graphs as graphs  # noqa: E402

REQUIRED_IOU = 0.99
REQUIRED_FILTER_AGREEMENT = 0.99
SCORE_THRESHOLD = 0.8
STABILITY_THRESHOLD = 0.95
STABILITY_OFFSET = 1.0
TORCH_CHUNK = 16


def peak_working_set_mb() -> float | None:
    """Peak working set of this process (Windows), to record the memory use of the check."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters),
                                           wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters),
                                      counters.cb):
        return None
    return counters.PeakWorkingSetSize / 1048576.0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def stability(logits: torch.Tensor) -> torch.Tensor:
    inner = (logits > STABILITY_OFFSET).flatten(-2).sum(-1).float()
    outer = (logits > -STABILITY_OFFSET).flatten(-2).sum(-1).float()
    return torch.where(outer > 0, inner / outer.clamp(min=1.0), torch.ones_like(outer))


def compare_batch(predictor, session, features, points: torch.Tensor) -> dict:
    """`points` are normalized [B, 2]; returns the comparison of one batch."""
    coords = points * graphs.IMAGE_SIZE
    # `_predict` also resizes every mask of the batch to the photo size in float32 (64 prompts x
    # 3 masks x 45.6 MP x 4 bytes = 35 GB on the largest photo). The comparison uses only the
    # 256 x 256 logits, so the resize target is the logit size. Point coordinates are already
    # model-input pixels and do not use this size.
    predictor._orig_hw = [(graphs.LOW_RES_MASK_SIZE, graphs.LOW_RES_MASK_SIZE)]
    # `_process_batch` of the generator: transform_coords(normalize=True) maps the normalized
    # points of the photo to model-input pixels, which is coords above. Each prompt is decoded
    # independently, so chunks of TORCH_CHUNK prompts give the same values with less memory.
    torch_scores, torch_logits = [], []
    with torch.no_grad():
        for start in range(0, coords.shape[0], TORCH_CHUNK):
            chunk = coords[start:start + TORCH_CHUNK]
            _, scores, logits = predictor._predict(
                chunk[:, None, :], torch.ones(chunk.shape[0], 1, dtype=torch.int),
                multimask_output=True, return_logits=True)
            torch_scores.append(scores)
            torch_logits.append(logits)
    torch_scores = torch.cat(torch_scores)
    torch_logits = torch.cat(torch_logits).float()
    torch_stability = stability(torch_logits).numpy()
    torch_logits = torch_logits.numpy()
    torch_scores = torch_scores.numpy()
    onnx_logits, onnx_scores, onnx_stability = session.run(None, {
        "image_embed": features[0], "high_res_feats_0": features[1],
        "high_res_feats_1": features[2],
        "point_coords": coords[:, None, :].numpy().astype(np.float32),
    })
    ious = np.array([[iou(onnx_logits[b, t] > 0, torch_logits[b, t] > 0) for t in range(3)]
                     for b in range(coords.shape[0])])
    kept_torch = (torch_scores > SCORE_THRESHOLD) & (torch_stability >= STABILITY_THRESHOLD)
    kept_onnx = (onnx_scores > SCORE_THRESHOLD) & (onnx_stability >= STABILITY_THRESHOLD)
    kept_ious = ious[kept_torch]
    return {
        "prompts": int(coords.shape[0]),
        "masks": int(ious.size),
        "minimum_iou": float(ious.min()),
        "masks_below_required_iou": int((ious < REQUIRED_IOU).sum()),
        "kept_masks": int(kept_torch.sum()),
        "kept_minimum_iou": float(kept_ious.min()) if kept_ious.size else None,
        "kept_masks_below_required_iou": int((kept_ious < REQUIRED_IOU).sum()),
        "filter_agreement": float((kept_torch == kept_onnx).mean()),
        "maximum_logit_difference": float(np.abs(onnx_logits - torch_logits).max()),
        "maximum_iou_score_difference": float(np.abs(onnx_scores - torch_scores).max()),
        "maximum_stability_difference": float(np.abs(onnx_stability - torch_stability).max()),
    }


def upstream_candidates(model, image: Image.Image, long_edge: int, out_path: Path) -> dict:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    scale = long_edge / max(image.size)
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)),
                             Image.BILINEAR)
    generator = SAM2AutomaticMaskGenerator(model)  # Upstream defaults (32 x 32 grid).
    with torch.no_grad():
        records = generator.generate(np.asarray(image))
    size = graphs.LOW_RES_MASK_SIZE
    masks = np.zeros((len(records), size, size), np.uint8)
    for index, record in enumerate(records):
        tensor = torch.from_numpy(record["segmentation"].astype(np.float32))[None, None]
        masks[index] = (F.interpolate(tensor, (size, size), mode="bilinear",
                                      align_corners=False)[0, 0] > 0.5).numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, masks=masks,
        predicted_iou=np.array([r["predicted_iou"] for r in records], np.float32),
        stability_score=np.array([r["stability_score"] for r in records], np.float32),
        point_coords=np.array([r["point_coords"][0] for r in records], np.float32)
        / np.array(image.size, np.float32))
    return {"candidates": len(records), "photo_size": list(image.size)}


def main() -> int:
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", required=True, choices=sorted(graphs.VARIANTS))
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--small-batch", type=int, default=13)
    parser.add_argument("--batches-per-photo", type=int, default=2,
                        help="full batches compared per photo, spread over the grid")
    parser.add_argument("--upstream-out", help="directory for the upstream generator masks")
    parser.add_argument("--upstream-long-edge", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4,
                        help="PyTorch and ONNX Runtime threads; keeps the machine usable")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    graphs.add_repository_to_path(args.repository)
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    _, checkpoint = graphs.VARIANTS[args.variant]
    model = graphs.build_model(args.variant, str(Path(args.weights) / checkpoint))
    predictor = SAM2ImagePredictor(model)
    onnx_dir = Path(args.onnx_dir)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = args.threads
    # Without the arena, ONNX Runtime returns the memory of a run instead of keeping its peak.
    session_options.enable_cpu_mem_arena = False
    encoder = ort.InferenceSession(str(onnx_dir / "point_encoder.onnx"), session_options,
                                   providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(onnx_dir / "candidate_decoder.onnx"), session_options,
                                   providers=["CPUExecutionProvider"])
    fixed_count = decoder.get_inputs()[3].shape[0]
    if isinstance(fixed_count, int):
        args.batch = fixed_count
    grid = graphs.grid_points(args.points_per_side)
    batch_starts = np.linspace(0, len(grid) - args.batch, args.batches_per_photo).astype(int)

    report = {"check": "CandidateDecoderLogitsMatchPytorchWithinTolerance",
              "variant": args.variant, "onnx_dir": str(onnx_dir),
              "prompt_count": fixed_count if isinstance(fixed_count, int) else "dynamic",
              "required_iou_of_kept_masks": REQUIRED_IOU,
              "required_filter_agreement": REQUIRED_FILTER_AGREEMENT, "photos": {}}
    failed = 0
    for path in sorted(Path(args.photos).glob("*.jpg")):
        image = Image.open(path).convert("RGB")
        predictor.set_image(image)
        pixels = graphs.preprocess_image(image).numpy()
        features = encoder.run(None, {"pixel_values": pixels})
        batches = [compare_batch(predictor, decoder, features,
                                 grid[start:start + args.batch]) for start in batch_starts]
        if not isinstance(fixed_count, int):
            batches.append(compare_batch(predictor, decoder, features, grid[:args.small_batch]))
        passed = all(b["kept_masks_below_required_iou"] == 0
                     and b["filter_agreement"] >= REQUIRED_FILTER_AGREEMENT for b in batches)
        failed += not passed
        entry = {"passed": passed, "batches": batches}
        if args.upstream_out:
            entry["upstream_generator"] = upstream_candidates(
                model, image, args.upstream_long_edge, Path(args.upstream_out) / f"{path.stem}.npz")
        report["photos"][path.name] = entry
        print(f"[compare] {args.variant} {path.name}: passed={passed} "
              f"min IoU {min(b['minimum_iou'] for b in batches):.5f} "
              f"kept {sum(b['kept_masks'] for b in batches)} "
              f"filter agreement {min(b['filter_agreement'] for b in batches):.4f} "
              f"max logit diff {max(b['maximum_logit_difference'] for b in batches):.4f}"
              + (f" upstream {entry['upstream_generator']['candidates']}"
                 if args.upstream_out else "")
              + f" peak working set {peak_working_set_mb() or 0:.0f} MB", flush=True)
    report["passed_photos"] = len(report["photos"]) - failed
    report["peak_working_set_mb"] = peak_working_set_mb()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
