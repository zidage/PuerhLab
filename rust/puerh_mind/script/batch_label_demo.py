#!/usr/bin/env python3

import argparse
import json
import sys
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

from clip_labeling import (
    CLIP_DEFAULT_CLASSES,
    DEFAULT_MAX_IN_FLIGHT,
    classify_images_by_prototypes,
    list_images,
    load_or_build_label_prototypes,
)
from semantic_client import (
    cargo_run_command,
    ensure_dependencies,
    start_server,
    stop_server,
    wait_until_ready,
)


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Batch image labeling demo: classify all images in testdata with "
            "CLIP_DEFAULT_CLASSES and print top-1 label."
        )
    )
    parser.add_argument("--host", default="127.0.0.1", help="gRPC host")
    parser.add_argument("--port", default=50051, type=int, help="gRPC port")
    parser.add_argument(
        "--testdata-dir",
        default=str(repo_root / "testdata"),
        help="image directory to classify",
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="do not spawn cargo run; use existing server",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PUERH_MIND_DEVICE value used when spawning server",
    )
    parser.add_argument(
        "--top-k",
        default=1,
        type=int,
        help="print top-k scores for each image (>=1)",
    )
    parser.add_argument(
        "--output-json",
        default=str(repo_root / "testdata" / "album_labels.json"),
        help="output json index path used by album TUI",
    )
    parser.add_argument(
        "--request-size",
        default=DEFAULT_MAX_IN_FLIGHT,
        type=int,
        help="max in-flight image embed requests (>=1)",
    )
    return parser.parse_args()


def write_album_index(output_json_path, address, testdata_dir, entries):
    success_count = sum(1 for item in entries if item.get("error") is None)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "address": address,
        "testdata_dir": str(testdata_dir.resolve()),
        "classes": CLIP_DEFAULT_CLASSES,
        "total": len(entries),
        "success_count": success_count,
        "failure_count": len(entries) - success_count,
        "items": entries,
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_batch_label(
    host="127.0.0.1",
    port=50051,
    testdata_dir=None,
    no_spawn=False,
    device="cuda",
    top_k=1,
    request_size=DEFAULT_MAX_IN_FLIGHT,
    output_json_path=None,
):
    if top_k < 1:
        raise RuntimeError("--top-k must be >= 1")
    if request_size < 1:
        raise RuntimeError("--request-size must be >= 1")

    address = f"{host}:{port}"
    repo_root = Path(__file__).resolve().parents[1]
    testdata_dir = Path(testdata_dir or (repo_root / "testdata"))
    output_json_path = Path(output_json_path or (testdata_dir / "album_labels.json"))

    if not testdata_dir.exists():
        raise RuntimeError(f"testdata dir not found: {testdata_dir}")

    ensure_dependencies(repo_root, require_cargo=False, require_grpc=True)

    server_proc = None
    owns_server = False

    try:
        if no_spawn:
            if not wait_until_ready(address, timeout_sec=3):
                raise RuntimeError(f"server is not reachable at {address}")
            print(f"using existing server at {address}")
        else:
            if wait_until_ready(address, timeout_sec=1):
                print(f"using existing server at {address} (skip spawn)")
            else:
                ensure_dependencies(repo_root, require_cargo=True, require_grpc=True)
                print(
                    f"starting rust server with {' '.join(cargo_run_command(repo_root))} "
                    f"and PUERH_MIND_DEVICE={device} at {repo_root} ..."
                )
                server_proc = start_server(repo_root, address, device=device)
                owns_server = True
                print(f"server started at {address}")

        images = list_images(testdata_dir)
        print(f"found {len(images)} images in {testdata_dir}")
        print(
            "building text prototypes for labels: "
            + ", ".join(CLIP_DEFAULT_CLASSES)
        )

        label_prototypes, text_cache_path, reused_cache = load_or_build_label_prototypes(
            address,
            CLIP_DEFAULT_CLASSES,
            testdata_dir,
        )
        if reused_cache:
            print(f"loaded text prototypes from cache: {text_cache_path}")
        else:
            print(f"generated text prototypes and cached to: {text_cache_path}")

        label_counter = Counter()
        failures = []
        entries = []

        print("\n=== Batch Label Results ===")
        results = classify_images_by_prototypes(
            address,
            images,
            label_prototypes,
            max_in_flight=request_size,
        )
        for idx, result in enumerate(results, start=1):
            image_path = result["image_path"]
            error = result["error"]

            try:
                if error:
                    raise RuntimeError(error)

                scores = result["scores"]
                best_label, best_score = scores[0]
                label_counter[best_label] += 1
                top_scores = scores[:top_k]
                entries.append(
                    {
                        "index": idx,
                        "name": image_path.name,
                        "path": str(image_path.resolve()),
                        "label": best_label,
                        "score": best_score,
                        "top_scores": [
                            {"label": label, "score": score}
                            for label, score in top_scores
                        ],
                        "error": None,
                    }
                )

                print(
                    f"{idx:03d}. {image_path.name:36s} -> "
                    f"{best_label:12s} score={best_score:.6f}"
                )

                if top_k > 1:
                    details = ", ".join(
                        f"{label}={score:.6f}" for label, score in top_scores
                    )
                    print(f"     top-{top_k}: {details}")

            except Exception as exc:
                failures.append((image_path.name, str(exc)))
                entries.append(
                    {
                        "index": idx,
                        "name": image_path.name,
                        "path": str(image_path.resolve()),
                        "label": None,
                        "score": None,
                        "top_scores": [],
                        "error": str(exc),
                    }
                )
                print(f"{idx:03d}. {image_path.name:36s} -> ERROR: {exc}")

        print("\n=== Label Distribution ===")
        if label_counter:
            for label, count in label_counter.most_common():
                print(f"{label:20s} {count}")
        else:
            print("no successful predictions")

        if failures:
            print("\n=== Failures ===")
            for name, message in failures:
                print(f"{name}: {message}")

        write_album_index(output_json_path, address, testdata_dir, entries)
        print(f"\nalbum json written to: {output_json_path}")

        return {
            "output_json_path": output_json_path,
            "entries": entries,
        }

    finally:
        if owns_server:
            print("\nstopping rust server ...")
            stop_server(server_proc)


def main():
    args = parse_args()
    try:
        run_batch_label(
            host=args.host,
            port=args.port,
            testdata_dir=args.testdata_dir,
            no_spawn=args.no_spawn,
            device=args.device,
            top_k=args.top_k,
            request_size=args.request_size,
            output_json_path=args.output_json,
        )
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
