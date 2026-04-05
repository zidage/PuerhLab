#!/usr/bin/env python3

import argparse
import signal
import sys
from pathlib import Path

from clip_labeling import (
    CLIP_DEFAULT_CLASSES,
    classify_image_by_labels,
    decide_prediction,
    dedupe_labels,
    list_images,
)
from semantic_client import (
    cargo_run_command,
    ensure_dependencies,
    start_server,
    stop_server,
    wait_until_ready,
)

def choose_image(images):
    print("\n=== Simple Album ===")
    for idx, img in enumerate(images, start=1):
        print(f"[{idx}] {img.name}")

    while True:
        user_input = input("\nSelect photo number: ").strip()
        if not user_input.isdigit():
            print("Please enter a valid number")
            continue

        index = int(user_input)
        if 1 <= index <= len(images):
            return images[index - 1]

        print("Number out of range")


def choose_labels():
    print("\nCLIP default classes:")
    print(", ".join(CLIP_DEFAULT_CLASSES))
    print("Enter labels (comma-separated). Press Enter to use default classes.")
    while True:
        raw = input("Please enter candidate labels (comma-separated): ").strip()
        if not raw:
            return CLIP_DEFAULT_CLASSES.copy()
        labels = dedupe_labels(raw.split(","))
        if labels:
            return labels
        print("Please enter at least one label")


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Simple album demo: spawn Rust server and classify image by user labels."
    )
    parser.add_argument("--host", default="127.0.0.1", help="gRPC host")
    parser.add_argument("--port", default=50051, type=int, help="gRPC port")
    parser.add_argument(
        "--testdata-dir",
        default=str(repo_root / "testdata"),
        help="image directory used as album",
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="do not spawn cargo run; use existing server",
    )
    parser.add_argument(
        "--min-score",
        default=0.20,
        type=float,
        help="minimum top-1 cosine score to accept prediction",
    )
    parser.add_argument(
        "--min-margin",
        default=0.02,
        type=float,
        help="minimum (top1-top2) margin to accept prediction",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PUERH_MIND_DEVICE value used when spawning server",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    ensure_dependencies(repo_root, require_cargo=False, require_grpcurl=True)

    address = f"{args.host}:{args.port}"
    testdata_dir = Path(args.testdata_dir)

    if not testdata_dir.exists():
        raise RuntimeError(f"testdata dir not found: {testdata_dir}")

    server_proc = None
    owns_server = False

    def handle_sigint(_sig, _frame):
        stop_server(server_proc)
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        if args.no_spawn:
            if not wait_until_ready(address, timeout_sec=3):
                raise RuntimeError(f"server is not reachable at {address}")
            print(f"using existing server at {address}")
        else:
            if wait_until_ready(address, timeout_sec=1):
                print(f"using existing server at {address} (skip spawn)")
            else:
                ensure_dependencies(repo_root, require_cargo=True, require_grpcurl=False)
                print(
                    f"starting rust server with {' '.join(cargo_run_command(repo_root))} "
                    f"and PUERH_MIND_DEVICE={args.device} at {repo_root} ..."
                )
                server_proc = start_server(repo_root, address, device=args.device)
                owns_server = True
                print(f"server started at {address}")

        images = list_images(testdata_dir)
        image_path = choose_image(images)
        labels = choose_labels()

        scores = classify_image_by_labels(address, image_path, labels)

        print("\n=== Top Results ===")
        for idx, (label, score) in enumerate(scores, start=1):
            print(f"{idx}. {label:20s} score={score:.6f}")

        decision = decide_prediction(
            scores,
            min_score=args.min_score,
            min_margin=args.min_margin,
        )
        print("\n=== Predicted Label ===")
        if decision["confident"]:
            print(
                f"image={image_path.name} -> label={decision['label']} "
                f"(score={decision['score']:.6f}, margin={decision['margin']:.6f})"
            )
        else:
            print(
                f"image={image_path.name} -> uncertain "
                f"(best={decision['label']}, score={decision['score']:.6f}, "
                f"margin={decision['margin']:.6f}; "
                f"thresholds: score>={args.min_score:.3f}, margin>={args.min_margin:.3f})"
            )

    finally:
        if owns_server:
            print("\nstopping rust server ...")
            stop_server(server_proc)


if __name__ == "__main__":
    main()
