#!/usr/bin/env python3

import argparse
import base64
import json
import math
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


SEMANTIC_SERVICE = "semantic.SemanticService"


def cosine_similarity(vec_a, vec_b):
    if len(vec_a) != len(vec_b):
        raise ValueError(f"embedding dimension mismatch: {len(vec_a)} vs {len(vec_b)}")

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)


def grpc_call(address, method, payload):
    cmd = [
        "grpcurl",
        "-plaintext",
        "-d",
        json.dumps(payload, ensure_ascii=False),
        address,
        f"{SEMANTIC_SERVICE}/{method}",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"grpcurl call failed for {method}\n"
            f"command: {' '.join(cmd)}\n"
            f"stderr: {proc.stderr.strip()}"
        )

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"invalid grpcurl json output for {method}: {proc.stdout.strip()}"
        ) from exc


def wait_until_ready(address, timeout_sec=120):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            grpc_call(
                address,
                "Ping",
                {"request_id": f"wait-{uuid.uuid4().hex[:8]}"},
            )
            return True
        except Exception:
            time.sleep(1)
    return False


def start_server(repo_root, address):
    server_proc = subprocess.Popen(
        ["cargo", "run"],
        cwd=repo_root,
        stdout=None,
        stderr=None,
    )

    ready = wait_until_ready(address)
    if not ready:
        if server_proc.poll() is None:
            server_proc.terminate()
        raise RuntimeError("server did not become ready in time")

    owns_server = server_proc.poll() is None
    return server_proc, owns_server


def stop_server(server_proc):
    if server_proc is None:
        return

    if server_proc.poll() is not None:
        return

    server_proc.terminate()
    try:
        server_proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        server_proc.kill()
        server_proc.wait(timeout=5)


def list_images(testdata_dir):
    allowed = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in sorted(testdata_dir.iterdir()) if p.suffix.lower() in allowed]
    if not images:
        raise RuntimeError(f"no images found in {testdata_dir}")
    return images


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
    print("\nExample labels: dog, cat, puppy")
    while True:
        raw = input("Please enter candidate labels (comma-separated): ").strip()
        labels = [item.strip() for item in raw.split(",") if item.strip()]
        if labels:
            return labels
        print("Please enter at least one label")


def embed_image(address, image_path):
    img_bytes = image_path.read_bytes()
    b64 = base64.b64encode(img_bytes).decode("ascii")
    payload = {
        "request_id": f"img-{uuid.uuid4().hex[:8]}",
        "image_bytes": b64,
        "image_format_hint": image_path.suffix.replace(".", "").lower(),
    }
    return grpc_call(address, "EmbedImage", payload)


def embed_text(address, text):
    payload = {
        "request_id": f"txt-{uuid.uuid4().hex[:8]}",
        "text": text,
    }
    return grpc_call(address, "EmbedText", payload)


def classify_image_by_labels(address, image_path, labels):
    img_result = embed_image(address, image_path)
    img_embedding = img_result.get("embedding", [])

    scores = []
    for label in labels:
        txt_result = embed_text(address, label)
        txt_embedding = txt_result.get("embedding", [])
        score = cosine_similarity(img_embedding, txt_embedding)
        scores.append((label, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


def ensure_dependencies():
    missing = [binary for binary in ["cargo", "grpcurl"] if shutil.which(binary) is None]
    if missing:
        raise RuntimeError(
            "missing required tools: "
            + ", ".join(missing)
            + "\nplease install them and retry"
        )


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
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dependencies()

    address = f"{args.host}:{args.port}"
    testdata_dir = Path(args.testdata_dir)
    repo_root = Path(__file__).resolve().parents[1]

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
            print(f"starting rust server with cargo run at {repo_root} ...")
            server_proc, owns_server = start_server(repo_root, address)
            if owns_server:
                print(f"server started at {address}")
            else:
                print(f"server reachable at {address} (process exited, likely existing server)")

        images = list_images(testdata_dir)
        image_path = choose_image(images)
        labels = choose_labels()

        scores = classify_image_by_labels(address, image_path, labels)

        print("\n=== Top Results ===")
        for idx, (label, score) in enumerate(scores, start=1):
            print(f"{idx}. {label:20s} score={score:.6f}")

        best_label, best_score = scores[0]
        print("\n=== Predicted Label ===")
        print(f"image={image_path.name} -> label={best_label} (score={best_score:.6f})")

    finally:
        if owns_server:
            print("\nstopping rust server ...")
            stop_server(server_proc)


if __name__ == "__main__":
    main()
