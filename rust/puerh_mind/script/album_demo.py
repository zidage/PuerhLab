#!/usr/bin/env python3

import argparse
import base64
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections import OrderedDict
from pathlib import Path


SEMANTIC_SERVICE = "semantic.SemanticService"

CLIP_DEFAULT_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CLIP_PROMPT_TEMPLATES = [
    'a photo of a {}.',
]

CLIP_CLASS_SYNONYMS = {
    "airplane": ["airplane"],
    "automobile": ["automobile"],
    "bird": ["bird"],
    "cat": ["cat"],
    "deer": ["deer"],
    "dog": ["dog"],
    "frog": ["frog"],
    "horse": ["horse"],
    "ship": ["ship"],
    "truck": ["truck"],
}


def cosine_similarity(vec_a, vec_b):
    if len(vec_a) != len(vec_b):
        raise ValueError(f"embedding dimension mismatch: {len(vec_a)} vs {len(vec_b)}")

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)


def average_embeddings(vectors):
    if not vectors:
        raise ValueError("no embeddings to average")

    dim = len(vectors[0])
    for idx, vec in enumerate(vectors, start=1):
        if len(vec) != dim:
            raise ValueError(
                f"embedding dimension mismatch in average: expected {dim}, got {len(vec)} at item {idx}"
            )

    summed = [0.0] * dim
    for vec in vectors:
        for i, value in enumerate(vec):
            summed[i] += value

    count = float(len(vectors))
    return [value / count for value in summed]


def l2_normalize(vector):
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError("embedding norm is zero")
    return [value / norm for value in vector]


def average_normalized_embeddings(vectors):
    normalized_vectors = [l2_normalize(vector) for vector in vectors]
    return l2_normalize(average_embeddings(normalized_vectors))


def grpc_call(address, method, payload):
    payload_json = json.dumps(payload, ensure_ascii=False)
    cmd = [
        "grpcurl",
        "-plaintext",
        "-d",
        "@",
        address,
        f"{SEMANTIC_SERVICE}/{method}",
    ]

    proc = subprocess.run(cmd, input=payload_json, capture_output=True, text=True)
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


def wait_until_ready(address, timeout_sec=None):
    deadline = None if timeout_sec is None else time.time() + timeout_sec
    while deadline is None or time.time() < deadline:
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


def cargo_msvc_cmd(repo_root):
    cmd_path = repo_root / "script" / "cargo_msvc.cmd"
    if not cmd_path.exists():
        raise RuntimeError(f"cargo wrapper not found: {cmd_path}")
    return cmd_path


def cargo_run_command(repo_root):
    cargo_cmd = cargo_msvc_cmd(repo_root)
    return [str(cargo_cmd), "run", "--release", "--features", "cuda"]


def server_environment():
    env = os.environ.copy()
    env["PUERH_MIND_DEVICE"] = "cuda"
    return env


def start_server(repo_root, address):
    server_proc = subprocess.Popen(
        cargo_run_command(repo_root),
        cwd=repo_root,
        env=server_environment(),
        stdout=None,
        stderr=None,
    )

    ready = wait_until_ready(address, timeout_sec=None)
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
    print("\nCLIP default classes:")
    print(", ".join(CLIP_DEFAULT_CLASSES))
    print("Enter labels (comma-separated). Press Enter to use default classes.")
    while True:
        raw = input("Please enter candidate labels (comma-separated): ").strip()
        if not raw:
            return CLIP_DEFAULT_CLASSES.copy()
        labels = [item.strip() for item in raw.split(",") if item.strip()]
        if labels:
            deduped = list(OrderedDict.fromkeys(labels))
            return deduped
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


def embed_text_prompt_ensemble(address, label):
    prompt_embeddings = []
    for template in CLIP_PROMPT_TEMPLATES:
        prompt = template.format(label)
        txt_result = embed_text(address, prompt)
        txt_embedding = txt_result.get("embedding", [])
        if not txt_embedding:
            raise RuntimeError(f"empty text embedding for prompt: {prompt}")
        prompt_embeddings.append(txt_embedding)

    return average_normalized_embeddings(prompt_embeddings)


def embed_text_label_prototype(address, label):
    synonyms = CLIP_CLASS_SYNONYMS.get(label.lower(), [label])
    synonym_embeddings = []
    for synonym in synonyms:
        synonym_embedding = embed_text_prompt_ensemble(address, synonym)
        synonym_embeddings.append(synonym_embedding)

    return average_normalized_embeddings(synonym_embeddings)


def classify_image_by_labels(address, image_path, labels):
    img_result = embed_image(address, image_path)
    img_embedding = img_result.get("embedding", [])
    if not img_embedding:
        raise RuntimeError("empty image embedding")
    img_embedding = l2_normalize(img_embedding)

    scores = []
    for label in labels:
        txt_embedding = embed_text_label_prototype(address, label)
        score = cosine_similarity(img_embedding, txt_embedding)
        scores.append((label, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


def decide_prediction(scores, min_score=0.20, min_margin=0.02):
    if not scores:
        raise RuntimeError("no score candidates")

    best_label, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else -1.0
    margin = best_score - second_score

    confident = best_score >= min_score and margin >= min_margin
    return {
        "label": best_label,
        "score": best_score,
        "second_score": second_score,
        "margin": margin,
        "confident": confident,
    }


def ensure_dependencies():
    repo_root = Path(__file__).resolve().parents[1]
    cargo_cmd = cargo_msvc_cmd(repo_root)

    missing = []
    if not cargo_cmd.exists():
        missing.append(str(cargo_cmd))
    if shutil.which("grpcurl") is None:
        missing.append("grpcurl")

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
            print(
                f"starting rust server with {' '.join(cargo_run_command(repo_root))} "
                f"and PUERH_MIND_DEVICE=cuda at {repo_root} ..."
            )
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
