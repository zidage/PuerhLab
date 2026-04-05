#!/usr/bin/env python3

import base64
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path


SEMANTIC_SERVICE = "semantic.SemanticService"


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


def ping(address, request_id=None):
    return grpc_call(
        address,
        "Ping",
        {"request_id": request_id or f"wait-{uuid.uuid4().hex[:8]}"},
    )


def wait_until_ready(address, timeout_sec=None, server_proc=None):
    deadline = None if timeout_sec is None else time.time() + timeout_sec
    while deadline is None or time.time() < deadline:
        if server_proc is not None and server_proc.poll() is not None:
            return False
        try:
            ping(address)
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
    return [str(cargo_msvc_cmd(repo_root)), "run", "--release", "--features", "cuda"]


def server_environment(device="cuda", rust_log=None):
    env = os.environ.copy()
    env["PUERH_MIND_DEVICE"] = device
    if rust_log is not None:
        env["RUST_LOG"] = rust_log
    return env


def start_server(repo_root, address, device="cuda", quiet_stdio=False, rust_log=None):
    stdout = subprocess.DEVNULL if quiet_stdio else None
    stderr = subprocess.DEVNULL if quiet_stdio else None

    server_proc = subprocess.Popen(
        cargo_run_command(repo_root),
        cwd=repo_root,
        env=server_environment(device=device, rust_log=rust_log),
        stdout=stdout,
        stderr=stderr,
    )

    ready = wait_until_ready(address, timeout_sec=None, server_proc=server_proc)
    if not ready:
        if server_proc.poll() is None:
            server_proc.terminate()
        raise RuntimeError("server did not become ready in time")

    if server_proc.poll() is not None:
        raise RuntimeError("spawned rust server exited unexpectedly after startup")

    return server_proc


def stop_server(server_proc):
    if server_proc is None:
        return

    if server_proc.poll() is not None:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(server_proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return

    server_proc.terminate()
    try:
        server_proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        server_proc.kill()
        server_proc.wait(timeout=5)


def ensure_dependencies(repo_root, require_cargo=True, require_grpcurl=True):
    missing = []

    if require_cargo:
        cargo_cmd = repo_root / "script" / "cargo_msvc.cmd"
        if not cargo_cmd.exists():
            missing.append(str(cargo_cmd))

    if require_grpcurl and shutil.which("grpcurl") is None:
        missing.append("grpcurl")

    if missing:
        raise RuntimeError(
            "missing required tools: "
            + ", ".join(missing)
            + "\nplease install them and retry"
        )


def embed_image(address, image_path, request_id=None):
    image_bytes = image_path.read_bytes()
    payload = {
        "request_id": request_id or f"img-{uuid.uuid4().hex[:8]}",
        "image_bytes": base64.b64encode(image_bytes).decode("ascii"),
        "image_format_hint": image_path.suffix.replace(".", "").lower(),
    }
    return grpc_call(address, "EmbedImage", payload)


def embed_text(address, text):
    payload = {
        "request_id": f"txt-{uuid.uuid4().hex[:8]}",
        "text": text,
    }
    return grpc_call(address, "EmbedText", payload)
