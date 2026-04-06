#!/usr/bin/env python3

import importlib
import importlib.util
import os
import subprocess
import time
import uuid
from pathlib import Path

GRPC_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
GRPC_CHANNEL_OPTIONS = [
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_BYTES),
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_BYTES),
]


def _load_grpc_modules():
    grpc = importlib.import_module("grpc")
    semantic_pb2 = importlib.import_module("semantic_pb2")
    semantic_pb2_grpc = importlib.import_module("semantic_pb2_grpc")
    return grpc, semantic_pb2, semantic_pb2_grpc


def _grpc_runtime_error(grpc, method, address, exc):
    if isinstance(exc, grpc.RpcError):
        status = exc.code()
        code_name = status.name if status is not None else "UNKNOWN"
        details = exc.details() or str(exc)
        return RuntimeError(
            f"grpc call failed for {method}\n"
            f"address: {address}\n"
            f"code: {code_name}\n"
            f"details: {details}"
        )

    return RuntimeError(
        f"grpc call failed for {method}\n"
        f"address: {address}\n"
        f"error: {exc}"
    )


def _call_semantic(address, method, request):
    grpc, _, semantic_pb2_grpc = _load_grpc_modules()

    with grpc.insecure_channel(address, options=GRPC_CHANNEL_OPTIONS) as channel:
        stub = semantic_pb2_grpc.SemanticServiceStub(channel)
        rpc = getattr(stub, method)
        try:
            return rpc(request)
        except Exception as exc:
            raise _grpc_runtime_error(grpc, method, address, exc) from exc


def _embedding_response_to_dict(response):
    return {
        "request_id": response.request_id,
        "embedding": list(response.embedding),
        "dimension": int(response.dimension),
        "model_name": response.model_name,
        "elapsed_ms": int(response.elapsed_ms),
    }


def ping(address, request_id=None):
    _, semantic_pb2, _ = _load_grpc_modules()
    request = semantic_pb2.PingRequest(
        request_id=request_id or f"wait-{uuid.uuid4().hex[:8]}"
    )
    response = _call_semantic(address, "Ping", request)
    return {
        "request_id": response.request_id,
        "message": response.message,
        "elapsed_ms": int(response.elapsed_ms),
    }


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


def cargo_command(repo_root, *cargo_args):
    wrapper = str(cargo_msvc_cmd(repo_root))

    # .cmd files should be launched through cmd.exe for reliable argument
    # forwarding and environment setup on Windows.
    if os.name == "nt":
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        return [comspec, "/d", "/s", "/c", wrapper, *cargo_args]

    return [wrapper, *cargo_args]


def cargo_run_command(repo_root):
    return cargo_command(repo_root, "run", "--release", "--features", "cuda")


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


def ensure_dependencies(
    repo_root,
    require_cargo=True,
    require_grpc=True,
    require_grpcurl=False,
):
    # Backward compatibility for existing callers that still pass require_grpcurl.
    if require_grpcurl:
        require_grpc = True

    missing = []

    if require_cargo:
        cargo_cmd = repo_root / "script" / "cargo_msvc.cmd"
        if not cargo_cmd.exists():
            missing.append(str(cargo_cmd))
        else:
            cargo_check = subprocess.run(
                cargo_command(repo_root, "--version"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if cargo_check.returncode != 0:
                missing.append("usable cargo_msvc.cmd (MSVC environment setup failed)")

    if require_grpc:
        python_modules = {
            "grpc": "grpcio",
            "google.protobuf": "protobuf",
        }
        missing_python = []
        for module, package in python_modules.items():
            try:
                available = importlib.util.find_spec(module) is not None
            except ModuleNotFoundError:
                available = False

            if not available:
                missing_python.append(package)

        if missing_python:
            missing.append("python packages: " + ", ".join(missing_python))

        script_dir = Path(__file__).resolve().parent
        missing_stubs = []
        for generated in ("semantic_pb2.py", "semantic_pb2_grpc.py"):
            if not (script_dir / generated).exists():
                missing_stubs.append(generated)

        if missing_stubs:
            missing.append("generated gRPC stubs: " + ", ".join(missing_stubs))

    if missing:
        install_hint = ""
        if require_grpc:
            install_hint = (
                "\ninstall gRPC deps with: pip install grpcio protobuf grpcio-tools"
                "\nthen generate stubs with: "
                "python -m grpc_tools.protoc -I proto --python_out=script "
                "--grpc_python_out=script proto/semantic.proto"
            )

        raise RuntimeError("missing required dependencies: " + ", ".join(missing) + install_hint)


def embed_image(address, image_path, request_id=None):
    _, semantic_pb2, _ = _load_grpc_modules()
    image_bytes = image_path.read_bytes()
    request = semantic_pb2.EmbedImageRequest(
        request_id=request_id or f"img-{uuid.uuid4().hex[:8]}",
        image_bytes=image_bytes,
        image_format_hint=image_path.suffix.replace(".", "").lower(),
    )
    response = _call_semantic(address, "EmbedImage", request)
    return _embedding_response_to_dict(response)


def embed_text(address, text):
    _, semantic_pb2, _ = _load_grpc_modules()
    request = semantic_pb2.EmbedTextRequest(
        request_id=f"txt-{uuid.uuid4().hex[:8]}",
        text=text,
    )
    response = _call_semantic(address, "EmbedText", request)
    return _embedding_response_to_dict(response)
