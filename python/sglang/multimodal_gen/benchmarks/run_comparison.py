"""Cross-framework comparison benchmark for diffusion serving.

Launches servers (SGLang, vLLM-Omni, LightX2V) for each test case, sends a
single request, measures end-to-end latency, and writes comparison-results.json.

Usage:
    # Full run (requires GPU)
    python3 -m sglang.multimodal_gen.benchmarks.run_comparison

    # Dry-run (config parsing + command preview only)
    python3 -m sglang.multimodal_gen.benchmarks.run_comparison --dry-run

    # Run only specific case(s)
    python3 -m sglang.multimodal_gen.benchmarks.run_comparison --case-ids flux1_dev_t2i_1024

    # Run only specific framework(s)
    python3 -m sglang.multimodal_gen.benchmarks.run_comparison --frameworks sglang
"""

import argparse
import base64
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIGS_PATH = Path(__file__).parent / "comparison_configs.json"
INSTALL_SCRIPT = (
    Path(__file__).parents[4] / "scripts" / "ci" / "install_comparison_frameworks.sh"
)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 30000
HEALTH_TIMEOUT = 1200  # seconds
REQUEST_TIMEOUT = 1200  # seconds
GPU_CLEAR_WAIT = 15  # seconds between framework runs

# Frameworks that need separate installation (conflict with sglang's deps)
INSTALLABLE_FRAMEWORKS = {"vllm-omni", "lightx2v"}

# Cached reference image (downloaded once)
_cached_ref_image: bytes | None = None
_cached_ref_image_path: str | None = None


# ---------------------------------------------------------------------------
# Server lifecycle — command builders
# ---------------------------------------------------------------------------


def _build_sglang_cmd(case: dict, fw_cfg: dict, port: int) -> list[str]:
    cmd = [
        "sglang",
        "serve",
        "--model-path",
        case["model"],
        "--port",
        str(port),
        "--host",
        DEFAULT_HOST,
    ]
    if case["num_gpus"] > 1:
        cmd += ["--num-gpus", str(case["num_gpus"])]
    if fw_cfg.get("serve_args", "").strip():
        cmd += fw_cfg["serve_args"].strip().split()
    return cmd


def _build_vllm_cmd(case: dict, fw_cfg: dict, port: int) -> list[str]:
    cmd = [
        "vllm",
        "serve",
        case["model"],
        "--omni",
        "--port",
        str(port),
        "--host",
        DEFAULT_HOST,
    ]
    if fw_cfg.get("serve_args", "").strip():
        cmd += fw_cfg["serve_args"].strip().split()
    return cmd


def _resolve_hf_model_path(model_id: str) -> str:
    """Resolve a HuggingFace model ID to a local cache path, or return as-is."""
    if os.path.isdir(model_id):
        return model_id
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(model_id)
        print(f"  Resolved {model_id} -> {path}")
        return path
    except Exception:
        return model_id


def _write_lightx2v_config(case: dict) -> str:
    """Write a minimal LightX2V config JSON and return its path."""
    cfg = {
        "infer_steps": case.get("num_inference_steps", 50),
        "guidance_scale": case.get("guidance_scale", 4.0),
        "seed": case.get("seed", 42),
    }
    if "num_frames" in case:
        cfg["target_video_length"] = case["num_frames"]
    if "height" in case:
        cfg["height"] = case["height"]
    if "width" in case:
        cfg["width"] = case["width"]

    config_path = os.path.join(
        tempfile.gettempdir(), f"lightx2v_config_{case['id']}.json"
    )
    with open(config_path, "w") as f:
        json.dump(cfg, f)
    return config_path


def _build_lightx2v_cmd(case: dict, fw_cfg: dict, port: int) -> list[str]:
    """Build LightX2V server launch command.

    Single GPU:  python -m lightx2v.server --model_path ... --model_cls ... --task ... --port ...
    Multi GPU:   torchrun --nproc_per_node=N -m lightx2v.server ...

    LightX2V requires a local model path and a config JSON with infer params.
    """
    model_cls = fw_cfg["model_cls"]
    task = fw_cfg["lightx2v_task"]
    num_gpus = case["num_gpus"]
    model_path = _resolve_hf_model_path(case["model"])
    config_path = _write_lightx2v_config(case)

    server_args = [
        "--model_path",
        model_path,
        "--model_cls",
        model_cls,
        "--task",
        task,
        "--config_json",
        config_path,
        "--host",
        DEFAULT_HOST,
        "--port",
        str(port),
    ]
    if fw_cfg.get("serve_args", "").strip():
        server_args += fw_cfg["serve_args"].strip().split()

    if num_gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "-m",
            "lightx2v.server",
        ] + server_args
    else:
        cmd = ["python3", "-m", "lightx2v.server"] + server_args

    return cmd


def build_server_cmd(framework: str, case: dict, fw_cfg: dict, port: int) -> list[str]:
    builders = {
        "sglang": _build_sglang_cmd,
        "vllm-omni": _build_vllm_cmd,
        "lightx2v": _build_lightx2v_cmd,
    }
    builder = builders.get(framework)
    if builder is None:
        raise ValueError(f"Unknown framework: {framework}")
    return builder(case, fw_cfg, port)


# ---------------------------------------------------------------------------
# Server lifecycle — health check & cleanup
# ---------------------------------------------------------------------------

# Health check endpoints per framework
HEALTH_ENDPOINTS = {
    "sglang": "/health",
    "vllm-omni": "/health",
    "lightx2v": "/v1/service/status",
}


def wait_for_health(
    base_url: str, framework: str = "sglang", timeout: int = HEALTH_TIMEOUT
) -> None:
    """Poll health endpoint until 200 or timeout."""
    endpoint = HEALTH_ENDPOINTS.get(framework, "/health")
    health_url = f"{base_url}{endpoint}"
    print(f"  Waiting for server at {health_url} ...")
    start = time.time()
    while True:
        try:
            resp = requests.get(health_url, timeout=2)
            if resp.status_code == 200:
                elapsed = time.time() - start
                print(f"  Server ready in {elapsed:.1f}s")
                return
        except requests.exceptions.RequestException:
            pass
        if time.time() - start > timeout:
            raise TimeoutError(
                f"Server at {health_url} did not start within {timeout}s"
            )
        time.sleep(2)


def kill_server(proc: subprocess.Popen) -> None:
    """Kill server process tree."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Reference image helpers
# ---------------------------------------------------------------------------


def _get_ref_image_bytes(config: dict) -> bytes:
    """Download and cache the shared test reference image."""
    global _cached_ref_image
    if _cached_ref_image is not None:
        return _cached_ref_image
    url = config.get("test_image_url", "")
    if not url:
        raise RuntimeError("No test_image_url in config for image-conditioned case")
    print(f"  Downloading reference image from {url} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    _cached_ref_image = resp.content
    return _cached_ref_image


def _get_ref_image_b64(config: dict) -> str:
    """Get reference image as base64 string."""
    return base64.b64encode(_get_ref_image_bytes(config)).decode("utf-8")


def _get_ref_image_path(config: dict) -> str:
    """Save reference image to a temp file and return path."""
    global _cached_ref_image_path
    if _cached_ref_image_path and os.path.exists(_cached_ref_image_path):
        return _cached_ref_image_path
    data = _get_ref_image_bytes(config)
    fd, path = tempfile.mkstemp(suffix=".png")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    _cached_ref_image_path = path
    return path


# ---------------------------------------------------------------------------
# Request helpers — SGLang (OpenAI-compatible)
# ---------------------------------------------------------------------------


def send_image_request_sglang(base_url: str, case: dict) -> float:
    """Send a single T2I request via SGLang's /v1/images/generations."""
    payload = {
        "model": case["model"],
        "prompt": case["prompt"],
        "size": f"{case['width']}x{case['height']}",
        "n": 1,
        "response_format": "b64_json",
    }
    if "num_inference_steps" in case:
        payload["num_inference_steps"] = case["num_inference_steps"]
    if "guidance_scale" in case:
        payload["guidance_scale"] = case["guidance_scale"]
    if "seed" in case:
        payload["seed"] = case["seed"]

    start = time.time()
    resp = requests.post(
        f"{base_url}/v1/images/generations",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    latency = time.time() - start
    resp.raise_for_status()
    data = resp.json()
    if "data" not in data or len(data["data"]) == 0:
        raise RuntimeError(f"Image request returned no data: {data}")
    print(f"  Image generated in {latency:.2f}s")
    return latency


def send_video_request_sglang(base_url: str, case: dict) -> float:
    """Send a single T2V request via SGLang's /v1/videos/generations (async)."""
    payload = {
        "model": case["model"],
        "prompt": case["prompt"],
        "size": f"{case['width']}x{case['height']}",
        "n": 1,
        "response_format": "b64_json",
    }
    for key in ("num_frames", "num_inference_steps", "guidance_scale", "seed"):
        if key in case:
            payload[key] = case[key]

    start = time.time()

    # Submit job
    resp = requests.post(
        f"{base_url}/v1/videos/generations",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    job = resp.json()
    job_id = job.get("id")
    if not job_id:
        raise RuntimeError(f"Video submit returned no job id: {job}")

    # Poll for completion
    poll_url = f"{base_url}/v1/videos/{job_id}"
    while True:
        time.sleep(1)
        poll_resp = requests.get(poll_url, timeout=30)
        poll_resp.raise_for_status()
        poll_data = poll_resp.json()
        status = poll_data.get("status")
        if status == "completed":
            break
        elif status == "failed":
            raise RuntimeError(f"Video generation failed: {poll_data}")
        if time.time() - start > REQUEST_TIMEOUT:
            raise TimeoutError(f"Video generation timed out after {REQUEST_TIMEOUT}s")

    latency = time.time() - start
    print(f"  Video generated in {latency:.2f}s")
    return latency


def send_image_conditioned_request_sglang(
    base_url: str, case: dict, config: dict
) -> float:
    """Send an image-conditioned request (edit/I2V/TI2V) via SGLang multipart API."""
    task = case["task"]
    ref_bytes = _get_ref_image_bytes(config)

    # Build multipart form
    files = {"image": ("ref.png", io.BytesIO(ref_bytes), "image/png")}
    data = {
        "model": case["model"],
        "prompt": case["prompt"],
        "size": f"{case['width']}x{case['height']}",
        "n": "1",
        "response_format": "b64_json",
    }
    for key in ("num_inference_steps", "guidance_scale", "seed", "num_frames"):
        if key in case:
            data[key] = str(case[key])

    # Choose endpoint based on task
    if task in ("image-edit", "image-to-image"):
        endpoint = "/v1/images/edits"
    elif task in ("image-to-video", "text-image-to-video"):
        endpoint = "/v1/videos/generations"
    else:
        endpoint = "/v1/images/generations"

    start = time.time()
    resp = requests.post(
        f"{base_url}{endpoint}",
        files=files,
        data=data,
        timeout=REQUEST_TIMEOUT,
    )

    # For video endpoints, need to poll
    if task in ("image-to-video", "text-image-to-video"):
        resp.raise_for_status()
        job = resp.json()
        job_id = job.get("id")
        if not job_id:
            raise RuntimeError(f"Video submit returned no job id: {job}")
        poll_url = f"{base_url}/v1/videos/{job_id}"
        while True:
            time.sleep(1)
            poll_resp = requests.get(poll_url, timeout=30)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            status = poll_data.get("status")
            if status == "completed":
                break
            elif status == "failed":
                raise RuntimeError(f"Video generation failed: {poll_data}")
            if time.time() - start > REQUEST_TIMEOUT:
                raise TimeoutError(f"Timed out after {REQUEST_TIMEOUT}s")
    else:
        resp.raise_for_status()

    latency = time.time() - start
    print(f"  Generated in {latency:.2f}s (sglang, image-conditioned)")
    return latency


# ---------------------------------------------------------------------------
# Request helpers — vLLM-Omni
# ---------------------------------------------------------------------------


def send_request_vllm_omni(base_url: str, case: dict, config: dict) -> float:
    """Send request via vLLM-Omni's /v1/chat/completions endpoint."""
    extra_body = {
        "height": case["height"],
        "width": case["width"],
        "num_inference_steps": case.get("num_inference_steps", 50),
        "guidance_scale": case.get("guidance_scale", 4.0),
        "seed": case.get("seed", 42),
    }
    if "num_frames" in case:
        extra_body["num_frames"] = case["num_frames"]

    # Build message content (text or text+image)
    content: list[dict] | str = case["prompt"]
    if case.get("reference_image"):
        ref_b64 = _get_ref_image_b64(config)
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{ref_b64}"},
            },
            {"type": "text", "text": case["prompt"]},
        ]

    payload = {
        "model": case["model"],
        "messages": [{"role": "user", "content": content}],
        "extra_body": extra_body,
    }

    start = time.time()
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    latency = time.time() - start
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"vLLM-Omni request returned no choices: {data}")
    print(f"  Generated in {latency:.2f}s (vllm-omni)")
    return latency


# ---------------------------------------------------------------------------
# Request helpers — LightX2V
# ---------------------------------------------------------------------------


def send_request_lightx2v(base_url: str, case: dict, config: dict) -> float:
    """Send request via LightX2V's async task API."""
    task = case["task"]
    if task in ("text-to-image", "image-edit"):
        endpoint = "/v1/tasks/image"
    else:
        endpoint = "/v1/tasks/video"

    payload = {
        "prompt": case["prompt"],
        "seed": case.get("seed", 42),
        "infer_steps": case.get("num_inference_steps", 50),
    }
    # LightX2V uses target_video_length for frames, height/width directly
    if "num_frames" in case:
        payload["target_video_length"] = case["num_frames"]
    if "height" in case:
        payload["height"] = case["height"]
    if "width" in case:
        payload["width"] = case["width"]
    if "guidance_scale" in case:
        payload["guidance_scale"] = case["guidance_scale"]
    # Image-conditioned: LightX2V accepts image_path (URL or local path)
    if case.get("reference_image"):
        payload["image_path"] = config.get("test_image_url", "")

    start = time.time()

    # Submit task
    resp = requests.post(
        f"{base_url}{endpoint}",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    task_data = resp.json()
    task_id = task_data.get("task_id")
    if not task_id:
        raise RuntimeError(f"LightX2V submit returned no task_id: {task_data}")

    # Poll for completion
    poll_url = f"{base_url}/v1/tasks/{task_id}/status"
    while True:
        time.sleep(1)
        poll_resp = requests.get(poll_url, timeout=30)
        poll_resp.raise_for_status()
        poll_data = poll_resp.json()
        status = poll_data.get("task_status", "").upper()
        if status == "COMPLETED":
            break
        elif status in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"LightX2V task {status}: {poll_data}")
        if time.time() - start > REQUEST_TIMEOUT:
            raise TimeoutError(f"LightX2V task timed out after {REQUEST_TIMEOUT}s")

    latency = time.time() - start
    print(f"  Generated in {latency:.2f}s (lightx2v)")
    return latency


# ---------------------------------------------------------------------------
# Unified request dispatcher
# ---------------------------------------------------------------------------


def send_request(
    base_url: str,
    case: dict,
    framework: str = "sglang",
    config: dict | None = None,
) -> float:
    config = config or {}
    if framework == "vllm-omni":
        return send_request_vllm_omni(base_url, case, config)
    elif framework == "lightx2v":
        return send_request_lightx2v(base_url, case, config)
    # SGLang — use OpenAI-compatible endpoints
    task = case["task"]
    if case.get("reference_image"):
        return send_image_conditioned_request_sglang(base_url, case, config)
    elif task == "text-to-image":
        return send_image_request_sglang(base_url, case)
    elif task == "text-to-video":
        return send_video_request_sglang(base_url, case)
    else:
        raise ValueError(f"Unknown task type: {task}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_single(
    case: dict,
    framework: str,
    fw_cfg: dict,
    port: int,
    log_dir: Path,
    config: dict | None = None,
) -> dict:
    """Run a single (case, framework) combination. Returns result dict."""
    result = {
        "case_id": case["id"],
        "framework": framework,
        "model": case["model"],
        "task": case["task"],
        "latency_s": None,
        "error": None,
    }

    cmd = build_server_cmd(framework, case, fw_cfg, port)
    print(f"\n  Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env.update(fw_cfg.get("extra_env", {}))

    log_file = log_dir / f"{case['id']}_{framework}.log"
    log_fh = open(log_file, "w", encoding="utf-8", buffering=1)
    log_thread = None

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
            text=True,
            bufsize=1,
        )

        # Tee server output to both log file and stdout (like test_server_utils)
        def _log_pipe(pipe, fh):
            try:
                for line in iter(pipe.readline, ""):
                    sys.stdout.write(f"  [server] {line}")
                    sys.stdout.flush()
                    fh.write(line)
            except ValueError:
                pass  # pipe closed

        log_thread = threading.Thread(
            target=_log_pipe, args=(proc.stdout, log_fh)
        )
        log_thread.daemon = True
        log_thread.start()

        base_url = f"http://{DEFAULT_HOST}:{port}"
        wait_for_health(base_url, framework)

        # Warmup request (not measured)
        print("  Sending warmup request...")
        try:
            send_request(base_url, case, framework, config)
        except Exception as e:
            print(f"  Warmup request failed (non-fatal): {e}")

        # Measured request
        print("  Sending measured request...")
        latency = send_request(base_url, case, framework, config)
        result["latency_s"] = round(latency, 3)

    except Exception as e:
        result["error"] = str(e)
        print(f"  ERROR: {e}")
    finally:
        if proc:
            kill_server(proc)
        if log_thread:
            log_thread.join(timeout=5)
        log_fh.close()

    return result


def _install_framework(fw_name: str, dry_run: bool = False) -> bool:
    """Install a comparison framework via the install script. Returns True on success."""
    if fw_name not in INSTALLABLE_FRAMEWORKS:
        return True
    if not INSTALL_SCRIPT.exists():
        print(f"  WARNING: Install script not found at {INSTALL_SCRIPT}")
        return False
    if dry_run:
        print(f"  [DRY-RUN] Would install: bash {INSTALL_SCRIPT} {fw_name}")
        return True
    print(f"\n{'='*60}")
    print(f"Installing framework: {fw_name}")
    print(f"{'='*60}")
    ret = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), fw_name],
        timeout=600,
    )
    if ret.returncode != 0:
        print(f"  WARNING: {fw_name} installation failed (exit {ret.returncode})")
        return False
    return True


def run_comparison(
    config: dict,
    case_ids: list[str] | None = None,
    frameworks: list[str] | None = None,
    port: int = DEFAULT_PORT,
    output: str = "comparison-results.json",
    dry_run: bool = False,
) -> dict:
    """Run all comparison cases, grouped by framework to minimize installs.

    Order: sglang first (already installed), then vllm-omni, then lightx2v.
    Each non-sglang framework is installed right before its cases run.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    commit_sha = os.environ.get("GITHUB_SHA", "unknown")
    run_id = os.environ.get("GITHUB_RUN_ID", "local")

    log_dir = Path("comparison-logs")
    log_dir.mkdir(exist_ok=True)

    # Collect all (case, framework) pairs, grouped by framework
    fw_order = ["sglang", "vllm-omni", "lightx2v"]
    fw_cases: dict[str, list[tuple[dict, dict]]] = {fw: [] for fw in fw_order}

    for case in config["cases"]:
        if case_ids and case["id"] not in case_ids:
            continue
        for fw_name, fw_cfg in case["frameworks"].items():
            if frameworks and fw_name not in frameworks:
                continue
            if fw_name not in fw_cases:
                fw_cases[fw_name] = []
            fw_cases[fw_name].append((case, fw_cfg))

    results = []
    installed_fws: set[str] = set()

    for fw_name in fw_order:
        pairs = fw_cases.get(fw_name, [])
        if not pairs:
            continue

        # Install framework if needed (once per framework)
        if fw_name not in installed_fws and fw_name in INSTALLABLE_FRAMEWORKS:
            if not _install_framework(fw_name, dry_run):
                # Skip all cases for this framework
                for case, _ in pairs:
                    results.append(
                        {
                            "case_id": case["id"],
                            "framework": fw_name,
                            "model": case["model"],
                            "task": case["task"],
                            "latency_s": None,
                            "error": f"{fw_name} installation failed",
                        }
                    )
                continue
            installed_fws.add(fw_name)

        for case, fw_cfg in pairs:
            print(f"\n{'='*60}")
            print(f"Case: {case['id']} | Model: {case['model']} | Framework: {fw_name}")
            print(f"{'='*60}")

            if dry_run:
                cmd = build_server_cmd(fw_name, case, fw_cfg, port)
                print(f"  [DRY-RUN] Would run: {' '.join(cmd)}")
                results.append(
                    {
                        "case_id": case["id"],
                        "framework": fw_name,
                        "model": case["model"],
                        "task": case["task"],
                        "latency_s": None,
                        "error": "dry-run",
                    }
                )
                continue

            result = run_single(case, fw_name, fw_cfg, port, log_dir, config)
            results.append(result)

            # Wait for GPU memory to clear
            print(f"  Waiting {GPU_CLEAR_WAIT}s for GPU memory to clear...")
            time.sleep(GPU_CLEAR_WAIT)

    output_data = {
        "timestamp": timestamp,
        "commit_sha": commit_sha,
        "run_id": run_id,
        "results": results,
    }

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults written to {output}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        lat = f"{r['latency_s']:.2f}s" if r["latency_s"] else r.get("error", "N/A")
        print(f"  {r['case_id']:30s} | {r['framework']:12s} | {lat}")

    return output_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Cross-framework diffusion serving comparison benchmark"
    )
    parser.add_argument(
        "--config",
        default=str(CONFIGS_PATH),
        help="Path to comparison_configs.json",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="Only run specific case IDs",
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=None,
        help="Only run specific frameworks (sglang, vllm-omni, lightx2v)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Server port",
    )
    parser.add_argument(
        "--output",
        default="comparison-results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and print commands without launching servers",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print(f"Loaded {len(config['cases'])} comparison case(s) from {args.config}")

    run_comparison(
        config=config,
        case_ids=args.case_ids,
        frameworks=args.frameworks,
        port=args.port,
        output=args.output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
