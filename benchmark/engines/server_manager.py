"""Server manager for local MLX-VLM servers.

Handles starting, stopping, health polling, cold-start latency measurement,
and fallback if a model cannot be loaded (e.g., 16GB RAM constraint).
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import httpx
import psutil

logger = logging.getLogger(__name__)


def find_process_on_port(port: int = 8080) -> Optional[int]:
    """Find PID of process listening on the given port."""
    try:
        out = subprocess.check_output(
            ["lsof", "-t", f"-i:{port}"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            pids = [int(p) for p in out.split() if p.isdigit()]
            return pids[0] if pids else None
    except Exception:
        pass
    return None


def get_loaded_model_on_server(base_url: str = "http://127.0.0.1:8080") -> Optional[str]:
    """Check what model is currently loaded on the server."""
    try:
        r = httpx.get(f"{base_url}/v1/models", timeout=2.0)
        if r.status_code == 200:
            data = r.json()
            for m in data.get("data", []):
                if m.get("loaded", False):
                    return m.get("id")
            # If none explicitly loaded=True, return first id
            if data.get("data"):
                return data["data"][0].get("id")
    except Exception:
        pass
    return None


def stop_server_on_port(port: int = 8080, timeout: float = 10.0) -> bool:
    """Terminate any process listening on the port."""
    try:
        subprocess.run(["pkill", "-9", "-f", "mlx_vlm.server"], check=False, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
    except Exception:
        pass

    pid = find_process_on_port(port)
    if not pid:
        return True

    try:
        p = psutil.Process(pid)
        p.terminate()
        p.wait(timeout=timeout)
        logger.info("Stopped server on port %d (PID %d)", port, pid)
        return True
    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
        try:
            os.kill(pid, signal.SIGKILL)
            time.sleep(1.0)
            return True
        except Exception as e:
            logger.warning("Failed to kill server PID %d: %s", pid, e)
            return False


def is_model_cached(model_name: str) -> bool:
    """Check if complete model weights are present in local HuggingFace cache."""
    cache_hub = Path.home() / ".cache" / "huggingface" / "hub"
    repo_folder_name = "models--" + model_name.replace("/", "--")
    repo_dir = cache_hub / repo_folder_name
    if not repo_dir.exists():
        return False
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return False
    for snap in snapshots_dir.iterdir():
        if not snap.is_dir():
            continue
        idx_file = snap / "model.safetensors.index.json"
        if idx_file.exists():
            try:
                import json
                with open(idx_file, "r") as f:
                    idx = json.load(f)
                required = set(idx.get("weight_map", {}).values())
                if required and all((snap / r).exists() and (snap / r).stat().st_size > 0 for r in required):
                    return True
            except Exception:
                pass
        else:
            single = snap / "model.safetensors"
            if single.exists() and single.stat().st_size > 0:
                return True
    return False


class MLXVLMServerManager:
    """Orchestrates MLX-VLM server lifecycle."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.current_process: Optional[subprocess.Popen] = None
        self.current_model: Optional[str] = None
        self.last_cold_start_ms: Optional[int] = None

    def ensure_server(
        self,
        model_name: str,
        fallback_model: Optional[str] = None,
        force_restart_for_cold_start: bool = False,
        startup_timeout: int = 180,
    ) -> Tuple[str, int]:
        """Ensure the server is running with the specified model.

        Returns
        -------
        Tuple[str, int]
            (actual_model_name, cold_start_ms)
        """
        currently_loaded = get_loaded_model_on_server(self.base_url)

        if not force_restart_for_cold_start and currently_loaded == model_name:
            logger.info("Server already running with model %s", model_name)
            pid = find_process_on_port(self.port)
            self.current_model = model_name
            return model_name, self.last_cold_start_ms or 0

        # If model is not cached locally and fallback is available and cached, fall back immediately
        if not is_model_cached(model_name) and fallback_model and is_model_cached(fallback_model):
            logger.warning(
                "Model %s is not cached locally on disk. Falling back to cached model %s (16GB RAM constraint).",
                model_name,
                fallback_model,
            )
            model_name = fallback_model

        # Need to start or restart
        logger.info("Starting MLX server with model %s (force_restart=%s)...", model_name, force_restart_for_cold_start)
        stop_server_on_port(self.port)
        time.sleep(1.0)

        success, elapsed_ms = self._launch(model_name, timeout=startup_timeout)
        if success:
            self.current_model = model_name
            self.last_cold_start_ms = elapsed_ms
            return model_name, elapsed_ms

        if fallback_model:
            logger.warning("Failed to start %s. Falling back to %s...", model_name, fallback_model)
            stop_server_on_port(self.port)
            time.sleep(1.0)
            success_fb, elapsed_ms_fb = self._launch(fallback_model, timeout=startup_timeout)
            if success_fb:
                self.current_model = fallback_model
                self.last_cold_start_ms = elapsed_ms_fb
                return fallback_model, elapsed_ms_fb

        raise RuntimeError(f"Failed to start MLX server for model {model_name} (and fallback {fallback_model})")

    def _launch(self, model_name: str, timeout: int = 180) -> Tuple[bool, int]:
        """Launch mlx_vlm.server subprocess and wait until ready."""
        cmd = [
            sys.executable,
            "-m",
            "mlx_vlm.server",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--model",
            model_name,
        ]

        server_log_path = Path("/tmp/mlx_vlm_benchmark_server.log")
        self.server_log_file = open(server_log_path, "w", encoding="utf-8")
        start_time = time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            stdout=self.server_log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.current_process = proc

        # Poll /v1/models until ready
        poll_interval = 0.5
        deadline = time.time() + timeout
        ready = False

        while time.time() < deadline:
            # Check if process terminated prematurely
            ret = proc.poll()
            if ret is not None:
                self.server_log_file.flush()
                err_log = server_log_path.read_text(encoding="utf-8")[-1000:] if server_log_path.exists() else ""
                logger.error("mlx_vlm.server exited with code %s. Output:\n%s", ret, err_log)
                return False, 0

            try:
                r = httpx.get(f"{self.base_url}/v1/models", timeout=1.0)
                if r.status_code == 200:
                    data = r.json()
                    models = data.get("data", [])
                    # Check if model is reported
                    if any(m.get("id") == model_name or m.get("loaded", False) for m in models) or len(models) > 0:
                        ready = True
                        break
            except Exception:
                pass

            time.sleep(poll_interval)

        cold_start_ms = int((time.perf_counter() - start_time) * 1000)

        if not ready:
            logger.error("Timed out waiting for MLX server to start model %s", model_name)
            stop_server_on_port(self.port)
            return False, 0

        logger.info("MLX server ready with model %s in %d ms", model_name, cold_start_ms)
        return True, cold_start_ms

    def get_server_rss_mb(self) -> float:
        """Get RSS memory of server process in MB."""
        pid = find_process_on_port(self.port)
        if pid:
            try:
                return round(psutil.Process(pid).memory_info().rss / (1024 * 1024), 2)
            except Exception:
                pass
        return 0.0
