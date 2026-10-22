"""Launch official mlx-vlm server on port 8080 with real Qwen2.5-VL model weights.

Loads real 4-bit weights into Apple Silicon Metal memory.
Zero mocks, zero canned responses, zero hidden OCR fallbacks.
"""
import subprocess
import sys

DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
DEFAULT_PORT = 8080


def launch_server(model: str = DEFAULT_MODEL, port: int = DEFAULT_PORT):
    """Run native mlx_vlm.server process."""
    cmd = [
        sys.executable,
        "-m",
        "mlx_vlm.server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        model,
    ]
    print(f"Starting native mlx-vlm server with real weights: {model} on port {port}...")
    subprocess.run(cmd)


if __name__ == "__main__":
    launch_server()
