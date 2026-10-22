#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "Unwrap - Vision Model Setup & Verification Script"
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Checking Apple Vision OCR framework..."
python3 -c "import Vision, Cocoa; print('✓ Apple Vision OCR framework available')" 2>/dev/null || {
    echo "Installing pyobjc Vision dependencies..."
    pip install pyobjc-framework-Vision pyobjc-framework-Cocoa
    python3 -c "import Vision, Cocoa; print('✓ Apple Vision OCR installed successfully')"
}

echo ""
echo "Checking Local VLM Server connectivity..."
VLM_URL="${VLM_BASE_URL:-http://localhost:8080/v1}"
if curl -s -f -m 3 "$VLM_URL/models" >/dev/null 2>&1; then
    echo "✓ Local VLM server is RUNNING at $VLM_URL"
else
    echo "ℹ Local VLM server not detected at $VLM_URL (Apple Vision OCR will act as primary fallback)"
    echo "  To start local MLX VLM: python3 -m mlx_vlm.server --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit --port 8080"
fi

echo ""
echo "Running smoke test on sample food packaging image..."
python3 scripts/smoke_test.py
