# Vision-Language Model (VLM) & OCR Setup Guide

This guide details exact setup instructions for running local vision models on Apple Silicon (MacBook Pro M5, 16GB RAM) and configuring the Unwrap transcription pipeline.

---

## 1. Hardware Architecture & Memory Policy
- **Target Machine**: Apple Silicon M5 (16GB unified RAM).
- **Hard Rule**: Only **ONE** model may be active in unified memory at any time.
- **Primary VLM**: `Qwen2.5-VL-7B-Instruct-4bit` / `Qwen3-VL-8B-Instruct` 4-bit quantization (~5.5 GB to 6.2 GB RAM).
- **Secondary Local Engine**: **Apple Vision OCR** (via macOS native `VNRecognizeTextRequest`). Uses negligible RAM (~50 MB), runs in < 200ms on Apple Silicon Neural Engine.

---

## 2. Option A: Native MLX Server (Recommended for Apple Silicon)

MLX is Apple Silicon's native unified-memory machine learning framework.

### Installation
```bash
# In the project root with your virtual environment active:
source .venv/bin/activate
pip install mlx mlx-vlm
```

### Launch the 4-bit VLM Server
```bash
# Starts an OpenAI-compatible API server on port 8080:
python3 -m mlx_vlm.server \
  --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
  --port 8080 \
  --host 127.0.0.1
```

### Fallback if 8B/7B is memory heavy:
```bash
python3 -m mlx_vlm.server \
  --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
  --port 8080
```

---

## 3. Option B: Ollama Server

### Installation
```bash
# Via Homebrew
brew install ollama

# Or download direct from https://ollama.ai/download/mac
```

### Pull and Run Model
```bash
# Start Ollama service
ollama serve

# In another terminal, pull the 4-bit quantized vision model:
ollama run qwen2.5-vl:7b
```

Ollama serves an OpenAI-compatible API at:
`http://localhost:11434/v1`

---

## 4. Secondary OCR: Native Apple Vision Framework

No external server or daemon required! Apple Vision is built directly into macOS.
It is installed in the project virtualenv via:
```bash
pip install pyobjc-framework-Vision pyobjc-framework-Cocoa
```
The Unwrap backend automatically falls back to Apple Vision whenever the local VLM server is not running, times out, or returns invalid JSON.

---

## 5. Configuration via `.env`

Copy `.env.example` to `.env` and set your preferred backend:

```ini
# Local MLX (Default)
VLM_BACKEND=local
VLM_BASE_URL=http://localhost:8080/v1
VLM_MODEL_NAME=mlx-community/Qwen2.5-VL-7B-Instruct-4bit
VLM_TIMEOUT_SECONDS=60

# Or Ollama:
# VLM_BACKEND=local
# VLM_BASE_URL=http://localhost:11434/v1
# VLM_MODEL_NAME=qwen2.5-vl:7b

# Remote Fallback Options (Optional):
# VLM_BACKEND=gemini
# GEMINI_API_KEY=your_gemini_api_key

# VLM_BACKEND=groq
# GROQ_API_KEY=your_groq_api_key
```

---

## 6. Smoke Test Verification

Run the automated smoke test script to verify both engines on a test image with exact timings:
```bash
source .venv/bin/activate
python3 scripts/smoke_test.py
```
