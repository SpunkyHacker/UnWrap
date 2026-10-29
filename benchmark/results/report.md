# Label OCR & VLM Transcription Benchmark Report

## Executive Summary
This benchmark evaluates on-device text transcription across **4 engines** on local Apple Silicon hardware without any cloud LLMs.

- **Safety Status**: Total False Clears across all evaluated engines = **516**.
> [!CAUTION]
> **SAFETY VIOLATION**: Detected 516 instance(s) of `FALSE_CLEAR` where allergens were expected but the engine output resulted in a CLEAR safety status!

## Overall Engine Performance (Default Config)

| Engine | CER (95% CI) | WER (95% CI) | Ingredient F1 (95% CI) | Allergen Acc | False Clears | Latency Mean (P95) | Peak RSS | Peak GPU | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `apple_vision_corrected` | 0.742 [0.582, 0.915] | 0.951 [0.738, 1.169] | 0.547 [0.481, 0.613] | 0.69 | **45** | 226 ms (660 ms) | 389 MB | 0 MB | 264.9 img/min |
| `apple_vision_raw` | 0.738 [0.563, 0.920] | 0.945 [0.721, 1.192] | 0.549 [0.480, 0.615] | 0.69 | **45** | 224 ms (651 ms) | 387 MB | 0 MB | 267.9 img/min |
| `vlm_qwen25vl_7b` | 0.626 [0.377, 0.915] | 0.750 [0.470, 1.075] | 0.792 [0.720, 0.853] | 0.81 | **51** | 15444 ms (50199 ms) | 1039 MB | 6657 MB | 3.9 img/min |
| `vlm_qwen3vl_8b` | 0.439 [0.283, 0.623] | 0.441 [0.327, 0.590] | 0.658 [0.594, 0.721] | 0.60 | **51** | 6307 ms (20480 ms) | 3602 MB | 4028 MB | 9.5 img/min |

## Efficiency & Resource Footprint

| Engine | Parameters | Model Size on Disk | Est. Total FLOPs / img | Cold-Start Latency |
| :--- | :---: | :---: | :---: | :---: |
| `apple_vision_corrected` | 50M | Built-in System | 50.00 GFLOPs* | 0 ms |
| `apple_vision_raw` | 50M | Built-in System | 50.00 GFLOPs* | 0 ms |
| `vlm_qwen25vl_7b` | 7.61B | Cached | 15.22 TFLOPs* | 5071 ms |
| `vlm_qwen3vl_8b` | 8.20B | Cached | 6.32 TFLOPs* | 3663 ms |

> \* *FLOPs are analytical estimates based on decoder (2 * params * tokens) + ViT patch encoder calculations.*

## Ablation Study

| Engine | Ablation | Latency Mean | Ingredient F1 | CER | False Clears |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `apple_vision_corrected` | `no_preproc` | 212 ms | 0.551 | 0.739 | 21 |
| `apple_vision_raw` | `no_preproc` | 211 ms | 0.554 | 0.735 | 21 |
| `vlm_qwen25vl_7b` | `max_dim_1024` | 16160 ms | 0.790 | 0.659 | 42 |
| `vlm_qwen25vl_7b` | `max_dim_1536` | 16197 ms | 0.790 | 0.659 | 42 |
| `vlm_qwen25vl_7b` | `no_preproc` | 15653 ms | 0.800 | 0.489 | 36 |
| `vlm_qwen3vl_8b` | `max_dim_1024` | 5777 ms | 0.678 | 0.334 | 48 |
| `vlm_qwen3vl_8b` | `max_dim_1536` | 5859 ms | 0.678 | 0.332 | 48 |
| `vlm_qwen3vl_8b` | `no_preproc` | 6802 ms | 0.709 | 0.617 | 66 |

## Visualizations
- Accuracy vs Latency: `accuracy_vs_latency.png`
- Latency Boxplots: `latency_boxplots.png`
- Peak Memory Footprint: `peak_memory_barchart.png`
- Per-Engine Error Tables: `table_<engine>_errors.png`
- Detailed Word-Level Diffs: `failures/*.md`

## Commands to Reproduce
```bash
# Run the complete deterministic benchmark:
python -m benchmark.run

# Run without ablations (quick mode):
python -m benchmark.run --no-ablations

# Run specific engines only:
python -m benchmark.run --engines apple_vision_raw apple_vision_corrected vlm_qwen25vl_7b
```
