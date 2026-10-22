# OCR vs VLM Transcription Comparison Report
**Date**: 2026-10-21 22:09:39
**VLM Config**: `local` @ `http://localhost:8080/v1` (Model: `qwen3-vl-8b`)

| Image | Apple Vision Text | OCR Time | VLM Text | VLM Time | Word Diff (OCR vs VLM) |
| :--- | :--- | :---: | :--- | :---: | :--- |
| `smoke_test_label.png` | CRISPY CHOCOLATE COOKIES INGREDIENTS Wheat Flour (62%), Sugar, Palm Oil Cocoa Po... | 395ms | Wheat Flour (62%), Sugar, Palm Oil Cocoa Powder, Emulsifier (Soy Lecithin, E322)... | 179ms | [-CRISPY CHOCOLATE COOKIES INGREDIENTS-] |
| `test_label.png` | INGREDIENTS. Wheat Flour, Sugar, Palm Oil Salt Contains Wheat... | 65ms | Wheat Flour, Sugar, Palm Oil Salt Contains Wheat... | 82ms | [-INGREDIENTS.-] |
