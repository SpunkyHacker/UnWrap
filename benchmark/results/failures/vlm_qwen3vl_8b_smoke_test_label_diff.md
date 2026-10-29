# Word Diff for vlm_qwen3vl_8b on smoke_test_label.jpg

- **F1**: 0.8571
- **CER**: 0.0067
- **WER**: 0.0500
- **False Clears**: 0

### Reference (Ground Truth)
```
Wheat Flour (62%), Sugar, Palm Oil, Cocoa Powder, Emulsifier (Soy Lecithin, E322), Salt, Raising Agent (Sodium Bicarbonate), Natural Vanilla Flavour.
```

### Hypothesis (Transcribed)
```
Wheat Flour (62%), Sugar, Palm Oil, Cocoa Powder, Emulsifier (Soy Lecithin E322), Salt, Raising Agent (Sodium Bicarbonate), Natural Vanilla Flavour.
```

### Word-level Diff
Legend: `- word` (missing), `+ word` (hallucinated/extra), `  word` (correct)

```
  Wheat   Flour   (62%),   Sugar,   Palm   Oil,   Cocoa   Powder,   Emulsifier   (Soy - Lecithin, ?         -
 + Lecithin   E322),   Salt,   Raising   Agent   (Sodium   Bicarbonate),   Natural   Vanilla   Flavour.
```
