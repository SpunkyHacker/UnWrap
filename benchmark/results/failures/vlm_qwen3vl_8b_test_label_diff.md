# Word Diff for vlm_qwen3vl_8b on test_label.jpg

- **F1**: 0.4444
- **CER**: 0.7805
- **WER**: 0.8261
- **False Clears**: 0

### Reference (Ground Truth)
```
Enriched Flour (Wheat Flour, Niacin, Reduced Iron), Sugar, Palm Oil, Chocolate Chips (Sugar, Cocoa Mass, Cocoa Butter, Soy Lecithin), Salt, Leavening (Baking Soda).
```

### Hypothesis (Transcribed)
```
INGREDIENTS Wheat Flour, Sugar, Palm Oil, Salt.
```

### Word-level Diff
Legend: `- word` (missing), `+ word` (hallucinated/extra), `  word` (correct)

```
+ INGREDIENTS - Enriched - Flour - (Wheat ? -
 + Wheat   Flour, - Niacin, - Reduced - Iron),   Sugar,   Palm   Oil, - Chocolate - Chips - (Sugar, - Cocoa - Mass, - Cocoa - Butter, - Soy - Lecithin), - Salt, ?     ^
 + Salt. ?     ^
 - Leavening - (Baking - Soda).
```
