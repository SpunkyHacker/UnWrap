# Word Diff for apple_vision_corrected on test_label.jpg

- **F1**: 0.2353
- **CER**: 0.7439
- **WER**: 0.8696
- **False Clears**: 0

### Reference (Ground Truth)
```
Enriched Flour (Wheat Flour, Niacin, Reduced Iron), Sugar, Palm Oil, Chocolate Chips (Sugar, Cocoa Mass, Cocoa Butter, Soy Lecithin), Salt, Leavening (Baking Soda).
```

### Hypothesis (Transcribed)
```
INGREDIENTS Wheat Flour, Sugar, Palm Oil Salt Contains Wheat
```

### Word-level Diff
Legend: `- word` (missing), `+ word` (hallucinated/extra), `  word` (correct)

```
+ INGREDIENTS - Enriched - Flour - (Wheat ? -
 + Wheat   Flour, - Niacin, - Reduced - Iron),   Sugar,   Palm - Oil, ?    -
 + Oil - Chocolate - Chips - (Sugar, - Cocoa - Mass, - Cocoa - Butter, - Soy - Lecithin), - Salt, ?     -
 + Salt + Contains + Wheat - Leavening - (Baking - Soda).
```
