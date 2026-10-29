# Word Diff for apple_vision_corrected on smoke_test_label.jpg

- **F1**: 0.6316
- **CER**: 0.4027
- **WER**: 0.5000
- **False Clears**: 0

### Reference (Ground Truth)
```
Wheat Flour (62%), Sugar, Palm Oil, Cocoa Powder, Emulsifier (Soy Lecithin, E322), Salt, Raising Agent (Sodium Bicarbonate), Natural Vanilla Flavour.
```

### Hypothesis (Transcribed)
```
CRISPY CHOCOLATE COOKIES INGREDIENTS Wheat Flour (62%), Sugar, Palm Oil Cocoa Powder, Emulsifier (Soy Lecithin, E322), Salt Raising Agent (Sodium Bicarbonate). Natural Vanilla Flavour. Contains Wheat Soy.
```

### Word-level Diff
Legend: `- word` (missing), `+ word` (hallucinated/extra), `  word` (correct)

```
+ CRISPY + CHOCOLATE + COOKIES + INGREDIENTS   Wheat   Flour   (62%),   Sugar,   Palm - Oil, ?    -
 + Oil   Cocoa   Powder,   Emulsifier   (Soy   Lecithin,   E322), - Salt, ?     -
 + Salt   Raising   Agent   (Sodium - Bicarbonate), ?             ^
 + Bicarbonate). ?             ^
   Natural   Vanilla   Flavour. + Contains + Wheat + Soy.
```
