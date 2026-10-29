# Word Diff for vlm_qwen3vl_8b on 012.jpeg

- **F1**: 0.0000
- **CER**: 0.3362
- **WER**: 0.3333
- **False Clears**: 0

### Reference (Ground Truth)
```
INGREDIENTS: Quinoa Flour, Potato Starch, Corn Starch, Sunflower And/Or Safflower Oil, Corn Flour, Organic Buttermilk, Salt, Sugar, Rice Flour, Organic Cheddar Cheese, Onion Powder, Natural Flavors, Organic Tapioca Syrup Solids, Organic Cream, Organic Whey, Yeast Extract, Lactic Acid, Extract of Paprika, Citric Acid, Garlic Powder, Extract of Turmeric.
```

### Hypothesis (Transcribed)
```
Quinoa Flour, Potato Starch, Corn Starch, Sunflower And/Or Safflower Oil, Corn Flour, Organic Cheddar Cheese, Onion Powder, Natural Flavors, Organic Tapioca Syrup Solids, Organic Cream, Organic Citric Acid, Garlic Powder, Extract of Paprika
```

### Word-level Diff
Legend: `- word` (missing), `+ word` (hallucinated/extra), `  word` (correct)

```
- INGREDIENTS:   Quinoa   Flour,   Potato   Starch,   Corn   Starch,   Sunflower   And/Or   Safflower   Oil,   Corn - Flour, - Organic - Buttermilk, - Salt, - Sugar, - Rice   Flour,   Organic   Cheddar   Cheese,   Onion   Powder,   Natural   Flavors,   Organic   Tapioca   Syrup   Solids,   Organic   Cream,   Organic - Whey, - Yeast - Extract, - Lactic - Acid, - Extract - of - Paprika,   Citric   Acid,   Garlic   Powder,   Extract   of - Turmeric. + Paprika
```
