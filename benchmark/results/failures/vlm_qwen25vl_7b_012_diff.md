# Word Diff for vlm_qwen25vl_7b on 012.jpeg

- **F1**: 0.0000
- **CER**: 0.1243
- **WER**: 0.1875
- **False Clears**: 9

### Reference (Ground Truth)
```
INGREDIENTS: Quinoa Flour, Potato Starch, Corn Starch, Sunflower And/Or Safflower Oil, Corn Flour, Organic Buttermilk, Salt, Sugar, Rice Flour, Organic Cheddar Cheese, Onion Powder, Natural Flavors, Organic Tapioca Syrup Solids, Organic Cream, Organic Whey, Yeast Extract, Lactic Acid, Extract of Paprika, Citric Acid, Garlic Powder, Extract of Turmeric.
```

### Hypothesis (Transcribed)
```
Quinoa Flour, Potato Starch, Corn Starch, Sunflower Oil And/or Safflower Oil, Corn Flour, Organic Buttermilk, Salt, Sugar, Rice Flour, Organic Cheddar Cheese, Onion Powder, Natural Flavors, Organic Paprika, Sym. Solids, Organic Cream, Organic Lactose, Organic Lactic Acid, Extract of Paprika, Citric Acid, Garlic Powder, Extract of Turmeric
```

### Word-level Diff
Legend: `- word` (missing), `+ word` (hallucinated/extra), `  word` (correct)

```
- INGREDIENTS:   Quinoa   Flour,   Potato   Starch,   Corn   Starch,   Sunflower + Oil - And/Or ?     ^
 + And/or ?     ^
   Safflower   Oil,   Corn   Flour,   Organic   Buttermilk,   Salt,   Sugar,   Rice   Flour,   Organic   Cheddar   Cheese,   Onion   Powder,   Natural   Flavors,   Organic - Tapioca - Syrup + Paprika, + Sym.   Solids,   Organic   Cream,   Organic + Lactose, + Organic - Whey, - Yeast - Extract,   Lactic   Acid,   Extract   of   Paprika,   Citric   Acid,   Garlic   Powder,   Extract   of - Turmeric. ?         -
 + Turmeric
```
