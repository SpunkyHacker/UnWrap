# Word Diff for vlm_qwen3vl_8b on 024.jpeg

- **F1**: 0.0000
- **CER**: 0.0473
- **WER**: 0.1552
- **False Clears**: 1

### Reference (Ground Truth)
```
Ingredients: Enriched Corn Meal (Corn Meal, Ferrous Sulfate, Niacin, Thiamin Mononitrate, Riboflavin, and Folic Acid), Vegetable Oil (Corn, Canola, and/or Sunflower Oil), Cheese Seasoning (Whey, Cheddar Cheese [Milk, Cheese Cultures, Salt, Enzymes], Canola Oil, Maltodextrin [Made From Corn], Salt, Whey Protein Concentrate, Monosodium Glutamate, Natural and Artificial Flavors, Lactic Acid, Citric Acid, Artificial Color [Yellow 6]), and Salt.
```

### Hypothesis (Transcribed)
```
Enriched Corn Meal (Corn Meal, Ferrous Sulfate, Niacin, Thiamin Mononitrate, Riboflavin, and Folic Acid), Vegetable Oil (Corn, Canola, and/or Sunflower Oil), Cheese Seasoning (Whey, Cheddar Cheese (Milk, Cheese Cultures, Salt, Enzymes), Canola Oil, Maltoextrin (Made from Corn), Salt, Whey Protein Concentrate, Monosodium Glutamate, Natural and Artificial Flavors, Lactic Acid, Citric Acid, Artificial Color (Yellow 6)), and Salt.
```

### Word-level Diff
Legend: `- word` (missing), `+ word` (hallucinated/extra), `  word` (correct)

```
- Ingredients:   Enriched   Corn   Meal   (Corn   Meal,   Ferrous   Sulfate,   Niacin,   Thiamin   Mononitrate,   Riboflavin,   and   Folic   Acid),   Vegetable   Oil   (Corn,   Canola,   and/or   Sunflower   Oil),   Cheese   Seasoning   (Whey,   Cheddar   Cheese - [Milk, ? ^
 + (Milk, ? ^
   Cheese   Cultures,   Salt, - Enzymes], ?        ^
 + Enzymes), ?        ^
   Canola   Oil, - Maltodextrin ?      -
 + Maltoextrin - [Made ? ^
 + (Made ? ^
 - From ? ^
 + from ? ^
 - Corn], ?     ^
 + Corn), ?     ^
   Salt,   Whey   Protein   Concentrate,   Monosodium   Glutamate,   Natural   and   Artificial   Flavors,   Lactic   Acid,   Citric   Acid,   Artificial   Color - [Yellow ? ^
 + (Yellow ? ^
 - 6]), ?  ^
 + 6)), ?  ^
   and   Salt.
```
