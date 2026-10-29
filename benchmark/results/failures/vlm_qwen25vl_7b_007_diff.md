# Word Diff for vlm_qwen25vl_7b on 007.jpeg

- **F1**: 0.0000
- **CER**: 0.2362
- **WER**: 0.2459
- **False Clears**: 0

### Reference (Ground Truth)
```
INGREDIENTS: Dark chocolate (26%) (cocoa mass, sugar, cocoa butter, emulsifier (soya lecithin), flavouring), milk chocolate (25%) (sugar, cocoa butter, whole milk powder, cocoa mass, emulsifier (soya lecithin), flavouring), white chocolate (13%) (sugar, cocoa butter, whole milk powder, emulsifier (soya lecithin), flavouring), sugar, rapeseed oil, palm oil, glucose syrup, caramelised biscuits (wheat flour, sugar, rapeseed oil, palm oil, raising agent (sodium bicarbonate), soya flour, salt, cinnamon), coconut oil, cocoa mass, hazelnuts, raspberry purée, invert sugar syrup, cocoa butter, milk powder, milk fat, sea salt, lemon juice, cherry juice, acid (citric acid), milk solids, whole milk powder, coffee, emulsifier (soya lecithin), raspberry pieces, meringue pieces (sugar, shea oil, palm oil, wheat starch, dried egg albumen, sunflower lecithin), flavouring, butter (milk), salt, milk whey powder.
```

### Hypothesis (Transcribed)
```
Dark chocolate (26%) (cocoa mass, sugar, cocoa butter, emulsifier (soya lecithin), flavouring), milk chocolate (25%) (sugar, cocoa butter, whole milk powder, cocoa mass, emulsifier (soya lecithin), flavouring), white chocolate (13%) (sugar, cocoa butter, whole milk powder, emulsifier (soya lecithin), flavouring), sugar, rapeseed oil, palm oil, glucose syrup, caramelised biscuits (wheat flour, sugar, rapeseed oil, palm oil, raising agent (sodium bicarbonate), soya flour, salt, cinnamon), coconut oil, cocoa mass, hazelnuts, raspberry purée, invert sugar syrup, cocoa butter, milk powder, milk fat, sea salt, lemon juice, cherry juice, acid (citric acid), milk solids, whole milk powder, coffee, emulsifier (soya lecithin), raspberry pieces, meringue pieces (sugar, shea oil, palm oil, wheat starch, dried egg albumen, sunflower lecithin), flavouring, butter (milk), salt, milk whey powder. Milk chocolate contains cocoa solids 30% minimum, milk solids 20% minimum. Dark chocolate contains cocoa solids 55% minimum. White chocolate contains cocoa solids 26% minimum, milk solids 26% minimum.
```

### Word-level Diff
Legend: `- word` (missing), `+ word` (hallucinated/extra), `  word` (correct)

```
- INGREDIENTS:   Dark   chocolate   (26%)   (cocoa   mass,   sugar,   cocoa   butter,   emulsifier   (soya   lecithin),   flavouring),   milk   chocolate   (25%)   (sugar,   cocoa   butter,   whole   milk   powder,   cocoa   mass,   emulsifier   (soya   lecithin),   flavouring),   white   chocolate   (13%)   (sugar,   cocoa   butter,   whole   milk   powder,   emulsifier   (soya   lecithin),   flavouring),   sugar,   rapeseed   oil,   palm   oil,   glucose   syrup,   caramelised   biscuits   (wheat   flour,   sugar,   rapeseed   oil,   palm   oil,   raising   agent   (sodium   bicarbonate),   soya   flour,   salt,   cinnamon),   coconut   oil,   cocoa   mass,   hazelnuts,   raspberry   purée,   invert   sugar   syrup,   cocoa   butter,   milk   powder,   milk   fat,   sea   salt,   lemon   juice,   cherry   juice,   acid   (citric   acid),   milk   solids,   whole   milk   powder,   coffee,   emulsifier   (soya   lecithin),   raspberry   pieces,   meringue   pieces   (sugar,   shea   oil,   palm   oil,   wheat   starch,   dried   egg   albumen,   sunflower   lecithin),   flavouring,   butter   (milk),   salt,   milk   whey   powder. + Milk + chocolate + contains + cocoa + solids + 30% + minimum, + milk + solids + 20% + minimum. + Dark + chocolate + contains + cocoa + solids + 55% + minimum. + White + chocolate + contains + cocoa + solids + 26% + minimum, + milk + solids + 26% + minimum.
```
