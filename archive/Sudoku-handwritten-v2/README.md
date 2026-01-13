# Sudoku-handwritten-v2

Attempt to apply the KenKen V2 CNN approach to Sudoku and HexaSudoku benchmarks.

## Key Finding: Domain Mismatch

**The KenKen V2 ImprovedCNN cannot be directly used for Sudoku/HexaSudoku puzzles.**

### Why It Doesn't Work

The KenKen V2 model was trained on characters **extracted from KenKen board images**. These characters have a specific visual style:
- Rendered with particular fonts/strokes
- Extracted using KenKen-specific cell detection
- 70% handwritten / 30% computer-generated mix from KenKen boards

The Sudoku/HexaSudoku puzzles use **MNIST-style handwritten digits** which have different visual characteristics:
- Different stroke patterns
- Different digit aspect ratios
- Different rendering style

When the KenKen V2 model is applied to Sudoku images:
- It predicts class 0 for almost all digits
- Confidence is high but predictions are wrong
- The model has learned KenKen-specific features that don't transfer

### Test Results

**Expected (V1 Sudoku model):**
```
Puzzle 0: [_, _, _, 2] [_, _, _, 4] [2, _, _, _] [3, _, _, _]
V1 predictions: 2(1.00), 4(1.00), 2(1.00), 3(1.00) - All correct
```

**Actual (KenKen V2 model):**
```
Puzzle 0: [_, _, _, 2] [_, _, _, 4] [2, _, _, _] [3, _, _, _]
V2 predictions: 0(0.67), 0(0.99), 0(0.96), 0(0.82) - All wrong
```

### Lesson Learned

The V2 approach of training on **board-extracted characters** only works when:
1. The training data comes from the **same puzzle type**
2. The **same rendering pipeline** is used for training and inference

Domain-specific training does not transfer between puzzle types.

## Options to Proceed

### Option 1: Train New V2 Models (Recommended)

Train separate ImprovedCNN models for each puzzle type:
- Extract characters from Sudoku board images
- Extract characters from HexaSudoku board images
- Apply 10x augmentation and Focal Loss training
- This would replicate the V2 methodology for each domain

### Option 2: Use Existing V1 Models

The V1 models already achieve excellent results:
- Sudoku 4x4: 99% with error correction
- Sudoku 9x9: 99% with error correction
- HexaSudoku Numeric: 58% with domain constraints

The V1 models were trained on MNIST/EMNIST which matches the source data used to generate the puzzle images.

### Option 3: Unified Multi-Domain Training

Train a single model on characters from ALL puzzle types:
- KenKen characters
- Sudoku characters
- HexaSudoku characters
- This would require significant data collection and training

## Files in This Directory

- `models/improved_cnn.py` - ImprovedCNN architecture (copied from KenKen V2)
- `models/improved_cnn_weights.pth` - KenKen V2 weights (domain-specific)
- `solver/solve_sudoku.py` - Sudoku solver (not functional with KenKen model)
- `solver/solve_hexasudoku_numeric.py` - HexaSudoku solver (not functional)

## Conclusion

To apply the V2 approach to Sudoku/HexaSudoku, new models must be trained on domain-specific data. The improved CNN architecture and training methodology can be reused, but the weights cannot transfer between puzzle domains.
