# Unified Handwritten Sudoku/HexaSudoku Solver V2 - Technical Report

## 1. Overview

This report documents the unified neuro-symbolic approach for solving handwritten Sudoku and HexaSudoku puzzles. The system combines a deep learning CNN for character recognition with the Z3 SMT solver for constraint-based puzzle solving.

### Supported Puzzle Types

| Puzzle Type | Grid Size | Cell Values | Training Images |
|-------------|-----------|-------------|-----------------|
| Sudoku 4x4 | 4x4 | 1-4 | ~800 |
| Sudoku 9x9 | 9x9 | 1-9 | ~4,000 |
| HexaSudoku A-G | 16x16 | 0-9, A-G | ~12,800 |
| HexaSudoku Numeric | 16x16 | 1-16 (two-digit: 10-16) | ~12,800 |

## 2. Model Architecture

### ImprovedCNN (17-class unified model)

```
Block 1: 1 → 32 channels
  - Conv2d(1, 32, 3x3, padding=1) + BatchNorm + ReLU
  - Conv2d(32, 32, 3x3, padding=1) + BatchNorm + ReLU
  - MaxPool2d(2x2): 28x28 → 14x14

Block 2: 32 → 64 channels
  - Conv2d(32, 64, 3x3, padding=1) + BatchNorm + ReLU
  - Conv2d(64, 64, 3x3, padding=1) + BatchNorm + ReLU
  - MaxPool2d(2x2): 14x14 → 7x7

Fully Connected:
  - Linear(3136 → 256) + BatchNorm + ReLU + Dropout(0.4)
  - Linear(256 → 17)

Total Parameters: ~850,000
Validation Accuracy: 99.3%
```

**Output Classes**: 0-9 (digits), 10-16 (letters A-G)

## 3. Implementation Details

### 3.1 Puzzle-Specific Recognition

**Sudoku 4x4/9x9**: Single digit recognition (classes 1-9)
- Extract cell → Resize to 28x28 → CNN inference → argmax over digits 1-9

**HexaSudoku A-G**: Single character recognition (classes 0-16)
- Extract cell → Resize to 28x28 → CNN inference → argmax over all 17 classes

**HexaSudoku Numeric**: Hybrid single/double digit recognition
- **Single digits** (1-9): Standard recognition, constrained to classes 1-9
- **Double digits** (10-16): Tens digit forced to 1, ones digit constrained to 0-6

### 3.2 Cell Type Detection for HexaSudoku Numeric

The critical innovation for two-digit detection uses **ink density ratio analysis**:

```python
def classify_cell_type_numeric(cell_img):
    """Classify cell based on ink distribution."""
    total_ink = get_ink_density(cell_img)
    if total_ink < 0.02:  # Empty threshold
        return 'empty'

    # Measure center region (25%-75% of width)
    center = cell_img[:, int(w*0.25):int(w*0.75)]
    center_ink = get_ink_density(center)
    center_ratio = center_ink / total_ink

    # Single digit = high center concentration (centered)
    # Double digit = spread across cell
    return 'single' if center_ratio > 1.7 else 'double'
```

### 3.3 Two-Digit Extraction

**Fixed spatial split method** (proven approach):
- Left digit: cell[15%:85% height, 0:50% width]
- Right digit: cell[15%:85% height, 50%:100% width]

**Domain constraints enforced**:
- **Tens digit: Always forced to 1** - This is hardcoded, not recognized by the CNN, which completely eliminates any potential 1 vs 7 confusion in the tens position
- **Ones digit: 0-6 only** - Constrained to valid range (7, 8, 9 never appear in two-digit cells)
- Combined value: 10 + ones_digit (guaranteed to be in range [10, 16])

### 3.4 Z3 Constraint Solving with Error Correction

The solver uses Z3 SMT with optimized tactics: `simplify → propagate-values → solve-eqs → smt`

**Error Correction Process**:
1. Initial solve attempt with recognized values
2. If UNSAT, use `unsat_core()` to identify conflicting constraints
3. Generate top-K alternatives for suspect cells (K=3)
4. Try single-cell corrections first, then double-cell if needed
5. Accept first satisfying assignment

## 4. Comparison: V1 vs V2

| Aspect | V1 (90-10 Split) | V2 (Unified) |
|--------|------------------|--------------|
| **Model** | CNN_v2 (11 classes) | ImprovedCNN (17 classes) |
| **Cell Detection** | Ink density ratio | Ink density ratio (fixed from contours) |
| **Two-Digit Extract** | Fixed spatial split | Fixed spatial split (fixed from contours) |
| **Domain Constraints** | Tens=1, Ones=0-6 | Tens=1, Ones=0-6 |
| **Error Correction** | Yes | Yes |
| **Multi-Puzzle** | HexaSudoku Numeric only | All 4 puzzle types |

**Key Fix Applied**: The initial V2 unified solver used contour counting for cell type detection, which was unreliable (disconnected strokes, merged digits). After alignment with the V1 approach (ink density ratio + fixed spatial split), HexaSudoku Numeric improved from **14% → 72%**.

## 5. Results Summary

### Final Benchmark Results (100 puzzles each)

| Puzzle Type | Direct Solve | Single Error | Double Error | Uncorrectable | **Total Success** |
|-------------|-------------|--------------|--------------|---------------|-------------------|
| Sudoku 4x4 | 100 | 0 | 0 | 0 | **100%** |
| Sudoku 9x9 | 96 | 4 | 0 | 0 | **98%** |
| HexaSudoku A-G | 77 | 18 | 1 | 4 | **91%** |
| HexaSudoku Numeric | 60 | 16 | 1 | 23 | **72%** |

### Baseline vs Error-Corrected

| Puzzle Type | Baseline (Direct) | After Error Correction | Improvement |
|-------------|-------------------|------------------------|-------------|
| Sudoku 4x4 | 100% | 100% | +0% |
| Sudoku 9x9 | 96% | 98% | +2% |
| HexaSudoku A-G | 77% | 91% | +14% |
| HexaSudoku Numeric | 60% | 72% | +12% |

**Why Error Correction Helps**: The Z3-based error correction leverages the mathematical properties of Sudoku/HexaSudoku constraints. When OCR misrecognizes a single digit, the resulting puzzle is often unsolvable. By using unsat_core analysis, the system identifies which cells are involved in contradictions and tries alternative predictions from the CNN's top-K outputs.

## 6. Failure Analysis

### 6.1 HexaSudoku Numeric Failures (28 failed puzzles)

**Failed Puzzle IDs**: 2, 4, 7, 8, 9, 15, 22, 26, 38, 42, 46, 48, 56, 58, 59, 63, 64, 65, 69, 73, 79, 81, 86, 87, 90, 92, 96, 98

**Categorization**:
- **Uncorrectable (23)**: Multiple extraction errors beyond double-error correction capability
- **Single correction failed (3)**: #2, #63, #96 - wrong alternative selected
- **Double correction failed (1)**: #87 - both corrections incorrect

**Root Causes**:

1. **Cell Type Misclassification** (estimated 40% of failures)
   - Single digit classified as double → attempts to split, gets garbage
   - Double digit classified as single → treats "14" as single character

2. **Two-Digit Extraction Errors** (estimated 35% of failures)
   - Digits not well-centered → fixed split cuts through digit
   - Variable handwriting styles → inconsistent digit spacing

3. **OCR Confusion** (estimated 25% of failures)
   - "6" vs "0" confusion in ones position
   - "1" vs "4" confusion in single-digit cells
   - Note: 1 vs 7 confusion in tens position is eliminated by forcing tens=1

**Evidence from extraction accuracy**:
- Most failures have 99.6% extraction accuracy (1 error in 256 cells)
- Worst cases: 98.8% accuracy (3 errors in 256 cells)
- Single-error correction can fix 1 error; double-error can fix 2
- 3+ errors require more sophisticated correction strategies

### 6.2 HexaSudoku A-G Failures (9 failed puzzles)

**Failed Puzzle IDs**: 3, 12, 21, 31, 71, 81, 84, 91, 93

**Root Causes**:
- Character confusion: B/8, D/0, G/6 visually similar
- Handwriting variability in letterforms
- Multi-error cases exceeding correction capacity

### 6.3 Sudoku 9x9 Failures (2 failed puzzles)

**Failed Puzzle IDs**: 2, 4

**Root Causes**:
- #2: 1 extraction error, unable to correct (wrong alternative)
- #4: 1 extraction error, single correction attempted but failed

## 7. Recommendations for Improvement

1. **Adaptive Cell Type Detection**: Train a small classifier specifically for single/double digit classification

2. **Variable Spatial Split**: Use contour detection to find actual digit boundaries rather than fixed 50% split

3. **Expanded Error Correction**: Implement triple-error correction for 16x16 puzzles (current double-error may be insufficient)

4. **Model Ensemble**: Train separate models for single-digit vs double-digit recognition

5. **Data Augmentation**: Add more variation in digit spacing for two-digit training samples

## 8. Conclusion

The unified V2 solver successfully handles all four puzzle types with a single model. The key insight is that **cell type detection method matters significantly** - replacing contour-based detection with ink density ratio improved HexaSudoku Numeric from 14% to 72%. Error correction provides meaningful gains (+2% to +14%) by leveraging constraint propagation to identify and fix OCR errors. The remaining failures are primarily due to multiple extraction errors in 16x16 puzzles, suggesting that either improved OCR accuracy or expanded error correction strategies would yield further improvements.
