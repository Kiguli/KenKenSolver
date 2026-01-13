# 90-10 Split Handwritten Digits Only

This experiment evaluates HexaSudoku (16x16) solving using **only MNIST digits** - no EMNIST letters. Values 10-16 are rendered as **two-digit numbers** (e.g., "16" = "1" + "6") rather than letters A-G.

## Key Difference from Letter-Based Approach

| Aspect | Letter-Based (A-G) | Digits Only (10-16) |
|--------|-------------------|---------------------|
| Values 10-16 | Single letter per cell | Two digits side-by-side |
| CNN classes | 17 (1-9, A-G, empty) | 11 (0-9, empty) |
| Data source | MNIST + EMNIST | MNIST only |
| Recognition | One prediction per cell | Two predictions for 10-16 |

## Results Summary

| Metric | Value |
|--------|-------|
| **Top-K Solve Rate** | **34%** |
| Detection Solve Rate | 22% |
| CNN Validation Accuracy | 99.62% |
| Total Extraction Errors | 310 across 100 puzzles |
| Avg Errors per Puzzle | 3.1 |

### Comparison with Letter-Based (90-10 Split)

| Approach | Solve Rate | Avg Errors |
|----------|------------|------------|
| Letters (A-G) | 40% | ~2.5 |
| **Digits Only** | **34%** | 3.1 |

## Error Analysis

### Error Distribution by Cell Type

| Cell Type | Count | Errors | Error Rate |
|-----------|-------|--------|------------|
| Single-digit (1-9) | 6,767 | 63 | 0.93% |
| Double-digit (10-16) | 5,233 | 247 | 4.72% |

### Double-Digit Error Breakdown

| Error Type | Count | Percentage |
|------------|-------|------------|
| Tens digit wrong | 159 | 64.4% |
| Ones digit wrong | 69 | 27.9% |
| Both digits wrong | 19 | 7.7% |

### Top Confusions

The most common error is **1→7 confusion** in the tens position:

| Confusion | Count | Note |
|-----------|-------|------|
| 15→75 | 19 | Tens: 1→7 |
| 11→17 | 18 | Ones: 1→7 |
| 10→70 | 16 | Tens: 1→7 |
| 16→76 | 16 | Tens: 1→7 |
| 11→71 | 15 | Tens: 1→7 |
| 14→74 | 15 | Tens: 1→7 |

The digit "1" is frequently misclassified as "7" when extracted from the left half of two-digit cells.

### True Rank Distribution

| Rank Range | Count | Percentage | Note |
|------------|-------|------------|------|
| Rank 2-4 | 271 | 87.4% | Correctable with Top-K |
| Rank 5-10 | 33 | 10.6% | Beyond K=4 |
| Rank 11+ | 6 | 1.9% | CNN fundamentally wrong |

## Key Insights

1. **Two-digit cells have 5x higher error rate** than single-digit cells (4.72% vs 0.93%)

2. **1→7 confusion dominates**: The handwritten digit "1" is often misrecognized as "7", especially when extracted from the smaller (40x40) regions of two-digit cells

3. **87% of errors are in top-4**: Most errors can theoretically be corrected with Top-K approach, but the constraint solver struggles with 3+ errors per puzzle

4. **More errors than letter-based**: 3.1 errors/puzzle vs ~2.5 for letters, likely due to:
   - Smaller digit regions (40x40 vs 70x70)
   - Similar appearance of "1" and "7" in small sizes
   - Two opportunities for error per cell instead of one

## File Structure

```
90-10 split handwritten digits only/
├── download_datasets.py    # MNIST download and 90-10 split
├── train_cnn.py            # Train 11-class CNN (0-9 + empty)
├── generate_images.py      # Two-digit rendering for 10-16
├── detect_errors.py        # Constraint-based error detection
├── predict_digits.py       # Top-K prediction correction
├── analyze_failures.py     # Error analysis
├── data/                   # MNIST train/test data
├── models/                 # Trained CNN
├── puzzles/                # Puzzle definitions
├── board_images/           # Generated 16x16 boards
└── results/                # Evaluation results
```

## Running the Experiment

```bash
# 1. Download MNIST data
python3 download_datasets.py

# 2. Train CNN
python3 train_cnn.py

# 3. Generate board images
python3 generate_images.py

# 4. Run evaluations
python3 detect_errors.py     # 22% solve rate
python3 predict_digits.py    # 34% solve rate

# 5. Analyze failures
python3 analyze_failures.py
```

## Cell Rendering

### Single Digit (1-9)
- Size: 70x70 pixels, centered in 100x100 cell
- Recognition: Standard MNIST classification

### Two Digits (10-16)
- Each digit: 40x40 pixels
- Left digit (tens): x=10, y=30
- Right digit (ones): x=50, y=30
- Recognition: Extract and classify each digit separately

### Cell Classification
Uses center ink density ratio to distinguish:
- Single digit: High center concentration (ratio > 1.7)
- Double digit: Ink spread across cell (ratio ≤ 1.7)

## Conclusion

The digits-only approach achieves **34% solve rate**, which is **lower than the 40%** achieved with letters (A-G). The main reasons are:

1. Two-digit cells have smaller individual digit regions (40x40 vs 70x70)
2. The 1→7 confusion is particularly problematic at small sizes
3. Each two-digit cell has two chances for recognition error

The letter-based approach benefits from larger character sizes and no need to combine multiple predictions per cell.
