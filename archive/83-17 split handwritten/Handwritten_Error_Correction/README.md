# Handwritten Error Detection and Correction (83-17 Split)

This module implements error detection and correction for handwritten Sudoku and HexaSudoku puzzles. When the CNN misclassifies digits causing the Z3 solver to fail (UNSAT), the system identifies and corrects errors by leveraging CNN confidence scores.

> **Note**: This folder uses results from the **83-17 train/test split** (5000 train / 1000 test samples per class).
> See also: [90-10 split handwritten](../../90-10%20split%20handwritten/) for experiments with more training data.

## How It Works

### The Problem

When recognizing handwritten digits, the CNN occasionally misclassifies characters. Even a single misclassification can make a Sudoku puzzle unsolvable (UNSAT) because the constraints become contradictory.

### The Solution: Confidence-Based Alternative Substitution

The CNN doesn't just predict a class - it outputs a probability distribution over all possible classes via softmax. If the top prediction is wrong, the second-best prediction might be correct.

**Algorithm:**

1. **Extract puzzle** from image with full CNN probability distribution for each clue
2. **Try to solve** - if successful, done
3. **If UNSAT**, attempt **single-error correction**:
   - Sort clues by confidence (lowest first = most likely errors)
   - For each clue, substitute with second-best CNN prediction
   - Try to solve the modified puzzle
   - If solvable, we found the correction
4. **If single-error fails**, attempt **two-error correction**:
   - Consider pairs of low-confidence clues
   - Substitute both with their second-best predictions
   - Try to solve the modified puzzle
5. **Verify** the corrected solution against ground truth (for metrics only)

### Example

```
Cell at (3, 2): Extracted as 'B' (confidence: 0.42)
                Second-best: 'D' (confidence: 0.31)

Original puzzle with 'B' → UNSAT (contradictory constraints)
Modified puzzle with 'D' → SOLVED!

Correction found: B → D at position (3, 2)
```

## File Structure

```
Handwritten_Error_Correction/
├── README.md                 # This file
├── correct_errors.py         # Main evaluation script
├── puzzles/
│   ├── puzzles_dict.json         # Sudoku puzzles (4x4 and 9x9)
│   └── unique_puzzles_dict.json  # HexaSudoku puzzles (16x16)
└── results/
    ├── correction_results.csv    # Detailed per-puzzle results
    └── summary.txt               # Accuracy comparison summary
```

## Requirements

Same as the parent Handwritten_Sudoku and Handwritten_HexaSudoku folders:

```bash
pip install z3-solver torch torchvision pillow numpy
```

**Pre-requisites:**
- Trained CNN models (from parent folders)
- Generated board images (from parent folders)

## Usage

```bash
cd Handwritten_Error_Correction
python correct_errors.py
```

The script will:
1. Load the trained CNN models from parent folders
2. Process all Sudoku (4x4 and 9x9) puzzles
3. Process all HexaSudoku (16x16) puzzles
4. Attempt single-error then two-error correction for any failed puzzles
5. Save detailed results to CSV

## Key Insights

### Why This Works

1. **CNN uncertainty correlates with errors**: Misclassified digits typically have lower confidence scores
2. **Second-best is often correct**: Confusion happens between similar characters (7↔1, B↔D, G↔C)
3. **Z3 is fast**: Trying many solve attempts is computationally feasible

### Limitations

- **Handles up to TWO errors**: If 3+ digits are wrong, correction will fail
- **HexaSudoku challenge**: With 120+ clues, puzzles may have 3+ errors
- **Not guaranteed**: The second-best prediction isn't always the truth
- **Computational cost**: Two-error correction is O(n²) vs O(n) for single-error

### Results (83-17 Split with Two-Error Correction)

| Puzzle Type | Original | Single | Two | Final | Improvement |
|-------------|----------|--------|-----|-------|-------------|
| Sudoku 4x4 | 92% | 7 | 0 | **96%** | +4 puzzles |
| Sudoku 9x9 | 79% | 18 | 2 | **91%** | +12 puzzles |
| HexaSudoku 16x16 | 6% | 13 | 14 | **30%** | +24 puzzles |

### Comparison with 90-10 Split

| Puzzle Type | 83-17 Original | 83-17 Final | 90-10 Original | 90-10 Final |
|-------------|----------------|-------------|----------------|-------------|
| Sudoku 4x4 | 92% | 96% | 92% | **99%** |
| Sudoku 9x9 | 79% | 91% | 75% | **99%** |
| HexaSudoku 16x16 | 6% | 30% | 10% | **37%** |

**Key observations:**
- Single-error corrections: avg 1.3 attempts (4x4), 3.0 (9x9), 3.2 (16x16)
- Two-error corrections: avg 2.5 attempts (9x9), 30.7 (16x16)
- Confidence-based sorting means errors are typically found in first few attempts
- Two-error correction significantly helps HexaSudoku (17% → 30%)

## Algorithm Details

### Confidence Sorting

Low-confidence predictions are checked first because:
- They're more likely to be errors
- This minimizes average attempts needed
- Typical correction is found in 1-5 attempts

### Efficiency Limits

- **Sudoku**: All clues checked (4-25 clues per puzzle)
- **HexaSudoku**: Limited to 50 single-error attempts, 500 pair attempts

### Ground Truth Usage

**Important**: The ground truth (expected solution) is NEVER used during correction. It's only used afterward to verify whether our correction was accurate. The algorithm works purely from CNN output + Z3 constraint satisfaction.

## Reproducing Results

1. Ensure the handwritten puzzles have been generated:
   ```bash
   cd ../Handwritten_Sudoku
   python generate_images.py

   cd ../Handwritten_HexaSudoku
   python generate_images.py
   ```

2. Ensure CNN models are trained:
   ```bash
   cd ../Handwritten_Sudoku
   python train_cnn.py

   cd ../Handwritten_HexaSudoku
   python train_cnn.py
   ```

3. Run error correction evaluation:
   ```bash
   cd ../Handwritten_Error_Correction
   python correct_errors.py
   ```

4. Results will be saved to `results/correction_results.csv`
