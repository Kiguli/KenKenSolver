# Handwritten Error Detection and Correction

This module implements error detection and correction for handwritten Sudoku and HexaSudoku puzzles. When the CNN misclassifies a digit causing the Z3 solver to fail (UNSAT), the system identifies and corrects single-digit errors by leveraging CNN confidence scores.

## How It Works

### The Problem

When recognizing handwritten digits, the CNN occasionally misclassifies characters. A single misclassification can make a Sudoku puzzle unsolvable (UNSAT) because the constraints become contradictory.

### The Solution: Confidence-Based Alternative Substitution

The CNN doesn't just predict a class - it outputs a probability distribution over all possible classes via softmax. If the top prediction is wrong, the second-best prediction might be correct.

**Algorithm:**

1. **Extract puzzle** from image with full CNN probability distribution for each clue
2. **Try to solve** - if successful, done
3. **If UNSAT**, sort clues by confidence (lowest first = most likely errors)
4. **For each clue** (starting with lowest confidence):
   - Substitute with second-best CNN prediction
   - Try to solve the modified puzzle
   - If solvable, we found the correction
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
4. Attempt correction for any failed puzzles
5. Save detailed results to CSV

## Output Format

### CSV Columns (correction_results.csv)

| Column | Description |
|--------|-------------|
| `puzzle_type` | "sudoku_4x4", "sudoku_9x9", or "hexasudoku_16x16" |
| `puzzle_idx` | Puzzle index (0-99) |
| `total_clues` | Number of given clues in the puzzle |
| `original_status` | "solved", "unsat", or "wrong_solution" |
| `original_extraction_accuracy` | % of cells matching ground truth |
| `num_misclassified_original` | Count of incorrectly recognized digits |
| `correction_attempted` | True/False - was correction tried? |
| `correction_found` | True/False - did substitution work? |
| `attempts_to_solve` | Number of substitutions tried |
| `error_row` | Row of corrected cell (if found) |
| `error_col` | Column of corrected cell (if found) |
| `original_value` | What CNN originally predicted |
| `corrected_value` | Second-best prediction that worked |
| `original_confidence` | CNN confidence for original prediction |
| `second_best_confidence` | CNN confidence for the corrected value |
| `time_first_solve_attempt` | Time for initial solve (seconds) |
| `time_per_correction_attempt` | Average time per substitution attempt |
| `time_total_all_attempts` | Total time including all correction attempts |
| `final_status` | "correct", "corrected", "uncorrectable", "wrong_correction" |
| `final_matches_ground_truth` | True/False - does final solution match expected? |
| `ground_truth_error_positions` | List of (row,col) where extraction differed from truth |

## Key Insights

### Why This Works

1. **CNN uncertainty correlates with errors**: Misclassified digits typically have lower confidence scores
2. **Second-best is often correct**: Confusion happens between similar characters (7↔1, B↔D, G↔C)
3. **Z3 is fast**: Trying 20-50 solve attempts is computationally feasible

### Limitations

- **Assumes exactly ONE error**: If multiple digits are wrong, the single-substitution approach won't help
- **HexaSudoku challenge**: With 120+ clues, puzzles often have 2+ errors, limiting correction success
- **Not guaranteed**: The second-best prediction isn't always the truth

### Results

| Puzzle Type | Original | Corrections Found | Final | Improvement |
|-------------|----------|-------------------|-------|-------------|
| Sudoku 4x4 | 92% | 7 | **96%** | +4 puzzles |
| Sudoku 9x9 | 79% | 18 | **90%** | +11 puzzles |
| HexaSudoku 16x16 | 6% | 13 | **17%** | +11 puzzles |

**Key observations:**
- Average attempts to find correction: 1.3 (4x4), 3.0 (9x9), 3.2 (16x16)
- Confidence-based sorting means errors are typically found in first few attempts
- HexaSudoku improvement is limited because most failures have 2+ misclassifications

## Algorithm Details

### Confidence Sorting

Low-confidence predictions are checked first because:
- They're more likely to be errors
- This minimizes average attempts needed
- Typical correction is found in 1-5 attempts

### Efficiency Limits

- **Sudoku**: All clues checked (4-25 clues per puzzle)
- **HexaSudoku**: Limited to 50 attempts per puzzle (out of 120+ clues)

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
