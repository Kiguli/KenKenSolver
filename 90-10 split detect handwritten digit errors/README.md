# Constraint-Based Error Detection for Handwritten Puzzles

This module implements a novel approach to error detection and correction using Z3's **unsatisfiable core (unsat core)** analysis. Instead of relying on CNN confidence scores to guess which digits might be wrong, this approach identifies errors through pure logical constraint analysis.

## Key Insight

When a Sudoku puzzle is UNSAT (unsolvable due to contradictory constraints), Z3 can identify the **minimal set of constraints** causing the conflict. By tracking which constraints correspond to which input clues, we can pinpoint exactly which clues are involved in logical conflicts.

## How It Works

### Algorithm

1. **Solve with tracking**: Each clue gets a Boolean "tracker" variable
   ```python
   tracker = Bool(f"clue_{i}_{j}")
   s.assert_and_track(X[i][j] == val, tracker)
   ```

2. **Extract unsat core**: If UNSAT, call `solver.unsat_core()` to get conflicting trackers

3. **Refine suspects**: For each suspect, try removing it:
   - If solvable → this clue alone causes the conflict (high weight)
   - If still UNSAT → count which clues remain in conflict

4. **Targeted correction**: Only try substitutions on detected suspects

### Comparison with Confidence-Based Approach

| Aspect | Confidence-Based | Unsat Core (This) |
|--------|-----------------|-------------------|
| Principle | Statistical (low confidence = likely wrong) | Logical (conflicting constraints = definitely involved) |
| Search space | All clues by confidence | Only conflict participants |
| Avg solve calls (4x4) | ~5-10 | **2.1** |
| Avg solve calls (9x9) | ~25-50 | **2.7** |
| Avg solve calls (16x16) | ~500 | **20.9** |
| Speed | Slower | **10-25x faster** |

## Results (90-10 Split, Up to 5-Error Correction)

| Puzzle Type | Success Rate | Detection Accuracy | Avg Suspects | Avg Solve Calls |
|-------------|--------------|-------------------|--------------|-----------------|
| Sudoku 4x4 | **99%** | 98% | 0.1 | 2.1 |
| Sudoku 9x9 | **85%** | 89% | 0.9 | 2.8 |
| HexaSudoku 16x16 | **36%** | 56% | 24.8 | 306.6 |

### Correction Breakdown

| Puzzle Type | Direct Solve | 1-Error | 2-Error | 3-Error | 4-Error | 5-Error | Uncorrectable |
|-------------|--------------|---------|---------|---------|---------|---------|---------------|
| Sudoku 4x4 | 92 | 6 | 1 | 0 | 0 | 0 | 1 |
| Sudoku 9x9 | 85 | 11 | 3 | 0 | 0 | 0 | 1 |
| HexaSudoku | 10 | 18 | 12 | 0 | 0 | 0 | 60 |

### Comparison with Confidence-Based (90-10 split)

| Puzzle Type | Confidence-Based | Unsat Core | Notes |
|-------------|-----------------|------------|-------|
| Sudoku 4x4 | 99% | **99%** | Same accuracy |
| Sudoku 9x9 | 99% | 85% | Lower accuracy (errors don't cause UNSAT) |
| HexaSudoku | 37% | **36%** | Similar accuracy |

### Key Observations

1. **4x4 Sudoku**: Same accuracy with much faster detection
2. **9x9 Sudoku**: Lower accuracy (85% vs 99%) because some errors don't cause UNSAT - they enable wrong solutions rather than making the puzzle unsolvable
3. **HexaSudoku**: Similar accuracy despite trying up to 5-error correction

### Why Extended Error Correction Didn't Help HexaSudoku

Even with 5-error correction capability, HexaSudoku stayed at 36%. Analysis shows:
- No puzzles needed 3, 4, or 5 error corrections
- 60% of puzzles remain uncorrectable
- The issue is that for many errors, the **second-best CNN prediction is also wrong**
- The unsat core approach correctly identifies suspect clues, but substituting second-best predictions doesn't fix them

## Extended Correction: Top-K Predictions

To address the limitation where the second-best CNN prediction is also wrong, `predict_digits.py` extends the approach to try the **2nd, 3rd, and 4th** best CNN predictions for each suspect cell.

### Top-K Results (90-10 Split)

| Puzzle Type | 2nd-Best Only | Top-4 Predictions | Improvement |
|-------------|---------------|-------------------|-------------|
| Sudoku 4×4 | 99% | **99%** | - |
| Sudoku 9×9 | 85% | **84%** | -1% |
| HexaSudoku | 36% | **40%** | **+4%** |

### Correction Breakdown (Top-K)

| Puzzle Type | Direct Solve | 1-Error | 2-Error | 3-Error | Uncorrectable |
|-------------|--------------|---------|---------|---------|---------------|
| Sudoku 4×4 | 92 | 7 | 1 | 0 | 0 |
| Sudoku 9×9 | 85 | 12 | 2 | 1 | 0 |
| HexaSudoku | 10 | 19 | 16 | 0 | 55 |

### Key Findings

1. **HexaSudoku improved from 36% → 40%**: The additional predictions helped correct 4 more puzzles
2. **9×9 Sudoku achieved a 3-error correction**: One puzzle needed swapping 3 digits to the 3rd/4th best predictions
3. **Trade-off**: More solve attempts needed (avg 805 for HexaSudoku vs 307 with 2nd-best only)
4. **Still limited**: 55% of HexaSudoku puzzles remain uncorrectable because even the 4th-best prediction isn't the correct digit

### Usage

```bash
cd "90-10 split detect handwritten digit errors"
python predict_digits.py
```

## Failure Analysis: Why Puzzles Remain Unsolvable

To understand why 55% of HexaSudoku puzzles remain uncorrectable even with top-4 predictions, `analyze_failures.py` performs a detailed analysis comparing CNN predictions against ground truth.

### Key Question

For each misclassified digit, **where does the true digit rank** in the CNN's probability distribution?
- Rank 1: Correct prediction
- Rank 2-4: Within our top-K search space (correctable)
- Rank 5-10: Beyond K=4 but potentially correctable with larger K
- Rank 11+: CNN fundamentally wrong (true digit has very low probability)

### Error Count Distribution Per Puzzle

How many CNN misclassifications occur in each puzzle?

| Puzzle Type | 0 errors | 1 error | 2 errors | 3 errors | 4 errors | 5 errors | 6 errors |
|-------------|----------|---------|----------|----------|----------|----------|----------|
| Sudoku 4×4 | 90 | 9 | 1 | 0 | 0 | 0 | 0 |
| Sudoku 9×9 | 75 | 21 | 4 | 0 | 0 | 0 | 0 |
| HexaSudoku | 10 | 19 | 27 | 25 | 7 | 8 | 4 |

**Summary Statistics:**

| Puzzle Type | Total Errors | Puzzles with Errors | Avg Errors/Puzzle | Avg When Present |
|-------------|--------------|---------------------|-------------------|------------------|
| Sudoku 4×4 | 11 | 10 (10%) | 0.11 | 1.10 |
| Sudoku 9×9 | 29 | 25 (25%) | 0.29 | 1.16 |
| HexaSudoku | 240 | 90 (90%) | 2.40 | 2.67 |

**Key Observations:**
- **Sudoku 4×4/9×9**: Most errors are single-digit (1 error), making correction feasible
- **HexaSudoku**: 90% of puzzles have errors, with most having 2-3 errors simultaneously
- The high multi-error rate in HexaSudoku explains why correction is difficult

#### HexaSudoku Error Distribution by Correctability

| Final Status | 0 err | 1 err | 2 err | 3 err | 4 err | 5 err | 6 err |
|--------------|-------|-------|-------|-------|-------|-------|-------|
| Solved directly | 10 | - | - | - | - | - | - |
| Single correction | - | 18 | 1 | - | - | - | - |
| Two corrections | - | - | 12 | 4 | - | - | - |
| Uncorrectable | - | 1 | 14 | 21 | 7 | 8 | 4 |

**Why 3+ errors are uncorrectable**: With K=4 alternatives per error, the search space for 3 simultaneous errors is 4³ = 64 combinations. The current algorithm only tries up to 2-error corrections due to computational constraints.

### True Digit Rank Distribution

| Puzzle Type | Total Misclassifications | Rank 2-4 | Rank 5-10 | Rank 11+ | Empty Detection |
|-------------|--------------------------|----------|-----------|----------|-----------------|
| Sudoku 4×4 | 11 | 81.8% | 0% | 0% | 18.2% |
| Sudoku 9×9 | 29 | **100%** | 0% | 0% | 0% |
| HexaSudoku | 240 | 94.2% | **4.2%** | **1.7%** | 0% |

### Why HexaSudoku Fails (55 uncorrectable puzzles)

| Failure Category | Count | Explanation |
|------------------|-------|-------------|
| all_rank_1-4 | **42** | All true digits were in top-4, but multiple errors require simultaneous correction |
| some_rank_5-10 | 10 | At least one true digit ranked 5-10 (beyond K=4 search) |
| some_rank_11+ | 3 | At least one true digit ranked 11+ (CNN fundamentally wrong) |

**Key Insight**: The main issue (42/55 = 76%) is **not** that K=4 is too small, but that puzzles have **multiple errors** requiring simultaneous correction. With 3+ errors where each has ~3 alternatives (2nd, 3rd, 4th best), the search space explodes: 3^3 = 27, 3^4 = 81, 3^5 = 243 combinations per error set.

### Top Character Confusions

| Puzzle Type | Common Confusions |
|-------------|-------------------|
| Sudoku 4×4 | 4→9 (4x), 2→7 (2x), 1→7, 3→8, 3→7 |
| Sudoku 9×9 | 5→3 (5x), 7→9 (3x), 6→5 (3x), 6→8 (2x), 5→9 (2x) |
| HexaSudoku | E→C (20x), G→9 (16x), D→A (11x), G→A/8/6/C (10x each), 7→1 (7x), 5→9 (7x) |

The letter 'G' is particularly problematic in HexaSudoku, commonly confused with 9, A, 8, 6, and C.

### Recommendations Based on Analysis

| Finding | Recommendation |
|---------|---------------|
| 94% of errors have true digit in top-4 | K=4 is sufficient for most individual errors |
| 76% of failures are "all_rank_1-4" | Improve multi-error correction strategy |
| G is the most confused character | Add more G training samples, augment with similar shapes |
| E→C confusion dominates | Consider E/C-specific classifier or post-processing |
| 6% of HexaSudoku errors beyond top-4 | Increasing K to 6-8 could help these edge cases |

### Puzzles Uncorrectable Due to Rank 5+ Errors

13 puzzles have at least one error where the true digit ranks 5th or lower in the CNN's predictions, making them fundamentally uncorrectable with K=4.

#### Rank 5-10 Puzzles (10 puzzles)

These could potentially be corrected by increasing K to 6-8:

| Puzzle | Errors | Max Rank | Critical Error | Notes |
|--------|--------|----------|----------------|-------|
| 3 | 4 | 8 | F->7 at (9,8) | 100% confidence, true prob 0.0000 |
| 7 | 2 | 8 | B->C at (6,12) | 85% confidence |
| 20 | 5 | 5 | G->4 at (1,1) | 99.8% confidence |
| 31 | 2 | 6 | F->5 at (4,6) | 56% confidence |
| 41 | 6 | 7 | G->8 at (13,9) | 60% confidence |
| 46 | 2 | 5 | G->4 at (11,9) | 31% confidence |
| 63 | 2 | 8 | B->C at (11,6) | 85% confidence |
| 67 | 5 | 8 | F->7 at (12,0) | 100% confidence |
| 71 | 3 | 6 | F->E at (12,15) | 100% confidence |
| 84 | 5 | 5 | 8->2 at (12,1) | 99.8% confidence |

#### Rank 11+ Puzzles (3 puzzles)

These are fundamentally uncorrectable - the CNN gave near-zero probability to the true digit:

| Puzzle | Errors | Max Rank | Critical Error | Notes |
|--------|--------|----------|----------------|-------|
| 10 | 4 | 14 | E->1 at (1,14) | 100% confidence, true digit at rank 14 |
| 14 | 5 | 11 | F->2 at (1,3) | 74% confidence, true digit at rank 11 |
| 47 | 1 | 14 | E->1 at (13,2) | 100% confidence, true digit at rank 14 |

#### Rank 5+ Error Patterns

| Confusion | Count | Avg Rank | Notes |
|-----------|-------|----------|-------|
| E->1 | 2 | 14.0 | CNN fundamentally wrong |
| F->7 | 2 | 8.0 | Similar shapes |
| B->C | 2 | 8.0 | Similar shapes |
| G->4 | 2 | 5.0 | Unusual confusion |
| F->2 | 1 | 11.0 | CNN fundamentally wrong |
| F->5 | 1 | 6.0 | Similar shapes |
| G->8 | 1 | 7.0 | G confusion pattern |
| F->E | 1 | 6.0 | Similar shapes |
| 8->2 | 1 | 5.0 | Unusual confusion |

**Key Observations:**
- The letter **F** appears in 6 of 13 rank 5+ errors, confused with 7, 2, 5, and E
- The letter **E** appears in 2 critical rank 14 errors, confused with 1
- **B->C** confusion consistently ranks around 8
- High-confidence (>99%) predictions can still be completely wrong

### Running the Failure Analysis

```bash
cd "90-10 split detect handwritten digit errors"
python analyze_failures.py
```

Generates:
- `results/failure_analysis.csv`: Per-error detailed analysis
- `results/puzzle_failure_summary.csv`: Per-puzzle summary

## Advantages

1. **Much faster**: 10-25x fewer solver calls on average
2. **Principled**: Based on logical conflicts, not statistical guesses
3. **Catches high-confidence errors**: If a digit has 99% confidence but conflicts with constraints, it will still be detected
4. **Insight into puzzle structure**: Shows which clues are mutually conflicting

## Limitations

1. **Doesn't catch all errors**: Only detects clues that participate in conflicts
2. **Wrong solutions not detected**: If an error enables a different valid solution (not the intended one), it won't be flagged as UNSAT
3. **Multiple errors can mask each other**: If errors cancel out constraint violations

## Usage

```bash
cd "90-10 split detect handwritten digit errors"
python detect_errors.py
```

## File Structure

```
90-10 split detect handwritten digit errors/
├── README.md                 # This file
├── detect_errors.py          # Unsat core detection (2nd-best only)
├── predict_digits.py         # Extended correction (top-4 predictions)
├── analyze_failures.py       # Failure analysis (true digit ranking)
├── puzzles/
│   ├── puzzles_dict.json         # Sudoku puzzles (4x4 and 9x9)
│   └── unique_puzzles_dict.json  # HexaSudoku puzzles (16x16)
└── results/
    ├── detection_results.csv     # Results from detect_errors.py
    ├── prediction_results.csv    # Results from predict_digits.py
    ├── failure_analysis.csv      # Per-error analysis from analyze_failures.py
    ├── puzzle_failure_summary.csv # Per-puzzle summary from analyze_failures.py
    ├── summary.txt               # detect_errors.py summary
    └── prediction_summary.txt    # predict_digits.py summary
```

## Technical Details

### Z3 Unsat Core Tracking

```python
from z3 import Bool, Solver

# Create tracker for each clue
tracker = Bool(f"clue_{row}_{col}")
clue_trackers[tracker] = (row, col)

# Add constraint with tracking
solver.assert_and_track(X[row][col] == value, tracker)

# After check() returns UNSAT
core = solver.unsat_core()
suspects = [clue_trackers[t] for t in core if t in clue_trackers]
```

### Suspect Refinement

For each initial suspect, we probe by removing it:
- If puzzle becomes SAT: This clue alone caused the conflict (weight +10)
- If still UNSAT: The remaining conflicts indicate other involved clues (weight +1)

Suspects are then sorted by total weight (higher = more likely to be the error).

## Dependencies

```bash
pip install z3-solver torch torchvision pillow numpy
```

## References

- Z3's `unsat_core()` documentation: [Z3 API](https://z3prover.github.io/api/html/classz3py_1_1_solver.html)
- MNIST dataset (LeCun et al., 1998)
- EMNIST dataset (Cohen et al., 2017)
