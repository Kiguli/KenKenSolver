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

## Results (90-10 Split)

| Puzzle Type | Success Rate | Detection Accuracy | Avg Suspects | Avg Solve Calls |
|-------------|--------------|-------------------|--------------|-----------------|
| Sudoku 4x4 | **99%** | 98% | 0.1 | 2.1 |
| Sudoku 9x9 | **85%** | 89% | 0.9 | 2.7 |
| HexaSudoku 16x16 | **36%** | 56% | 24.5 | 20.9 |

### Comparison with Confidence-Based (90-10 split)

| Puzzle Type | Confidence-Based | Unsat Core | Speedup |
|-------------|-----------------|------------|---------|
| Sudoku 4x4 | 99% | **99%** | ~5x fewer solve calls |
| Sudoku 9x9 | 99% | 85% | ~20x fewer solve calls |
| HexaSudoku | 37% | **36%** | ~25x fewer solve calls |

### Key Observations

1. **4x4 Sudoku**: Same accuracy, much faster
2. **9x9 Sudoku**: Lower accuracy but dramatically fewer solve calls
3. **HexaSudoku**: Similar accuracy, 25x fewer solve calls

The lower 9x9 accuracy suggests some errors don't directly participate in constraint conflicts (they enable wrong solutions rather than causing UNSAT). The confidence-based approach can catch these through exhaustive search.

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
├── detect_errors.py          # Main implementation
├── puzzles/
│   ├── puzzles_dict.json         # Sudoku puzzles (4x4 and 9x9)
│   └── unique_puzzles_dict.json  # HexaSudoku puzzles (16x16)
└── results/
    ├── detection_results.csv     # Detailed per-puzzle results
    └── summary.txt               # Summary statistics
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
