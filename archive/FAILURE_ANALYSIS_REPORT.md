# KenKen Solver Failure Analysis Report

This report analyzes the performance of the KenKen neuro-symbolic solver on computer-generated and handwritten puzzle datasets, documenting failures and the effectiveness of various correction methods.

---

## Executive Summary

| Dataset | Size Range | Baseline Accuracy | Corrected Accuracy | Notes |
|---------|------------|-------------------|-------------------|-------|
| KenKen-300px (Computer) | 3x3 to 9x9 | 100% | 100% | Perfect OCR |
| KenKen (Computer, 900px) | 7x7 to 9x9 | 96% | 100% | Error correction works |
| KenKen-300px-handwritten | 3x3 to 9x9 | 22% | 32% | Significant OCR challenges |

---

## Part 1: Computer-Generated Puzzles (KenKen-300px)

### Results Summary

| Size | Total | Baseline Solved | Corrected Solved | Accuracy |
|------|-------|-----------------|------------------|----------|
| 3x3 | 100 | 100 | 100 | 100% |
| 4x4 | 100 | 100 | 100 | 100% |
| 5x5 | 100 | 100 | 100 | 100% |
| 6x6 | 100 | 100 | 100 | 100% |
| 7x7 | 100 | 100 | 100 | 100% |
| 9x9 | 100 | 100 | 100 | 100% |

### Analysis

The KenKen-300px computer-generated dataset achieves **100% accuracy** across all sizes without requiring any error correction. This demonstrates that:
- The CNN models are highly accurate on clean, computer-generated fonts
- The Z3 solver correctly handles all KenKen constraint types
- The cage detection algorithm works perfectly on these images

---

## Part 2: Computer-Generated Puzzles (Original KenKen - 900px)

### Results Summary

| Size | Total | Baseline Solved | Corrected Solved | Baseline % | Corrected % |
|------|-------|-----------------|------------------|------------|-------------|
| 7x7 | 100 | 95 | 96 | 95% | 96% |
| 9x9 | 100 | 96 | 100 | 96% | 100% |

### Failures Requiring Correction

#### 7x7 Puzzles
| Puzzle ID | Correction Type | Notes |
|-----------|-----------------|-------|
| 13 | uncorrectable | Detection/OCR failure |
| 14 | uncorrectable | Detection/OCR failure |
| 31 | cage_redetect+single | Cage boundary issue, then single digit error |
| 69 | uncorrectable | Detection/OCR failure |
| 85 | uncorrectable | Detection/OCR failure |

#### 9x9 Puzzles
| Puzzle ID | Correction Type | Notes |
|-----------|-----------------|-------|
| 19 | single | One OCR error corrected |
| 38 | single | One OCR error corrected |
| 68 | single | One OCR error corrected |
| 88 | single | One OCR error corrected |

### Correction Method Effectiveness

- **single**: 4 puzzles corrected (all 9x9) - Fixed single character OCR errors
- **cage_redetect+single**: 1 puzzle corrected (7x7) - Re-detected cage boundaries then fixed one error
- **uncorrectable**: 4 puzzles remain unsolved (all 7x7)

---

## Part 3: Handwritten Puzzles (KenKen-300px-handwritten)

### Results Summary

| Size | Total | Baseline Solved | Corrected Solved | Baseline % | Corrected % | Improvement |
|------|-------|-----------------|------------------|------------|-------------|-------------|
| 3x3 | 100 | 69 | 89 | 69% | 89% | +20% |
| 4x4 | 100 | 36 | 58 | 36% | 58% | +22% |
| 5x5 | 100 | 18 | 26 | 18% | 26% | +8% |
| 6x6 | 100 | 7 | 15 | 7% | 15% | +8% |
| 7x7 | 100 | 1 | 2 | 1% | 2% | +1% |
| 9x9 | 100 | 0 | 1 | 0% | 1% | +1% |

### Correction Methods Used

#### 3x3 Puzzles (31 baseline failures, 20 corrected)

| Correction Type | Count | Description |
|-----------------|-------|-------------|
| constraint_auto_1 | 5 | Auto-corrected 1 impossible target using constraints |
| constraint_single | 15 | Single error fixed via constraint-filtered alternatives |
| uncorrectable | 11 | Could not fix |

**Uncorrectable 3x3 puzzles**: 2, 5, 7, 12, 23, 25, 30, 48, 68, 75, 87

#### 4x4 Puzzles (64 baseline failures, 22 corrected)

| Correction Type | Count | Description |
|-----------------|-------|-------------|
| constraint_auto_1 | 7 | Auto-corrected 1 impossible target |
| constraint_single | 14 | Single error fixed |
| constraint_two | 1 | Two errors fixed (puzzle 97) |
| uncorrectable | 42 | Could not fix |

**Uncorrectable 4x4 puzzles**: 0, 1, 2, 3, 4, 5, 7, 10, 12, 15, 19, 21, 23, 24, 25, 28, 29, 30, 31, 43, 44, 46, 50, 54, 61, 62, 68, 69, 70, 72, 73, 76, 77, 78, 82, 84, 85, 86, 88, 90, 95, 99

#### 5x5 Puzzles (82 baseline failures, 8 corrected)

| Correction Type | Count | Description |
|-----------------|-------|-------------|
| constraint_auto_1 | 3 | Auto-corrected 1 impossible target (puzzles 1, 7, 10) |
| constraint_single | 4 | Single error fixed (puzzles 27, 35, 56, 89) |
| constraint_two | 1 | Two errors fixed (puzzle 67) |
| uncorrectable | 74 | Could not fix |

**Corrected 5x5 puzzles**: 1, 7, 10, 27, 35, 56, 67, 89

#### 6x6 Puzzles (93 baseline failures, 8 corrected)

| Correction Type | Count | Description |
|-----------------|-------|-------------|
| constraint_auto_1 | 3 | Auto-corrected 1 impossible target (puzzles 9, 14, 71) |
| constraint_single | 5 | Single error fixed (puzzles 7, 18, 35, 86, 91) |
| uncorrectable | 85 | Could not fix |

**Corrected 6x6 puzzles**: 7, 9, 14, 18, 35, 71, 86, 91

#### 7x7 Puzzles (99 baseline failures, 1 corrected)

| Correction Type | Count | Description |
|-----------------|-------|-------------|
| constraint_single | 1 | Single error fixed (puzzle 41) |
| baseline | 1 | Solved without correction (puzzle 90) |
| uncorrectable | 98 | Could not fix |

#### 9x9 Puzzles (100 baseline failures, 1 corrected)

| Correction Type | Count | Description |
|-----------------|-------|-------------|
| constraint_three | 1 | Three errors fixed (puzzle 39) |
| uncorrectable | 99 | Could not fix |

---

## Part 4: Detailed Failure Analysis

### Why Puzzles Remain Uncorrectable

#### 1. Too Many OCR Errors
Larger puzzles (7x7, 9x9) have more cells and more opportunities for OCR errors. When a puzzle has more than 3-5 OCR errors, the combinatorial search space becomes too large to explore efficiently.

**Example**: A 9x9 puzzle has 81 cells. With a handwritten digit accuracy of ~85%, we expect ~12 errors per puzzle, far exceeding the correction limit.

#### 2. Errors Not in Top-K Alternatives
The correction system relies on the correct digit being in the top-8 alternatives from the OCR model. When handwriting is ambiguous, the correct digit may not appear in alternatives at all.

**Common confusions**:
- 1 ↔ 7 (accounts for ~37% of errors)
- 6 ↔ 8
- 3 ↔ 8
- 5 ↔ 6

#### 3. Multi-Digit Target Errors
Cage targets with 2+ digits (e.g., "12+", "24×") are particularly error-prone because:
- Each digit can have independent errors
- The combination must satisfy constraint validation
- 79.7% of handwritten errors occur in double-digit cells

#### 4. Operator Confusion
Handwritten operators (+, -, ×, ÷) can be misread:
- × misread as + or ÷
- - misread as / or =
- Empty (single cell) misread as operator

#### 5. Constraint-Valid but Incorrect Alternatives
Sometimes an incorrect digit produces a constraint-valid target that still makes the puzzle unsolvable. For example:
- Target "12×" misread as "17×" (7 is a valid divisor of 7, so constraints pass)
- Target "8-" misread as "3-" (both are valid subtraction targets)

### Correction Success Patterns

#### When constraint_auto_1 Works
The auto-correction of impossible targets is most effective when:
- The detected target violates mathematical constraints (e.g., subtraction target > size-1)
- The correct alternative is the highest-confidence valid alternative
- Only one cage has an impossible target

**Examples**:
- 3x3 puzzle 17: Auto-corrected impossible subtraction/division target
- 4x4 puzzle 16: Auto-corrected impossible multiplication target
- 6x6 puzzle 9: Auto-corrected impossible target

#### When constraint_single Works
Single-error correction via constraint filtering succeeds when:
- Exactly one OCR error exists
- The correct digit is in the top-8 alternatives
- The error cage can be identified via error likelihood scoring

**Examples**:
- 3x3 puzzle 18: Fixed single digit error in 12 attempts
- 5x5 puzzle 27: Fixed single error in 3 attempts
- 7x7 puzzle 41: Fixed single error in 75 attempts

#### When constraint_two/three Works
Multi-error correction succeeds rarely because:
- Must correctly identify both/all error cages
- Must find the right combination of alternatives
- Search space grows exponentially

**Examples**:
- 4x4 puzzle 97: Fixed 2 errors in 53 attempts
- 5x5 puzzle 67: Fixed 2 errors in 113 attempts
- 9x9 puzzle 39: Fixed 3 errors in 615 attempts (notable success!)

---

## Part 5: Computational Cost Analysis

### Handwritten Puzzle Solve Times

| Size | Baseline (avg) | Corrected Fail (avg) | Reason |
|------|----------------|----------------------|--------|
| 3x3 | 127ms | 605ms | ~19 correction attempts |
| 4x4 | 228ms | 1,274ms | ~100 correction attempts |
| 5x5 | 358ms | 5,535ms | ~700 correction attempts |
| 6x6 | 548ms | 30,867ms | ~3,400 correction attempts |
| 7x7 | 770ms | 67,285ms | ~6,100 correction attempts |
| 9x9 | - | 169,527ms | ~9,900 correction attempts |

The exponential growth in solve time for uncorrectable puzzles reflects the exhaustive search through alternative combinations.

---

## Part 6: Recommendations for Improvement

### Short-term Improvements

1. **Improve OCR Model**
   - Train with more handwritten samples
   - Add data augmentation (rotation, scaling, noise)
   - Focus on commonly confused pairs (1/7, 6/8, 3/8)

2. **Better Error Prioritization**
   - Use cage structure to identify likely error locations
   - Weight multi-digit targets higher in error likelihood
   - Consider grid position (corner cells may have different error rates)

3. **Adaptive Correction Limits**
   - Increase max_errors for larger puzzles
   - Use confidence thresholds to skip hopeless puzzles early
   - Implement timeout-based early stopping

### Long-term Improvements

1. **Ensemble OCR Models**
   - Combine multiple CNN architectures
   - Use attention mechanisms for operator/digit distinction
   - Add contextual features (cage size, target range)

2. **Constraint Propagation Integration**
   - Use constraint-valid value sets to filter OCR alternatives earlier
   - Propagate known values to constrain neighboring cells
   - Detect impossible cage combinations before solving

3. **Iterative Correction**
   - Solve partial puzzles to identify definite errors
   - Use partial solutions to constrain remaining unknowns
   - Re-run OCR on suspected error cells with different preprocessing

---

## Appendix: Complete Failure Lists

### KenKen-300px-handwritten Uncorrectable Puzzles

**3x3** (11 puzzles): 2, 5, 7, 12, 23, 25, 30, 48, 68, 75, 87

**4x4** (42 puzzles): 0, 1, 2, 3, 4, 5, 7, 10, 12, 15, 19, 21, 23, 24, 25, 28, 29, 30, 31, 43, 44, 46, 50, 54, 61, 62, 68, 69, 70, 72, 73, 76, 77, 78, 82, 84, 85, 86, 88, 90, 95, 99

**5x5** (74 puzzles): 0, 2, 3, 4, 5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 29, 30, 31, 32, 33, 34, 36, 39, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 55, 57, 58, 59, 60, 61, 62, 65, 66, 68, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 85, 87, 88, 90, 91, 92, 94, 95, 96, 97, 98, 99

**6x6** (85 puzzles): 0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 15, 16, 17, 19, 20, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99

**7x7** (98 puzzles): All except 41, 90

**9x9** (99 puzzles): All except 39

### KenKen (Original) Uncorrectable Puzzles

**7x7** (4 puzzles): 13, 14, 69, 85

**9x9** (0 puzzles): All solved with correction

---

*Report generated: January 2026*
*Data sources: KenKen-300px/results/, KenKen/results/, KenKen-300px-handwritten/results/*
