# 100-0 Split Handwritten Digit Error Detection

This folder contains experiments with a **100-0 train/test split** where board images use the same data the CNN was trained on.

## Key Insight

Unlike 90-10 and 95-5 splits where board images use held-out test data, this experiment uses **training data for board images**. This tests:
1. CNN performance when recognizing digits it has already seen (memorized)
2. The upper bound of the neuro-symbolic system's performance when perception errors are minimized

## Configuration

| Parameter | 90-10 Split | 95-5 Split | 100-0 Split |
|-----------|-------------|------------|-------------|
| Train samples per class | 5,400 | 5,700 | **ALL** (~6,500-7,900 digits, 5,600 letters) |
| Test samples per class | 600 | 300 | **SAME as training** |
| Board images use | Test data | Test data | **Training data** |
| CNN sees board digits during training | No | No | **Yes** |

## Results

### HexaSudoku 16x16

| Approach | Success Rate | Avg Solve Calls |
|----------|--------------|-----------------|
| **Constraint-Based** | 61% | 111.2 |
| **Top-K Prediction** | 68% | 260.6 |

### Breakdown

| Metric | Value |
|--------|-------|
| Total Puzzles | 100 |
| Direct Solve (0 errors) | 24% |
| 1-Error Corrections | 31 |
| 2-Error Corrections | 18 |
| Uncorrectable | 27 |
| Total Misclassifications | 145 |
| Avg Errors per Puzzle | 1.45 |
| CNN Validation Accuracy | 99.17% |

### True Digit Rank Distribution

| Rank | Count | Percentage |
|------|-------|------------|
| Rank 2-4 | 143 | 98.6% |
| Rank 5-10 | 2 | 1.4% |
| Rank 11+ | 0 | 0.0% |

### Top Confusions

| Confusion | Count | True Digit Avg Rank |
|-----------|-------|---------------------|
| G -> A | 14 | 2.1 |
| E -> C | 11 | 2.0 |
| B -> 8 | 6 | 2.8 |
| E -> F | 5 | 2.0 |
| A -> G | 4 | 2.0 |

## Comparison with 95-5 Split

| Metric | 95-5 Split (Seed 9010) | 100-0 Split |
|--------|------------------------|-------------|
| HexaSudoku (Constraint) | 29% | **61%** |
| HexaSudoku (Top-K) | 37% | **68%** |
| Total Misclassifications | 249 | **145** |
| Avg Errors per Puzzle | 2.5 | **1.45** |
| Direct Solve (0 errors) | 5% | **24%** |
| CNN Validation Accuracy | 99.11% | 99.17% |

## Key Findings

1. **42% fewer misclassifications** when using training data (145 vs 249)
   - Still not perfect recognition despite CNN having memorized these samples
   - Data augmentation during training creates variations that don't exactly match board images

2. **Nearly 2x improvement in solve rate** (68% vs 37% for Top-K)
   - Reducing errors from 2.5 to 1.45 per puzzle dramatically improves correction success
   - More puzzles (24% vs 5%) have zero errors and solve directly

3. **Same confusion patterns persist** (G->A, E->C)
   - These character pairs are visually similar regardless of whether CNN has memorized them
   - The confusions are inherent to the character shapes

4. **98.6% of errors still rank 2-4**
   - K=4 remains appropriate even with training data
   - No fundamental CNN failures (rank 11+)

5. **Upper bound is ~68% with current architecture**
   - Even with perfect familiarity, character confusions limit performance
   - Further improvements require better character discrimination or multi-character reasoning

## Files

| File | Description |
|------|-------------|
| `download_datasets.py` | Downloads MNIST/EMNIST, combines ALL data |
| `train_cnn.py` | Trains CNN with standard augmentation |
| `generate_images.py` | Creates board images from **training** data |
| `detect_errors.py` | Constraint-based error correction (unsat core) |
| `predict_digits.py` | Top-K prediction error correction |
| `analyze_failures.py` | Failure analysis by digit rank |

## Running the Pipeline

```bash
# 1. Download datasets (combines ALL data)
python3 download_datasets.py

# 2. Train CNN
python3 train_cnn.py

# 3. Generate board images (uses training data)
python3 generate_images.py

# 4. Run error correction
python3 detect_errors.py      # Constraint-based
python3 predict_digits.py     # Top-K prediction

# 5. Analyze failures
python3 analyze_failures.py
```

## Conclusions

1. **Using training data improves results significantly** (68% vs 37%)
2. **Recognition is still imperfect** due to data augmentation variations
3. **Character confusion patterns are inherent** to visual similarity, not memorization
4. **~68% represents the practical upper bound** for this CNN architecture on HexaSudoku
5. **Multi-error correction is essential** even with training data (76% of solvable puzzles needed correction)
