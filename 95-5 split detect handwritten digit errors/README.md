# 95-5 Split Handwritten Digit Error Detection

This folder contains experiments with a 95-5 train/test split for handwritten puzzle recognition. The CNN training is **identical** to the 90-10 split - only the data split differs.

## Configuration

| Parameter | 90-10 Split | 95-5 Split |
|-----------|-------------|------------|
| Train samples per class | 5,400 | 5,700 |
| Test samples per class | 600 | 300 |
| Rotation | +/-5 deg | +/-5 deg |
| Translation | +/-2 px | +/-2 px |
| Scale | +/-10% | +/-10% |

## Results by Random Seed

### Seed 9505 (Different Test Set)

| Approach | Success Rate | Avg Solve Calls |
|----------|--------------|-----------------|
| **Constraint-Based** | 29% | 135.0 |
| **Top-K Prediction** | 42% | 389.1 |

| Metric | Value |
|--------|-------|
| Total Misclassifications | 256 |
| Avg Errors per Puzzle | 2.6 |
| Puzzles with 0 errors | 5% |
| CNN Validation Accuracy | 99.03% |

**True Digit Rank Distribution (Seed 9505)**:
| Rank | Count | Percentage |
|------|-------|------------|
| Rank 2-4 | 237 | 92.6% |
| Rank 5-10 | 19 | 7.4% |
| Rank 11+ | 0 | 0.0% |

**Top Confusions (Seed 9505)**:
| Confusion | Count | True Digit Avg Rank |
|-----------|-------|---------------------|
| G -> A | 30 | 2.3 |
| E -> C | 24 | 2.0 |
| 2 -> A | 11 | 2.0 |
| 5 -> 3 | 11 | 2.0 |
| B -> 4 | 10 | 4.0 |

### Seed 9010 (Same Test Set as 90-10)

| Approach | Success Rate | Avg Solve Calls |
|----------|--------------|-----------------|
| **Constraint-Based** | 29% | 183.2 |
| **Top-K Prediction** | 37% | 448.3 |

| Metric | Value |
|--------|-------|
| Total Misclassifications | 249 |
| Avg Errors per Puzzle | 2.5 |
| Puzzles with 0 errors | 5% |
| CNN Validation Accuracy | 99.11% |

**True Digit Rank Distribution (Seed 9010)**:
| Rank | Count | Percentage |
|------|-------|------------|
| Rank 2-4 | 238 | 95.6% |
| Rank 5-10 | 11 | 4.4% |
| Rank 11+ | 0 | 0.0% |

**Top Confusions (Seed 9010)**:
| Confusion | Count | True Digit Avg Rank |
|-----------|-------|---------------------|
| E -> C | 26 | 2.0 |
| G -> A | 18 | 2.0 |
| B -> G | 10 | 2.0 |
| C -> E | 10 | 2.0 |
| G -> C | 9 | 2.0 |

## Full Comparison

| Metric | 90-10 Split | 95-5 (Seed 9505) | 95-5 (Seed 9010) |
|--------|-------------|------------------|------------------|
| Random Seed | 9010 | 9505 | 9010 |
| Train samples/class | 5,400 | 5,700 | 5,700 |
| Test samples/class | 600 | 300 | 300 |
| HexaSudoku (Constraint) | 36% | 29% | 29% |
| HexaSudoku (Top-K) | **40%** | **42%** | 37% |
| Total Misclassifications | ~240 | 256 | 249 |
| CNN Validation Accuracy | ~99% | 99.03% | 99.11% |

## Key Findings

1. **Seed 9505 vs 9010 comparison**: With seed 9505, 95-5 achieves 42% (better than 90-10's 40%). With seed 9010, 95-5 achieves 37% (worse than 90-10's 40%).

2. **Test set composition matters more than split ratio**: The 3% difference between seeds (42% vs 37%) is larger than the effect of having 300 more training samples per class.

3. **More training data has marginal effect**: 5700 vs 5400 samples per class (5.5% more) doesn't significantly improve results.

4. **95.6% of errors rank 2-4 with seed 9010**: K=4 is appropriate for both test sets.

## Files

| File | Description |
|------|-------------|
| `download_datasets.py` | Downloads MNIST/EMNIST with 95-5 split |
| `train_cnn.py` | Trains CNN with standard augmentation (same as 90-10) |
| `generate_images.py` | Creates board images from test set |
| `detect_errors.py` | Constraint-based error correction (unsat core) |
| `predict_digits.py` | Top-K prediction error correction |
| `analyze_failures.py` | Failure analysis by digit rank |

## Running the Pipeline

```bash
# 1. Download datasets
python3 download_datasets.py

# 2. Train CNN
python3 train_cnn.py

# 3. Generate board images
python3 generate_images.py

# 4. Run error correction
python3 detect_errors.py      # Constraint-based
python3 predict_digits.py     # Top-K prediction

# 5. Analyze failures
python3 analyze_failures.py
```

## Conclusions

1. **Split ratio has minimal impact**: The difference between 90-10 and 95-5 is negligible compared to test set composition
2. **Random seed significantly affects results**: Different seeds select different samples, leading to 5% variation in results
3. **K=4 is appropriate**: 92-96% of errors have true digit in rank 2-4
4. **Letters E, G remain most problematic**: Confused with C, A respectively across all experiments
