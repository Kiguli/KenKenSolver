# KenKen Handwritten Digit Evaluation

Evaluation of the KenKen neuro-symbolic solver on puzzles rendered with **handwritten MNIST digits** instead of computer-generated fonts.

## Results Summary

| Size | Baseline | Corrected | Improvement |
|------|----------|-----------|-------------|
| 3x3  | 65%      | 87%       | +22%        |
| 4x4  | 23%      | 60%       | +37%        |
| 5x5  | 11%      | 33%       | +22%        |
| 6x6  | 2%       | 11%       | +9%         |
| 7x7  | 0%       | 1%        | +1%         |
| 9x9  | 0%       | 0%        | +0%         |

**Key Finding**: Error correction provides significant improvement for smaller puzzles but cannot fully recover from the high error rate of handwritten digit recognition on larger puzzles.

## Comparison with Computer-Generated

| Size | Computer Baseline | Handwritten Baseline | Handwritten Corrected |
|------|-------------------|----------------------|-----------------------|
| 3x3  | 100%              | 65%                  | 87%                   |
| 4x4  | 100%              | 23%                  | 60%                   |
| 5x5  | 100%              | 11%                  | 33%                   |
| 6x6  | 100%              | 2%                   | 11%                   |
| 7x7  | 100%              | 0%                   | 1%                    |
| 9x9  | 100%              | 0%                   | 0%                    |

## Error Analysis

### Error Distribution by Size

| Size | Puzzles with Errors | Total Errors | Target | Operator | Cage Boundary |
|------|---------------------|--------------|--------|----------|---------------|
| 3x3  | 41%                 | 49           | 49     | 0        | 0             |
| 4x4  | 81%                 | 125          | 125    | 0        | 0             |
| 5x5  | 89%                 | 204          | 204    | 0        | 0             |
| 6x6  | 98%                 | 367          | 367    | 0        | 0             |
| 7x7  | 100%                | 664          | 573    | 4        | 87            |
| 9x9  | 100%                | 1338         | 1223   | 66       | 49            |

### Most Common Digit Confusions

| Expected | Detected | Count |
|----------|----------|-------|
| 9        | 4        | 100   |
| 9        | 7        | 85    |
| 7        | 1        | 70    |
| 6        | 0        | 36    |
| 9        | 1        | 34    |
| 8        | 6        | 30    |
| 5        | 7        | 27    |
| 4        | 7        | 24    |
| 8        | 1        | 22    |
| 5        | 3        | 21    |

The CNN struggles most with distinguishing:
- **9 vs 4/7/1**: Handwritten 9s often look like these digits
- **7 vs 1**: Similar stroke patterns
- **6 vs 0**: Closed loops can be confused
- **8 vs 6/1**: Multi-loop digit challenging

### Error Correction Types

The error correction pipeline uses multiple strategies:
- **Unsat Core Detection**: Identifies clues causing constraint conflicts
- **Top-K Alternatives**: Uses 2nd, 3rd, 4th best CNN predictions
- **Multi-error correction**: Can correct up to 3 errors (5 for 9x9)

However, for larger puzzles (7x7, 9x9), the cumulative effect of many digit recognition errors exceeds the correction capacity.

## Methodology

### Data Split (90-10)
- **Training**: 5,400 samples per digit class (0-9) from MNIST
- **Testing**: 600 samples per digit class (unseen during training)
- **Board generation**: Uses test samples only

### Pipeline
```
Board Image (900x900)
        |
    Grid_CNN --> Detect puzzle size (3-9)
        |
    OpenCV --> Extract cage boundaries
        |
   Unified CNN --> Read digits + operators (14 classes)
        |
   Z3 Solver --> Compute solution
        |
Error Correction --> Retry with alternatives if unsatisfiable
```

### Model Training
- **Architecture**: CNN_v2 (14 classes: digits 0-9 + operators add/div/mul/sub)
- **MNIST Preprocessing**: Inverted to match board convention (ink=LOW, background=HIGH)
- **Operator Augmentation**: 5,000 augmented samples per operator from PNG templates
- **Training**: 30 epochs, batch size 64, Adam optimizer
- **Validation Accuracy**: 99.39%

## Quick Start

```bash
# 1. Download MNIST and create 90-10 split
python download_datasets.py

# 2. Train unified 14-class CNN
python train_cnn.py

# 3. Generate board images with handwritten digits
python generate_images.py

# 4. Run baseline evaluation
python evaluate.py

# 5. Run with error correction
python detect_errors.py

# 6. Analyze failures
python analyze_failures.py
```

## File Structure

```
KenKen handwritten/
├── README.md                  # This file
├── download_datasets.py       # Download MNIST, create 90-10 split
├── train_cnn.py              # Train unified 14-class CNN
├── generate_images.py        # Generate board images with handwritten digits
├── evaluate.py               # Baseline evaluation (no error correction)
├── detect_errors.py          # Full error correction pipeline
├── analyze_failures.py       # Analyze errors and create visualizations
├── handwritten_data/         # MNIST split (54k train, 6k test)
├── models/
│   ├── grid_detection_model_weights.pth  # Copied from KenKen/
│   └── unified_kenken_cnn.pth            # Trained 14-class model
├── puzzles/
│   └── puzzles_dict.json     # Copied from KenKen/puzzles/
├── symbols/
│   └── operators/            # PNG templates (add, div, mul, sub)
├── board_images/             # 600 generated images (100 per size)
├── results/                  # Evaluation CSVs
│   ├── baseline_evaluation.csv
│   ├── baseline_summary.csv
│   ├── corrected_evaluation.csv
│   └── corrected_summary.csv
└── failure_analysis/         # Error visualizations
    ├── error_details.csv
    ├── summary.txt
    └── {size}x{size}_errors/ # Images with errors highlighted
```

## Key Insights

1. **CNN accuracy matters significantly**: Even 99%+ character recognition leads to many unsolvable puzzles due to cumulative errors in larger puzzles.

2. **Error correction helps but has limits**: Baseline accuracy drops dramatically for larger puzzles (0% for 7x7 and 9x9), and error correction can only partially recover.

3. **Target digits are the main error source**: 97%+ of errors are target digit misreads, not operator or cage boundary issues.

4. **Handwritten 9 is problematic**: The digit 9 accounts for 219 confusions alone (100 as 4, 85 as 7, 34 as 1).

5. **Puzzle size impacts correction feasibility**: Error correction works well for 3x3-4x4 (87%/60%) but fails for larger puzzles where too many errors accumulate.
