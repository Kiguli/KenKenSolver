# Handwritten Sudoku Solver (90-10 Split)

A neuro-symbolic solver for Sudoku puzzles using **real handwritten digits** from the MNIST dataset.

> **Note**: This folder uses the **90-10 train/test split** (5400 train / 600 test samples per class).
> See also: [83-17 split handwritten](../../83-17%20split%20handwritten/) for the original experiments.

## Overview

Unlike the standard Sudoku solver (which uses synthetic font-rendered digits), this version:
- **Trains** on real handwritten digits from MNIST
- **Tests** on unseen handwritten samples to evaluate true generalization
- Demonstrates the neuro-symbolic approach on realistic handwritten input
- Supports both 4×4 and 9×9 puzzle sizes

## Dataset Used

### MNIST (Digits 0-9)
- **Source**: http://yann.lecun.com/exdb/mnist/
- **Citation**: LeCun, Y., Cortes, C., & Burges, C. (1998). *The MNIST database of handwritten digits.*
- **Description**: 70,000 grayscale images (28×28) of handwritten digits
- **Usage**: Digits 1-9 for Sudoku values (digit 0 not used in solutions)

## Data Split Strategy

**Configuration**: 90-10 split (5400 train / 600 test samples per class)

| Split | Source | Samples/Class | Purpose |
|-------|--------|---------------|---------|
| Training | MNIST train | 5400 | CNN model training |
| Test | MNIST test | 600 | Board image generation |

The CNN **never sees** the exact handwritten samples that appear on the board images. This ensures the evaluation measures true generalization to unseen handwriting styles.

## Results (90-10 Split)

| Size | Extraction Accuracy | Solve Rate | + Error Correction |
|------|---------------------|------------|-------------------|
| 4×4 | 99.31% | 92% | **99%** |
| 9×9 | 99.64% | 75% | **99%** |
| **Overall** | **99.48%** | **83.5%** | **99%** |

*Error correction (single + two-digit) results from [Handwritten_Error_Correction](../Handwritten_Error_Correction/)*

### Comparison with 83-17 Split

| Size | 83-17 Solve | 90-10 Solve | 83-17 Corrected | 90-10 Corrected |
|------|-------------|-------------|-----------------|-----------------|
| 4×4 | 92% | 92% | 96% | **99%** |
| 9×9 | 79% | 75% | 91% | **99%** |

The 90-10 split with two-error correction achieves near-perfect accuracy (99%) for both puzzle sizes.

## Quick Start

```bash
# Navigate to this directory
cd "90-10 split handwritten/Handwritten_Sudoku"

# 1. Download and prepare MNIST dataset
python download_datasets.py

# 2. Train CNN on handwritten digits
python train_cnn.py

# 3. Generate board images with handwritten test samples
python generate_images.py

# 4. Run full evaluation
python evaluate.py
```

## Pipeline Architecture

```
Board Image (900×900)
         ↓
   Cell Extraction (size×size cells)
         ↓
   Resize to 28×28
         ↓
   CNN Recognition (10 classes)
         ↓
   Z3 Constraint Solver (Sudoku rules)
         ↓
   Solution (size×size grid)
```

## CNN Architecture

Same architecture as other solvers (CNN_v2):

```
Input: 28×28 grayscale
  ↓
Conv2d(1→32, 3×3) + ReLU + MaxPool(2×2)  → 14×14×32
  ↓
Conv2d(32→64, 3×3) + ReLU + MaxPool(2×2) → 7×7×64
  ↓
Flatten → 3136
  ↓
Linear(3136→128) + ReLU + Dropout(0.25)
  ↓
Linear(128→10)
  ↓
Output: 10 classes (0-9)
```

## Training Details

- **Augmentation**: Rotation (±5°), scaling (±10%), position jitter (±2px)
- **Empty cells**: Generated as near-white images with slight noise
- **Epochs**: 30
- **Batch size**: 64
- **Optimizer**: Adam with learning rate scheduling
- **Validation split**: 15%

## Dependencies

```
torch
torchvision
numpy
scipy
Pillow
z3-solver
```

Install with:
```bash
pip install torch torchvision numpy scipy pillow z3-solver
```

## License

This project uses the publicly available MNIST dataset (public domain).

## References

1. LeCun, Y., Cortes, C., & Burges, C. (1998). The MNIST database of handwritten digits.
