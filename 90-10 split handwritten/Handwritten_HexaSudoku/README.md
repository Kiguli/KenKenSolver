# Handwritten HexaSudoku Solver (90-10 Split)

A neuro-symbolic solver for 16×16 Sudoku puzzles using **real handwritten characters** from MNIST and EMNIST datasets.

> **Note**: This folder uses the **90-10 train/test split** (5400 train / 600 test samples per class).
> See also: [83-17 split handwritten](../../83-17%20split%20handwritten/) for the original experiments.

## Overview

Unlike the standard HexaSudoku solver (which uses synthetic font-rendered characters), this version:
- **Trains** on real handwritten digits and letters
- **Tests** on unseen handwritten samples to evaluate true generalization
- Demonstrates the neuro-symbolic approach on realistic handwritten input

## Datasets Used

### MNIST (Digits 0-9)
- **Source**: http://yann.lecun.com/exdb/mnist/
- **Citation**: LeCun, Y., Cortes, C., & Burges, C. (1998). *The MNIST database of handwritten digits.*
- **Description**: 70,000 grayscale images (28×28) of handwritten digits
- **Usage**: Digits 1-9 for HexaSudoku values 1-9

### EMNIST-Letters (Letters A-Z)
- **Source**: https://www.nist.gov/itl/products-and-services/emnist-dataset
- **Citation**: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). *EMNIST: Extending MNIST to handwritten letters.* arXiv:1702.05373
- **Description**: 145,600 grayscale images (28×28) of handwritten letters
- **Usage**: Letters A-G filtered for HexaSudoku values 10-16

## Data Split Strategy

**Configuration**: 90-10 split (5400 train / 600 test samples per class)

| Split | Source | Samples/Class | Purpose |
|-------|--------|---------------|---------|
| Training | MNIST train + EMNIST train | 5400 | CNN model training |
| Test | MNIST test + EMNIST test | 600 | Board image generation |

The CNN **never sees** the exact handwritten samples that appear on the board images. This ensures the evaluation measures true generalization to unseen handwriting styles.

## Results (90-10 Split)

| Metric | Value | + Error Correction |
|--------|-------|-------------------|
| Extraction Accuracy | 99.06% | - |
| Solve Rate | 10% | **27%** |

*Error correction results from [Handwritten_Error_Correction](../Handwritten_Error_Correction/)*

### Comparison with 83-17 Split

| Metric | 83-17 | 90-10 |
|--------|-------|-------|
| Extraction Accuracy | 98.89% | **99.06%** |
| Original Solve Rate | 6% | **10%** |
| After Error Correction | 17% | **27%** |

The 90-10 split achieves significantly better results due to more training data, with a 67% relative improvement in original solve rate (6% → 10%) and 59% relative improvement after error correction (17% → 27%).

## Quick Start

```bash
# Navigate to this directory
cd "90-10 split handwritten/Handwritten_HexaSudoku"

# 1. Download and prepare datasets
python download_datasets.py

# 2. Train CNN on handwritten data
python train_cnn.py

# 3. Generate board images with handwritten test samples
python generate_images.py

# 4. Run full evaluation
python evaluate.py
```

## Pipeline Architecture

```
Board Image (1600×1600)
         ↓
   Cell Extraction (16×16 cells, 100×100 px each)
         ↓
   Resize to 28×28
         ↓
   CNN Recognition (17 classes)
         ↓
   Z3 Constraint Solver (Sudoku rules)
         ↓
   Solution (16×16 grid)
```

## Class Mapping

| CNN Class | Value | Display | Dataset Source |
|-----------|-------|---------|----------------|
| 0 | Empty | (blank) | Generated white cells |
| 1-9 | 1-9 | 1-9 | MNIST digits |
| 10-16 | 10-16 | A-G | EMNIST letters |

## CNN Architecture

Same architecture as standard HexaSudoku (CNN_v2):

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
Linear(128→17)
  ↓
Output: 17 classes
```

## Training Details

- **Augmentation**: Rotation (±5°), scaling (±10%), position jitter (±2px)
- **Empty cells**: Generated as near-white images with slight noise
- **Epochs**: 30
- **Batch size**: 64
- **Optimizer**: Adam with learning rate scheduling
- **Validation split**: 15%

## Per-Class Recognition Accuracy

Key recognition challenges (from evaluation):
- **Digits (1-9)**: 98.1% - 100% accuracy
- **Letter 'G' (92.1%)**: Often confused with '9', 'A', and '8'
- **Letter 'E' (95.8%)**: Often confused with 'C'
- **Letter 'D' (97.5%)**: Often confused with 'A'

These confusions stem from visual similarity in handwritten forms.

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

This project uses publicly available datasets:
- MNIST: Public domain
- EMNIST: Public domain (NIST)

## References

1. LeCun, Y., Cortes, C., & Burges, C. (1998). The MNIST database of handwritten digits.
2. Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: Extending MNIST to handwritten letters. arXiv:1702.05373.
