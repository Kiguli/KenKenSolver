# Handwritten Sudoku Solver (83-17 Split)

A neuro-symbolic solver for Sudoku puzzles using **real handwritten digits** from the MNIST dataset.

> **Note**: This folder uses the **83-17 train/test split** (5000 train / 1000 test samples per class).
> See also: [90-10 split handwritten](../90-10%20split%20handwritten/) for experiments with more training data.

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
- **Split**: 60,000 train / 10,000 test

## Data Split Strategy

**Configuration**: 83-17 split (5000 train / 1000 test samples per class)

| Split | Source | Samples/Class | Purpose |
|-------|--------|---------------|---------|
| Training | MNIST train | 5000 | CNN model training |
| Test | MNIST test | 1000 | Board image generation |

The CNN **never sees** the exact handwritten samples that appear on the board images. This ensures the evaluation measures true generalization to unseen handwriting styles.

## Class Mapping

| CNN Class | Value | Display | Usage |
|-----------|-------|---------|-------|
| 0 | Empty | (blank) | Empty cells |
| 1-9 | 1-9 | 1-9 | Sudoku digits |

## Puzzle Sizes

| Size | Box Dimension | Cells | Board Resolution | Cell Size |
|------|---------------|-------|------------------|-----------|
| 4×4 | 2×2 | 16 | 900×900 px | 225 px |
| 9×9 | 3×3 | 81 | 900×900 px | 100 px |

## Quick Start

```bash
# Navigate to this directory
cd Handwritten_Sudoku

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

## Directory Structure

```
Handwritten_Sudoku/
├── README.md                  # This file
├── download_datasets.py       # Download MNIST via torchvision
├── train_cnn.py               # Train CNN on handwritten digits
├── generate_images.py         # Create board images using test samples
├── evaluate.py                # Full evaluation pipeline
├── datasets/                  # torchvision cache (auto-created)
├── handwritten_data/          # Prepared numpy arrays
│   ├── train_images.npy       # Training images (N, 28, 28)
│   ├── train_labels.npy       # Training labels (N,)
│   ├── test_images.npy        # Test images for boards
│   └── test_labels.npy        # Test labels
├── puzzles/
│   └── puzzles_dict.json      # 200 puzzles (100 per size, from Sudoku/)
├── board_images/              # Generated handwritten boards
│   ├── board4_0.png ... board4_99.png   # 100 4×4 boards
│   └── board9_0.png ... board9_99.png   # 100 9×9 boards
├── models/
│   └── handwritten_sudoku_cnn.pth   # Trained model weights
├── symbols/
│   └── handwritten_sudoku_data.csv  # Training data sample (TMNIST format)
└── results/
    └── detailed_evaluation.csv   # Evaluation metrics
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

## Results (83-17 Split)

| Size | Extraction Accuracy | Solve Rate | + Error Correction |
|------|---------------------|------------|-------------------|
| 4×4 | 99.50% | 92% | **96%** |
| 9×9 | 99.67% | 79% | **91%** |
| **Overall** | **99.58%** | **85.5%** | **93.5%** |

*Error correction (single + two-digit) results from [Handwritten_Error_Correction](../Handwritten_Error_Correction/)*

### Analysis

The **99.58% extraction accuracy** demonstrates excellent digit recognition on handwritten input. The solve rates (92% for 4×4, 79% for 9×9) are significantly better than Handwritten HexaSudoku (6%) because:

1. **Fewer classes**: Only 10 classes (0-9) vs 17 classes (with confusable letters A-G)
2. **No letter confusion**: MNIST digits are more distinct than EMNIST letters
3. **Fewer clues per puzzle**: 4×4 has ~4-5 clues, 9×9 has ~20-25 clues (vs 120+ for 16×16)
4. **Higher per-class accuracy**: All digit classes achieve >99.8% accuracy

Even with near-perfect recognition, a single misclassified clue can make the puzzle unsolvable or produce an incorrect solution, explaining why the solve rate is lower than extraction accuracy.

## Expected Performance

Handwritten Sudoku should achieve **higher accuracy** than Handwritten HexaSudoku because:
- **Fewer classes** (10 vs 17): Only digits 0-9, no letters
- **MNIST quality**: Well-established dataset with clear handwriting
- **No letter confusion**: No A/C/G or E/C confusion issues
- **Smaller puzzles**: 4×4 has only ~4-5 clues, 9×9 has ~20-25 clues (vs 120+ for 16×16)

## Technical Notes

1. **MNIST only**: No EMNIST needed (standard Sudoku only uses digits 1-9)

2. **Image convention - WHITE background, DARK digits**:
   - MNIST stores: ink=HIGH (255=white), background=LOW (0=black)
   - Board images need: ink=LOW (0=black), background=HIGH (255=white)
   - Inversion is applied during board generation

3. **Empty Cell Detection**: Uses mean intensity threshold (0.98)

4. **Two puzzle sizes**: 4×4 and 9×9, each with different box constraints

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
