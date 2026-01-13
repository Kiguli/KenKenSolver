# Handwritten HexaSudoku Solver (83-17 Split)

A neuro-symbolic solver for 16×16 Sudoku puzzles using **real handwritten characters** from MNIST and EMNIST datasets.

> **Note**: This folder uses the **83-17 train/test split** (5000 train / 1000 test samples per class).
> See also: [90-10 split handwritten](../../90-10%20split%20handwritten/) for experiments with more training data.

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
- **Split**: 60,000 train / 10,000 test

### EMNIST-Letters (Letters A-Z)
- **Source**: https://www.nist.gov/itl/products-and-services/emnist-dataset
- **Citation**: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). *EMNIST: Extending MNIST to handwritten letters.* arXiv:1702.05373
- **Description**: 145,600 grayscale images (28×28) of handwritten letters
- **Usage**: Letters A-G filtered for HexaSudoku values 10-16
- **Note**: EMNIST images are stored transposed and require rotation correction

## Data Split Strategy

**Configuration**: 83-17 split (5000 train / 1000 test samples per class)

| Split | Source | Samples/Class | Purpose |
|-------|--------|---------------|---------|
| Training | MNIST train + EMNIST train | 5000 | CNN model training |
| Test | MNIST test + EMNIST test | 1000 | Board image generation |

The CNN **never sees** the exact handwritten samples that appear on the board images. This ensures the evaluation measures true generalization to unseen handwriting styles.

## Class Mapping

| CNN Class | Value | Display | Dataset Source |
|-----------|-------|---------|----------------|
| 0 | Empty | (blank) | Generated white cells |
| 1-9 | 1-9 | 1-9 | MNIST digits |
| 10-16 | 10-16 | A-G | EMNIST letters |

## Quick Start

```bash
# Navigate to this directory
cd Handwritten_HexaSudoku

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

## Directory Structure

```
Handwritten_HexaSudoku/
├── README.md                  # This file
├── download_datasets.py       # Download MNIST + EMNIST via torchvision
├── train_cnn.py               # Train CNN on handwritten data
├── generate_images.py         # Create board images using test samples
├── evaluate.py                # Full evaluation pipeline
├── datasets/                  # torchvision cache (auto-created)
├── handwritten_data/          # Prepared numpy arrays
│   ├── train_images.npy       # Training images (N, 28, 28)
│   ├── train_labels.npy       # Training labels (N,)
│   ├── test_images.npy        # Test images for boards
│   └── test_labels.npy        # Test labels
├── puzzles/
│   └── unique_puzzles_dict.json  # 100 puzzles (from HexaSudoku)
├── board_images/              # Generated handwritten boards
│   ├── board16_0.png
│   ├── board16_1.png
│   └── ...
├── models/
│   └── handwritten_hex_cnn.pth   # Trained model weights
├── symbols/
│   └── handwritten_hex_data.csv  # Training data sample (TMNIST format)
└── results/
    └── detailed_evaluation.csv   # Evaluation metrics
```

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

## Results (83-17 Split)

| Metric | Value | + Error Correction |
|--------|-------|-------------------|
| Extraction Accuracy | 98.89% | - |
| Solve Rate | 6% | **30%** |

*Error correction (single + two-digit) results from [Handwritten_Error_Correction](../Handwritten_Error_Correction/)*

### Analysis

The **98.89% extraction accuracy** demonstrates strong character recognition performance on handwritten input. However, the **6% puzzle solve rate** reveals that HexaSudoku is highly sensitive to recognition errors - even a few misrecognized clues can make the puzzle unsolvable or produce incorrect solutions.

Key recognition challenges (per-class accuracy):
- **Digits (1-9)**: 97.4% - 100% accuracy
- **Letter 'G' (89.2%)**: Often confused with 'A' and 'C'
- **Letter 'E' (94.2%)**: Often confused with 'C'
- **Letter 'B' (96.2%)**: Often confused with 'D' and digit '6'

These confusions stem from visual similarity in handwritten forms. The neuro-symbolic approach successfully demonstrates generalization to handwritten input, but achieving high solve rates on 16x16 puzzles requires near-perfect character recognition (~99.5%+ per character for 120+ clues).

## Technical Notes

1. **EMNIST Orientation**: EMNIST images are stored transposed. The download script applies `.transpose(1, 2)` to correct this.

2. **Image Convention**:
   - Board images: WHITE background (255), DARK characters (0)
   - MNIST/EMNIST: ink=HIGH (255), background=LOW (0)
   - Inversion is applied during board generation

3. **Empty Cell Detection**: Uses mean intensity threshold (0.98). May need tuning for very faint handwriting.

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
