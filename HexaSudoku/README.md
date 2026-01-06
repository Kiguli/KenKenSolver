# HexaSudoku Solver

A neuro-symbolic solver for 16×16 Sudoku puzzles supporting both hexadecimal and numeric notations.

## Overview

HexaSudoku puzzles are 16×16 grids divided into 4×4 boxes. Values range from 1-16, displayed using two notation systems:

### Hex Notation (Default)
- **1-9**: Standard digits
- **A-G**: Letters for values 10-16

### Numeric Notation
- **1-9**: Standard digits
- **10-16**: Two-digit numbers

## Results

| Notation | CNN Classes | Extraction Accuracy | Solve Accuracy |
|----------|-------------|---------------------|----------------|
| Hex (A-G) | 17 | 100% | 100% |
| Numeric (10-16) | 17 | 100% | 100% |

Both notation systems achieve 100% accuracy on computer-generated puzzles.

## Pipeline

```
Image (1600×1600) → CNN (character recognition) → Z3 (constraint solving) → Solution
```

## Files

| File | Description |
|------|-------------|
| `SymbolicPuzzleGenerator.ipynb` | Generate 16×16 puzzles using Z3 |
| `BoardImageGeneration.ipynb` | Create 1600×1600 PNG board images (hex) |
| `generate_images.py` | Generate board images (hex notation) |
| `generate_images_numeric.py` | Generate board images (numeric notation) |
| `train_cnn.py` | Train CNN for hex notation (0-9 + A-G) |
| `train_cnn_numeric.py` | Train CNN for numeric notation (0-16) |
| `evaluate.py` | Evaluate hex notation pipeline |
| `evaluate_numeric.py` | Evaluate numeric notation pipeline |
| `NeuroSymbolicSolver.ipynb` | Main solver pipeline and evaluation |

## Directories

| Directory | Contents |
|-----------|----------|
| `puzzles/` | `puzzles_dict.json` - Generated puzzles |
| `board_images/` | PNG images - hex notation (1600×1600) |
| `board_images_numeric/` | PNG images - numeric notation (1600×1600) |
| `models/` | Trained CNN weights |
| `symbols/` | Training data CSV |
| `results/` | Evaluation results |

## Quick Start

### 1. Generate Puzzles
```bash
# Run SymbolicPuzzleGenerator.ipynb
# Generates 100 puzzles in puzzles/puzzles_dict.json
```

### 2. Generate Board Images
```bash
# Hex notation
python generate_images.py
# Creates board_images/board16_*.png

# Numeric notation
python generate_images_numeric.py
# Creates board_images_numeric/board16_*.png
```

### 3. Train CNN
```bash
# Hex notation
python train_cnn.py
# Trains models/hex_character_cnn.pth

# Numeric notation
python train_cnn_numeric.py
# Trains models/numeric_character_cnn.pth
```

### 4. Run Evaluation
```bash
# Hex notation
python evaluate.py
# Results: results/detailed_evaluation.csv

# Numeric notation
python evaluate_numeric.py
# Results: results/numeric_evaluation.csv
```

## Z3 Constraints

```python
# Cell range: 1 ≤ X[i][j] ≤ 16
# Row uniqueness: Distinct(row)
# Column uniqueness: Distinct(column)
# 4×4 Box uniqueness: Distinct(box)
```

## CNN Architecture

- Input: 28×28 grayscale images
- Output: 17 classes (0-16 values)
- Architecture: CNN_v2 (same as KenKen/Sudoku)

## Character Mapping

### Hex Notation

| Value | Display | CNN Class |
|-------|---------|-----------|
| 0 | (empty) | 0 |
| 1-9 | 1-9 | 1-9 |
| 10-16 | A-G | 10-16 |

### Numeric Notation

| Value | Display | CNN Class |
|-------|---------|-----------|
| 0 | (empty) | 0 |
| 1-9 | 1-9 | 1-9 |
| 10-16 | 10-16 | 10-16 |
